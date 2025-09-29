#!/usr/bin/env python3
import sys
import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_factory import create_model
from tasks.t2t_tasks import OperatorInductionTextTask
from utils.prompt_utils import build_vl_icl_prompt
from utils.model_utils import get_model_tokenizer, get_language_model
from interp.helper import categorize_tokens_text_fixed, extract_embeddings_and_tokens, get_transformer_layers, HookManager


# categorize_tokens_text_fixed is now imported from helper.py


class CaptureForAttention:
    def __init__(self):
        self.captured_input_ids = None
        self.captured_inputs_embeds = None

    def generate_hook(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is not None:
            self.captured_input_ids = input_ids
        return self.orig_generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def first_layer_pre_hook(self):
        def hook_fn(module, inputs):
            if isinstance(inputs, (tuple, list)) and len(inputs) > 0:
                hidden_states = inputs[0]
                if hidden_states is not None:
                    self.captured_inputs_embeds = hidden_states.detach()
            return None
        return hook_fn

def get_actual_embeddings_and_tokens(model, tokenizer, language_model, prompt, debug=True):
    return extract_embeddings_and_tokens(model, tokenizer, language_model, prompt, CaptureForAttention)


def extract_attention_with_correct_hooks(language_model, inputs_embeds, attention_mask, debug=True):
    transformer_layers = get_transformer_layers(language_model)
    
    n_layers = len(transformer_layers)
    
    attention_weights = []  
    attention_outputs = [] 
    
    def create_attention_hook(layer_idx):
        def hook_fn(module, inputs, outputs):
            if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
                attn_output = outputs[0]  # [batch, seq_len, hidden_size]
                attn_weights = outputs[1]  # [batch, num_heads, seq_len, seq_len]
                
                if attn_weights is not None and hasattr(attn_weights, 'shape'):
                    weights = attn_weights[0].detach().cpu()
                    attention_weights.append(weights)
                    
                    batch_size, seq_len, hidden_size = attn_output.shape
                    num_heads = attn_weights.shape[1]
                    head_dim = hidden_size // num_heads
                    
                    # Reshape: [batch, seq_len, hidden] -> [batch, seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
                    reshaped_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)
                    outputs_per_head = reshaped_output[0].permute(1, 0, 2).detach().cpu()  # [num_heads, seq_len, head_dim]
                    attention_outputs.append(outputs_per_head)
            
            return outputs
        return hook_fn
    
    hooks = []
    for layer_idx, layer in enumerate(transformer_layers):
        if hasattr(layer, 'self_attn'):
            hook = layer.self_attn.register_forward_hook(create_attention_hook(layer_idx))
            hooks.append(hook)
    
    try:
        original_attention_type = None
        if hasattr(language_model.config, '_attn_implementation'):
            original_attention_type = language_model.config._attn_implementation
            language_model.config._attn_implementation = 'eager'
        
        with torch.no_grad():
            outputs = language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_attentions=True)
        
        if original_attention_type is not None:
            language_model.config._attn_implementation = original_attention_type
        
        return attention_weights, attention_outputs
        
    finally:
        for hook in hooks:
            hook.remove()


def run_qwen3_attention_extraction(model_name="Qwen/Qwen3-8B", 
                                  data_dir="../VL-ICL", 
                                  n_shot=4, 
                                  num_samples=2,
                                  debug=True):
    
    model = create_model("qwen3", model_name)
    tokenizer = get_model_tokenizer(model, "qwen3")
    language_model = get_language_model(model, "qwen3")
    task = OperatorInductionTextTask(data_dir)
    
    config = language_model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // n_heads

    all_samples = task.query_data[:num_samples]
    results = []
    
    for sample_idx, query in enumerate(all_samples):
        try:
            demonstrations = task.select_demonstrations(query, n_shot=n_shot)
            prompt = build_vl_icl_prompt(task, demonstrations, query, mode="constrained")
            
            actual_embeddings, actual_token_ids, actual_token_texts = get_actual_embeddings_and_tokens(
                model, tokenizer, language_model, prompt, debug=debug
            )
            
            token_categories = categorize_tokens_text_fixed(
                actual_token_ids, actual_token_texts, n_shot, tokenizer, debug=debug
            )
            
            attention_mask = torch.ones((1, actual_embeddings.shape[1]), dtype=torch.long, device=actual_embeddings.device)
            
            attention_weights, attention_outputs = extract_attention_with_correct_hooks(
                language_model, actual_embeddings, attention_mask, debug=debug
            )
            print()
            
            success = True
            if len(attention_weights) != n_layers:
                success = False
            
            if attention_weights and debug:
                qf_pos = token_categories.get('query_forerunner')
                demo_positions = []
                for d in range(n_shot):
                    demo_pos = token_categories.get(f'demo_{d+1}_forerunner')
                    if demo_pos is not None:
                        demo_positions.append(demo_pos)
                
                if qf_pos is not None and demo_positions:
                    for layer_idx in range(min(5, len(attention_weights))): 
                        layer_weights = attention_weights[layer_idx]  # [n_heads, seq_len, seq_len]
                        layer_outputs = attention_outputs[layer_idx]  # [n_heads, seq_len, head_dim]
                        
                        for head_idx in range(min(3, n_heads)):
                            if qf_pos < layer_weights.shape[1]:
                                qf_to_demos = layer_weights[head_idx, qf_pos, demo_positions]  
                                qf_output = layer_outputs[head_idx, qf_pos, :]
                                
                                demo_attn_str = ", ".join([f"{a:.4f}" for a in qf_to_demos])
                                output_norm = torch.norm(qf_output).item()
                                
                                print(f"[LAYER {layer_idx:2d}] Head {head_idx:2d}: QF->demos [{demo_attn_str}], output_norm {output_norm:.3f}")
                    
                    for layer_idx in range(min(10, len(attention_weights))):
                        if qf_pos < attention_weights[layer_idx].shape[1] and demo_positions[0] < attention_weights[layer_idx].shape[2]:
                            attn_val = attention_weights[layer_idx][0, qf_pos, demo_positions[0]].item()
                            print(f"[LAYER {layer_idx:2d}] {attn_val:.6f}")
            
            sample_result = {
                'sample_idx': sample_idx,
                'query': query,
                'token_categories': token_categories,
                'actual_token_ids': actual_token_ids,
                'actual_token_texts': actual_token_texts,
                'actual_embeddings': actual_embeddings,
                'attention_weights': attention_weights,  # [layer][head, seq_len, seq_len]
                'attention_outputs': attention_outputs,  # [layer][head, seq_len, head_dim]
                'success': success
            }
            results.append(sample_result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
    
    return results


def main():
    try:
        results = run_qwen3_attention_extraction(
            model_name="Qwen/Qwen3-8B",
            data_dir="../VL-ICL", 
            n_shot=4,
            num_samples=60,
            debug=True
        )
        return results
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()