import torch
from typing import Dict, List, Tuple, Optional, Callable, Any

def get_transformer_layers(model):
    """Get transformer layers from various model architectures."""
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return model.model.language_model.model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'model'):
        return model.model.model.layers
    elif hasattr(model, 'model'):
        return model.model.layers
    elif hasattr(model, 'layers'):
        return model.layers
    else:
        raise AttributeError(f"Cannot find transformer layers in {type(model)}")

def categorize_tokens_text_fixed(actual_token_ids, actual_token_texts, n_shot, tokenizer, debug=True):
    """Categorize tokens into demonstration and query components."""
    token_categories = {
        'query_forerunner': None,
        'query_label': None
    }
    
    for d in range(n_shot):
        token_categories[f'demo_{d+1}_forerunner'] = None
        token_categories[f'demo_{d+1}_label'] = None

    answer_positions = []
    for i, token in enumerate(actual_token_texts):
        if token.strip().lower().startswith("answer"):
            answer_positions.append(i)
    
    if len(answer_positions) < n_shot + 1:
        return token_categories
    
    relevant_answer_positions = answer_positions[-(n_shot + 1):]
    
    for d in range(n_shot):
        answer_pos = relevant_answer_positions[d]

        for pos in range(answer_pos, min(answer_pos + 3, len(actual_token_texts))):
            if ':' in actual_token_texts[pos]:
                token_categories[f'demo_{d+1}_forerunner'] = pos
                
                label_pos = pos + 1
                while (label_pos < len(actual_token_texts) and 
                       actual_token_texts[label_pos].strip() in ['', ' ', '\n']):
                    label_pos += 1
                if label_pos < len(actual_token_texts):
                    token_categories[f'demo_{d+1}_label'] = label_pos
                break
    
    query_answer_pos = relevant_answer_positions[-1]
    for pos in range(query_answer_pos, min(query_answer_pos + 3, len(actual_token_texts))):
        if ':' in actual_token_texts[pos]:
            token_categories['query_forerunner'] = pos
            
            label_pos = pos + 1
            while (label_pos < len(actual_token_texts) and 
                   actual_token_texts[label_pos].strip() in ['', ' ', '\n']):
                label_pos += 1
            token_categories['query_label'] = label_pos if label_pos < len(actual_token_texts) else len(actual_token_texts)
            break
    
    return token_categories

class HookManager:
    """Manages model hooks for extracting activations and attention weights."""
    def __init__(self):
        self.hooks = []
        self.activations = {}
        self.attention_weights = {}
        
    def register_activation_hook(self, module, name):
        """Register a hook to capture activations."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.activations[name] = hidden_states.detach().cpu()
        
        h = module.register_forward_hook(hook)
        self.hooks.append(h)
        return h
    
    def register_attention_hook(self, module, name):
        """Register a hook to capture attention weights."""
        def hook(module, input, output):
            if hasattr(output, 'attentions') and output.attentions is not None:
                self.attention_weights[name] = output.attentions.detach().cpu()
            elif isinstance(output, (tuple, list)) and len(output) > 1 and output[1] is not None:
                self.attention_weights[name] = output[1].detach().cpu()
        
        h = module.register_forward_hook(hook)
        self.hooks.append(h)
        return h
    
    def create_attention_extraction_hook(self, layer_idx, store_outputs=True):
        """Create a hook function for extracting attention weights and optionally outputs."""
        attention_data = []
        
        def hook_fn(module, inputs, outputs):
            if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
                attn_output = outputs[0]
                attn_weights = outputs[1]
                
                if attn_weights is not None and hasattr(attn_weights, 'shape'):
                    weights = attn_weights[0].detach().cpu()
                    
                    data = {'weights': weights, 'layer_idx': layer_idx}
                    
                    if store_outputs:
                        batch_size, seq_len, hidden_size = attn_output.shape
                        num_heads = attn_weights.shape[1]
                        head_dim = hidden_size // num_heads
                        
                        reshaped_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)
                        outputs_per_head = reshaped_output[0].permute(1, 0, 2).detach().cpu()
                        data['outputs'] = outputs_per_head
                    
                    attention_data.append(data)
            
            return outputs
        
        return hook_fn, attention_data
    
    def clear(self):
        """Remove all hooks and clear stored data."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self.attention_weights = {}

def extract_embeddings_and_tokens(model, tokenizer, language_model, prompt, capture_hook_class=None):
    """Extract actual embeddings and tokens from model generation."""
    
    class CaptureHook:
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
    
    capture = capture_hook_class() if capture_hook_class else CaptureHook()
    capture.orig_generate = model.model.generate
    model.model.generate = capture.generate_hook
    
    transformer_layers = get_transformer_layers(language_model)
    hook = transformer_layers[0].register_forward_pre_hook(capture.first_layer_pre_hook())
    
    try:
        gen_cfg = dict(max_new_tokens=1, do_sample=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.model.device)
        model.model.generate(**inputs, **gen_cfg)
        
    finally:
        model.model.generate = capture.orig_generate
        hook.remove()
    
    actual_token_ids = capture.captured_input_ids[0].tolist()
    actual_token_texts = [tokenizer.decode([tid]) for tid in actual_token_ids]
    actual_embeddings = capture.captured_inputs_embeds.detach().clone()
    
    return actual_embeddings, actual_token_ids, actual_token_texts

def register_layer_hooks(language_model, layers_to_extract: Optional[List[int]] = None, 
                        hook_manager: Optional[HookManager] = None) -> HookManager:
    """Register hooks on transformer layers for activation extraction."""
    if hook_manager is None:
        hook_manager = HookManager()
    
    transformer_layers = get_transformer_layers(language_model)
    
    if layers_to_extract is None:
        layers_to_extract = list(range(len(transformer_layers)))
    
    for layer_idx in layers_to_extract:
        if layer_idx >= len(transformer_layers):
            continue
        
        layer = transformer_layers[layer_idx]
        hook_manager.register_activation_hook(layer, f'layer_{layer_idx}')
        if hasattr(layer, 'self_attn'):
            hook_manager.register_attention_hook(layer.self_attn, f'attn_{layer_idx}')
    
    return hook_manager