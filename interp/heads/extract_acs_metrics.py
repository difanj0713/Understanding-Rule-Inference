import os
import sys
import torch
import numpy as np
import pickle
import random
import re
from typing import Dict, List, Tuple
from collections import defaultdict
import traceback
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from models.model_factory import create_model
from tasks.t2t_tasks import OperatorInductionTextTask
from utils.model_utils import get_model_tokenizer, get_language_model
from interp.helper import categorize_tokens_text_fixed

def get_embeddings_text(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.model.device)
    input_ids = inputs.input_ids
    token_texts = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
    
    with torch.no_grad():
        embedding_layer = model.model.model.embed_tokens
        embeddings = embedding_layer(input_ids)
    
    return embeddings, input_ids[0].tolist(), token_texts

def extract_attention_weights_and_outputs(language_model, embeddings, attention_mask):
    attention_weights = []
    attention_outputs = []
    
    def create_hook(layer_idx):
        def hook_fn(module, input_tuple, output):
            attn_output = output[0]  # [batch, seq_len, hidden_size]
            attn_weights = output[1]  # [batch, num_heads, seq_len, seq_len] 
            
            # Store weights: [num_heads, seq_len, seq_len]
            attention_weights.append(attn_weights[0].detach().cpu())
            
            # Reshape attn_output to [num_heads, seq_len, head_dim]
            batch_size, seq_len, hidden_size = attn_output.shape
            num_heads = attn_weights.shape[1]
            head_dim = hidden_size // num_heads
            
            reshaped_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)
            reshaped_output = reshaped_output.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
            attention_outputs.append(reshaped_output[0].detach().cpu())  # [num_heads, seq_len, head_dim]
            
            return output
        return hook_fn
    
    hooks = []
    for layer_idx, layer in enumerate(language_model.layers):
        hook = layer.self_attn.register_forward_hook(create_hook(layer_idx))
        hooks.append(hook)
    
    try:
        with torch.no_grad():
            outputs = language_model(inputs_embeds=embeddings, attention_mask=attention_mask, output_attentions=True)
    finally:
        for hook in hooks:
            hook.remove()
    
    return attention_weights, attention_outputs


def extract_numbers_from_text_question(question):
    numbers = re.findall(r'\d+', question)
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    return None, None

def compute_answer(num1, num2, operator):
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == 'x':
        return num1 * num2
    return 0


def create_corrupted_demonstration(demo, corruption_operator):
    corrupted_demo = demo.copy()
    
    num1, num2 = extract_numbers_from_text_question(demo['question'])
    if num1 is not None and num2 is not None:
        corrupted_answer = compute_answer(num1, num2, corruption_operator)
        corrupted_demo['answer'] = corrupted_answer
    
    return corrupted_demo


def create_text_prompt(task, query, n_shot, corrupt_position=None, corruption_operator=None):
    demonstrations = task.select_demonstrations(query, n_shot)
    
    if corrupt_position is not None and corruption_operator is not None:
        if 0 <= corrupt_position < len(demonstrations):
            demonstrations[corrupt_position] = create_corrupted_demonstration(
                demonstrations[corrupt_position], corruption_operator
            )
    
    prompt_parts = []
    for demo in demonstrations:
        prompt_parts.append(f"{demo['question']}\nAnswer: {demo['answer']}\n\n")
    
    prompt_parts.append(f"{query['question']}\nAnswer:")
    prompt = "".join(prompt_parts)
    
    return prompt

def compute_attention_allocation(attention_weights, token_categories, n_shot):
    allocation_scores = {}
    qf_pos = token_categories.get('query_forerunner')
    if qf_pos is None:
        return allocation_scores
    
    for layer_idx, layer_weights in enumerate(attention_weights):
        num_heads, seq_len, _ = layer_weights.shape
        
        if qf_pos >= seq_len:
            continue
            
        for head_idx in range(num_heads):
            qf_attention = layer_weights[head_idx, qf_pos, :]
            for demo_pos in range(n_shot):
                demo_attention = 0.0
                demo_forerunner = token_categories.get(f'demo_{demo_pos+1}_forerunner')
                demo_label = token_categories.get(f'demo_{demo_pos+1}_label')

                demo_tokens = []
                if demo_forerunner is not None:
                    demo_tokens.append(demo_forerunner)
                if demo_label is not None:
                    demo_tokens.append(demo_label)
                
                for token_pos in demo_tokens:
                    if token_pos < seq_len:
                        demo_attention += float(qf_attention[token_pos].item())
                
                if demo_tokens:
                    demo_attention /= len(demo_tokens)
                
                allocation_scores[(layer_idx, head_idx, demo_pos)] = demo_attention
    
    return allocation_scores


def compute_corruption_sensitivity(model, tokenizer, language_model, task, query, n_shot,
                                 attention_outputs_clean, token_categories, position,
                                 all_operators=['+', '-', 'x'], epsilon=1e-8):
    qf_pos = token_categories.get('query_forerunner')
    if qf_pos is None:
        return {}

    correct_operator = query.get('operator', '+')
    wrong_operators = [op for op in all_operators if op != correct_operator]
    head_outputs_clean = {}

    for layer_idx, layer_outputs in enumerate(attention_outputs_clean):
        num_heads, seq_len, head_dim = layer_outputs.shape
        if qf_pos < seq_len:
            for head_idx in range(num_heads):
                head_output = layer_outputs[head_idx, qf_pos, :]
                head_outputs_clean[(layer_idx, head_idx)] = head_output
    
    corruption_deviations = defaultdict(list)
    for wrong_operator in wrong_operators:
        corrupted_prompt = create_text_prompt(task, query, n_shot, 
                                            corrupt_position=position, 
                                            corruption_operator=wrong_operator)
        
        try:
            embeddings, _, _ = get_embeddings_text(model, tokenizer, corrupted_prompt)
            attention_mask = torch.ones((1, embeddings.shape[1]), dtype=torch.long, device=embeddings.device)
            
            _, attention_outputs_corrupted = extract_attention_weights_and_outputs(
                language_model, embeddings, attention_mask
            )
            for layer_idx, layer_outputs in enumerate(attention_outputs_corrupted):
                num_heads, seq_len, head_dim = layer_outputs.shape
                if qf_pos < seq_len:
                    for head_idx in range(num_heads):
                        if (layer_idx, head_idx) in head_outputs_clean:
                            clean_output = head_outputs_clean[(layer_idx, head_idx)]
                            corrupted_output = layer_outputs[head_idx, qf_pos, :]

                            deviation = corrupted_output - clean_output
                            deviation_norm = torch.norm(deviation, p=2).item()
                            corruption_deviations[(layer_idx, head_idx)].append(deviation_norm)
        
        except Exception as e:
            continue

    cs_scores = {}
    for (layer_idx, head_idx) in head_outputs_clean:
        if (layer_idx, head_idx) not in corruption_deviations:
            continue
        if len(corruption_deviations[(layer_idx, head_idx)]) != len(wrong_operators):
            continue
        clean_output = head_outputs_clean[(layer_idx, head_idx)]
        clean_norm = torch.norm(clean_output, p=2).item()

        deviations = corruption_deviations[(layer_idx, head_idx)]
        avg_deviation = np.mean(deviations)
        cs_score = avg_deviation / (clean_norm + epsilon)
        cs_scores[(layer_idx, head_idx, position)] = cs_score
    
    return cs_scores

def run_acs_extraction(model_name="Qwen/Qwen3-8B", data_dir="../../VL-ICL", 
                      n_shot=4, num_samples=60, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    model = create_model("qwen3", model_name)
    tokenizer = get_model_tokenizer(model, "qwen3")
    language_model = get_language_model(model, "qwen3")
    task = OperatorInductionTextTask(data_dir)
    all_queries = task.query_data[:num_samples]
    
    attention_allocation_results = {}
    corruption_sensitivity_results = {}
    
    for sample_idx, query in enumerate(all_queries):
        try:
            clean_prompt = create_text_prompt(task, query, n_shot)
            embeddings, token_ids, token_texts = get_embeddings_text(model, tokenizer, clean_prompt)
            
            token_categories = categorize_tokens_text_fixed(token_ids, token_texts, n_shot, tokenizer, debug=False)
            attention_mask = torch.ones((1, embeddings.shape[1]), dtype=torch.long, device=embeddings.device)
            attention_weights_clean, attention_outputs_clean = extract_attention_weights_and_outputs(
                language_model, embeddings, attention_mask
            )
            
            allocation_scores = compute_attention_allocation(attention_weights_clean, token_categories, n_shot)
            for key, score in allocation_scores.items():
                if key not in attention_allocation_results:
                    attention_allocation_results[key] = []
                attention_allocation_results[key].append(score)

            for position in range(n_shot):
                cs_scores = compute_corruption_sensitivity(
                    model, tokenizer, language_model, task, query, n_shot,
                    attention_outputs_clean, token_categories, position,
                    all_operators=['+', '-', 'x']
                )
                
                for key, score in cs_scores.items():
                    if key not in corruption_sensitivity_results:
                        corruption_sensitivity_results[key] = []
                    corruption_sensitivity_results[key].append(score)
            
        except Exception as e:
            traceback.print_exc()
            continue

    final_attention_allocation = {}
    for key, scores in attention_allocation_results.items():
        final_attention_allocation[key] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'count': len(scores)
        }

    final_corruption_sensitivity = {}
    for key, scores in corruption_sensitivity_results.items():
        final_corruption_sensitivity[key] = {
            'mean': np.mean(scores),
            'std': np.std(scores), 
            'count': len(scores)
        }
    
    # sorted_allocation = sorted(final_attention_allocation.items(), key=lambda x: x[1]['mean'], reverse=True)
    # sorted_cs = sorted(final_corruption_sensitivity.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    return {
        'attention_allocation': final_attention_allocation,
        'corruption_sensitivity': final_corruption_sensitivity,
        'metadata': {
            'model_name': model_name,
            'num_samples': len(all_queries),
            'n_shot': n_shot,
            'seed': seed,
            'task': 'operator_induction_text',
            'framework': 'single_corruption_sensitivity'
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Extract Attention Allocation and Corruption Sensitivity Metrics")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B",
                       help="Model name (default: Qwen/Qwen3-8B)")
    parser.add_argument("--data_dir", type=str, default="../../VL-ICL",
                       help="Data directory (default: ../../VL-ICL)")
    parser.add_argument("--n_shot", type=int, default=4,
                       help="Number of shots (default: 4)")
    parser.add_argument("--num_samples", type=int, default=60,
                       help="Number of samples (default: 60)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    results = run_acs_extraction(
        model_name=args.model_name,
        data_dir=args.data_dir,
        n_shot=args.n_shot,
        num_samples=args.num_samples,
        seed=args.seed
    )

    os.makedirs('../../results', exist_ok=True)
    model_suffix = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    output_path = f'../../results/acs_metrics_{model_suffix}_{args.n_shot}shot_{args.num_samples}samples.pkl'

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    return results

if __name__ == "__main__":
    main()