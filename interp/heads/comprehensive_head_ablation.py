import torch
import numpy as np
import pickle
import os
import sys
import random
import re
from tqdm import tqdm
from typing import List, Dict, Tuple, Set

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(root_dir)

from models.model_factory import create_model
from tasks.t2t_tasks import OperatorInductionTextTask
from utils.prompt_utils import build_vl_icl_prompt


class AttentionHeadMaskingHook:
    def __init__(self, heads_to_mask: Set[Tuple[int, int]]):
        self.heads_to_mask = heads_to_mask
        self.hooks = []
    
    def mask_attention_head(self, module, input_, output):
        if not self.heads_to_mask:
            return output
        
        layer_idx = getattr(module, '_layer_idx', None)
        if layer_idx is None:
            return output
        
        heads_in_layer = [head_idx for l_idx, head_idx in self.heads_to_mask if l_idx == layer_idx]
        if not heads_in_layer:
            return output
        
        masked_output = output.clone()
        batch_size, seq_len, hidden_dim = masked_output.shape
        num_heads = getattr(module, '_num_heads', 32)
        head_dim = hidden_dim // num_heads
        
        reshaped = masked_output.view(batch_size, seq_len, num_heads, head_dim)
        for head_idx in heads_in_layer:
            if head_idx < num_heads:
                reshaped[:, :, head_idx, :] = 0.0
        
        return reshaped.view(batch_size, seq_len, hidden_dim)
    
    def register_hooks(self, model):
        self.remove_hooks()
        
        if hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
            transformer = model.model.model
        else:
            transformer = model.model
        
        for layer_idx, layer in enumerate(transformer.layers):
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
                layer.self_attn.o_proj._layer_idx = layer_idx
                layer.self_attn.o_proj._num_heads = getattr(layer.self_attn, 'num_heads', 32)
                hook = layer.self_attn.o_proj.register_forward_hook(self.mask_attention_head)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def run_batch_inference(model, tokenizer, prompts: List[str], max_new_tokens: int, batch_size: int = 60) -> List[str]:
    all_responses = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Batch inference (max_tokens={max_new_tokens})", leave=False):
        batch_prompts = prompts[i:i+batch_size]
        
        batch_texts = []
        for prompt in batch_prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_texts.append(text)
        
        model_inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.model.device)
        
        with torch.no_grad():
            outputs = model.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        for j, output in enumerate(outputs):
            response = tokenizer.decode(
                output[len(model_inputs['input_ids'][j]):], 
                skip_special_tokens=True
            )
            all_responses.append(response)
    
    return all_responses


def evaluate_responses(queries: List[Dict], responses: List[str]) -> List[bool]:
    results = []
    for query, response in zip(queries, responses):
        expected_answer = query.get('answer')
        if expected_answer is None:
            results.append(False)
            continue
        
        numbers_in_response = re.findall(r'\b\d+\b', response)
        expected_str = str(expected_answer)
        results.append(expected_str in numbers_in_response)
    
    return results


def create_corrupted_demonstrations(demonstrations: List[Dict], corruption_position: int, corruption_operator: str) -> List[Dict]:
    corrupted = [demo.copy() for demo in demonstrations]
    
    if 0 <= corruption_position < len(corrupted):
        demo = corrupted[corruption_position]
        numbers = re.findall(r'\d+', demo['question'])
        if len(numbers) >= 2:
            num1, num2 = int(numbers[0]), int(numbers[1])
            
            if corruption_operator == '+':
                corrupted_answer = num1 + num2
            elif corruption_operator == '-':
                corrupted_answer = num1 - num2
            elif corruption_operator == 'x':
                corrupted_answer = num1 * num2
            else:
                corrupted_answer = demo['answer']
            
            demo['answer'] = corrupted_answer
    
    return corrupted


def load_acs_position_rankings(file_path):
    with open(file_path, 'rb') as f:
        acs_data = pickle.load(f)

    position_rankings = acs_data['position_rankings']
    return position_rankings


def load_resolution_heads(top_k=20):
    file_path = 'results/anti_resolution_heads_rigorous.pkl'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    head_rankings = data['final_head_rankings']
    
    positive_heads = []
    for head_name, head_data in head_rankings.items():
        if head_data['anti_resolution_score'] > 0.01:
            positive_heads.append((head_name, head_data['anti_resolution_score']))

    positive_heads.sort(key=lambda x: x[1], reverse=True)
    resolution_heads = set()
    for head_name, score in positive_heads[:top_k]:
        layer_idx = int(head_name[1:head_name.index('H')])
        head_idx = int(head_name[head_name.index('H')+1:])
        resolution_heads.add((layer_idx, head_idx))
    
    return resolution_heads

def generate_random_heads(num_layers, num_attention_heads, num_heads=20, exclude_heads=None):
    all_heads = [(layer, head) for layer in range(num_layers) for head in range(num_attention_heads)]
    if exclude_heads:
        available = [h for h in all_heads if h not in exclude_heads]
    else:
        available = all_heads
    
    return set(random.sample(available, min(num_heads, len(available))))


def main():
    MODEL_NAME = "Qwen/Qwen3-8B"
    MODEL_TYPE = "qwen3"
    DATA_DIR = "../../VL-ICL"
    N_SHOT = 4
    NUM_SAMPLES = 60
    acs_file_path = 'acs_analysis_results.pkl'
    num_rollouts = 3
    ablation_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
    SEED = 42
    
    model = create_model(MODEL_TYPE, MODEL_NAME)
    tokenizer = model.tokenizer
    task = OperatorInductionTextTask(DATA_DIR)
    position_rankings = load_acs_position_rankings(acs_file_path)
    
    resolution_heads_all = load_resolution_heads(top_k=50)
    num_layers = model.model.config.num_hidden_layers
    num_attention_heads = model.model.config.num_attention_heads
    
    query_samples = task.query_data[:NUM_SAMPLES]
    all_operators = ['+', '-', 'x']
    positions = list(range(N_SHOT))
    
    results = {}

    for num_heads in ablation_sizes:
        rollout_results = []
        
        for rollout in range(num_rollouts):
            rollout_seed = SEED + rollout
            random.seed(rollout_seed)
            np.random.seed(rollout_seed)
            torch.manual_seed(rollout_seed)
            
            corruption_plan = {}
            for operator in all_operators:
                other_operators = [op for op in all_operators if op != operator]
                for pos in positions:
                    corruption_op = random.choice(other_operators)
                    corruption_plan[(operator, pos)] = corruption_op
            
            fixed_demonstrations = {}
            for query in tqdm(query_samples, desc="Pre-generating demonstrations", leave=False):
                query_id = query.get('id', f'query_{query_samples.index(query)}')
                demonstrations = task.select_demonstrations(query, N_SHOT, seed=rollout_seed)
                fixed_demonstrations[query_id] = demonstrations
            
            position_results = {
                'baseline': [],
                'original': [],
                'vulnerability': [],
                'resolution': [],
                'random': []
            }
            
            for pos in positions:
                if num_heads <= len(position_rankings[pos]):
                    vulnerability_heads = set((layer, head) for layer, head, _ in position_rankings[pos][:num_heads])
                else:
                    vulnerability_heads = set((layer, head) for layer, head, _ in position_rankings[pos])
                
                top_vuln_heads = set((layer, head) for layer, head, _ in position_rankings[pos][:20])
                random_heads = generate_random_heads(num_layers, num_attention_heads, num_heads, exclude_heads=top_vuln_heads)
                baseline_prompts = []
                corrupted_prompts = []
                queries_batch = []
                
                for query in query_samples:
                    query_id = query.get('id', f'query_{query_samples.index(query)}')
                    demonstrations = fixed_demonstrations[query_id]
                    
                    baseline_prompt = build_vl_icl_prompt(task, demonstrations, query, mode="free")
                    baseline_prompts.append(baseline_prompt)
                    
                    corruption_op = corruption_plan.get((query.get('operator'), pos))
                    if corruption_op:
                        corrupted_demos = create_corrupted_demonstrations(demonstrations, pos, corruption_op)
                        corrupted_prompt = build_vl_icl_prompt(task, corrupted_demos, query, mode="free")
                    else:
                        corrupted_prompt = baseline_prompt
                    corrupted_prompts.append(corrupted_prompt)
                    queries_batch.append(query)
                
                baseline_responses = run_batch_inference(model, tokenizer, baseline_prompts, 256)
                baseline_results = evaluate_responses(queries_batch, baseline_responses)
                baseline_acc = np.mean(baseline_results)
                
                original_responses = run_batch_inference(model, tokenizer, corrupted_prompts, 256)
                original_results = evaluate_responses(queries_batch, original_responses)
                original_acc = np.mean(original_results)
                
                if len(vulnerability_heads) > 0:
                    hook_manager = AttentionHeadMaskingHook(vulnerability_heads)
                    hook_manager.register_hooks(model)
                    vuln_responses = run_batch_inference(model, tokenizer, corrupted_prompts, 512)
                    vuln_results = evaluate_responses(queries_batch, vuln_responses)
                    vuln_acc = np.mean(vuln_results)
                    hook_manager.remove_hooks()
                else:
                    vuln_acc = original_acc
                
                if num_heads <= len(resolution_heads_all):
                    resolution_heads_sample = set(random.sample(list(resolution_heads_all), num_heads))
                    hook_manager = AttentionHeadMaskingHook(resolution_heads_sample)
                    hook_manager.register_hooks(model)
                    res_responses = run_batch_inference(model, tokenizer, corrupted_prompts, 512)
                    res_results = evaluate_responses(queries_batch, res_responses)
                    res_acc = np.mean(res_results)
                    hook_manager.remove_hooks()
                else:
                    res_acc = original_acc
                
                hook_manager = AttentionHeadMaskingHook(random_heads)
                hook_manager.register_hooks(model)
                random_responses = run_batch_inference(model, tokenizer, corrupted_prompts, 256)
                random_results = evaluate_responses(queries_batch, random_responses)
                random_acc = np.mean(random_results)
                hook_manager.remove_hooks()
                
                position_results['baseline'].append(baseline_acc)
                position_results['original'].append(original_acc)
                position_results['vulnerability'].append(vuln_acc)
                position_results['resolution'].append(res_acc)
                position_results['random'].append(random_acc)
            
            # rollout_baseline_mean = np.mean(position_results['baseline'])
            # rollout_original_mean = np.mean(position_results['original'])
            # rollout_vuln_mean = np.mean(position_results['vulnerability'])
            # rollout_res_mean = np.mean(position_results['resolution'])
            # rollout_random_mean = np.mean(position_results['random'])
            rollout_results.append(position_results)
        
        all_baselines = []
        all_originals = []
        all_vulnerabilities = []
        all_resolutions = []
        all_randoms = []
        
        for rollout in rollout_results:
            all_baselines.extend(rollout['baseline'])
            all_originals.extend(rollout['original'])
            all_vulnerabilities.extend(rollout['vulnerability'])
            all_resolutions.extend(rollout['resolution'])
            all_randoms.extend(rollout['random'])
        
        avg_baseline = np.mean(all_baselines)
        avg_original = np.mean(all_originals)
        avg_vulnerability = np.mean(all_vulnerabilities)
        avg_resolution = np.mean(all_resolutions)
        avg_random = np.mean(all_randoms)

        vuln_improvement = avg_vulnerability - avg_original
        res_improvement = avg_resolution - avg_original
        random_improvement = avg_random - avg_original
        
        results[num_heads] = {
            'avg_baseline': avg_baseline,
            'avg_original': avg_original,
            'avg_vulnerability': avg_vulnerability,
            'avg_resolution': avg_resolution,
            'avg_random': avg_random,
            'vuln_improvement': vuln_improvement,
            'res_improvement': res_improvement,
            'random_improvement': random_improvement,
            'rollout_results': rollout_results
        }
    
    os.makedirs('figs', exist_ok=True)
    with open('figs/comprehensive_head_ablation_results.pkl', 'wb') as f:
        pickle.dump({
            'model_name': MODEL_NAME,
            'results': results,
            'ablation_sizes': ablation_sizes,
            'num_rollouts': num_rollouts
        }, f)

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    main()