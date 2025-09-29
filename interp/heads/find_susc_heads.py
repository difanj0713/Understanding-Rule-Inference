import sys
import os
import torch
from typing import Dict, List
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(os.path.dirname(parent_dir))

from models.model_factory import create_model
from tasks.t2t_tasks import OperatorInductionTextTask
from interp.helper import categorize_tokens_text_fixed

class AntiResolutionHeadFinder:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B", model_type: str = "qwen3"):
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.operator_words = {'+': 'plus', '-': 'minus', '*': 'multiplication', 'x': 'multiplication'}
        
    def load_model(self):
        self.model = create_model(self.model_type, self.model_name)
        self.tokenizer = self.model.tokenizer
        
    def create_prompt(self, demonstrations: List[Dict], query: Dict, corruption_pattern: List[int], corrupted_op: str) -> str:
        prompt_parts = []
        for i, demo in enumerate(demonstrations):
            question = demo['question']
            correct_answer = demo['answer'][0] if isinstance(demo['answer'], list) else demo['answer']
            
            if corruption_pattern[i] == 1:
                num1, num2 = self.extract_numbers(question)
                if num1 is not None and num2 is not None:
                    corrupted_answer = self.compute_answer(num1, num2, corrupted_op)
                    prompt_parts.append(f"{question} Answer: {corrupted_answer}")
                else:
                    prompt_parts.append(f"{question} Answer: {correct_answer}")
            else:
                prompt_parts.append(f"{question} Answer: {correct_answer}")
        
        prompt_parts.extend([
            f"{query['question']}",
            "What mathematical operation does ? represent? Choose from: plus, minus, multiplication",
            "Answer:"
        ])
        return '\n'.join(prompt_parts)
    
    def extract_numbers(self, question: str):
        import re
        numbers = re.findall(r'\d+', question)
        return (int(numbers[0]), int(numbers[1])) if len(numbers) >= 2 else (None, None)
    
    def compute_answer(self, num1: int, num2: int, operator: str) -> int:
        if operator == '+': return num1 + num2
        elif operator == '-': return num1 - num2
        elif operator == '*': return num1 * num2
        return 0
    
    def get_operator_token_ids(self):
        operator_tokens = {}
        for word in ['plus', 'minus', 'multiplication']:
            tokens = self.tokenizer.encode(word, add_special_tokens=False)
            operator_tokens[word] = tokens[0] if tokens else None
        return operator_tokens
    
    def extract_head_contributions(self, prompt: str, layers_to_analyze: List[int]) -> Dict:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.model.device)
        input_ids = model_inputs['input_ids']
        
        actual_token_texts = [self.tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
        token_categories = categorize_tokens_text_fixed(
            input_ids[0].tolist(), actual_token_texts, 4, self.tokenizer, debug=False
        )
        
        if token_categories['query_forerunner'] is None:
            return None
            
        query_pos = token_categories['query_forerunner']
        head_contributions = {}
        
        def attention_hook(layer_idx):
            def hook(module, _, outputs):
                attn_output = outputs[0][0, query_pos, :]
                config = self.model.model.config
                num_heads = config.num_attention_heads
                head_dim = config.hidden_size // num_heads
                head_outputs = attn_output.view(num_heads, head_dim)
                
                W_O = module.o_proj.weight
                for head_idx in range(num_heads):
                    head_out = head_outputs[head_idx]
                    
                    start_idx = head_idx * head_dim
                    end_idx = (head_idx + 1) * head_dim
                    W_O_head = W_O[:, start_idx:end_idx].T
                    residual_contrib = torch.matmul(head_out, W_O_head)
                    
                    head_contributions[f"L{layer_idx}H{head_idx}"] = residual_contrib.detach().cpu()
            return hook
        
        hooks = []
        for layer_idx in layers_to_analyze:
            layer = self.model.model.model.layers[layer_idx]
            hook = layer.self_attn.register_forward_hook(attention_hook(layer_idx))
            hooks.append(hook)
        
        try:
            with torch.no_grad():
                outputs = self.model.model(**model_inputs)
                logits = outputs.logits[0, -1, :]
                unembedding = self.model.model.lm_head.weight
                
                # Compute DLA scores for each head
                operator_tokens = self.get_operator_token_ids()
                head_scores = {}
                
                for head_name, residual_contrib in head_contributions.items():
                    logit_attribution = torch.matmul(unembedding, residual_contrib.to(unembedding.device))
                    head_logits = {}
                    for word, token_id in operator_tokens.items():
                        if token_id is not None:
                            head_logits[word] = logit_attribution[token_id].item()
                    
                    head_scores[head_name] = head_logits
                    
        finally:
            for hook in hooks:
                hook.remove()
        
        return {
            'head_scores': head_scores,
            'baseline_logits': {word: logits[token_id].item() 
                              for word, token_id in operator_tokens.items() if token_id is not None}
        }
    
    def rank_anti_resolution_heads(self, query: Dict, demonstrations: List[Dict], 
                                  layers_to_analyze: List[int] = None) -> Dict:
        if layers_to_analyze is None:
            layers_to_analyze = list(range(24, 36))
            
        correct_op = query['operator']
        if correct_op == 'x': correct_op = '*'
        available_ops = [op for op in ['+', '-', '*'] if op != correct_op]
        corrupted_op = available_ops[0]
        correct_word = self.operator_words[query['operator']]
        
        clean_pattern = [0, 0, 0, 0] 
        minority_corrupt_pattern = [1, 0, 0, 0]
        
        clean_prompt = self.create_prompt(demonstrations, query, clean_pattern, corrupted_op)
        minority_corrupt_prompt = self.create_prompt(demonstrations, query, minority_corrupt_pattern, corrupted_op)
        clean_result = self.extract_head_contributions(clean_prompt, layers_to_analyze)
        minority_corrupt_result = self.extract_head_contributions(minority_corrupt_prompt, layers_to_analyze)

        anti_resolution_scores = {}
        
        for head_name in clean_result['head_scores']:
            if head_name not in minority_corrupt_result['head_scores']:
                continue
                
            clean_logits = clean_result['head_scores'][head_name]
            minority_corrupt_logits = minority_corrupt_result['head_scores'][head_name]

            clean_correct = clean_logits.get(correct_word, 0)
            minority_corrupt_correct = minority_corrupt_logits.get(correct_word, 0)
            anti_resolution_score = clean_correct - minority_corrupt_correct
            
            anti_resolution_scores[head_name] = {
                'anti_resolution_score': anti_resolution_score,
                'clean_correct_contrib': clean_correct,
                'minority_corrupt_correct_contrib': minority_corrupt_correct,
                'clean_logits': clean_logits,
                'minority_corrupt_logits': minority_corrupt_logits
            }
        
        return anti_resolution_scores

def main():
    finder = AntiResolutionHeadFinder()
    finder.load_model()
    
    data_dir = "../../VL-ICL"
    task = OperatorInductionTextTask(data_dir)
    all_queries = task.query_data
    layers_to_analyze = list(range(36))
    head_score_aggregates = {}
    
    for query_idx, test_query in enumerate(all_queries):
        demonstrations = task.select_demonstrations(test_query, n_shot=4, seed=42)
        
        head_rankings = finder.rank_anti_resolution_heads(
            test_query, demonstrations, layers_to_analyze
        )
        
        if head_rankings:
            for head_name, data in head_rankings.items():
                if head_name not in head_score_aggregates:
                    head_score_aggregates[head_name] = {
                        'anti_resolution_scores': [],
                        'clean_contribs': [],
                        'minority_corrupt_contribs': []
                    }
                
                head_score_aggregates[head_name]['anti_resolution_scores'].append(data['anti_resolution_score'])
                head_score_aggregates[head_name]['clean_contribs'].append(data['clean_correct_contrib'])
                head_score_aggregates[head_name]['minority_corrupt_contribs'].append(data['minority_corrupt_correct_contrib'])
    
    final_head_rankings = {}
    for head_name, aggregates in head_score_aggregates.items():
        if aggregates['anti_resolution_scores']:  # Only if we have valid scores
            avg_anti_resolution = sum(aggregates['anti_resolution_scores']) / len(aggregates['anti_resolution_scores'])
            avg_clean = sum(aggregates['clean_contribs']) / len(aggregates['clean_contribs'])
            avg_minority_corrupt = sum(aggregates['minority_corrupt_contribs']) / len(aggregates['minority_corrupt_contribs'])
            
            final_head_rankings[head_name] = {
                'anti_resolution_score': avg_anti_resolution,
                'clean_correct_contrib': avg_clean,
                'minority_corrupt_correct_contrib': avg_minority_corrupt,
                'n_samples': len(aggregates['anti_resolution_scores'])
            }
    
    if final_head_rankings:
        rigorous_results = {
            'final_head_rankings': final_head_rankings,
            'total_queries': len(all_queries),
            'total_layers': len(layers_to_analyze),
            'methodology': 'DLA averaged across all queries and layers'
        }
        
        with open('results/anti_resolution_heads_rigorous.pkl', 'wb') as f:
            pickle.dump(rigorous_results, f)

if __name__ == "__main__":
    main()