#!/usr/bin/env python3
import sys
import os
from typing import List, Dict
import numpy as np
import re
import argparse
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(root_dir)

from interp.representation_extractor import RepresentationExtractor
from interp.logit_lens.logit_lens import LogitLens
from tasks.t2t_tasks import OperatorInductionTextTask

class GeneralizedLogitLensAnalyzer:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B", model_type: str = "qwen3"):
        self.model_name = model_name
        self.model_type = model_type
        self.extractor = RepresentationExtractor(model_name, model_type)
        self.logit_lens = LogitLens(model_name, model_type)
        self.model_config = self._get_model_config(model_name)
        
        self.operator_words = {
            '+': 'plus',
            '-': 'minus', 
            '*': 'multiplication',
            'x': 'multiplication'
        }
        
        # Extended semantic equivalents for robust matching
        self.operator_synonyms = {
            'plus': ['plus', 'addition', 'add', 'adding', 'sum', 'summation'],
            'minus': ['minus', 'subtraction', 'subtract', 'subtracting', 'difference'],
            'multiplication': ['multiplication', 'multiply', 'multiplying', 'times', 'product']
        }
    
    def _get_model_config(self, model_name: str) -> Dict:
        model_configs = {
            "Qwen/Qwen3-4B": {"num_layers": 32, "hidden_dim": 3584},
            "Qwen/Qwen3-8B": {"num_layers": 36, "hidden_dim": 4096}, 
            "Qwen/Qwen3-14B": {"num_layers": 40, "hidden_dim": 5120},
            "Qwen/Qwen3-32B": {"num_layers": 64, "hidden_dim": 5120}
        }
        
        if model_name in model_configs:
            return model_configs[model_name]
        else:
            print(f"Warning: Unknown model {model_name}, using default config")
            return {"num_layers": 36, "hidden_dim": 4096}
        
    def load_models(self):
        self.extractor.load_model()
        self.logit_lens.model = self.extractor.model
        self.logit_lens.tokenizer = self.extractor.tokenizer
        self.logit_lens._find_model_components()
    
    def get_rule_classification(self, correct_op: str, corruption_config: Dict) -> Dict[str, str]:
        if correct_op == 'x':
            correct_op = '*'
        
        if corruption_config['num_corrupted'] == 0:
            corrupted_op = None
        else:
            corrupted_op = corruption_config['corrupted_op']
            if corrupted_op == 'x':
                corrupted_op = '*'
        
        rule_map = {
            'correct': self.operator_words[correct_op],
            'corrupted': self.operator_words[corrupted_op] if corrupted_op else None
        }
        
        return rule_map
    
    def extract_numbers_from_question(self, question: str) -> tuple:
        numbers = re.findall(r'\d+', question)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
        return None, None
    
    def compute_answer(self, num1: int, num2: int, operator: str) -> int:
        if operator == '+':
            return num1 + num2
        elif operator == '-':
            return num1 - num2
        elif operator == '*' or operator == 'x':
            return num1 * num2
        return 0
    
    def create_demonstration_prompt(self, demonstrations: List[Dict], query: Dict, 
                                  correct_op: str, num_corrupted: int = 0) -> Dict:
        prompt_parts = []
        all_ops = ['+', '-', '*']
        available_corrupt_ops = [op for op in all_ops if op != correct_op]
        corrupt_op = available_corrupt_ops[0]
        
        for i, demo in enumerate(demonstrations):
            question = demo['question']
            
            if isinstance(demo['answer'], list):
                correct_answer = demo['answer'][0]
            else:
                correct_answer = demo['answer']
            
            if i < num_corrupted:
                num1, num2 = self.extract_numbers_from_question(question)
                if num1 is not None and num2 is not None:
                    corrupted_answer = self.compute_answer(num1, num2, corrupt_op)
                    prompt_parts.append(f"{question} Answer: {corrupted_answer}")
                else:
                    prompt_parts.append(f"{question} Answer: {correct_answer}")
            else:
                prompt_parts.append(f"{question} Answer: {correct_answer}")
        
        prompt_parts.append(f"{query['question']}")
        prompt_parts.append("What mathematical operation does ? represent? Choose from: plus, minus, multiplication")
        prompt_parts.append("Answer:")
        
        prompt = '\n'.join(prompt_parts)
        
        corruption_config = {
            'num_corrupted': num_corrupted,
            'corrupted_op': corrupt_op if num_corrupted > 0 else None
        }
        
        rule_map = self.get_rule_classification(correct_op, corruption_config)
        
        return {
            'prompt': prompt,
            'rule_map': rule_map,
            'corruption_config': corruption_config
        }
    
    def get_operator_word_tokens(self):
        operator_tokens = {}
        
        for base_word in ['plus', 'minus', 'multiplication']:
            operator_tokens[base_word] = []
            synonyms = self.operator_synonyms.get(base_word, [base_word])
            
            for synonym in synonyms:
                tokens = self.logit_lens.tokenizer.encode(synonym, add_special_tokens=False)
                operator_tokens[base_word].extend(tokens)
                
                tokens_with_space = self.logit_lens.tokenizer.encode(' ' + synonym, add_special_tokens=False)
                operator_tokens[base_word].extend(tokens_with_space)
                
                tokens_cap = self.logit_lens.tokenizer.encode(synonym.capitalize(), add_special_tokens=False)
                operator_tokens[base_word].extend(tokens_cap)
                
                tokens_cap_space = self.logit_lens.tokenizer.encode(' ' + synonym.capitalize(), add_special_tokens=False)
                operator_tokens[base_word].extend(tokens_cap_space)
            
            operator_tokens[base_word] = list(set(operator_tokens[base_word]))
        
        return operator_tokens
    
    def check_operator_prefix_match(self, predicted_token: str, target_word: str) -> bool:
        if not target_word:
            return False
            
        predicted_clean = predicted_token.strip().lower()
        if not predicted_clean:
            return False
        
        synonyms = self.operator_synonyms.get(target_word, [target_word])
        
        for synonym in synonyms:
            synonym_lower = synonym.lower()
            if predicted_clean == synonym_lower or synonym_lower.startswith(predicted_clean):
                return True
        
        return False
    
    def analyze_rule_probabilities(self, prompt_data: Dict, layers_to_analyze: List[int] = None) -> Dict:
        if layers_to_analyze is None:
            layers_to_analyze = list(range(self.model_config["num_layers"]))
        
        prompt = prompt_data['prompt']
        rule_map = prompt_data['rule_map']
        
        representations = self.extractor.extract_representations([prompt], layers_to_analyze, return_attention=False)
        sample_data = representations['sample_0']
        
        operator_token_map = self.get_operator_word_tokens()
        
        results = {
            'prompt': prompt,
            'rule_map': rule_map,
            'layers': {},
            'corruption_config': prompt_data['corruption_config']
        }
        
        for layer_name, hidden_states in sample_data['hidden_states'].items():
            layer_idx = int(layer_name.split('_')[1])
            
            final_hidden_state = hidden_states[0, -1, :].unsqueeze(0).unsqueeze(0)
            final_hidden_state = final_hidden_state.to(self.logit_lens.model.model.device)
            
            lens_results = self.logit_lens.apply_logit_lens(final_hidden_state, top_k=100)
            
            rule_probs = {}
            
            for rule_type, word in rule_map.items():
                if word is None:
                    continue
                    
                max_prob = 0.0
                
                for token_id in operator_token_map[word]:
                    prob = lens_results['probabilities'][0, 0, token_id].item()
                    max_prob = max(max_prob, prob)
                
                for i in range(100):
                    token_id = lens_results['top_k_indices'][0, 0, i].item()
                    token_str = self.logit_lens.tokenizer.decode([token_id])
                    prob = lens_results['top_k_probs'][0, 0, i].item()
                    
                    if self.check_operator_prefix_match(token_str, word):
                        max_prob = max(max_prob, prob)
                
                rule_probs[rule_type] = max_prob
            
            top_tokens = []
            for i in range(3):
                token_id = lens_results['top_k_indices'][0, 0, i].item()
                token_str = self.logit_lens.tokenizer.decode([token_id])
                prob = lens_results['top_k_probs'][0, 0, i].item()
                top_tokens.append({'token': token_str, 'prob': prob})
            
            results['layers'][layer_idx] = {
                'rule_probs': rule_probs,
                'top_tokens': top_tokens
            }
        
        return results
    
    def full_dataset_analysis(self, task: OperatorInductionTextTask, n_shot: int = 4, max_queries: int = None) -> Dict:
        if max_queries is None:
            max_queries = len(task.query_data)

        all_layers = list(range(self.model_config["num_layers"]))
        
        aggregated_results = {
            'n_shot': n_shot,
            'total_queries': max_queries,
            'layers_analyzed': all_layers,
            'corruption_aggregate': {
                level: {
                    layer: {
                        'correct': [], 
                        'corrupted': []
                    } for layer in all_layers
                } for level in [0, 1]
            },
            'sample_results': [],
            'metadata': {
                'model_name': self.model_name,
                'operator_words': self.operator_words
            }
        }
        
        for query_idx in range(min(max_queries, len(task.query_data))):
            query = task.query_data[query_idx]
            demonstrations = task.select_demonstrations(query, n_shot=n_shot, seed=42)
            correct_operator = query['operator']
            
            query_results = {
                'query_idx': query_idx,
                'query': query,
                'correct_operator': correct_operator,
                'corruption_levels': {}
            }

            for num_corrupted in [0, 1]:
                prompt_data = self.create_demonstration_prompt(demonstrations, query, correct_operator, num_corrupted)
                analysis = self.analyze_rule_probabilities(prompt_data, all_layers)
                
                query_results['corruption_levels'][num_corrupted] = analysis
                
                for layer_idx, layer_data in analysis['layers'].items():
                    for rule_type in ['correct', 'corrupted']:
                        if rule_type in layer_data['rule_probs']:
                            prob = layer_data['rule_probs'][rule_type]
                            aggregated_results['corruption_aggregate'][num_corrupted][layer_idx][rule_type].append(prob)
            
            if query_idx < 3:
                aggregated_results['sample_results'].append(query_results)
        
        for corruption_level in aggregated_results['corruption_aggregate']:
            for layer_idx in aggregated_results['corruption_aggregate'][corruption_level]:
                for rule_type in ['correct', 'corrupted']:
                    probs = aggregated_results['corruption_aggregate'][corruption_level][layer_idx][rule_type]
                    if probs:
                        aggregated_results['corruption_aggregate'][corruption_level][layer_idx][rule_type] = {
                            'mean': float(np.mean(probs)),
                            'std': float(np.std(probs)),
                            'count': len(probs)
                        }
        
        return aggregated_results

def main():
    parser = argparse.ArgumentParser(description='Run Logit Lens Analysis')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-8B", 
                        help='Model name (e.g., Qwen/Qwen3-8B)')
    parser.add_argument('--model_type', type=str, default="qwen3", 
                        help='Model type (qwen3, etc.)')
    parser.add_argument('--task', type=str, default="operator_induction_text", 
                        help='Task name (operator_induction_text, operator_induction_interleaved_text)')
    parser.add_argument('--n_shot', type=int, default=4, 
                        help='Number of shots')
    parser.add_argument('--max_queries', type=int, default=60, 
                        help='Maximum number of queries to process')
    
    args = parser.parse_args()
    
    analyzer = GeneralizedLogitLensAnalyzer(args.model_name, args.model_type)
    analyzer.load_models()
    data_dir = "../../VL-ICL"

    if args.task == "operator_induction_text":
        from tasks.t2t_tasks import OperatorInductionTextTask
        task = OperatorInductionTextTask(data_dir)
    elif args.task == "operator_induction_interleaved_text":
        from tasks.t2t_tasks import OperatorInductionInterleavedTextTask
        task = OperatorInductionInterleavedTextTask(data_dir)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    results = analyzer.full_dataset_analysis(task, n_shot=args.n_shot, max_queries=args.max_queries)
    model_name_clean = analyzer.model_name.replace("/", "_").replace("-", "_")
    task_clean = args.task.replace("/", "_").replace("-", "_")
    results_filename = f'generalized_logit_lens_results_{model_name_clean}_{task_clean}_{args.n_shot}shot.pkl'
    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()