#!/usr/bin/env python3
import argparse
import json
import numpy as np
import os
import sys
import re
from typing import Dict, List
from tqdm import tqdm
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from tasks.t2t_tasks import (
    OperatorInductionTextTask, 
    OperatorInductionInterleavedTextTask, 
)
from utils.prompt_utils import build_vl_icl_prompt

class CleanBaselineEvaluator:
    def __init__(self, model_name: str, model_type: str, data_dir: str, dataset: str = "operator_induction_text"):
        self.model_name = model_name
        self.model_type = model_type
        self.data_dir = data_dir
        self.dataset = dataset
        
        if self.dataset == "operator_induction_text":
            self.task = OperatorInductionTextTask(self.data_dir)
        elif self.dataset == "operator_induction_interleaved_text":
            self.task = OperatorInductionInterleavedTextTask(self.data_dir)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        self.engine = None
        
    def _initialize_vllm(self):
        from vllm import LLM, SamplingParams
        
        print(f"Initializing vLLM engine for {self.model_name}...")
        self.engine = LLM(
            model=self.model_name,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            trust_remote_code=True
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2,
            stop=None
        )

        return True
    
    def create_clean_demonstrations(self, query: Dict, n_shot: int):
        if 'google_analogy' in self.dataset:
            query_id = query.get('id', 'default')
            seed = hash(query_id) % 10000
            demonstrations = self.task.select_demonstrations(query, n_shot, seed=seed, corrupted=False)
        else:
            demonstrations = self.task.select_demonstrations(query, n_shot)
        
        return demonstrations
    
    def _evaluate_response_harsh(self, query: Dict, response: str) -> bool:
        correct_answer = query.get('answer')
        if isinstance(correct_answer, list):
            correct_answer = correct_answer[0]
        
        response_clean = response.strip()
        
        numbers = re.findall(r'\d+', response_clean)
        if numbers:
            try:
                first_number = int(numbers[0])
                return first_number == correct_answer
            except ValueError:
                return False
        return False
    
    def evaluate_batch(self, queries: List[Dict], prompts: List[str], n_shot: int):
        try:
            outputs = self.engine.generate(prompts, self.sampling_params)
            responses = [output.outputs[0].text for output in outputs]
        except Exception as e:
            return None
        
        correct_count = 0
        
        for query, response in zip(queries, responses):
            is_correct = self._evaluate_response_harsh(query, response)
            if is_correct:
                correct_count += 1
        
        accuracy = correct_count / len(queries)
        return accuracy
    
    def run_clean_baseline_evaluation(self, n_shots: List[int] = [0, 1, 2, 4, 6, 8], num_samples: int = 60):
        available_samples = len(self.task.query_data)
        actual_samples = min(num_samples, available_samples)
        
        query_samples = self.task.query_data[:actual_samples]
        
        results = {
            'experiment_info': {
                'model_name': self.model_name,
                'dataset': self.dataset,
                'n_shots': n_shots,
                'num_samples': actual_samples,
                'timestamp': datetime.now().isoformat()
            },
            'shot_results': {}
        }
        
        try:
            for n_shot in n_shots:
                print(f"\n{'='*60}")
                print(f"EVALUATING {n_shot}-SHOT")
                print("="*60)
                
                queries = []
                prompts = []
                
                for query in tqdm(query_samples, desc=f"Building {n_shot}-shot prompts"):
                    demonstrations = self.create_clean_demonstrations(query, n_shot)
                    
                    prompt = build_vl_icl_prompt(
                        self.task, demonstrations, query,
                        mode="constrained",
                        warned=False,
                        max_images=8
                    )
                    
                    queries.append(query)
                    prompts.append(prompt)
                
                accuracy = self.evaluate_batch(queries, prompts, n_shot)
                
                if accuracy is not None:
                    results['shot_results'][f'{n_shot}_shot'] = {
                        'accuracy': float(accuracy),
                        'n_shot': n_shot,
                        'correct_count': int(accuracy * len(queries)),
                        'total_count': len(queries)
                    }
        
        finally:
            if hasattr(self, 'engine') and self.engine is not None:
                try:
                    del self.engine
                    self.engine = None
                except:
                    pass
        
        model_short = self.model_name.split('/')[-1]
        filename = f"results/clean_baseline_{self.dataset}_{model_short}.json"
        os.makedirs("results", exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Clean baseline evaluation across different shots')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name (e.g., Qwen/Qwen3-8B)')
    parser.add_argument('--model_type', type=str, default='qwen3', 
                        choices=['qwen3', 'llama3', 'internvl', 'qwen25'],
                        help='Model type for compatibility')
    parser.add_argument('--dataset', type=str, default='operator_induction_text',
                        choices=['operator_induction_text', 'operator_induction_interleaved_text'],
                        help='Dataset to evaluate')
    parser.add_argument('--data_dir', type=str, default='./VL-ICL',
                        help='Data directory path')
    parser.add_argument('--n_shots', type=int, nargs='+', default=[0, 1, 2, 4, 6, 8],
                        help='List of shot numbers to evaluate')
    parser.add_argument('--num_samples', type=int, default=60,
                        help='Number of query samples to test')
    
    args = parser.parse_args()
    
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Shots: {args.n_shots}")
    print(f"Samples: {args.num_samples}")
    
    evaluator = CleanBaselineEvaluator(
        model_name=args.model_name,
        model_type=args.model_type,
        data_dir=args.data_dir,
        dataset=args.dataset
    )
    
    results = evaluator.run_clean_baseline_evaluation(
        n_shots=args.n_shots,
        num_samples=args.num_samples
    )
    
    if results:
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())