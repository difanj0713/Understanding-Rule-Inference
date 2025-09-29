#!/usr/bin/env python3
import argparse
import json
import numpy as np
import os
import sys
import re
import random
from typing import Dict, List, Tuple
from tqdm import tqdm
from datetime import datetime
from scipy import stats

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from tasks.t2t_tasks import (
    OperatorInductionTextTask, 
    OperatorInductionInterleavedTextTask, 
)
from utils.prompt_utils import build_vl_icl_prompt
from utils.corruption_utils import CorruptionManager
from evaluation.llm_judge import HybridEvaluator

class ComprehensiveEvaluator:
    def __init__(self, model_name: str, model_type: str, data_dir: str, dataset: str = "operator_induction_text"):
        self.model_name = model_name
        self.model_type = model_type
        self.data_dir = data_dir
        self.dataset = dataset
        self.corruption_manager = CorruptionManager()
        self.hybrid_evaluator = HybridEvaluator()
        
        if self.dataset == "operator_induction_text":
            self.task = OperatorInductionTextTask(self.data_dir)
        elif self.dataset == "operator_induction_interleaved_text":
            self.task = OperatorInductionInterleavedTextTask(self.data_dir)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        self.engine = None
        
    def _initialize_vllm(self):
        from vllm import LLM, SamplingParams
        
        self.engine = LLM(
            model=self.model_name,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.99,
            trust_remote_code=True
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            stop=None
        )

        return True
    
    def create_corrupted_demonstrations(self, query: Dict, n_shot: int, corruption_position: int, rollout: int = 1):
        random.seed(rollout * 12345)
        
        demonstrations = self.task.select_demonstrations(query, n_shot)
        
        if corruption_position is not None:
            demonstrations = self.corruption_manager.create_corrupted_demonstrations(
                self.dataset, query, demonstrations, self.task.support_data, corruption_position
            )
    
        return demonstrations
    
    def _evaluate_response(self, query: Dict, response: str, is_baseline: bool = False) -> bool:
        mode = "constrained" if is_baseline else "free"
        task_type = "operator_induction"
        return self.hybrid_evaluator.evaluate_response(query, response, mode, task_type)
    
    def evaluate_batch(self, queries: List[Dict], prompts: List[str], evaluation_type: str):
        is_baseline = "BASELINE" in evaluation_type
        sampling_params = self.sampling_params
        
        try:
            outputs = self.engine.generate(prompts, sampling_params)
            responses = [output.outputs[0].text for output in outputs]
        except Exception as e:
            return None
        
        correct_count = 0
        
        for query, response in zip(queries, responses):
            is_correct = self._evaluate_response(query, response, is_baseline=is_baseline)
            if is_correct:
                correct_count += 1
        
        accuracy = correct_count / len(queries)
        return accuracy
    
    def run_comprehensive_evaluation(self, n_shot: int = 4, num_samples: int = 60, num_rollouts: int = 3):
        available_samples = len(self.task.query_data)
        actual_samples = min(num_samples, available_samples)
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE EVALUATION")
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset}")
        print(f"Shots: {n_shot}")
        print(f"Samples: {actual_samples} (requested: {num_samples}, available: {available_samples})")
        print(f"Rollouts: {num_rollouts}")
        print("="*80)
        
        query_samples = self.task.query_data[:actual_samples]
        
        results = {
            'experiment_info': {
                'model_name': self.model_name,
                'dataset': self.dataset,
                'n_shot': n_shot,
                'num_samples': actual_samples,
                'num_rollouts': num_rollouts,
                'timestamp': datetime.now().isoformat()
            },
            'position_results': {}
        }
        
        try:
            for position in range(n_shot):
                position_data = {
                    'upperbound_rollouts': [],
                    'baseline_rollouts': [],
                    'upperbound_mean': None,
                    'upperbound_std': None,
                    'baseline_mean': None,
                    'baseline_std': None,
                    'improvement_mean': None,
                    'improvement_std': None,
                    'paired_differences': [],
                    'p_value': None
                }
                
                upperbound_accuracies = []
                baseline_accuracies = []
                paired_differences = []
                
                for rollout in range(num_rollouts):
                    print(f"\n--- Rollout {rollout + 1}/{num_rollouts} for Position {position} ---")
                    
                    upperbound_queries = []
                    upperbound_prompts = []
                    baseline_queries = []
                    baseline_prompts = []
                    
                    for query in tqdm(query_samples, desc=f"Building paired prompts R{rollout+1}"):
                        demonstrations = self.create_corrupted_demonstrations(query, n_shot, position, rollout=rollout)
                        
                        upperbound_prompt = build_vl_icl_prompt(
                            self.task, demonstrations, query,
                            mode="free",
                            warned=True,
                            max_images=8
                        )
                        upperbound_queries.append(query)
                        upperbound_prompts.append(upperbound_prompt)
                        
                        baseline_mode =  "free"
                        baseline_prompt = build_vl_icl_prompt(
                            self.task, demonstrations, query,
                            mode=baseline_mode,
                            warned=False,
                            max_images=8
                        )
                        baseline_queries.append(query)
                        baseline_prompts.append(baseline_prompt)
                    
                    upperbound_accuracy = self.evaluate_batch(upperbound_queries, upperbound_prompts, f"UPPERBOUND R{rollout+1}")
                    baseline_accuracy = self.evaluate_batch(baseline_queries, baseline_prompts, f"BASELINE R{rollout+1}")
                    
                    if upperbound_accuracy is not None and baseline_accuracy is not None:
                        upperbound_accuracies.append(upperbound_accuracy)
                        baseline_accuracies.append(baseline_accuracy)
                        difference = upperbound_accuracy - baseline_accuracy
                        paired_differences.append(difference)
                    else:
                        print(f"Position {position} R{rollout+1}: Failed")
                
                if upperbound_accuracies and baseline_accuracies:
                    position_data['upperbound_rollouts'] = upperbound_accuracies
                    position_data['baseline_rollouts'] = baseline_accuracies
                    position_data['upperbound_mean'] = float(np.mean(upperbound_accuracies))
                    position_data['upperbound_std'] = float(np.std(upperbound_accuracies)) if len(upperbound_accuracies) > 1 else 0.0
                    position_data['baseline_mean'] = float(np.mean(baseline_accuracies))
                    position_data['baseline_std'] = float(np.std(baseline_accuracies)) if len(baseline_accuracies) > 1 else 0.0
                    position_data['paired_differences'] = paired_differences
                    position_data['improvement_mean'] = float(np.mean(paired_differences))
                    position_data['improvement_std'] = float(np.std(paired_differences)) if len(paired_differences) > 1 else 0.0
                    
                    if len(paired_differences) > 1:
                        t_stat, p_value = stats.ttest_rel(upperbound_accuracies, baseline_accuracies)
                        position_data['p_value'] = float(p_value)
                
                results['position_results'][f'pos_{position}'] = position_data
        
        finally:
            if hasattr(self, 'engine') and self.engine is not None:
                try:
                    del self.engine
                    self.engine = None
                except:
                    pass
        
        model_short = self.model_name.split('/')[-1]
        filename = f"results/bounded_corruption_analysis/comprehensive_{self.dataset}_{model_short}_{n_shot}shot_{num_rollouts}rollouts.json"
        os.makedirs("results", exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation with upperbound and baseline')
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
    parser.add_argument('--n_shot', type=int, default=4,
                        help='Number of shots (demonstrations)')
    parser.add_argument('--num_samples', type=int, default=60,
                        help='Number of query samples to test')
    parser.add_argument('--num_rollouts', type=int, default=3,
                        help='Number of rollouts for baseline evaluation')
    
    args = parser.parse_args()
    
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Shots: {args.n_shot}")
    print(f"Samples: {args.num_samples}")
    print(f"Rollouts: {args.num_rollouts}")
    
    evaluator = ComprehensiveEvaluator(
        model_name=args.model_name,
        model_type=args.model_type,
        data_dir=args.data_dir,
        dataset=args.dataset
    )
    
    results = evaluator.run_comprehensive_evaluation(
        n_shot=args.n_shot,
        num_samples=args.num_samples,
        num_rollouts=args.num_rollouts
    )
    
    if results:
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())