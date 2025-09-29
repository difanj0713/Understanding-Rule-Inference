import sys
import os
import json
import pickle
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(root_dir)

from interp.representation_extractor import RepresentationExtractor
from tasks.t2t_tasks import OperatorInductionTextTask

def get_model_config(model_name: str) -> Dict:
    model_configs = {
        "Qwen/Qwen3-4B": {"num_layers": 32, "hidden_dim": 3584},
        "Qwen/Qwen3-8B": {"num_layers": 36, "hidden_dim": 4096}, 
        "Qwen/Qwen3-14B": {"num_layers": 40, "hidden_dim": 5120},
        "Qwen/Qwen3-32B": {"num_layers": 64, "hidden_dim": 5120}
    }
    
    if model_name in model_configs:
        return model_configs[model_name]
    else:
        return {"num_layers": 36, "hidden_dim": 4096}

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

def load_trained_probes(probes_path: str):
    with open(probes_path, 'rb') as f:
        trained_probes_data = pickle.load(f)
    
    trained_probes = {}
    for layer_idx, probe_data in trained_probes_data.items():
        input_dim = probe_data['input_dim']
        probe = LinearProbe(input_dim)
        probe.load_state_dict(probe_data['probe_state_dict'])
        probe_data_copy = probe_data.copy()
        probe_data_copy['probe'] = probe
        trained_probes[layer_idx] = probe_data_copy
    return trained_probes

def extract_numbers_from_question(question: str) -> Tuple[int, int]:
    import re
    numbers = re.findall(r'\d+', question)
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    return None, None

def create_evaluation_prompt(demonstrations: List[Dict], query: Dict, 
                           scenario: str = "all_correct", 
                           corrupted_position: int = 2) -> str:
    prompt_parts = []
    
    for i, demo in enumerate(demonstrations):
        question = demo['question']
        
        if isinstance(demo['answer'], list):
            correct_answer = demo['answer'][0]
        else:
            correct_answer = demo['answer']
        
        if scenario == "one_corrupted" and i == corrupted_position:
            num1, num2 = extract_numbers_from_question(question)
            if num1 is not None and num2 is not None:
                corrupted_answer = num1 - num2  # Force minus operation
                prompt_parts.append(f"{question} Answer: {corrupted_answer}")
            else:
                prompt_parts.append(f"{question} Answer: {correct_answer}")
        else:
            prompt_parts.append(f"{question} Answer: {correct_answer}")
    
    prompt_parts.append(f"{query['question']}")
    prompt_parts.append("What mathematical operation does ? represent? Choose from: plus, minus, multiplication")
    prompt_parts.append("Answer:")
    
    return '\n'.join(prompt_parts)

def generate_evaluation_dataset(num_samples_per_scenario: int = 300, seed: int = 100) -> Dict:
    np.random.seed(seed)
    data_dir = "../../VL-ICL"
    task = OperatorInductionTextTask(data_dir)
    
    plus_queries = [q for q in task.query_data if q['operator'] == '+']
    minus_queries = [q for q in task.query_data if q['operator'] == '-']
    
    evaluation_data = {
        'all_correct': [],
        'one_corrupted': []
    }
    
    for i in range(num_samples_per_scenario):
        query = np.random.choice(plus_queries)
        demonstrations = task.select_demonstrations(query, n_shot=4, seed=seed+i)
        
        prompt = create_evaluation_prompt(demonstrations, query, "all_correct")
        
        evaluation_data['all_correct'].append({
            'prompt': prompt,
            'demonstrations': demonstrations,
            'query': query
        })

    corrupted_position = 2
    
    for i in range(num_samples_per_scenario):
        query = np.random.choice(plus_queries)
        demonstrations = task.select_demonstrations(query, n_shot=4, seed=seed+i+1000)
        
        prompt = create_evaluation_prompt(demonstrations, query, "one_corrupted", corrupted_position)
        
        evaluation_data['one_corrupted'].append({
            'prompt': prompt,
            'demonstrations': demonstrations,
            'query': query,
            'corrupted_position': corrupted_position
        })
    
    return evaluation_data

def evaluate_probes_two_scenarios(num_samples_per_scenario: int = 300) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plus_probes_path = "trained_probes/probes_Qwen_Qwen3_8B_+.pkl"
    minus_probes_path = "trained_probes/probes_Qwen_Qwen3_8B_-.pkl"

    plus_probes = load_trained_probes(plus_probes_path)
    minus_probes = load_trained_probes(minus_probes_path)
    
    extractor = RepresentationExtractor("Qwen/Qwen3-8B", "qwen3")
    extractor.load_model()
    
    eval_data = generate_evaluation_dataset(num_samples_per_scenario)
    scenarios = ["all_correct", "one_corrupted"]
    all_layers = sorted(plus_probes.keys())
    
    results = {
        'all_correct': {
            'plus_probe_confidences': {layer: [] for layer in all_layers},
            'minus_probe_confidences': {layer: [] for layer in all_layers}
        },
        'one_corrupted': {
            'plus_probe_confidences': {layer: [] for layer in all_layers},
            'minus_probe_confidences': {layer: [] for layer in all_layers}
        },
        'layers': all_layers,
        'num_samples_per_scenario': num_samples_per_scenario,
        'corrupted_position': 2
    }
    
    for scenario in scenarios:
        scenario_samples = eval_data[scenario]
        
        for sample_idx, sample in enumerate(scenario_samples):
            prompt = sample['prompt']
            representations = extractor.extract_representations(
                [prompt], all_layers, return_attention=False
            )
            
            for layer_idx in all_layers:
                layer_name = f"layer_{layer_idx}"
                if layer_name in representations['sample_0']['hidden_states']:
                    hidden_state = representations['sample_0']['hidden_states'][layer_name]
                    final_token_repr = hidden_state[0, -1, :].unsqueeze(0).to(device).float()
                    
                    plus_probe = plus_probes[layer_idx]['probe'].to(device)
                    plus_probe.eval()
                    
                    with torch.no_grad():
                        plus_logits = plus_probe(final_token_repr)
                        plus_probs = F.softmax(plus_logits, dim=1)
                        plus_confidence = plus_probs[0, 1].item()
                        
                        results[scenario]['plus_probe_confidences'][layer_idx].append(plus_confidence)
                    
                    minus_probe = minus_probes[layer_idx]['probe'].to(device)
                    minus_probe.eval()
                    
                    with torch.no_grad():
                        minus_logits = minus_probe(final_token_repr)
                        minus_probs = F.softmax(minus_logits, dim=1)
                        minus_confidence = minus_probs[0, 1].item() 
                        
                        results[scenario]['minus_probe_confidences'][layer_idx].append(minus_confidence)
    
    summary_results = {
        'all_correct': {
            'plus_probe_avg': [],
            'minus_probe_avg': [],
            'plus_probe_std': [],
            'minus_probe_std': []
        },
        'one_corrupted': {
            'plus_probe_avg': [],
            'minus_probe_avg': [],
            'plus_probe_std': [],
            'minus_probe_std': []
        },
        'layers': all_layers
    }
    
    for scenario in scenarios:
        for layer_idx in all_layers:
            plus_confidences = results[scenario]['plus_probe_confidences'][layer_idx]
            minus_confidences = results[scenario]['minus_probe_confidences'][layer_idx]
            
            plus_avg = np.mean(plus_confidences)
            plus_std = np.std(plus_confidences)
            minus_avg = np.mean(minus_confidences)
            minus_std = np.std(minus_confidences)
            
            summary_results[scenario]['plus_probe_avg'].append(plus_avg)
            summary_results[scenario]['plus_probe_std'].append(plus_std)
            summary_results[scenario]['minus_probe_avg'].append(minus_avg)
            summary_results[scenario]['minus_probe_std'].append(minus_std)
    
    results['summary'] = summary_results
    os.makedirs('probe_results', exist_ok=True)
    results_path = 'probe_results/two_scenario_evaluation_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    return results

def main():
    
    
    parser = argparse.ArgumentParser(description='Evaluate Linear Probes in 2 Scenarios')
    parser.add_argument('--num_samples_per_scenario', type=int, default=300,
                        help='Number of samples per scenario')
    
    args = parser.parse_args()
    results = evaluate_probes_two_scenarios(num_samples_per_scenario=args.num_samples_per_scenario)

if __name__ == "__main__":
    main()