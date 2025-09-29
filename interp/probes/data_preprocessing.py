import sys
import os
import json
import pickle
from typing import List, Dict, Tuple
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(root_dir)

from tasks.t2t_tasks import OperatorInductionTextTask

def extract_numbers_from_question(question: str) -> Tuple[int, int]:
    import re
    numbers = re.findall(r'\d+', question)
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    return None, None

def create_demonstration_prompt(demonstrations: List[Dict], query: Dict) -> str:
    """Create demonstration prompt for probe training"""
    prompt_parts = []
    
    for demo in demonstrations:
        question = demo['question']
        if isinstance(demo['answer'], list):
            answer = demo['answer'][0]
        else:
            answer = demo['answer']
        prompt_parts.append(f"{question} Answer: {answer}")
    
    prompt_parts.append(f"{query['question']}")
    prompt_parts.append("What mathematical operation does ? represent? Choose from: plus, minus, multiplication")
    prompt_parts.append("Answer:")
    
    return '\n'.join(prompt_parts)

def generate_probe_training_data(target_operator: str, 
                                num_samples: int = 600,
                                task: str = "operator_induction_text",
                                seed: int = 42) -> List[Dict]:
    np.random.seed(seed)
    data_dir = "../../VL-ICL"
    task_obj = OperatorInductionTextTask(data_dir)
    
    if target_operator == '+':
        other_operator = '-'
    elif target_operator == '-':
        other_operator = '+'
    else:
        other_operator = '+'
    
    target_queries = [q for q in task_obj.query_data if q['operator'] == target_operator]
    other_queries = [q for q in task_obj.query_data if q['operator'] == other_operator]
    training_data = []
    num_negative = num_samples // 2

    for i in range(num_negative):
        mixed_demonstrations = []
        for j in range(4):
            query = np.random.choice(other_queries)
            demo = task_obj.select_demonstrations(query, n_shot=1, seed=seed+i*100+j)[0]
            mixed_demonstrations.append(demo)
        
        test_query = np.random.choice(target_queries)
        prompt = create_demonstration_prompt(mixed_demonstrations, test_query)
        
        training_data.append({
            'prompt': prompt,
            'label': 0,
            'num_target': 0,
            'num_other': 4,
            'target_operator': target_operator,
            'other_operator': other_operator,
            'demonstrations': mixed_demonstrations,
            'query': test_query
        })
    
    num_positive = num_samples - num_negative
    num_per_presence = num_positive // 4
    
    for num_target in [1, 2, 3, 4]:
        num_other = 4 - num_target
        print(f"Generating {num_per_presence} positive samples ({num_target}/4 {target_operator} demonstrations)...")
        
        for i in range(num_per_presence):
            mixed_demonstrations = []
            
            for j in range(num_target):
                query = np.random.choice(target_queries)
                demo = task_obj.select_demonstrations(query, n_shot=1, seed=seed+i*200+num_target*50+j)[0]
                mixed_demonstrations.append(demo)

            for j in range(num_other):
                query = np.random.choice(other_queries)
                demo = task_obj.select_demonstrations(query, n_shot=1, seed=seed+i*200+num_target*50+num_target+j)[0]
                mixed_demonstrations.append(demo)
            
            np.random.shuffle(mixed_demonstrations)
            
            test_query = np.random.choice(target_queries)
            prompt = create_demonstration_prompt(mixed_demonstrations, test_query)
            
            training_data.append({
                'prompt': prompt,
                'label': 1,  
                'num_target': num_target,
                'num_other': num_other,
                'target_operator': target_operator,
                'other_operator': other_operator,
                'demonstrations': mixed_demonstrations,
                'query': test_query
            })
    
    np.random.shuffle(training_data)
    positive_count = sum(1 for sample in training_data if sample['label'] == 1)
    negative_count = len(training_data) - positive_count

    return training_data

def save_training_data(training_data: List[Dict], target_operator: str, save_dir: str = "probe_data"):
    os.makedirs(save_dir, exist_ok=True)
    op_clean = target_operator.replace("*", "mult")
    filename = f"training_data_{op_clean}.pkl"
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(training_data, f)
    
    return filepath

def main():
    operators = ['+', '-', 'x']
    
    for operator in operators:
        
        training_data = generate_probe_training_data(
            target_operator=operator,
            num_samples=600,
            seed=42
        )
        
        save_training_data(training_data, operator)

if __name__ == "__main__":
    main()