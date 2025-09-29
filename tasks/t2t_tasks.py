import random
import re
import copy
import os
import json
from typing import List, Dict, Optional, Tuple
from .base_task import BaseTask
import logging
from utils.evaluation_utils import extract_number_from_response, extract_text_from_response

logger = logging.getLogger(__name__)

class OperatorInductionTextTask(BaseTask):
    """Text-only Operator Induction Task for LLM extension"""
    
    def __init__(self, data_dir: str):
        super().__init__("operator_induction_text", data_dir)
        self.hybrid_evaluator = None
        if self.support_data:
            logger.info(f"Sample support data: {self.support_data[0]}")
        if self.query_data:
            logger.info(f"Sample query data: {self.query_data[0]}")
    
    def _get_evaluator(self):
        if self.hybrid_evaluator is None:
            from evaluation.llm_judge import HybridEvaluator
            self.hybrid_evaluator = HybridEvaluator()
        return self.hybrid_evaluator
    
    def get_task_instruction(self, mode="constrained", warned=False) -> str:
        base_instruction = ("The text contains two digit numbers and a ? representing the mathematical operator. "
                           "Induce the mathematical operator (addition, multiplication, minus) according to the "
                           "results of the in-context examples and calculate the result.")
        
        if warned:
            warning_instruction = (" IMPORTANT: Follow these steps:\n"
                                 "1) Look at each example and identify what operation it shows\n"
                                 "2) Count how many examples show addition (+), subtraction (-), and multiplication (*)\n"
                                 "3) The operation that appears most frequently is the correct one\n"
                                 "4) Apply that operation to solve the final problem\n")
            base_instruction = base_instruction + warning_instruction
        
        if mode == "constrained":
            return base_instruction + " Answer with only the final number. Answer: "
        elif mode == "free":
            return base_instruction + " Reason carefully step by step and provide the final answer. Answer: "
        else:
            return base_instruction
    
    def format_demonstration(self, support_item: Dict, include_image_token=True, mode="constrained") -> str:
        if mode == "constrained":
            return f"{support_item['question']}\nAnswer: {support_item['answer']}"
        elif mode == "free":
            return f"{support_item['question']}\nAnswer: {support_item['answer']}"
        else:
            return f"{support_item['question']}\n{support_item['answer']}"
    
    def select_demonstrations(self, query: Dict, n_shot: int, seed: Optional[int] = None) -> List[Dict]:
        if n_shot == 0:
            return []
            
        operator_index = {'+': 0, '-': 1, 'x': 2}
        operator = query['operator']
        operator_idx = operator_index[operator]
        
        if seed is not None:
            random_state = random.getstate()
            random.seed(seed)
        
        selected = random.sample(self.support_data, n_shot)
        demonstrations = []
        
        for support in selected:
            demo = copy.deepcopy(support)
            if isinstance(demo['answer'], list):
                demo['answer'] = demo['answer'][operator_idx]
            demonstrations.append(demo)
        
        if seed is not None:
            random.setstate(random_state)
        
        return demonstrations
    
    def format_query(self, query: Dict, include_image_token=True, mode="constrained") -> str:
        if mode == "free":
            return f"{query['question']} Think for this question step by step."
        else:
            return f"{query['question']}"
    
    def evaluate_response(self, query: Dict, response: str, mode="constrained") -> bool:
        evaluator = self._get_evaluator()
        return evaluator.evaluate_response(query, response, mode, "operator_induction")


class OperatorInductionInterleavedTextTask(BaseTask):
    """Interleaved Text Operator Induction Task - numbers are separated in the question format"""
    
    def __init__(self, data_dir: str):
        super().__init__("operator_induction_interleaved_text", data_dir)
        self.hybrid_evaluator = None
        if self.support_data:
            logger.info(f"Sample support data: {self.support_data[0]}")
        if self.query_data:
            logger.info(f"Sample query data: {self.query_data[0]}")
    
    def _get_evaluator(self):
        if self.hybrid_evaluator is None:
            from evaluation.llm_judge import HybridEvaluator
            self.hybrid_evaluator = HybridEvaluator()
        return self.hybrid_evaluator
    
    def get_task_instruction(self, mode="constrained", warned=False) -> str:
        base_instruction = ("The text contains mathematical expressions with two numbers. "
                           "Induce the mathematical operator (addition, multiplication, minus) according to the "
                           "results of the in-context examples and calculate the result.")
        
        if warned:
            warning_instruction = (" IMPORTANT: Follow these steps:\n"
                                 "1) Look at each example and identify what operation it shows\n"
                                 "2) Count how many examples show addition (+), subtraction (-), and multiplication (*)\n"
                                 "3) The operation that appears most frequently is the correct one\n"
                                 "4) Apply that operation to solve the final problem\n")
            base_instruction = base_instruction + warning_instruction
        
        if mode == "constrained":
            return base_instruction + " Answer with only the final number. Answer: "
        elif mode == "free":
            return base_instruction + " Reason carefully step by step and provide the final answer. Answer: "
        else:
            return base_instruction
    
    def format_demonstration(self, support_item: Dict, include_image_token=True, mode="constrained") -> str:
        if mode == "constrained":
            return f"{support_item['question']}\nAnswer: {support_item['answer']}"
        elif mode == "free":
            return f"{support_item['question']}\nAnswer: {support_item['answer']}"
        else:
            return f"{support_item['question']}\n{support_item['answer']}"
    
    def select_demonstrations(self, query: Dict, n_shot: int, seed: Optional[int] = None) -> List[Dict]:
        if n_shot == 0:
            return []
            
        operator_index = {'+': 0, '-': 1, 'x': 2}
        operator = query['operator']
        operator_idx = operator_index[operator]
        
        if seed is not None:
            random_state = random.getstate()
            random.seed(seed)
        
        selected = random.sample(self.support_data, n_shot)
        demonstrations = []
        
        for support in selected:
            demo = copy.deepcopy(support)
            if isinstance(demo['answer'], list):
                demo['answer'] = demo['answer'][operator_idx]
            demonstrations.append(demo)
        
        if seed is not None:
            random.setstate(random_state)
        
        return demonstrations
    
    def format_query(self, query: Dict, include_image_token=True, mode="constrained") -> str:
        if mode == "free":
            return f"{query['question']} Think for this question step by step."
        else:
            return f"{query['question']}"
    
    def evaluate_response(self, query: Dict, response: str, mode="constrained") -> bool:
        evaluator = self._get_evaluator()
        return evaluator.evaluate_response(query, response, mode, "operator_induction")