import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

class LLMJudge:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            logger.info(f"Loading LLM Judge: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("LLM Judge loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM Judge: {e}")
            raise
    
    def evaluate_mathematical_reasoning(self, response: str, expected_answer: Any, task_type: str = "operator_induction") -> bool:
        """
        Evaluate if a mathematical reasoning response contains the correct answer
        
        Args:
            response: The model's free-form response
            expected_answer: The expected numerical answer
            task_type: Type of mathematical task
            
        Returns:
            bool: True if the response contains the correct answer
        """

        if task_type == "operator_induction":
            prompt = f"""You are evaluating a mathematical reasoning response. 

The task is: Given images showing mathematical expressions with unknown operators, induce the operator and calculate the result.

Expected Answer: {expected_answer}

Model Response: "{response}"

Question: Does the model's response contain the correct final answer of {expected_answer}? 

Look for the final numerical answer in the response, even if the reasoning process contains errors or is verbose. The response is correct if it arrives at the final answer {expected_answer}, regardless of the path taken.

Answer with exactly one word: "YES" if the response contains the correct answer {expected_answer}, "NO" if it does not."""

        elif task_type == "clevr":
            prompt = f"""You are evaluating a counting/reasoning response about objects in an image.

Expected Answer: {expected_answer}

Model Response: "{response}"

Question: Does the model's response contain the correct final answer of {expected_answer}?

Answer with exactly one word: "YES" if the response contains the correct count/answer {expected_answer}, "NO" if it does not."""
        
        else:
            prompt = f"""You are evaluating a mathematical reasoning response.

Expected Answer: {expected_answer}

Model Response: "{response}"

Question: Does the model's response contain the correct final answer of {expected_answer}?

Answer with exactly one word: "YES" if correct, "NO" if incorrect."""
        
        try:
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            input_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10, 
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            judge_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            judge_response_clean = judge_response.upper().strip()
            
            if "YES" in judge_response_clean:
                return True
            elif "NO" in judge_response_clean:
                return False
            else:
                logger.warning(f"Judge gave unclear response: '{judge_response}', falling back to keyword matching")
                return self._fallback_evaluation(response, expected_answer)
                
        except Exception as e:
            logger.error(f"LLM Judge evaluation failed: {e}")
            return self._fallback_evaluation(response, expected_answer)
    
    def _fallback_evaluation(self, response: str, expected_answer: Any) -> bool:
        try:
            numbers = re.findall(r'-?\d+', response)
            if not numbers:
                return False
            
            predicted = int(numbers[-1]) 
            expected = int(expected_answer)
            
            return predicted == expected
        except:
            return False

class HybridEvaluator:
    def __init__(self, judge_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.llm_judge = None
        self.judge_model_name = judge_model_name
    
    def _get_llm_judge(self):
        if self.llm_judge is None:
            self.llm_judge = LLMJudge(self.judge_model_name)
        return self.llm_judge
    
    def evaluate_response(self, query: Dict, response: str, mode: str = "constrained", task_type: str = "operator_induction") -> bool:
        if mode == "constrained":
            return self._regex_evaluation(response, query, task_type)
        
        elif mode == "free":
            expected_answer = self._extract_expected_answer(query, task_type)
            judge = self._get_llm_judge()
            return judge.evaluate_mathematical_reasoning(response, expected_answer, task_type)
        
        else:
            raise ValueError(f"Unknown evaluation mode: {mode}")
    
    def _regex_evaluation(self, response: str, query: Dict, task_type: str) -> bool:
        response = response.strip()
        
        if task_type == "operator_induction":
            # Extract numbers (including negative)
            numbers = re.findall(r'-?\d+', response)
            if not numbers:
                return False
            
            try:
                predicted = int(numbers[-1])
            except ValueError:
                return False
            
            ground_truth = query['answer']
            if isinstance(ground_truth, list):
                operator_index = {'+': 0, '-': 1, 'x': 2}
                operator = query['operator']
                operator_idx = operator_index[operator]
                ground_truth = ground_truth[operator_idx]
            
            return predicted == int(ground_truth)
        
        elif task_type == "clevr":
            numbers = re.findall(r'-?\d+', response)
            if not numbers:
                return False
            try:
                predicted = int(numbers[-1])
                return predicted == query['answer']
            except ValueError:
                return False
        
        return False
    
    def _extract_expected_answer(self, query: Dict, task_type: str) -> Any:
        if task_type == "operator_induction":
            ground_truth = query['answer']
            if isinstance(ground_truth, list):
                operator_index = {'+': 0, '-': 1, 'x': 2}
                operator = query['operator']
                operator_idx = operator_index[operator]
                return ground_truth[operator_idx]
            return ground_truth
        
        elif task_type in ["clevr", "textocr", "open_mi"]:
            return query['answer']
        
        return query.get('answer', 'unknown')