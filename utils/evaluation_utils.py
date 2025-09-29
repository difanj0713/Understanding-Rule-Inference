"""
Robust evaluation utilities for extracting answers from model responses
"""
import re
from typing import Optional, List

def extract_number_from_response(response: str) -> Optional[int]:
    """
    Robustly extract numerical answer from model response using priority-based approach.
    
    Priority order:
    1. Numbers after explicit answer markers ("Answer:", "Final answer:", etc.)
    2. Numbers after equals sign ("= number")
    3. Numbers after contextual words ("equals", "is", "result")
    4. Last number in the response (fallback)
    
    Args:
        response: Model response string
        
    Returns:
        Extracted integer or None if no valid number found
    """
    if not response or not isinstance(response, str):
        return None
    
    response = response.strip()
    
    # Priority 1: Look for explicit answer markers (case insensitive)
    answer_patterns = [
        r'final\s+answer\s*:\s*(-?\d+)',  # "Final answer: 42"
        r'answer\s*:\s*(-?\d+)',          # "Answer: 42"
        r'result\s*:\s*(-?\d+)',          # "Result: 42"
        r'solution\s*:\s*(-?\d+)',        # "Solution: 42"
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            try:
                # Take the last match in case of multiple
                return int(matches[-1])
            except ValueError:
                continue
    
    # Priority 2: Look for numbers immediately after equals sign
    equals_patterns = [
        r'=\s*(-?\d+)',           # "= 42" or "=-5"
        r'equals\s+(-?\d+)',      # "equals 42"
    ]
    
    for pattern in equals_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            try:
                # Take the last match
                return int(matches[-1])
            except ValueError:
                continue
    
    # Priority 3: Look for numbers after contextual keywords
    context_patterns = [
        r'(?:the\s+)?(?:answer|result|solution)\s+(?:is|equals?)\s+(-?\d+)',  # "the answer is 42"
        r'(?:is|equals?)\s+(-?\d+)',     # "is 42"
    ]
    
    for pattern in context_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            try:
                # Take the last match
                return int(matches[-1])
            except ValueError:
                continue
    
    # Priority 4: Fallback - find all numbers and take the last one
    all_numbers = re.findall(r'-?\d+', response)
    if all_numbers:
        try:
            return int(all_numbers[-1])
        except ValueError:
            pass
    
    return None


def extract_text_from_response(response: str, query_answer: List[str]) -> Optional[List[str]]:
    """
    Extract text answers for tasks like COBSAT where answers are strings.
    """
    if not response or not isinstance(response, str):
        return None
    
    response = response.strip().lower()
    
    if len(query_answer) == 2:
        found_components = []
        for component in query_answer:
            if component.lower() in response:
                found_components.append(component.lower())
        
        if len(found_components) == len(query_answer):
            return found_components
    
    for component in query_answer:
        if component.lower() in response:
            return [component.lower()]
    
    return None