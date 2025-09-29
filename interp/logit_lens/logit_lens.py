import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_factory import create_model

class LogitLens:
    def __init__(self, model_name: str, model_type: str = "qwen3"):
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.ln_f = None
        self.lm_head = None
        
    def load_model(self):
        self.model = create_model(self.model_type, self.model_name)
        self.tokenizer = self.model.tokenizer
        self._find_model_components()
            
    def _find_model_components(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model'):
            transformer = self.model.model.language_model.model
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'model'):
            transformer = self.model.model.model
        else:
            transformer = self.model.model
            
        if hasattr(transformer, 'norm'):
            self.ln_f = transformer.norm
        elif hasattr(transformer, 'layer_norm'):
            self.ln_f = transformer.layer_norm
        else:
            self.ln_f = None
            
        if hasattr(transformer, 'lm_head'):
            self.lm_head = transformer.lm_head
        elif hasattr(transformer, 'output_projection'):
            self.lm_head = transformer.output_projection
        elif hasattr(self.model.model, 'lm_head'):
            self.lm_head = self.model.model.lm_head
        else:
            self.lm_head = None
            
    def apply_logit_lens(self, 
                        hidden_states: torch.Tensor,
                        top_k: int = 10,
                        apply_ln: bool = True) -> Dict[str, torch.Tensor]:
        
        if apply_ln and self.ln_f is not None:
            hidden_states = self.ln_f(hidden_states)
            
        logits = self.lm_head(hidden_states)
        probabilities = F.softmax(logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(probabilities, top_k, dim=-1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'top_k_probs': top_k_probs,
            'top_k_indices': top_k_indices
        }
    
    def analyze_layer_predictions(self,
                                text: str,
                                target_position: int = -1,
                                layers_to_analyze: Optional[List[int]] = None,
                                top_k: int = 5) -> Dict[int, Dict[str, any]]:
        
        from .representation_extractor import RepresentationExtractor
        extractor = RepresentationExtractor(self.model_name, self.model_type)
        extractor.model = self.model
        extractor.tokenizer = self.tokenizer
        
        representations = extractor.extract_representations([text], layers_to_analyze, return_attention=False)
        sample_data = representations['sample_0']
        
        results = {}
        
        for layer_name, hidden_states in sample_data['hidden_states'].items():
            layer_idx = int(layer_name.split('_')[1])
            
            if target_position == -1:
                target_position = hidden_states.shape[1] - 1
                
            target_hidden_state = hidden_states[0, target_position, :].unsqueeze(0).unsqueeze(0)
            target_hidden_state = target_hidden_state.to(self.model.model.device)
            
            lens_results = self.apply_logit_lens(target_hidden_state, top_k=top_k)
            
            top_tokens = []
            for i in range(top_k):
                token_id = lens_results['top_k_indices'][0, 0, i].item()
                token_str = self.tokenizer.decode([token_id])
                prob = lens_results['top_k_probs'][0, 0, i].item()
                top_tokens.append({'token': token_str, 'prob': prob, 'token_id': token_id})
            
            results[layer_idx] = {
                'top_predictions': top_tokens,
                'position': target_position,
                'layer_name': layer_name
            }
        
        return results
    
    def probe_answer_encoding(self,
                             icl_prompt: str,
                             answer_candidates: List[str],
                             layers_to_analyze: Optional[List[int]] = None) -> Dict[int, Dict[str, float]]:
        
        answer_token_ids = []
        for answer in answer_candidates:
            tokens = self.tokenizer.encode(str(answer), add_special_tokens=False)
            if len(tokens) == 1:
                answer_token_ids.append(tokens[0])
            else:
                answer_token_ids.append(tokens[0])
        
        from .representation_extractor import RepresentationExtractor
        extractor = RepresentationExtractor(self.model_name, self.model_type)
        extractor.model = self.model
        extractor.tokenizer = self.tokenizer
        
        representations = extractor.extract_representations([icl_prompt], layers_to_analyze, return_attention=False)
        sample_data = representations['sample_0']
        
        results = {}
        
        for layer_name, hidden_states in sample_data['hidden_states'].items():
            layer_idx = int(layer_name.split('_')[1])
            
            final_hidden_state = hidden_states[0, -1, :].unsqueeze(0).unsqueeze(0)
            final_hidden_state = final_hidden_state.to(self.model.model.device)
            
            lens_results = self.apply_logit_lens(final_hidden_state, top_k=len(self.tokenizer))
            
            layer_results = {}
            for i, answer in enumerate(answer_candidates):
                token_id = answer_token_ids[i]
                prob = lens_results['probabilities'][0, 0, token_id].item()
                layer_results[answer] = prob
            
            results[layer_idx] = layer_results
        
        return results
    
    def probe_operator_encoding(self,
                               icl_prompt: str,
                               correct_operator: str,
                               layers_to_analyze: Optional[List[int]] = None) -> Dict[int, Dict[str, float]]:
        
        operators = ['+', '-', '*']
        operator_token_ids = []
        for op in operators:
            tokens = self.tokenizer.encode(op, add_special_tokens=False)
            if len(tokens) == 1:
                operator_token_ids.append(tokens[0])
            else:
                operator_token_ids.append(tokens[0])
        
        operator_question = icl_prompt + "\n\nWhat mathematical operator should be used here? Choose from: + - *\nAnswer:"
        
        from .representation_extractor import RepresentationExtractor
        extractor = RepresentationExtractor(self.model_name, self.model_type)
        extractor.model = self.model
        extractor.tokenizer = self.tokenizer
        
        representations = extractor.extract_representations([operator_question], layers_to_analyze, return_attention=False)
        sample_data = representations['sample_0']
        
        results = {}
        
        for layer_name, hidden_states in sample_data['hidden_states'].items():
            layer_idx = int(layer_name.split('_')[1])
            
            final_hidden_state = hidden_states[0, -1, :].unsqueeze(0).unsqueeze(0)
            final_hidden_state = final_hidden_state.to(self.model.model.device)
            
            lens_results = self.apply_logit_lens(final_hidden_state, top_k=len(self.tokenizer))
            
            layer_results = {}
            for i, op in enumerate(operators):
                token_id = operator_token_ids[i]
                prob = lens_results['probabilities'][0, 0, token_id].item()
                layer_results[op] = prob
            
            layer_results['correct_operator'] = correct_operator
            layer_results['correct_prob'] = layer_results[correct_operator]
            
            results[layer_idx] = layer_results
        
        return results
    
    def compare_clean_vs_corrupted_operators(self,
                                           clean_prompt: str,
                                           corrupted_prompt: str,
                                           correct_operator: str,
                                           layers_to_analyze: Optional[List[int]] = None) -> Dict[str, Dict[int, Dict[str, float]]]:
        
        clean_results = self.probe_operator_encoding(clean_prompt, correct_operator, layers_to_analyze)
        corrupted_results = self.probe_operator_encoding(corrupted_prompt, correct_operator, layers_to_analyze)
        
        return {
            'clean': clean_results,
            'corrupted': corrupted_results
        }
    
    def compare_clean_vs_corrupted(self,
                                  clean_prompt: str,
                                  corrupted_prompt: str,
                                  answer_candidates: List[str],
                                  layers_to_analyze: Optional[List[int]] = None) -> Dict[str, Dict[int, Dict[str, float]]]:
        
        clean_results = self.probe_answer_encoding(clean_prompt, answer_candidates, layers_to_analyze)
        corrupted_results = self.probe_answer_encoding(corrupted_prompt, answer_candidates, layers_to_analyze)
        
        return {
            'clean': clean_results,
            'corrupted': corrupted_results
        }