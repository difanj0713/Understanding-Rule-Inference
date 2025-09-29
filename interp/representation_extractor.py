import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_factory import create_model
from interp.helper import get_transformer_layers, HookManager, register_layer_hooks

class RepresentationExtractor:
    def __init__(self, model_name: str, model_type: str = "qwen3"):
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.hook_manager = HookManager()
        
    def load_model(self):
        self.model = create_model(self.model_type, self.model_name)
        self.tokenizer = self.model.tokenizer
        
    def _get_transformer_layers(self):
        return get_transformer_layers(self.model)
    
    def _register_hooks(self, layers_to_extract: Optional[List[int]] = None):
        language_model = self.model.model if hasattr(self.model, 'model') else self.model
        register_layer_hooks(language_model, layers_to_extract, self.hook_manager)
    
    def _clear_hooks(self):
        self.hook_manager.clear()
    
    def extract_representations(self, 
                              texts: List[str], 
                              layers_to_extract: Optional[List[int]] = None,
                              return_attention: bool = True) -> Dict[str, torch.Tensor]:
        
        self._register_hooks(layers_to_extract)
        
        results = {}
        
        for i, text in enumerate(texts):
            self.hook_manager.activations = {}
            self.hook_manager.attention_weights = {}
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.model.device)
            
            with torch.no_grad():
                outputs = self.model.model(**model_inputs, output_attentions=return_attention)
            
            results[f'sample_{i}'] = {
                'text': text,
                'input_ids': model_inputs['input_ids'].cpu(),
                'hidden_states': dict(self.hook_manager.activations),
                'attention_weights': dict(self.hook_manager.attention_weights) if return_attention else {}
            }
        
        self._clear_hooks()
        return results
    
    def extract_token_representations(self, 
                                    text: str,
                                    target_tokens: List[str],
                                    layers_to_extract: Optional[List[int]] = None) -> Dict[str, Any]:
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.model.device)
        input_ids = model_inputs['input_ids'][0]
        
        token_positions = {}
        for target_token in target_tokens:
            target_token_id = self.tokenizer.encode(target_token, add_special_tokens=False)[0]
            positions = (input_ids == target_token_id).nonzero(as_tuple=True)[0].tolist()
            token_positions[target_token] = positions
        
        representations = self.extract_representations([text], layers_to_extract)
        sample_data = representations['sample_0']
        
        token_reprs = {}
        for token, positions in token_positions.items():
            token_reprs[token] = {}
            for layer_name, hidden_states in sample_data['hidden_states'].items():
                token_reprs[token][layer_name] = []
                for pos in positions:
                    if pos < hidden_states.shape[1]:
                        token_reprs[token][layer_name].append(hidden_states[0, pos, :].numpy())
        
        return {
            'token_representations': token_reprs,
            'token_positions': token_positions,
            'input_ids': input_ids.cpu().numpy(),
            'text': text
        }