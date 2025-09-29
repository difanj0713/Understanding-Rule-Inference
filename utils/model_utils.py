import torch
from typing import Dict, List, Tuple, Any, Callable

def get_model_tokenizer(model, model_type: str):
    if model_type == "qwen25":
        return model.processor.tokenizer
    elif model_type == "qwen3":
        return model.tokenizer  
    elif model_type == "internvl":
        return model.tokenizer
    else:
        raise ValueError(f"Unsupported model type for tokenizer access: {model_type}")

def get_language_model(model, model_type: str):
    if model_type == "qwen25":
        return model.model.language_model  
    elif model_type == "qwen3":
        return model.model.model  
    elif model_type == "internvl":
        return model.model.language_model  
    else:
        raise ValueError(f"Unsupported model type for language model access: {model_type}")

def get_transformer_layers(language_model, model_type: str):
    if model_type == "qwen25":
        return language_model.model.layers  
    elif model_type == "qwen3":
        return language_model.layers 
    elif model_type == "internvl":
        return language_model.model.layers  
    else:
        raise ValueError(f"Unsupported model type for transformer layers: {model_type}")


def register_attention_hooks(language_model, model_type: str, hook_fn_factory: Callable[[int], Callable]) -> List[Any]:
    transformer_layers = get_transformer_layers(language_model, model_type)
    hooks = []
    
    if model_type == "internvl":
        for layer_idx in range(len(transformer_layers)):
            layer = transformer_layers[layer_idx]
            hook = layer.self_attn.o_proj.register_forward_hook(hook_fn_factory(layer_idx))
            hooks.append(hook)
            
    elif model_type == "qwen25":
        for layer_idx in range(len(transformer_layers)):
            layer = transformer_layers[layer_idx]
            hook = layer.self_attn.o_proj.register_forward_hook(hook_fn_factory(layer_idx))
            hooks.append(hook)
            
    elif model_type == "qwen3":
        for layer_idx in range(len(transformer_layers)):
            layer = transformer_layers[layer_idx]
            hook = layer.self_attn.o_proj.register_forward_hook(hook_fn_factory(layer_idx))
            hooks.append(hook)
            
    else:
        raise ValueError(f"Unsupported model type for attention hooks: {model_type}")
    
    return hook


def get_model_embeddings_method(model, model_type: str):
    return "standard"

def get_model_specific_config(model_type: str) -> Dict[str, Any]:
    base_config = {
        "supports_flash_attention": True,
        "default_max_tokens": 8,
        "attention_head_masking": True,
    }
    
    if model_type == "internvl":
        return {
            **base_config,
            "vision_encoder": True,
            "language_model_nested": True,
        }
        
    elif model_type == "qwen25":
        return {
            **base_config,
            "vision_encoder": True,
            "language_model_direct": True,
            "processor_tokenizer": True,
        }
        
    elif model_type == "qwen3":
        return {
            **base_config,
            "vision_encoder": False,
            "language_model_direct": True,
            "pure_llm": True,
        }
        
    else:
        raise ValueError(f"Unsupported model type for config: {model_type}")

def validate_model_architecture(model, model_type: str) -> bool:
    try:
        tokenizer = get_model_tokenizer(model, model_type)
        language_model = get_language_model(model, model_type)
        transformer_layers = get_transformer_layers(language_model, model_type)
        
        assert tokenizer is not None, "Tokenizer access failed"
        assert language_model is not None, "Language model access failed"
        assert len(transformer_layers) > 0, "No transformer layers found"

        first_layer = transformer_layers[0]
        assert hasattr(first_layer, 'self_attn'), "No self_attn module found"
        assert hasattr(first_layer.self_attn, 'o_proj'), "No o_proj module found"
        
        return True
        
    except Exception as e:
        print(f"Model architecture validation failed for {model_type}: {e}")
        import traceback
        traceback.print_exc()
        return False