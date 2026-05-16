"""
Specialized model architectures for Bangkong LLM Training System
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from transformers import GPT2Config
from ..config.schemas import BangkongConfig

def create_specialized_model(config: BangkongConfig) -> nn.Module:
    """
    Create a specialized model based on domain configuration.
    Uses the Native Bangkong Architecture (BangkongGPT2LMHeadModel) for all domains,
    applying domain-specific hyperparameters to the configuration.

    Args:
        config: Bangkong configuration

    Returns:
        Specialized model (BangkongGPT2LMHeadModel)
    """
    domain = getattr(config.model, 'domain', 'general')
    
    # Base configuration mapping
    model_kwargs: Dict[str, Any] = {
        "vocab_size": config.model.vocab_size,
        "n_positions": config.model.sequence_length,
        "n_embd": config.model.hidden_size,
        "n_layer": config.model.num_layers,
        "n_head": config.model.num_heads,
        "pi_latent_dim": config.model.hidden_size
    }

    # Apply domain-specific hyperparameter tweaks
    if domain == 'code':
        model_kwargs.update({
            "resid_pdrop": 0.1,  # Slightly higher dropout for code
            "embd_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "layer_norm_epsilon": 1e-5,  # Tighter layer norm for code
        })
    elif domain == 'math':
        model_kwargs.update({
            "resid_pdrop": 0.05,  # Lower dropout for precise math
            "embd_pdrop": 0.05,
            "attn_pdrop": 0.05,
            "layer_norm_epsilon": 1e-6,  # Very tight layer norm for math
        })
    elif domain == 'scientific':
        model_kwargs.update({
            "resid_pdrop": 0.1,
            "embd_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "layer_norm_epsilon": 1e-5,
        })
    else:
        # Defaults for 'general' or others
        pass

    # Instantiate using Native Bangkong GPT-2 model
    from bangkong.models.bangkong_model import BangkongGPT2LMHeadModel, BangkongConfig as NativeBangkongConfig
    
    bangkong_hf_config = NativeBangkongConfig(**model_kwargs)
    return BangkongGPT2LMHeadModel(bangkong_hf_config)