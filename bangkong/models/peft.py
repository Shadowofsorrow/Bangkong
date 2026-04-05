"""
Parameter-Efficient Fine-Tuning (PEFT) for Bangkong LLM Training System
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from ..config.schemas import BangkongConfig


class LoRALayer(nn.Module):
    """Low-Rank Adaptation (LoRA) layer for parameter-efficient fine-tuning."""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: int = 1):
        """
        Initialize LoRA layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            rank: Rank of the low-rank matrices
            alpha: Scaling factor for the LoRA output
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # Initialize low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Scaling factor
        self.scaling = alpha / rank
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=1)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LoRA layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with LoRA adaptation applied
        """
        # Compute low-rank adaptation
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return lora_output


class PEFTAdapter:
    """Adapter for applying PEFT techniques to models."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize PEFT adapter.
        
        Args:
            config: Bangkong configuration
        """
        self.config = config
        self.peft_type = getattr(config.training, 'peft_type', 'lora')
        self.lora_rank = getattr(config.training, 'lora_rank', 8)
        self.lora_alpha = getattr(config.training, 'lora_alpha', 1)
    
    def apply_lora_to_linear_layers(self, model: nn.Module) -> nn.Module:
        """
        Apply LoRA to linear layers in the model.
        
        Args:
            model: Model to apply LoRA to
            
        Returns:
            Model with LoRA applied to linear layers
        """
        # Keep track of modified layers
        modified_layers = []
        
        # Iterate through model modules
        for name, module in model.named_modules():
            # Apply LoRA to linear layers (but not to LoRA layers themselves)
            if isinstance(module, nn.Linear) and not isinstance(module, LoRALayer):
                # Replace the linear layer with a LoRA-enhanced version
                lora_layer = LoRALayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha
                )
                
                # Store the original weight and bias
                with torch.no_grad():
                    if hasattr(module, 'weight') and module.weight is not None:
                        # Create a new linear layer that combines original and LoRA
                        class LoRALinear(nn.Module):
                            def __init__(self, original_layer, lora_layer):
                                super().__init__()
                                self.original = original_layer
                                self.lora = lora_layer
                            
                            def forward(self, x):
                                original_output = self.original(x)
                                lora_output = self.lora(x)
                                return original_output + lora_output
                        
                        # Replace the module
                        parent_name = '.'.join(name.split('.')[:-1])
                        module_name = name.split('.')[-1]
                        
                        if parent_name:
                            parent_module = dict(model.named_modules())[parent_name]
                        else:
                            parent_module = model
                            
                        setattr(parent_module, module_name, LoRALinear(module, lora_layer))
                        modified_layers.append(name)
        
        print(f"Applied LoRA to {len(modified_layers)} linear layers")
        return model
    
    def apply_peft(self, model: nn.Module) -> nn.Module:
        """
        Apply parameter-efficient fine-tuning to the model.
        
        Args:
            model: Model to apply PEFT to
            
        Returns:
            Model with PEFT applied
        """
        if self.peft_type == 'lora':
            return self.apply_lora_to_linear_layers(model)
        elif self.peft_type == 'none':
            return model
        else:
            raise ValueError(f"Unsupported PEFT type: {self.peft_type}")


def apply_peft_to_model(model: nn.Module, config: BangkongConfig) -> nn.Module:
    """
    Apply parameter-efficient fine-tuning to a model.
    
    Args:
        model: Model to apply PEFT to
        config: Bangkong configuration
        
    Returns:
        Model with PEFT applied
    """
    adapter = PEFTAdapter(config)
    return adapter.apply_peft(model)