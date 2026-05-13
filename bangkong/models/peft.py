"""
Parameter-Efficient Fine-Tuning (PEFT) for Bangkong LLM Training System
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from ..config.schemas import BangkongConfig


class LoRAConfig:
    """LoRA configuration class for PEFT manager."""

    def __init__(
        self,
        r: int = 8,
        alpha: int = 1,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
        task_type: str = "FEATURE_EXTRACTION"
    ):
        """
        Initialize LoRA configuration.

        Args:
            r: LoRA rank
            alpha: LoRA alpha scaling parameter
            target_modules: List of module names to apply LoRA to
            bias: Bias type ('none', 'all', 'lora_only')
            task_type: Type of task for LoRA
        """
        self.r = r
        self.alpha = alpha
        self.target_modules = target_modules
        self.bias = bias
        self.task_type = task_type


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
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Scaling factor
        self.scaling = alpha / rank

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
        # Keep track of modifications to apply later
        # Format: (parent_module, child_name, original_module)
        modifications = []

        # First pass: collect all linear layers that need LoRA applied
        for name, module in model.named_modules():
            # Prevent wrapping lm_head to avoid breaking HuggingFace weight tying
            if 'lm_head' in name:
                continue

            # Apply LoRA to linear layers (but not to LoRA layers themselves)
            if isinstance(module, nn.Linear) and not isinstance(module, LoRALayer):
                modifications.append((name, module))

        # Second pass: apply modifications
        for name, module in modifications:
            # Store original weight and bias
            weight = module.weight
            bias = module.bias

            # Get device
            device = weight.device if weight is not None else torch.device('cpu')

            # Create LoRA layer
            lora_layer = LoRALayer(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=self.lora_rank,
                alpha=self.lora_alpha
            ).to(device)

            # Replace the module with a LoRA-enhanced version
            # We need to be careful not to create nested modules that cause recursion
            class LoRALinear(nn.Module):
                def __init__(self, linear_layer, lora_adapter):
                    super().__init__()
                    self.linear = linear_layer
                    self.lora = lora_adapter

                def forward(self, x):
                    # Ensure both linear layer and LoRA adapter are on the same device
                    linear_device = next(self.linear.parameters()).device
                    lora_device = next(self.lora.parameters()).device
                    
                    # If devices don't match, move LoRA to linear device
                    if linear_device != lora_device:
                        self.lora.to(linear_device)
                    
                    # Ensure input is on the same device as the linear layer
                    x = x.to(linear_device)
                    return self.linear(x) + self.lora(x)

            # Replace the module
            lora_linear = LoRALinear(module, lora_layer)

            # Navigate to parent module and replace the child
            if '.' in name:
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent_module = model.get_submodule(parent_name)
            else:
                parent_name = ''
                child_name = name
                parent_module = model

            setattr(parent_module, child_name, lora_linear)

        print(f"Applied LoRA to {len(modifications)} linear layers")
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


class LoRAPEFTManager:
    """Manager for applying PEFT techniques to Bangkong models."""

    def __init__(self, config: LoRAConfig):
        """
        Initialize LoRA PEFT manager.

        Args:
            config: LoRA configuration
        """
        self.config = config
        self.lora_rank = config.r
        self.lora_alpha = config.alpha

    def apply_lora_to_linear_layers(self, model: nn.Module) -> nn.Module:
        """
        Apply LoRA to linear layers in the model based on target_modules.

        Args:
            model: Model to apply LoRA to

        Returns:
            Model with LoRA applied to specified linear layers
        """
        # Keep track of modifications to apply later
        # Format: (parent_module, child_name, original_module)
        modifications = []

        # First pass: collect all linear layers that need LoRA applied
        for name, module in model.named_modules():
            # Prevent wrapping lm_head to avoid breaking HuggingFace weight tying
            if 'lm_head' in name:
                continue

            # Apply LoRA to linear layers (but not to LoRA layers themselves)
            if isinstance(module, nn.Linear) and not isinstance(module, LoRALayer):
                # Check if this module should be targeted based on config
                should_apply = False
                
                if self.config.target_modules:
                    # Check if any target module pattern matches the module name
                    for target in self.config.target_modules:
                        if target in name:
                            should_apply = True
                            break
                else:
                    # If no target modules specified, apply to all linear layers
                    should_apply = True
                    
                if should_apply:
                    modifications.append((name, module))

        # Second pass: apply modifications
        for name, module in modifications:
            # Store original weight and bias
            weight = module.weight
            bias = module.bias
            
            # Get device
            device = weight.device if weight is not None else torch.device('cpu')
            
            # Create LoRA layer
            lora_layer = LoRALayer(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=self.lora_rank,
                alpha=self.lora_alpha
            ).to(device)
            
            # Replace the module with a LoRA-enhanced version
            # We need to be careful not to create nested modules that cause recursion
            class LoRALinear(nn.Module):
                def __init__(self, linear_layer, lora_adapter):
                    super().__init__()
                    self.linear = linear_layer
                    self.lora = lora_adapter

                def forward(self, x):
                    # Ensure both linear layer and LoRA adapter are on the same device
                    linear_device = next(self.linear.parameters()).device
                    lora_device = next(self.lora.parameters()).device
                    
                    # If devices don't match, move LoRA to linear device
                    if linear_device != lora_device:
                        self.lora.to(linear_device)
                    
                    # Ensure input is on the same device as the linear layer
                    x = x.to(linear_device)
                    return self.linear(x) + self.lora(x)
            
            # Replace the module
            lora_linear = LoRALinear(module, lora_layer)
            
            # Navigate to parent module and replace the child
            if '.' in name:
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent_module = model.get_submodule(parent_name)
            else:
                parent_name = ''
                child_name = name
                parent_module = model
            
            setattr(parent_module, child_name, lora_linear)

        print(f"Applied LoRA to {len(modifications)} linear layers")
        return model

    def apply_peft(self, model: nn.Module) -> nn.Module:
        """
        Apply parameter-efficient fine-tuning to the model.

        Args:
            model: Model to apply PEFT to

        Returns:
            Model with PEFT applied
        """
        return self.apply_lora_to_linear_layers(model)


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