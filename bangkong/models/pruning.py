"""
Pruning and sparsity for Bangkong LLM Training System
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from ..config.schemas import BangkongConfig


class MagnitudePruning:
    """Magnitude-based pruning for neural networks."""
    
    def __init__(self, sparsity_ratio: float = 0.5):
        """
        Initialize magnitude pruning.
        
        Args:
            sparsity_ratio: Ratio of weights to prune (0.0 to 1.0)
        """
        self.sparsity_ratio = sparsity_ratio
    
    def prune_model(self, model: nn.Module) -> nn.Module:
        """
        Apply magnitude pruning to a model.
        
        Args:
            model: Model to prune
            
        Returns:
            Pruned model
        """
        for module in model.modules():
            if isinstance(module, nn.Linear):
                self._prune_linear_layer(module)
        return model
    
    def _prune_linear_layer(self, layer: nn.Linear):
        """
        Prune a linear layer using magnitude-based pruning.
        
        Args:
            layer: Linear layer to prune
        """
        # Get the weight tensor
        weight = layer.weight.data
        
        # Calculate threshold for pruning
        num_weights = weight.numel()
        num_prune = int(num_weights * self.sparsity_ratio)
        
        # Find the threshold value
        threshold = torch.kthvalue(weight.abs().flatten(), num_prune).values
        
        # Create mask for weights to keep
        mask = weight.abs() > threshold
        
        # Apply pruning by setting pruned weights to zero
        weight *= mask.float()
        
        # Apply mask to the layer
        layer.weight.data = weight


class StructuredPruning:
    """Structured pruning for neural networks (pruning entire neurons/channels)."""
    
    def __init__(self, sparsity_ratio: float = 0.5):
        """
        Initialize structured pruning.
        
        Args:
            sparsity_ratio: Ratio of neurons to prune (0.0 to 1.0)
        """
        self.sparsity_ratio = sparsity_ratio
    
    def prune_model(self, model: nn.Module) -> nn.Module:
        """
        Apply structured pruning to a model.
        
        Args:
            model: Model to prune
            
        Returns:
            Pruned model
        """
        for module in model.modules():
            if isinstance(module, nn.Linear):
                self._prune_linear_layer(module)
        return model
    
    def _prune_linear_layer(self, layer: nn.Linear):
        """
        Prune a linear layer using structured pruning.
        
        Args:
            layer: Linear layer to prune
        """
        # Calculate L2 norm for each output neuron
        neuron_norms = torch.norm(layer.weight.data, p=2, dim=1)
        
        # Determine number of neurons to prune
        num_neurons = neuron_norms.size(0)
        num_prune = int(num_neurons * self.sparsity_ratio)
        
        # Find neurons with smallest norms
        _, indices = torch.sort(neuron_norms)
        prune_indices = indices[:num_prune]
        
        # Create mask for neurons to keep
        mask = torch.ones(num_neurons, dtype=torch.bool)
        mask[prune_indices] = False
        
        # Apply pruning
        layer.weight.data = layer.weight.data[mask]
        if layer.bias is not None:
            layer.bias.data = layer.bias.data[mask]
        
        # Update output features
        layer.out_features = mask.sum().item()


class PruningController:
    """Controller for applying pruning to models."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize pruning controller.
        
        Args:
            config: Bangkong configuration
        """
        self.config = config
        self.pruning_type = getattr(config.training, 'pruning_type', 'none')
        self.sparsity_ratio = getattr(config.training, 'sparsity_ratio', 0.0)
    
    def apply_pruning_to_model(self, model: nn.Module) -> nn.Module:
        """
        Apply pruning to a model.
        
        Args:
            model: Model to prune
            
        Returns:
            Pruned model
        """
        if self.pruning_type == 'none' or self.sparsity_ratio <= 0.0:
            return model
        
        if self.pruning_type == 'magnitude':
            pruner = MagnitudePruning(sparsity_ratio=self.sparsity_ratio)
        elif self.pruning_type == 'structured':
            pruner = StructuredPruning(sparsity_ratio=self.sparsity_ratio)
        else:
            raise ValueError(f"Unsupported pruning type: {self.pruning_type}")
        
        return pruner.prune_model(model)


def apply_pruning_to_model(model: nn.Module, config: BangkongConfig) -> nn.Module:
    """
    Apply pruning to a model.
    
    Args:
        model: Model to prune
        config: Bangkong configuration
        
    Returns:
        Pruned model
    """
    controller = PruningController(config)
    return controller.apply_pruning_to_model(model)