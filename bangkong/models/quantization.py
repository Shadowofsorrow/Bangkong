"""
Quantization for Bangkong LLM Training System
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from ..config.schemas import BangkongConfig


class QuantizedLinear(nn.Module):
    """Quantized linear layer with configurable bit-width."""
    
    def __init__(self, in_features: int, out_features: int, bits: int = 8):
        """
        Initialize quantized linear layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bits: Number of bits for quantization (default: 8)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        # Original full-precision weights and biases
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # Quantization parameters
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('weight_zero_point', torch.tensor(0.0))
        self.register_buffer('bias_scale', torch.tensor(1.0))
        self.register_buffer('bias_zero_point', torch.tensor(0.0))
        
        # Quantized weights and biases (as int8 tensors)
        self.register_buffer('weight_quant', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('bias_quant', torch.zeros(out_features, dtype=torch.int8))
    
    def _quantize_tensor(self, tensor: torch.Tensor, bits: int) -> tuple:
        """
        Quantize a tensor to specified bit-width.
        
        Args:
            tensor: Tensor to quantize
            bits: Number of bits for quantization
            
        Returns:
            Tuple of (quantized_tensor, scale, zero_point)
        """
        # Calculate quantization parameters
        min_val = tensor.min()
        max_val = tensor.max()
        
        # For signed integers with specified bits
        qmin = -2**(bits-1)
        qmax = 2**(bits-1) - 1
        
        # Calculate scale and zero point
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
        
        # Clamp zero point to valid range
        zero_point = torch.clamp(zero_point, qmin, qmax).round()
        
        # Quantize tensor
        quantized = torch.round(tensor / scale + zero_point).clamp(qmin, qmax).to(torch.int8)
        
        return quantized, scale, zero_point
    
    def _dequantize_tensor(self, quantized: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        """
        Dequantize a quantized tensor.
        
        Args:
            quantized: Quantized tensor
            scale: Scale parameter
            zero_point: Zero point parameter
            
        Returns:
            Dequantized tensor
        """
        return scale * (quantized.to(torch.float32) - zero_point)
    
    def quantize_weights(self):
        """Quantize the weights and biases."""
        self.weight_quant, self.weight_scale, self.weight_zero_point = self._quantize_tensor(self.weight.data, self.bits)
        if self.bias is not None:
            self.bias_quant, self.bias_scale, self.bias_zero_point = self._quantize_tensor(self.bias.data, self.bits)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantized linear layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Dequantize weights and biases for computation
        weight_dequant = self._dequantize_tensor(self.weight_quant, self.weight_scale, self.weight_zero_point)
        if self.bias is not None:
            bias_dequant = self._dequantize_tensor(self.bias_quant, self.bias_scale, self.bias_zero_point)
        else:
            bias_dequant = None
        
        # Perform linear operation with dequantized parameters
        return nn.functional.linear(x, weight_dequant, bias_dequant)


class QuantizationController:
    """Controller for applying quantization to models."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize quantization controller.
        
        Args:
            config: Bangkong configuration
        """
        self.config = config
        self.quantization_precision = getattr(config.packaging.quantization, 'default_precision', 'none')
        self.bits = self._get_bits_from_precision(self.quantization_precision)
    
    def _get_bits_from_precision(self, precision: str) -> int:
        """
        Get number of bits from precision string.
        
        Args:
            precision: Precision string (e.g., 'int8', 'int4')
            
        Returns:
            Number of bits
        """
        if precision == 'int8':
            return 8
        elif precision == 'int4':
            return 4
        elif precision == 'int16':
            return 16
        else:
            return 32  # Full precision
    
    def apply_quantization_to_model(self, model: nn.Module) -> nn.Module:
        """
        Apply quantization to a model.
        
        Args:
            model: Model to quantize
            
        Returns:
            Quantized model
        """
        if self.quantization_precision == 'none':
            return model
        
        # Iterate through model modules and replace linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Replace with quantized linear layer
                quantized_layer = QuantizedLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bits=self.bits
                )
                
                # Copy weights and biases
                with torch.no_grad():
                    quantized_layer.weight.data.copy_(module.weight.data)
                    if module.bias is not None:
                        quantized_layer.bias.data.copy_(module.bias.data)
                
                # Quantize the weights
                quantized_layer.quantize_weights()
                
                # Replace the module
                parent_name = '.'.join(name.split('.')[:-1])
                module_name = name.split('.')[-1]
                
                if parent_name:
                    parent_module = dict(model.named_modules())[parent_name]
                else:
                    parent_module = model
                    
                setattr(parent_module, module_name, quantized_layer)
        
        return model
    
    def quantize_model_weights(self, model: nn.Module) -> nn.Module:
        """
        Quantize model weights.
        
        Args:
            model: Model to quantize weights for
            
        Returns:
            Model with quantized weights
        """
        if self.quantization_precision == 'none':
            return model
        
        # Iterate through model modules and quantize weights
        for module in model.modules():
            if isinstance(module, QuantizedLinear):
                module.quantize_weights()
        
        return model


def apply_quantization_to_model(model: nn.Module, config: BangkongConfig) -> nn.Module:
    """
    Apply quantization to a model.
    
    Args:
        model: Model to quantize
        config: Bangkong configuration
        
    Returns:
        Quantized model
    """
    controller = QuantizationController(config)
    return controller.apply_quantization_to_model(model)


def quantize_model_weights(model: nn.Module, config: BangkongConfig) -> nn.Module:
    """
    Quantize model weights.
    
    Args:
        model: Model to quantize weights for
        config: Bangkong configuration
        
    Returns:
        Model with quantized weights
    """
    controller = QuantizationController(config)
    return controller.quantize_model_weights(model)