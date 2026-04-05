#!/usr/bin/env python3
"""
Attention head specialization for pre-intelligent initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
from ..config.schemas import BangkongConfig
from .config_loader import get_models_config

class AttentionHeadSpecializer:
    """Specialize attention heads for reasoning subspaces."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize attention head specializer.
        
        Args:
            config: Bangkong configuration
        """
        self.config = config
        self.models_config = get_models_config()
        self.hidden_size = config.model.hidden_size
        self.num_heads = config.model.num_heads
        self.prior_knowledge = getattr(config.model, 'prior_knowledge', 'general')
        
    def _get_reasoning_patterns(self) -> List[Dict[str, float]]:
        """
        Get reasoning patterns for attention head specialization.
        
        Returns:
            List of pattern dictionaries
        """
        # Get pattern weights from config
        attention_config = self.models_config.get("models.attention_specialization", {})
        
        if self.prior_knowledge == 'reasoning':
            patterns = attention_config.get("reasoning_patterns", {
                "causal": 0.3,
                "logical": 0.25,
                "temporal": 0.2,
                "spatial": 0.15,
                "general": 0.1
            })
            return [{"pattern": name, "weight": weight} for name, weight in patterns.items()]
        elif self.prior_knowledge == 'math':
            patterns = attention_config.get("math_patterns", {
                "sequential": 0.35,
                "structural": 0.3,
                "numerical": 0.2,
                "general": 0.15
            })
            return [{"pattern": name, "weight": weight} for name, weight in patterns.items()]
        elif self.prior_knowledge == 'code':
            patterns = attention_config.get("code_patterns", {
                "syntactic": 0.3,
                "semantic": 0.25,
                "control": 0.2,
                "data": 0.15,
                "general": 0.1
            })
            return [{"pattern": name, "weight": weight} for name, weight in patterns.items()]
        else:
            # Configurable patterns
            patterns_config = getattr(self.config.model, 'preint_attention_patterns', None)
            if patterns_config:
                total_weight = sum(patterns_config.values())
                if abs(total_weight - 1.0) > 1e-6:
                    raise ValueError(f"Attention pattern weights must sum to 1.0, got {total_weight}")
                return [{"pattern": name, "weight": weight} 
                        for name, weight in patterns_config.items()]
            else:
                # Default uniform distribution
                return [
                    {"pattern": "general", "weight": 1.0}
                ]
            
    def _create_causal_pattern(self, head_idx: int) -> torch.Tensor:
        """
        Create causal attention pattern.
        
        Args:
            head_idx: Head index
            
        Returns:
            Causal attention pattern tensor
        """
        # Create causal mask pattern
        pattern = torch.tril(torch.ones(self.hidden_size, self.hidden_size))
        # Add some randomness while maintaining causality
        noise = torch.randn_like(pattern) * 0.1
        pattern = pattern + noise
        return torch.softmax(pattern, dim=-1)
        
    def _create_logical_pattern(self, head_idx: int) -> torch.Tensor:
        """
        Create logical attention pattern.
        
        Args:
            head_idx: Head index
            
        Returns:
            Logical attention pattern tensor
        """
        # Get local focus parameters from config
        local_focus_config = self.models_config.get("models.attention_specialization.local_focus", {})
        step_size = local_focus_config.get("step_size", 8)
        window_size = local_focus_config.get("window_size", 4)
        long_range_distance = local_focus_config.get("long_range_distance", 8)
        long_range_weight = local_focus_config.get("long_range_weight", 0.5)
        
        # Create pattern that focuses on operators and connectors
        pattern = torch.zeros(self.hidden_size, self.hidden_size)
        # Emphasize connections between logical tokens
        for i in range(0, self.hidden_size, step_size):
            end_idx = min(i + window_size, self.hidden_size)
            pattern[i:end_idx, i:end_idx] = 1.0  # Local focus
            if i + long_range_distance < self.hidden_size:
                pattern[i, i + long_range_distance] = long_range_weight  # Long-range logical connections
        return torch.softmax(pattern + torch.randn_like(pattern) * 0.1, dim=-1)
        
    def _create_temporal_pattern(self, head_idx: int) -> torch.Tensor:
        """
        Create temporal attention pattern.
        
        Args:
            head_idx: Head index
            
        Returns:
            Temporal attention pattern tensor
        """
        # Get temporal reasoning parameters from config
        temporal_config = self.models_config.get("models.attention_specialization.temporal_reasoning", {})
        window_range = temporal_config.get("window_range", 10)
        
        # Create pattern that captures temporal dependencies
        pattern = torch.zeros(self.hidden_size, self.hidden_size)
        # Diagonal emphasis with temporal spread
        for i in range(self.hidden_size):
            for j in range(max(0, i - window_range), min(self.hidden_size, i + window_range + 1)):
                pattern[i, j] = 1.0 / (1.0 + abs(i - j))
        return torch.softmax(pattern + torch.randn_like(pattern) * 0.1, dim=-1)
        
    def _create_spatial_pattern(self, head_idx: int) -> torch.Tensor:
        """
        Create spatial attention pattern.
        
        Args:
            head_idx: Head index
            
        Returns:
            Spatial attention pattern tensor
        """
        # Create 2D spatial pattern (assuming hidden_size is perfect square or rectangular)
        h, w = int(np.sqrt(self.hidden_size)), int(np.sqrt(self.hidden_size))
        if h * w != self.hidden_size:
            # Find closest rectangle
            for i in range(int(np.sqrt(self.hidden_size)), 0, -1):
                if self.hidden_size % i == 0:
                    h, w = i, self.hidden_size // i
                    break
                    
        pattern = torch.zeros(self.hidden_size, self.hidden_size)
        # Create spatial neighborhood connections
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                # Connect to immediate neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            nidx = ni * w + nj
                            pattern[idx, nidx] = 1.0
        return torch.softmax(pattern + torch.randn_like(pattern) * 0.1, dim=-1)
        
    def _create_sequential_pattern(self, head_idx: int) -> torch.Tensor:
        """
        Create sequential attention pattern.
        
        Args:
            head_idx: Head index
            
        Returns:
            Sequential attention pattern tensor
        """
        # Create pattern that emphasizes sequential processing
        pattern = torch.zeros(self.hidden_size, self.hidden_size)
        # Strong diagonal with immediate neighbors
        for i in range(self.hidden_size):
            pattern[i, i] = 1.0
            if i > 0:
                pattern[i, i-1] = 0.8
            if i < self.hidden_size - 1:
                pattern[i, i+1] = 0.8
        return torch.softmax(pattern + torch.randn_like(pattern) * 0.1, dim=-1)
        
    def _create_structural_pattern(self, head_idx: int) -> torch.Tensor:
        """
        Create structural attention pattern.
        
        Args:
            head_idx: Head index
            
        Returns:
            Structural attention pattern tensor
        """
        # Get structural pattern parameters from config
        structural_config = self.models_config.get("models.attention_specialization.structural_patterns", {})
        block_size_base = structural_config.get("block_size_base", 8)
        block_size_divisor = structural_config.get("block_size_divisor", 16)
        
        # Create pattern that focuses on structural elements
        pattern = torch.ones(self.hidden_size, self.hidden_size) / self.hidden_size
        # Add some structure
        block_size = max(block_size_base, self.hidden_size // block_size_divisor)
        for i in range(0, self.hidden_size, block_size):
            end_i = min(i + block_size, self.hidden_size)
            for j in range(0, self.hidden_size, block_size):
                end_j = min(j + block_size, self.hidden_size)
                pattern[i:end_i, j:end_j] += 0.5  # Block structure
        return torch.softmax(pattern + torch.randn_like(pattern) * 0.1, dim=-1)
        
    def _create_numerical_pattern(self, head_idx: int) -> torch.Tensor:
        """
        Create numerical attention pattern.
        
        Args:
            head_idx: Head index
            
        Returns:
            Numerical attention pattern tensor
        """
        # Create pattern that focuses on numerical tokens
        pattern = torch.zeros(self.hidden_size, self.hidden_size)
        # Emphasize connections between numerical positions
        for i in range(0, self.hidden_size, 4):
            for j in range(0, self.hidden_size, 4):
                if abs(i - j) <= 16:  # Focus on nearby numerical tokens
                    pattern[i:min(i+4, self.hidden_size), j:min(j+4, self.hidden_size)] = 1.0
        return torch.softmax(pattern + torch.randn_like(pattern) * 0.1, dim=-1)
        
    def _create_syntactic_pattern(self, head_idx: int) -> torch.Tensor:
        """
        Create syntactic attention pattern.
        
        Args:
            head_idx: Head index
            
        Returns:
            Syntactic attention pattern tensor
        """
        # Create pattern that captures syntactic relationships
        pattern = torch.zeros(self.hidden_size, self.hidden_size)
        # Focus on hierarchical structure
        for i in range(self.hidden_size):
            # Connect to parent and children in a tree-like structure
            parent_idx = max(0, (i - 1) // 2)
            pattern[i, parent_idx] = 1.0
            if 2*i + 1 < self.hidden_size:
                pattern[i, 2*i + 1] = 1.0
            if 2*i + 2 < self.hidden_size:
                pattern[i, 2*i + 2] = 1.0
        return torch.softmax(pattern + torch.randn_like(pattern) * 0.1, dim=-1)
        
    def _create_semantic_pattern(self, head_idx: int) -> torch.Tensor:
        """
        Create semantic attention pattern.
        
        Args:
            head_idx: Head index
            
        Returns:
            Semantic attention pattern tensor
        """
        # Create pattern that captures semantic relationships
        pattern = torch.ones(self.hidden_size, self.hidden_size) / self.hidden_size
        # Add semantic clustering
        cluster_size = max(8, self.hidden_size // 32)
        for i in range(0, self.hidden_size, cluster_size):
            end_i = min(i + cluster_size, self.hidden_size)
            for j in range(0, self.hidden_size, cluster_size):
                end_j = min(j + cluster_size, self.hidden_size)
                pattern[i:end_i, j:end_j] += 0.5  # Semantic clusters
        return torch.softmax(pattern + torch.randn_like(pattern) * 0.1, dim=-1)
        
    def _create_control_pattern(self, head_idx: int) -> torch.Tensor:
        """
        Create control flow attention pattern.
        
        Args:
            head_idx: Head index
            
        Returns:
            Control flow attention pattern tensor
        """
        # Create pattern that captures control flow
        pattern = torch.zeros(self.hidden_size, self.hidden_size)
        # Emphasize jump and branch patterns
        for i in range(self.hidden_size):
            # Forward connections
            for j in range(i, min(i + 32, self.hidden_size)):
                pattern[i, j] = 1.0 / (1.0 + abs(i - j) * 0.1)
            # Backward connections (loops, conditionals)
            for j in range(max(0, i - 16), i):
                pattern[i, j] = 0.5 / (1.0 + abs(i - j) * 0.2)
        return torch.softmax(pattern + torch.randn_like(pattern) * 0.1, dim=-1)
        
    def _create_data_pattern(self, head_idx: int) -> torch.Tensor:
        """
        Create data structure attention pattern.
        
        Args:
            head_idx: Head index
            
        Returns:
            Data structure attention pattern tensor
        """
        # Get data structure parameters from config
        data_config = self.models_config.get("models.attention_specialization.data_structures", {})
        stride_min = data_config.get("stride_min", 4)
        stride_divisor = data_config.get("stride_divisor", 64)
        
        # Create pattern that focuses on data structures
        pattern = torch.zeros(self.hidden_size, self.hidden_size)
        # Emphasize array/list-like access patterns
        stride = max(stride_min, self.hidden_size // stride_divisor)
        for i in range(self.hidden_size):
            # Sequential access
            for j in range(max(0, i - 2), min(self.hidden_size, i + 3)):
                pattern[i, j] = 1.0
            # Strided access (array indexing)
            for k in range(1, 5):
                if i + k * stride < self.hidden_size:
                    pattern[i, i + k * stride] = 0.7
                if i - k * stride >= 0:
                    pattern[i, i - k * stride] = 0.7
        return torch.softmax(pattern + torch.randn_like(pattern) * 0.1, dim=-1)
        
    def _create_general_pattern(self, head_idx: int) -> torch.Tensor:
        """
        Create general attention pattern.
        
        Args:
            head_idx: Head index
            
        Returns:
            General attention pattern tensor
        """
        # Uniform attention with some structure
        pattern = torch.ones(self.hidden_size, self.hidden_size) / self.hidden_size
        return pattern + torch.randn_like(pattern) * 0.1
        
    def _get_pattern_function(self, pattern_name: str):
        """
        Get pattern creation function by name.
        
        Args:
            pattern_name: Name of pattern
            
        Returns:
            Pattern creation function
        """
        pattern_functions = {
            "causal": self._create_causal_pattern,
            "logical": self._create_logical_pattern,
            "temporal": self._create_temporal_pattern,
            "spatial": self._create_spatial_pattern,
            "sequential": self._create_sequential_pattern,
            "structural": self._create_structural_pattern,
            "numerical": self._create_numerical_pattern,
            "syntactic": self._create_syntactic_pattern,
            "semantic": self._create_semantic_pattern,
            "control": self._create_control_pattern,
            "data": self._create_data_pattern,
            "general": self._create_general_pattern
        }
        return pattern_functions.get(pattern_name, self._create_general_pattern)
        
    def specialize_attention_heads(self, model: nn.Module) -> nn.Module:
        """
        Specialize attention heads in the model by registering forward hooks
        that apply pattern-based bias tensors to attention outputs.

        Args:
            model: Model to specialize

        Returns:
            Specialized model
        """
        if not getattr(self.config.model, 'preint_attention_specialization', True):
            print("Attention head specialization disabled in config")
            return model

        print(f"Specializing attention heads for {self.prior_knowledge} domain...")

        # Get reasoning patterns
        patterns = self._get_reasoning_patterns()

        # Assign heads to patterns based on weights
        head_assignments = []
        for pattern_info in patterns:
            pattern_name = pattern_info["pattern"]
            weight = pattern_info["weight"]
            num_heads_for_pattern = int(self.num_heads * weight)
            head_assignments.extend([pattern_name] * num_heads_for_pattern)

        # Fill remaining heads with general pattern if needed
        while len(head_assignments) < self.num_heads:
            head_assignments.append("general")

        # Trim if we have too many (due to rounding)
        head_assignments = head_assignments[:self.num_heads]

        # Pre-compute pattern bias tensors for each pattern name
        pattern_biases = {}
        for pattern_name in set(head_assignments):
            pattern_func = self._get_pattern_function(pattern_name)
            bias = pattern_func(0)
            pattern_biases[pattern_name] = bias

        # Apply specialization by registering forward hooks on attention layers
        attention_layers_found = 0
        for name, module in model.named_modules():
            type_name = type(module).__name__
            if isinstance(module, nn.MultiheadAttention):
                attention_layers_found += 1
                self._register_hook(module, head_assignments, pattern_biases, name)
            elif type_name.endswith('Attention'):
                # HuggingFace-style: GPT2Attention, LlamaAttention, MistralAttention
                attention_layers_found += 1
                self._register_hook(module, head_assignments, pattern_biases, name)
            elif hasattr(module, 'self_attn') and module.self_attn is not None:
                attention_layers_found += 1
                self._register_hook(module.self_attn, head_assignments, pattern_biases, name)
            elif name.endswith('attn') and hasattr(module, 'c_attn'):
                # GPT2 Conv1D-based attention (h.0.attn)
                attention_layers_found += 1
                self._register_hook(module, head_assignments, pattern_biases, name)
            elif name.endswith('attention') and hasattr(module, 'query'):
                attention_layers_found += 1
                self._register_hook(module, head_assignments, pattern_biases, name)

        if attention_layers_found == 0:
            print("Warning: No attention layers found in model for specialization")
        else:
            print(f"Registered attention specialization hooks on {attention_layers_found} layers")

        print(f"Specialized {len(head_assignments)} attention heads:")
        for pattern_name in set(head_assignments):
            count = head_assignments.count(pattern_name)
            print(f"  - {pattern_name}: {count} heads")

        # Store on model for reference
        model._attention_head_assignments = head_assignments
        model._attention_pattern_biases = pattern_biases

        return model

    def _register_hook(self, module, head_assignments, pattern_biases, layer_name):
        """Register a forward hook that biases attention output with pattern tensors."""
        # Blend all pattern biases according to head assignments
        blended_bias = self._blend_pattern_bias(head_assignments, pattern_biases)

        def hook_fn(m, args, output):
            # Handle tuple outputs (common: (hidden_states, ...) or (attn_output, attn_weights))
            if isinstance(output, tuple) and len(output) > 0:
                primary = output[0]
            elif isinstance(output, torch.Tensor):
                primary = output
            else:
                return output

            # Resize bias to match primary output shape
            bias = self._resize_bias(blended_bias, primary.shape)

            # Apply small additive bias (0.01 scale to avoid destabilizing training)
            biased = primary + 0.01 * bias

            if isinstance(output, tuple):
                return (biased,) + output[1:]
            return biased

        module.register_forward_hook(hook_fn)

    def _blend_pattern_bias(self, head_assignments, pattern_biases):
        """Blend pattern biases according to head assignment proportions."""
        if not head_assignments or not pattern_biases:
            return torch.zeros(self.hidden_size, self.hidden_size)

        device = next(iter(pattern_biases.values())).device
        blended = torch.zeros(self.hidden_size, self.hidden_size, device=device)

        for pattern_name in set(head_assignments):
            count = head_assignments.count(pattern_name)
            weight = count / len(head_assignments)
            bias = pattern_biases[pattern_name]
            if bias.device != device:
                bias = bias.to(device)
            blended += weight * bias

        total = sum(head_assignments.count(p) for p in set(head_assignments))
        if total > 0:
            blended /= total

        return blended

    def _resize_bias(self, bias, target_shape):
        """Resize bias tensor to match target output shape."""
        if bias.shape == target_shape:
            return bias

        if len(target_shape) == 2:
            return F.interpolate(
                bias.unsqueeze(0).unsqueeze(0),
                size=target_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        elif len(target_shape) == 3:
            bias_2d = F.interpolate(
                bias.unsqueeze(0).unsqueeze(0),
                size=(target_shape[1], target_shape[2]),
                mode='bilinear',
                align_corners=False
            )
            return bias_2d.squeeze(0)
        elif len(target_shape) == 4:
            bias_2d = F.interpolate(
                bias.unsqueeze(0).unsqueeze(0),
                size=(target_shape[2], target_shape[3]),
                mode='bilinear',
                align_corners=False
            )
            return bias_2d.expand(target_shape[0], target_shape[1], -1, -1)
        else:
            result = torch.zeros(target_shape, device=bias.device)
            slices = tuple(slice(0, min(ts, bs)) for ts, bs in zip(target_shape, bias.shape))
            result[slices] = bias[slices]
            return result