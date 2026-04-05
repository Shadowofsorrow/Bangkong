#!/usr/bin/env python3
"""
Cosine-clustered embedding initialization for Bangkong LLM Training System
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from ..config.schemas import BangkongConfig
from .config_loader import get_models_config

class CosineClusteredEmbeddings:
    """Generate embeddings with group-structured prototypes using cosine clustering."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize cosine-clustered embeddings generator.
        
        Args:
            config: Bangkong configuration
        """
        self.config = config
        self.models_config = get_models_config()
        self.vocab_size = config.model.vocab_size
        self.hidden_size = config.model.hidden_size
        self.domain_groups = self._get_domain_groups()
        
    def _get_domain_groups(self) -> List[Tuple[str, int]]:
        """
        Define domain groups for cosine clustering.
        
        Returns:
            List of (group_name, group_size) tuples
        """
        # Based on prior knowledge, define groups
        prior_knowledge = getattr(self.config.model, 'prior_knowledge', 'general')
        
        if prior_knowledge == 'reasoning':
            return [
                ('reasoning', int(self.vocab_size * 0.15)),
                ('logic', int(self.vocab_size * 0.15)),
                ('planning', int(self.vocab_size * 0.20)),
                ('general', int(self.vocab_size * 0.50))
            ]
        elif prior_knowledge == 'math':
            return [
                ('numbers', int(self.vocab_size * 0.20)),
                ('operators', int(self.vocab_size * 0.15)),
                ('variables', int(self.vocab_size * 0.10)),
                ('math_general', int(self.vocab_size * 0.55))
            ]
        elif prior_knowledge == 'code':
            return [
                ('keywords', int(self.vocab_size * 0.15)),
                ('operators', int(self.vocab_size * 0.15)),
                ('identifiers', int(self.vocab_size * 0.20)),
                ('literals', int(self.vocab_size * 0.10)),
                ('code_general', int(self.vocab_size * 0.40))
            ]
        else:
            # General case
            groups_config = getattr(self.config.model, 'preint_cluster_groups', None)
            if groups_config:
                # Use configured groups
                total_allocated = sum(groups_config.values())
                if abs(total_allocated - 1.0) > 1e-6:
                    raise ValueError(f"Cluster group proportions must sum to 1.0, got {total_allocated}")
                
                return [(name, int(self.vocab_size * proportion)) 
                        for name, proportion in groups_config.items()]
            else:
                # Default single group
                return [('general', self.vocab_size)]
            
    def _small_factorial(self, n: int) -> int:
        """
        Compute small factorial avoiding overflow.
        
        Args:
            n: Input number
            
        Returns:
            Factorial of n
        """
        factorial_cap = self.models_config.get("models.cosine_clustered_embeddings.factorial_cap", 11)
        if n <= 1:
            return 1
        result = 1
        for i in range(2, min(n + 1, factorial_cap)):  # Cap to avoid overflow
            result *= i
        return result
        
    def _generate_prototypes(self, dim: int, k: int, seed: int = 0, 
                           repel_iters: int = None, repel_scale: float = None) -> torch.Tensor:
        """
        Generate k prototypes on unit sphere with repulsion.
        
        Args:
            dim: Dimension of prototypes
            k: Number of prototypes
            seed: Random seed
            repel_iters: Number of repulsion iterations
            repel_scale: Repulsion scale factor
            
        Returns:
            Tensor of prototypes (k, dim)
        """
        # Get default values from config if not provided
        if repel_iters is None:
            repel_iters = self.models_config.get("models.cosine_clustered_embeddings.repulsion.default_iters", 50)
        if repel_scale is None:
            repel_scale = self.models_config.get("models.cosine_clustered_embeddings.repulsion.default_scale", 0.02)
            
        torch.manual_seed(seed)
        P = torch.randn(k, dim)
        P = F.normalize(P, dim=-1)
        
        # Simple repulsion to spread prototypes
        for _ in range(repel_iters):
            grads = torch.zeros_like(P)
            for i in range(k):
                # Compute attraction/repulsion based on dot product
                dots = (P @ P[i])  # similarity of all to P[i]
                # Push away proportional to similarity
                push = (dots.unsqueeze(-1) * P)  # (k, dim)
                grads[i] += push.sum(dim=0)
            # Normalize grads and update
            grads = grads / (grads.norm(dim=-1, keepdim=True) + 1e-6)
            P = P + repel_scale * grads
            P = F.normalize(P, dim=-1)
            
        return P
        
    def build_cosine_clustered_embeddings(self, groups: List[Tuple[str, int]], 
                                        prototypes_per_group: int = None) -> Tuple[torch.Tensor, List[dict]]:
        """
        Build cosine-clustered embeddings.
        
        Args:
            groups: List of (group_name, group_size) tuples
            prototypes_per_group: Base number of prototypes per group
            
        Returns:
            Tuple of (embeddings tensor, group metadata)
        """
        # Get default value from config if not provided
        if prototypes_per_group is None:
            prototypes_per_group = self.models_config.get("models.cosine_clustered_embeddings.default_prototypes_per_group", 8)
            
        total_size = sum(size for _, size in groups)
        # Adjust last group to account for rounding
        if total_size != self.vocab_size:
            diff = self.vocab_size - total_size
            groups = groups[:-1] + [(groups[-1][0], groups[-1][1] + diff)]
        
        E = torch.zeros(self.vocab_size, self.hidden_size)
        group_meta = []
        idx = 0
        
        # Get complexity scaling parameters
        complexity_config = self.models_config.get("models.cosine_clustered_embeddings.complexity_scaling", {})
        repel_iters_base = complexity_config.get("repel_iters_base", 8)
        repel_iters_increment_factor = complexity_config.get("repel_iters_increment_factor", 2)
        repel_scale_base = complexity_config.get("repel_scale_base", 0.03)
        repel_scale_increment_factor = complexity_config.get("repel_scale_increment_factor", 0.005)
        noise_scale_base = complexity_config.get("noise_scale_base", 0.01)
        noise_scale_increment_factor = complexity_config.get("noise_scale_increment_factor", 0.3)
        noise_scale_complexity_factor = complexity_config.get("noise_scale_complexity_factor", 10.0)
        
        # Get prototype count parameters
        prototype_config = self.models_config.get("models.cosine_clustered_embeddings.prototype_count", {})
        base_increment_factor = prototype_config.get("base_increment_factor", 0.2)
        min_prototypes = prototype_config.get("min_prototypes", 1)
        
        for g_i, (group_name, group_size) in enumerate(groups):
            # Factorial-like complexity weight
            fac = self._small_factorial(max(1, min(10, g_i + 1)))  # Cap to avoid overflow
            complexity = math.pow(fac, 1/3.0)  # Cube-root to reduce range
            
            # Prototypes count increases with complexity slightly
            k = max(min_prototypes, int(prototypes_per_group * (1 + base_increment_factor * g_i)))
            
            # Repel iterations and noise scale scale with complexity
            repel_iters = int(repel_iters_base + repel_iters_increment_factor * g_i + complexity)
            repel_scale = repel_scale_base + repel_scale_increment_factor * g_i
            P = self._generate_prototypes(self.hidden_size, k, seed=100+g_i, 
                                        repel_iters=repel_iters, repel_scale=repel_scale)
            
            # Noise scale: allow higher complexity groups more intra-cluster variance
            noise_scale = noise_scale_base * (1 + noise_scale_increment_factor * g_i + complexity/noise_scale_complexity_factor)
            
            # Assign tokens: map tokens to prototypes cyclically with noise
            for j in range(group_size):
                proto = P[j % k]
                noise = torch.randn(self.hidden_size) * noise_scale
                v = proto + noise
                v = F.normalize(v, dim=-1)
                E[idx] = v
                idx += 1
                
            group_meta.append({
                "group_index": g_i,
                "group_name": group_name,
                "group_size": group_size,
                "prototypes": k,
                "complexity": complexity,
                "repel_iters": repel_iters,
                "repel_scale": repel_scale,
                "noise_scale": noise_scale
            })
            
        return E, group_meta
        
    def initialize_embeddings(self, embedding_layer: torch.nn.Embedding) -> torch.nn.Embedding:
        """
        Initialize embedding layer with cosine-clustered embeddings.
        
        Args:
            embedding_layer: Embedding layer to initialize
            
        Returns:
            Initialized embedding layer
        """
        print("Initializing embeddings with cosine-clustered approach...")
        
        # Build cosine-clustered embeddings
        embeddings, meta = self.build_cosine_clustered_embeddings(self.domain_groups)
        
        # Apply to embedding layer
        with torch.no_grad():
            embedding_layer.weight.copy_(embeddings)
            
        print(f"Initialized embeddings with {len(self.domain_groups)} domain groups:")
        for group_info in meta:
            print(f"  - {group_info['group_name']}: {group_info['group_size']} tokens, "
                  f"{group_info['prototypes']} prototypes, complexity {group_info['complexity']:.2f}")
                  
        return embedding_layer