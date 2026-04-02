#!/usr/bin/env python3
"""
Benchmark script to compare initialization strategies
"""

import sys
import os
import torch
import torch.nn as nn
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bangkong.config.schemas import BangkongConfig, ModelConfig
from bangkong.models.intelligent_init import IntelligentInitializer

def create_benchmark_model():
    """Create a benchmark model."""
    class BenchmarkModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(5000, 256)
            # Multiple transformer layers for comprehensive testing
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024)
                for _ in range(6)
            ])
            self.output = nn.Linear(256, 5000)
        
        def forward(self, x):
            x = self.embedding(x)
            x = x.transpose(0, 1)  # For transformer
            for layer in self.layers:
                x = layer(x)
            x = x.transpose(0, 1)  # Back to batch first
            x = self.output(x)
            return x
    
    return BenchmarkModel()

def benchmark_initialization_strategy(strategy, prior_knowledge=None):
    """Benchmark a specific initialization strategy."""
    print(f"Benchmarking {strategy} initialization{' with ' + prior_knowledge if prior_knowledge else ''}...")
    
    # Create model config
    model_config = ModelConfig(
        name=f"benchmark-{strategy}",
        initialization_strategy=strategy,
        prior_knowledge=prior_knowledge
    )
    
    config = BangkongConfig(model=model_config)
    
    # Create model
    model = create_benchmark_model()
    
    # Time the initialization
    start_time = time.time()
    initializer = IntelligentInitializer(config)
    initialized_model = initializer.initialize_model(model)
    end_time = time.time()
    
    initialization_time = end_time - start_time
    
    # Count parameters
    total_params = sum(p.numel() for p in initialized_model.parameters())
    
    # Check for NaN values (indicates initialization issues)
    has_nan = False
    for param in initialized_model.parameters():
        if torch.isnan(param).any():
            has_nan = True
            break
    
    print(f"  Parameters: {total_params:,}")
    print(f"  Time: {initialization_time:.4f}s")
    print(f"  NaN values: {'Yes' if has_nan else 'No'}")
    
    return {
        'strategy': strategy,
        'prior_knowledge': prior_knowledge,
        'time': initialization_time,
        'parameters': total_params,
        'has_nan': has_nan
    }

def run_comprehensive_benchmark():
    """Run comprehensive benchmark of all initialization strategies."""
    print("Running comprehensive initialization benchmark...")
    print("=" * 60)
    
    strategies = [
        ('random', None),
        ('xavier', None),
        ('kaiming', None),
        ('structured', 'reasoning'),
        ('structured', 'math'),
        ('structured', 'code')
    ]
    
    results = []
    
    for strategy, prior_knowledge in strategies:
        result = benchmark_initialization_strategy(strategy, prior_knowledge)
        results.append(result)
        print()
    
    # Print summary
    print("=" * 60)
    print("Benchmark Summary:")
    print("-" * 60)
    print(f"{'Strategy':<15} {'Knowledge':<12} {'Time (s)':<10} {'Parameters':<12} {'NaN'}")
    print("-" * 60)
    
    for result in results:
        strategy = result['strategy']
        knowledge = result['prior_knowledge'] or 'N/A'
        time_taken = result['time']
        params = result['parameters']
        has_nan = 'Yes' if result['has_nan'] else 'No'
        
        print(f"{strategy:<15} {knowledge:<12} {time_taken:<10.4f} {params:<12,} {has_nan}")
    
    print("-" * 60)
    print("Benchmark completed successfully!")
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_benchmark()