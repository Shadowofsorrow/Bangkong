#!/usr/bin/env python3
"""
Example script demonstrating the use of intelligent initialization in Bangkong
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bangkong.config.schemas import BangkongConfig, ModelConfig
from bangkong.models.intelligent_init import apply_intelligent_initialization
import torch
import torch.nn as nn

def create_sample_model():
    """Create a sample model for demonstration."""
    class SampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 256)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=256, nhead=8), 
                num_layers=6
            )
            self.output = nn.Linear(256, 1000)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            x = self.output(x)
            return x
    
    return SampleModel()

def demonstrate_reasoning_initialization():
    """Demonstrate reasoning-focused initialization."""
    print("=== Reasoning-Focused Initialization ===")
    
    # Create model config with reasoning initialization
    model_config = ModelConfig(
        name="reasoning-demo",
        initialization_strategy="structured",
        prior_knowledge="reasoning"
    )
    
    config = BangkongConfig(model=model_config)
    
    # Create and initialize model
    model = create_sample_model()
    initialized_model = apply_intelligent_initialization(model, config)
    
    print("Model initialized with reasoning-focused patterns")
    print(f"Model name: {config.model.name}")
    print(f"Initialization strategy: {config.model.initialization_strategy}")
    print(f"Prior knowledge: {config.model.prior_knowledge}")
    print()

def demonstrate_math_initialization():
    """Demonstrate math-focused initialization."""
    print("=== Math-Focused Initialization ===")
    
    # Create model config with math initialization
    model_config = ModelConfig(
        name="math-demo",
        initialization_strategy="structured",
        prior_knowledge="math"
    )
    
    config = BangkongConfig(model=model_config)
    
    # Create and initialize model
    model = create_sample_model()
    initialized_model = apply_intelligent_initialization(model, config)
    
    print("Model initialized with math-focused patterns")
    print(f"Model name: {config.model.name}")
    print(f"Initialization strategy: {config.model.initialization_strategy}")
    print(f"Prior knowledge: {config.model.prior_knowledge}")
    print()

def demonstrate_code_initialization():
    """Demonstrate code-focused initialization."""
    print("=== Code-Focused Initialization ===")
    
    # Create model config with code initialization
    model_config = ModelConfig(
        name="code-demo",
        initialization_strategy="structured",
        prior_knowledge="code"
    )
    
    config = BangkongConfig(model=model_config)
    
    # Create and initialize model
    model = create_sample_model()
    initialized_model = apply_intelligent_initialization(model, config)
    
    print("Model initialized with code-focused patterns")
    print(f"Model name: {config.model.name}")
    print(f"Initialization strategy: {config.model.initialization_strategy}")
    print(f"Prior knowledge: {config.model.prior_knowledge}")
    print()

def demonstrate_other_initialization_strategies():
    """Demonstrate other initialization strategies."""
    print("=== Other Initialization Strategies ===")
    
    # Xavier initialization
    model_config = ModelConfig(
        name="xavier-demo",
        initialization_strategy="xavier"
    )
    
    config = BangkongConfig(model=model_config)
    model = create_sample_model()
    initialized_model = apply_intelligent_initialization(model, config)
    print("Model initialized with Xavier uniform initialization")
    
    # Kaiming initialization
    model_config = ModelConfig(
        name="kaiming-demo",
        initialization_strategy="kaiming"
    )
    
    config = BangkongConfig(model=model_config)
    model = create_sample_model()
    initialized_model = apply_intelligent_initialization(model, config)
    print("Model initialized with Kaiming normal initialization")
    
    print()

if __name__ == "__main__":
    print("Bangkong Intelligent Initialization Examples")
    print("=" * 50)
    print()
    
    demonstrate_reasoning_initialization()
    demonstrate_math_initialization()
    demonstrate_code_initialization()
    demonstrate_other_initialization_strategies()
    
    print("All examples completed successfully!")