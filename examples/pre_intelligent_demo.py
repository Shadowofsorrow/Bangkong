#!/usr/bin/env python3
"""
Comprehensive example of pre-intelligent initialization system
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import bangkong components
from bangkong.config.schemas import BangkongConfig, ModelConfig
from bangkong.models.intelligent_init import IntelligentInitializer

# Import pre-intelligent components
from bangkong.pre_intelligent import PreIntelligentInitializer, create_pre_intelligent_config

class PreIntelligentModel(nn.Module):
    """Example model enhanced with pre-intelligent components."""
    
    def __init__(self, hidden_size: int = 768, num_layers: int = 12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Standard transformer components
        self.embedding = nn.Embedding(50257, hidden_size)  # GPT-2 vocab size
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=12,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_size, 50257)
        
        # Pre-intelligent components (will be added during initialization)
        self.reasoning_organs = None
        self.memory_system = None
        self.consistency_layer = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard transformer forward pass
        x = self.embedding(x)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        x = self.output_layer(x)
        return x

def create_pre_intelligent_model(config: BangkongConfig) -> PreIntelligentModel:
    """
    Create a model enhanced with pre-intelligent initialization.
    
    Args:
        config: Bangkong configuration
        
    Returns:
        Pre-intelligent enhanced model
    """
    # Create base model
    model = PreIntelligentModel(
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers
    )
    
    # Create pre-intelligent initializer
    pre_intelligent_config = create_pre_intelligent_config()
    initializer = PreIntelligentInitializer(pre_intelligent_config)
    
    # Determine domain concepts based on configuration
    domain_concepts = []
    if config.model.domain == "math":
        domain_concepts = ['math', 'arithmetic', 'algebra']
    elif config.model.domain == "code":
        domain_concepts = ['code', 'programming', 'logic']
    elif config.model.domain == "reasoning":
        domain_concepts = ['reasoning', 'logic', 'planning']
    else:
        domain_concepts = ['general', 'reasoning', 'language']
    
    # Initialize model with pre-intelligent components
    initialized_model = initializer.initialize_model(model, domain_concepts)
    
    # Generate curriculum for this domain
    curriculum_dir = f"curriculum_{config.model.domain}"
    curriculum_files = initializer.generate_curriculum(curriculum_dir, num_stages=5)
    
    print(f"Created pre-intelligent model for domain: {config.model.domain}")
    print(f"Domain concepts: {domain_concepts}")
    print(f"Generated {len(curriculum_files)} curriculum stages")
    
    return initialized_model

def demonstrate_pre_intelligent_training():
    """Demonstrate training with pre-intelligent initialization."""
    print("Demonstrating pre-intelligent training workflow...")
    
    # Create configuration for different domains
    domains = ['math', 'code', 'reasoning', 'general']
    
    for domain in domains:
        print(f"\n--- Creating model for {domain} domain ---")
        
        # Create model configuration
        model_config = ModelConfig(
            name=f"pre-intelligent-{domain}-model",
            architecture="gpt2",
            size="small",
            domain=domain,
            hidden_size=768,
            num_layers=6,  # Smaller for demonstration
            num_heads=12,
            sequence_length=512
        )
        
        config = BangkongConfig(model=model_config)
        
        # Create pre-intelligent model
        model = create_pre_intelligent_model(config)
        
        # Verify model components
        print(f"Model components:")
        print(f"  - Reasoning organs: {hasattr(model, 'reasoning_organs') and model.reasoning_organs is not None}")
        print(f"  - Memory system: {hasattr(model, 'memory_system') and model.memory_system is not None}")
        print(f"  - Consistency layer: {hasattr(model, 'consistency_layer') and model.consistency_layer is not None}")
        
        # Test forward pass
        dummy_input = torch.randint(0, 50257, (2, 10))  # Batch of 2, sequence length 10
        with torch.no_grad():
            output = model(dummy_input)
            print(f"  - Forward pass successful: {output.shape}")
    
    print("\nPre-intelligent training demonstration completed!")

def compare_with_standard_initialization():
    """Compare pre-intelligent initialization with standard initialization."""
    print("\n--- Comparing initialization approaches ---")
    
    # Create model configuration
    model_config = ModelConfig(
        name="comparison-model",
        architecture="gpt2",
        size="small",
        domain="reasoning",
        hidden_size=768,
        num_layers=6,
        num_heads=12,
        sequence_length=512,
        initialization_strategy="structured",
        prior_knowledge="reasoning"
    )
    
    config = BangkongConfig(model=model_config)
    
    # Standard initialization
    print("1. Standard structured initialization:")
    standard_model = PreIntelligentModel(hidden_size=768, num_layers=6)
    standard_initializer = IntelligentInitializer(config)
    standard_initialized = standard_initializer.initialize_model(standard_model)
    print(f"   Model initialized with structured approach")
    
    # Pre-intelligent initialization
    print("2. Pre-intelligent initialization:")
    pre_intelligent_model = create_pre_intelligent_model(config)
    print(f"   Model initialized with pre-intelligent approach")
    
    # Compare parameter counts
    standard_params = sum(p.numel() for p in standard_initialized.parameters())
    pre_intelligent_params = sum(p.numel() for p in pre_intelligent_model.parameters())
    
    print(f"   Standard model parameters: {standard_params:,}")
    print(f"   Pre-intelligent model parameters: {pre_intelligent_params:,}")
    print(f"   Additional parameters: {pre_intelligent_params - standard_params:,}")

if __name__ == "__main__":
    demonstrate_pre_intelligent_training()
    compare_with_standard_initialization()
    
    print("\n" + "="*60)
    print("PRE-INTELLIGENT INITIALIZATION SUMMARY")
    print("="*60)
    print("Key features implemented:")
    print("1. Meta-learning initialization with MAML/Reptile")
    print("2. Hypernetwork-based prior generation")
    print("3. Hierarchical differentiable memory system")
    print("4. Reasoning organs (graph, validator, temporal heads)")
    print("5. Energy-based global consistency layer")
    print("6. Curriculum learning with synthetic traces")
    print("7. Domain-specific initialization")
    print("\nBenefits:")
    print("- Models start with reasoning priors")
    print("- Structured knowledge embedding at initialization")
    print("- Built-in memory and consistency mechanisms")
    print("- Reduced training data requirements")
    print("- Better generalization from fewer examples")