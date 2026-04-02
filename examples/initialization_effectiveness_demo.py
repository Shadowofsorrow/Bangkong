#!/usr/bin/env python3
"""
Practical demonstration of intelligent initialization effectiveness
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bangkong.config.schemas import BangkongConfig, ModelConfig
from bangkong.models.intelligent_init import IntelligentInitializer

def create_reasoning_task_model():
    """Create a model for a reasoning task (simple logical inference)."""
    class ReasoningModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 64)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256),
                num_layers=4
            )
            self.classifier = nn.Linear(64, 2)  # Binary classification
        
        def forward(self, x):
            x = self.embedding(x)
            x = x.transpose(0, 1)  # For transformer
            x = self.transformer(x)
            x = x.transpose(0, 1)  # Back to batch first
            # Use the first token for classification
            x = self.classifier(x[:, 0, :])
            return x
    
    return ReasoningModel()

def create_math_task_model():
    """Create a model for a math task (simple arithmetic)."""
    class MathModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(20, 32)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=128),
                num_layers=3
            )
            self.regressor = nn.Linear(32, 1)  # Regression output
        
        def forward(self, x):
            x = self.embedding(x)
            x = x.transpose(0, 1)  # For transformer
            x = self.transformer(x)
            x = x.transpose(0, 1)  # Back to batch first
            # Use the first token for regression
            x = self.regressor(x[:, 0, :])
            return x
    
    return MathModel()

def create_code_task_model():
    """Create a model for a code task (simple pattern recognition)."""
    class CodeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(50, 48)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=48, nhead=6, dim_feedforward=192),
                num_layers=4
            )
            self.classifier = nn.Linear(48, 5)  # Multi-class classification
        
        def forward(self, x):
            x = self.embedding(x)
            x = x.transpose(0, 1)  # For transformer
            x = self.transformer(x)
            x = x.transpose(0, 1)  # Back to batch first
            # Use the first token for classification
            x = self.classifier(x[:, 0, :])
            return x
    
    return CodeModel()

def generate_reasoning_data(batch_size=32, seq_length=10):
    """Generate synthetic reasoning data (logical patterns)."""
    # Simple pattern: if sequence contains both 1 and 2, output class 1, else class 0
    sequences = torch.randint(0, 10, (batch_size, seq_length))
    labels = torch.zeros(batch_size, dtype=torch.long)
    
    # Label based on presence of specific tokens
    for i in range(batch_size):
        if (sequences[i] == 1).any() and (sequences[i] == 2).any():
            labels[i] = 1
    
    return sequences, labels

def generate_math_data(batch_size=32, seq_length=8):
    """Generate synthetic math data (arithmetic patterns)."""
    # Simple pattern: sum of first two numbers in sequence
    sequences = torch.randint(0, 10, (batch_size, seq_length))
    targets = sequences[:, 0].float() + sequences[:, 1].float()
    
    return sequences, targets

def generate_code_data(batch_size=32, seq_length=12):
    """Generate synthetic code data (pattern recognition)."""
    # Simple pattern: classify based on sequence structure
    sequences = torch.randint(0, 20, (batch_size, seq_length))
    labels = torch.zeros(batch_size, dtype=torch.long)
    
    # Label based on pattern in sequence
    for i in range(batch_size):
        # Simple pattern: if sequence starts with 0,1,2 then class 1, etc.
        if seq_length >= 3 and sequences[i, 0] == 0 and sequences[i, 1] == 1 and sequences[i, 2] == 2:
            labels[i] = 1
        elif seq_length >= 2 and sequences[i, 0] == 5 and sequences[i, 1] == 5:
            labels[i] = 2
        elif (sequences[i] == 9).sum() > 2:  # More than 2 occurrences of 9
            labels[i] = 3
        elif seq_length >= 4 and sequences[i, -1] == 3 and sequences[i, -2] == 3:
            labels[i] = 4
        # Else class 0
    
    return sequences, labels

def train_model(model, data_generator, criterion, optimizer, epochs=50):
    """Train a model and return loss history."""
    model.train()
    losses = []
    
    for epoch in range(epochs):
        # Generate batch
        inputs, targets = data_generator()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return losses

def compare_initialization_effectiveness():
    """Compare effectiveness of different initialization strategies."""
    print("Comparing initialization strategies for different tasks...")
    print("=" * 60)
    
    tasks = [
        ("Reasoning", create_reasoning_task_model, generate_reasoning_data, nn.CrossEntropyLoss()),
        ("Math", create_math_task_model, generate_math_data, nn.MSELoss()),
        ("Code", create_code_task_model, generate_code_data, nn.CrossEntropyLoss())
    ]
    
    strategies = [
        ("random", None),
        ("xavier", None),
        ("kaiming", None),
        ("structured", "reasoning"),
        ("structured", "math"),
        ("structured", "code")
    ]
    
    results = {}
    
    for task_name, model_fn, data_fn, criterion in tasks:
        print(f"\nTask: {task_name}")
        print("-" * 30)
        results[task_name] = {}
        
        for strategy, prior_knowledge in strategies:
            print(f"  Testing {strategy} initialization{' with ' + prior_knowledge if prior_knowledge else ''}...")
            
            # Create model
            model = model_fn()
            
            # Apply initialization
            model_config = ModelConfig(
                name=f"{task_name.lower()}-{strategy}",
                initialization_strategy=strategy,
                prior_knowledge=prior_knowledge
            )
            config = BangkongConfig(model=model_config)
            initializer = IntelligentInitializer(config)
            initialized_model = initializer.initialize_model(model)
            
            # Set up training
            optimizer = optim.Adam(initialized_model.parameters(), lr=0.001)
            
            # Train model
            try:
                losses = train_model(initialized_model, data_fn, criterion, optimizer, epochs=30)
                final_loss = losses[-1]
                print(f"    Final loss: {final_loss:.4f}")
                results[task_name][f"{strategy}-{prior_knowledge or 'none'}"] = final_loss
            except Exception as e:
                print(f"    Training failed: {e}")
                results[task_name][f"{strategy}-{prior_knowledge or 'none'}"] = float('inf')
    
    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary:")
    print("=" * 60)
    
    for task_name in results:
        print(f"\n{task_name} Task:")
        print("-" * 20)
        best_loss = float('inf')
        best_strategy = ""
        
        for strategy_key, loss in results[task_name].items():
            print(f"  {strategy_key:<20}: {loss:.4f}")
            if loss < best_loss:
                best_loss = loss
                best_strategy = strategy_key
        
        print(f"  Best strategy: {best_strategy} (loss: {best_loss:.4f})")
    
    return results

if __name__ == "__main__":
    results = compare_initialization_effectiveness()
    print("\nDemonstration completed!")