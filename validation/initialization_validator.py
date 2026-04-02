#!/usr/bin/env python3
"""
Comprehensive validation framework for intelligent initialization
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from bangkong.config.schemas import BangkongConfig, ModelConfig
from bangkong.models.intelligent_init import IntelligentInitializer

class InitializationValidator:
    """Validator for intelligent initialization methods."""
    
    def __init__(self):
        self.results = {}
    
    def validate_parameter_distribution(self, model, strategy, knowledge):
        """Validate that parameter distributions match expected patterns."""
        print(f"Validating parameter distribution for {strategy}-{knowledge}...")
        
        stats = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                mean = param.data.mean().item()
                std = param.data.std().item()
                min_val = param.data.min().item()
                max_val = param.data.max().item()
                
                stats[name] = {
                    'mean': mean,
                    'std': std,
                    'min': min_val,
                    'max': max_val
                }
        
        # Check for expected patterns
        issues = []
        if strategy == 'structured':
            if knowledge == 'reasoning':
                # Reasoning should have lower variance weights
                for name, stat in stats.items():
                    if 'attention' in name and stat['std'] > 0.15:
                        issues.append(f"Attention weight std too high: {stat['std']:.4f}")
            elif knowledge == 'math':
                # Math should have very controlled weights
                for name, stat in stats.items():
                    if stat['std'] > 0.2:
                        issues.append(f"Weight std too high for math: {stat['std']:.4f}")
            elif knowledge == 'code':
                # Code should have higher variance for pattern recognition
                attention_std_sum = 0
                attention_count = 0
                for name, stat in stats.items():
                    if 'attention' in name:
                        attention_std_sum += stat['std']
                        attention_count += 1
                
                if attention_count > 0:
                    avg_attention_std = attention_std_sum / attention_count
                    if avg_attention_std < 0.1:
                        issues.append(f"Attention weights std too low for code: {avg_attention_std:.4f}")
        
        return {
            'stats': stats,
            'issues': issues
        }
    
    def validate_gradient_flow(self, model, strategy, knowledge):
        """Validate gradient flow properties."""
        print(f"Validating gradient flow for {strategy}-{knowledge}...")
        
        # Create dummy input and compute gradients
        dummy_input = torch.randint(0, 100, (4, 16))  # Batch of 4, sequence length 16
        
        # Simple model for gradient testing
        class TestModel(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                self.loss_fn = nn.MSELoss()
            
            def forward(self, x, target):
                output = self.base_model(x)
                return self.loss_fn(output.mean(), target)
        
        test_model = TestModel(model)
        target = torch.randn(1)
        
        # Compute gradients
        loss = test_model(dummy_input, target)
        loss.backward()
        
        # Check gradient magnitudes
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
        
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
        
        # Check for gradient vanishing/exploding
        issues = []
        if avg_grad_norm < 1e-6:
            issues.append("Gradient vanishing detected")
        elif avg_grad_norm > 100:
            issues.append("Gradient exploding detected")
        
        return {
            'avg_gradient_norm': avg_grad_norm,
            'issues': issues
        }
    
    def validate_initialization_effectiveness(self, strategy, knowledge, task_type):
        """Validate initialization effectiveness for a specific task."""
        print(f"Validating effectiveness for {task_type} task with {strategy}-{knowledge}...")
        
        # Create appropriate model for task
        if task_type == 'reasoning':
            model = self._create_reasoning_model()
        elif task_type == 'math':
            model = self._create_math_model()
        elif task_type == 'code':
            model = self._create_code_model()
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Apply initialization
        model_config = ModelConfig(
            name=f"{task_type}-{strategy}-{knowledge}",
            initialization_strategy=strategy,
            prior_knowledge=knowledge
        )
        config = BangkongConfig(model=model_config)
        initializer = IntelligentInitializer(config)
        initialized_model = initializer.initialize_model(model)
        
        # Validate parameter distribution
        param_validation = self.validate_parameter_distribution(initialized_model, strategy, knowledge)
        
        # Validate gradient flow
        gradient_validation = self.validate_gradient_flow(initialized_model, strategy, knowledge)
        
        # Store results
        key = f"{strategy}-{knowledge}-{task_type}"
        self.results[key] = {
            'parameter_validation': param_validation,
            'gradient_validation': gradient_validation
        }
        
        return self.results[key]
    
    def _create_reasoning_model(self):
        """Create a model for reasoning tasks."""
        class ReasoningModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=64, nhead=4),
                    num_layers=3
                )
                self.classifier = nn.Linear(64, 2)
            
            def forward(self, x):
                x = self.embedding(x)
                x = x.transpose(0, 1)
                x = self.transformer(x)
                x = x.transpose(0, 1)
                x = self.classifier(x[:, 0, :])
                return x
        
        return ReasoningModel()
    
    def _create_math_model(self):
        """Create a model for math tasks."""
        class MathModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(20, 32)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=32, nhead=4),
                    num_layers=2
                )
                self.regressor = nn.Linear(32, 1)
            
            def forward(self, x):
                x = self.embedding(x)
                x = x.transpose(0, 1)
                x = self.transformer(x)
                x = x.transpose(0, 1)
                x = self.regressor(x[:, 0, :])
                return x
        
        return MathModel()
    
    def _create_code_model(self):
        """Create a model for code tasks."""
        class CodeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(50, 48)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=48, nhead=6),
                    num_layers=3
                )
                self.classifier = nn.Linear(48, 5)
            
            def forward(self, x):
                x = self.embedding(x)
                x = x.transpose(0, 1)
                x = self.transformer(x)
                x = x.transpose(0, 1)
                x = self.classifier(x[:, 0, :])
                return x
        
        return CodeModel()
    
    def run_comprehensive_validation(self):
        """Run comprehensive validation suite."""
        print("Running comprehensive initialization validation...")
        print("=" * 50)
        
        strategies = [
            ('random', None),
            ('xavier', None),
            ('kaiming', None),
            ('structured', 'reasoning'),
            ('structured', 'math'),
            ('structured', 'code')
        ]
        
        tasks = ['reasoning', 'math', 'code']
        
        for task in tasks:
            print(f"\nValidating for {task} task:")
            print("-" * 30)
            
            for strategy, knowledge in strategies:
                try:
                    self.validate_initialization_effectiveness(strategy, knowledge, task)
                    print(f"  ✓ {strategy}-{knowledge or 'none'}: Completed")
                except Exception as e:
                    print(f"  ✗ {strategy}-{knowledge or 'none'}: Failed ({e})")
        
        print("\n" + "=" * 50)
        print("Validation completed!")
        return self.results

def main():
    """Main validation function."""
    validator = InitializationValidator()
    results = validator.run_comprehensive_validation()
    
    # Save results
    with open('validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to validation_results.json")
    
    # Print summary
    print("\nValidation Summary:")
    print("-" * 30)
    for key, result in results.items():
        param_issues = len(result['parameter_validation']['issues'])
        grad_issues = len(result['gradient_validation']['issues'])
        print(f"{key:<30} | Param issues: {param_issues} | Grad issues: {grad_issues}")

if __name__ == "__main__":
    main()