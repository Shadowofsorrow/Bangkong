#!/usr/bin/env python3
"""
Enhanced validation script for fine-tuned intelligent initialization
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from bangkong.config.schemas import BangkongConfig, ModelConfig
from bangkong.models.intelligent_init import IntelligentInitializer

class EnhancedInitializationValidator:
    """Enhanced validator for fine-tuned intelligent initialization methods."""
    
    def __init__(self):
        self.results = {}
    
    def validate_fine_tuned_initialization(self, strategy, knowledge, task_type):
        """Validate fine-tuned initialization effectiveness."""
        print(f"Validating fine-tuned {strategy}-{knowledge} for {task_type} task...")
        
        # Create appropriate model for task
        model = self._create_task_model(task_type)
        
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
        param_validation = self._validate_parameter_distribution(initialized_model, strategy, knowledge)
        
        # Validate gradient flow
        gradient_validation = self._validate_gradient_flow(initialized_model, strategy, knowledge)
        
        # Validate adaptive parameters
        adaptive_validation = self._validate_adaptive_parameters(initializer, strategy, knowledge)
        
        # Store results
        key = f"{strategy}-{knowledge}-{task_type}"
        self.results[key] = {
            'parameter_validation': param_validation,
            'gradient_validation': gradient_validation,
            'adaptive_validation': adaptive_validation
        }
        
        return self.results[key]
    
    def _create_task_model(self, task_type):
        """Create a model for specific task type."""
        if task_type == 'reasoning':
            return self._create_reasoning_model()
        elif task_type == 'math':
            return self._create_math_model()
        elif task_type == 'code':
            return self._create_code_model()
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
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
    
    def _validate_parameter_distribution(self, model, strategy, knowledge):
        """Validate that parameter distributions match expected patterns."""
        print(f"  Validating parameter distribution...")
        
        stats = {}
        issues = []
        
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
                
                # Check for expected patterns based on fine-tuned parameters
                if strategy == 'structured':
                    if knowledge == 'reasoning':
                        # Reasoning should have controlled weights
                        if 'embedding' in name and std > 0.1:
                            issues.append(f"Reasoning embedding std too high: {std:.4f}")
                        elif 'attention' in name and std > 0.15:
                            issues.append(f"Reasoning attention std too high: {std:.4f}")
                    elif knowledge == 'math':
                        # Math should have very controlled weights
                        if 'embedding' in name and std > 0.05:
                            issues.append(f"Math embedding std too high: {std:.4f}")
                        elif std > 0.1:
                            issues.append(f"Math weight std too high: {std:.4f}")
                    elif knowledge == 'code':
                        # Code should have moderate variance for pattern recognition
                        if 'embedding' in name and (std < 0.05 or std > 0.2):
                            issues.append(f"Code embedding std out of range: {std:.4f}")
        
        return {
            'stats': stats,
            'issues': issues
        }
    
    def _validate_gradient_flow(self, model, strategy, knowledge):
        """Validate gradient flow properties."""
        print(f"  Validating gradient flow...")
        
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
        
        # Check for domain-specific gradient expectations
        if strategy == 'structured':
            if knowledge == 'reasoning' and avg_grad_norm > 10:
                issues.append(f"Reasoning gradient norm too high: {avg_grad_norm:.4f}")
            elif knowledge == 'math' and avg_grad_norm > 5:
                issues.append(f"Math gradient norm too high: {avg_grad_norm:.4f}")
            elif knowledge == 'code' and avg_grad_norm > 20:
                issues.append(f"Code gradient norm too high: {avg_grad_norm:.4f}")
        
        return {
            'avg_gradient_norm': avg_grad_norm,
            'issues': issues
        }
    
    def _validate_adaptive_parameters(self, initializer, strategy, knowledge):
        """Validate that adaptive parameters are correctly set."""
        print(f"  Validating adaptive parameters...")
        
        issues = []
        
        if strategy == 'structured':
            expected_params = initializer.adaptive_params
            
            if knowledge == 'reasoning':
                if expected_params['attention_bias_mean'] != -0.1:
                    issues.append(f"Reasoning attention bias incorrect: {expected_params['attention_bias_mean']}")
                if expected_params['embedding_gain'] != 0.05:
                    issues.append(f"Reasoning embedding gain incorrect: {expected_params['embedding_gain']}")
            elif knowledge == 'math':
                if expected_params['attention_bias_mean'] != -0.2:
                    issues.append(f"Math attention bias incorrect: {expected_params['attention_bias_mean']}")
                if expected_params['embedding_gain'] != 0.02:
                    issues.append(f"Math embedding gain incorrect: {expected_params['embedding_gain']}")
            elif knowledge == 'code':
                if expected_params['attention_bias_mean'] != 0.15:
                    issues.append(f"Code attention bias incorrect: {expected_params['attention_bias_mean']}")
                if expected_params['embedding_gain'] != 0.1:
                    issues.append(f"Code embedding gain incorrect: {expected_params['embedding_gain']}")
        
        return {
            'issues': issues
        }
    
    def run_enhanced_validation(self):
        """Run enhanced validation suite."""
        print("Running enhanced initialization validation...")
        print("=" * 50)
        
        strategies = [
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
                    self.validate_fine_tuned_initialization(strategy, knowledge, task)
                    print(f"  ✓ {strategy}-{knowledge}: Completed")
                except Exception as e:
                    print(f"  ✗ {strategy}-{knowledge}: Failed ({e})")
        
        print("\n" + "=" * 50)
        print("Enhanced validation completed!")
        return self.results

def main():
    """Main validation function."""
    validator = EnhancedInitializationValidator()
    results = validator.run_enhanced_validation()
    
    # Save results
    with open('enhanced_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to enhanced_validation_results.json")
    
    # Print summary
    print("\nEnhanced Validation Summary:")
    print("-" * 40)
    for key, result in results.items():
        param_issues = len(result['parameter_validation']['issues'])
        grad_issues = len(result['gradient_validation']['issues'])
        adaptive_issues = len(result['adaptive_validation']['issues'])
        print(f"{key:<35} | Param: {param_issues} | Grad: {grad_issues} | Adapt: {adaptive_issues}")

if __name__ == "__main__":
    main()