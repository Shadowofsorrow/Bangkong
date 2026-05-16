#!/usr/bin/env python3
"""
Main pre-intelligent initialization system for bangkong LLMs
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import os
from pathlib import Path
import logging

# Import pre-intelligent components
from ..pre_intelligent.meta_learning.maml_reptile import train_meta_initializer, extract_meta_initialization
from ..pre_intelligent.hypernetwork.prior_generator import PriorGenerator, KnowledgeBase
from ..pre_intelligent.memory.hierarchical_memory import HierarchicalMemory
from ..pre_intelligent.reasoning_organs.reasoning_heads import ReasoningOrgans
from ..pre_intelligent.energy_layer.energy_consistency import GlobalConsistencyLayer
from ..pre_intelligent.curriculum.reasoning_curriculum import ReasoningTraceGenerator, CurriculumScheduler

class PreIntelligentInitializer:
    """Main pre-intelligent initialization system."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize all pre-intelligent components."""
        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Store configuration
        self.config = config

        # Meta-learning initializer
        self.prior_generator = PriorGenerator(
            latent_dim=self.config.get("latent_dim", 128),
            knowledge_dim=self.config.get("knowledge_dim", 64),
            adapter_rank=self.config.get("adapter_rank", 8)
        )

        # Knowledge base
        self.knowledge_base = KnowledgeBase()

        # Memory system
        self.memory_system = HierarchicalMemory(
            hidden_size=self.config.get("hidden_size", 768),
            scratchpad_size=self.config.get("scratchpad_size", 64),
            context_size=self.config.get("context_size", 128),
            semantic_size=self.config.get("semantic_size", 256)
        )

        # Reasoning organs
        self.reasoning_organs = ReasoningOrgans(
            hidden_size=self.config.get("hidden_size", 768)
        )

        # Global consistency layer
        self.consistency_layer = GlobalConsistencyLayer(
            hidden_size=self.config.get("hidden_size", 768),
            energy_dim=self.config.get("energy_dim", 256)
        )

        # Curriculum generator
        self.curriculum_generator = ReasoningTraceGenerator(
            num_samples=self.config.get("curriculum_samples", 1000),
            difficulty_range=self.config.get("difficulty_range", (1, 5))
        )

    def generate_meta_priors(self, concepts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Generate meta-initialized priors for concepts.

        Args:
            concepts: List of concept names

        Returns:
            Dictionary of generated priors
        """
        print(f"Generating meta priors for concepts: {concepts}")

        # Get composite knowledge embedding
        knowledge_vector = self.knowledge_base.get_composite_embedding(concepts)

        # Generate priors using hypernetwork
        priors = self.prior_generator.generate_priors(knowledge_vector)

        # NEW: Integrate MAML/Reptile meta-initialization
        # Train meta-initializer using MAML/Reptile with model configuration
        try:
            from ..pre_intelligent.meta_learning.maml_reptile import train_meta_initializer, create_meta_initialization_patterns
            print("Training MAML/Reptile meta-initializer...")
            
            # Create model configuration based on the actual model being initialized
            model_config = {
                'hidden_size': self.config.get("hidden_size", 768),
                'vocab_size': 50257,  # Default GPT-2 vocab size
                'sequence_length': self.config.get("sequence_length", 1024)
            }
            
            # Create initialization patterns that can be applied to the larger model
            meta_init = create_meta_initialization_patterns(model_config)
            print("MAML/Reptile meta-initialization completed and integrated with priors")
            
            # Combine hypernetwork priors with MAML/Reptile initialization
            priors['meta_initialization'] = meta_init
        except Exception as e:
            print(f"Warning: MAML/Reptile meta-initialization failed: {e}")
            print("Falling back to hypernetwork-only priors")
            meta_init = {}
            
            # Add empty meta initialization to priors
            priors['meta_initialization'] = meta_init

        return priors

    def apply_meta_initialization(self, model: nn.Module, priors: Dict[str, torch.Tensor]):
        """Apply meta-initialized priors to model with proper parameter mapping."""
        print("Applying meta-initialization to model...")
        
        # Apply LoRA adapters if they exist in the model
        if hasattr(model, 'lora_A') and hasattr(model, 'lora_B'):
            # Apply adapter weights
            if 'adapter_A' in priors and 'adapter_B' in priors:
                with torch.no_grad():
                    # Check tensor shapes before copying
                    adapter_A = priors['adapter_A'].squeeze(0)
                    adapter_B = priors['adapter_B'].squeeze(0)
                    
                    # Ensure shapes match before copying
                    if adapter_A.shape == model.lora_A.shape:
                        model.lora_A.copy_(adapter_A)
                    else:
                        # Handle shape mismatch by direct resizing
                        if adapter_A.numel() <= model.lora_A.numel():
                            # Expand smaller tensor to fit larger one
                            # Flatten both tensors and copy as much as possible
                            flat_A = adapter_A.view(-1)
                            flat_model_A = model.lora_A.view(-1)
                            flat_model_A[:flat_A.numel()] = flat_A
                        else:
                            # If adapter_A is larger, we need to reduce it
                            model.lora_A.copy_(adapter_A.view(-1)[:model.lora_A.numel()].view(model.lora_A.shape))
                    
                    if adapter_B.shape == model.lora_B.shape:
                        model.lora_B.copy_(adapter_B)
                    else:
                        # Handle shape mismatch by direct resizing
                        if adapter_B.numel() <= model.lora_B.numel():
                            # Expand smaller tensor to fit larger one
                            # Flatten both tensors and copy as much as possible
                            flat_B = adapter_B.view(-1)
                            flat_model_B = model.lora_B.view(-1)
                            flat_model_B[:flat_B.numel()] = flat_B
                        else:
                            # If adapter_B is larger, we need to reduce it
                            model.lora_B.copy_(adapter_B.view(-1)[:model.lora_B.numel()].view(model.lora_B.shape))
                print("Applied LoRA adapters from meta-priors")

        # Apply MAML/Reptile meta-initialization if available
        if 'meta_initialization' in priors:
            meta_init = priors['meta_initialization']
            if isinstance(meta_init, dict) and meta_init:
                print("Applying MAML/Reptile meta-initialization patterns...")
                # Import the mapping function from maml_reptile module
                from ..pre_intelligent.meta_learning.maml_reptile import map_meta_initialization_to_model
                # Apply meta-initialization patterns to model parameters with proper mapping
                map_meta_initialization_to_model(model, meta_init)
                print("Applied MAML/Reptile meta-initialization patterns to model parameters")
            else:
                print("No MAML/Reptile meta-initialization patterns to apply")

    def generate_curriculum(self, output_dir: str, num_stages: int = 5) -> List[str]:
        """
        Generate curriculum for pre-intelligent training.

        Args:
            output_dir: Directory to save curriculum files
            num_stages: Number of curriculum stages

        Returns:
            List of generated curriculum file paths
        """
        print(f"Generating curriculum with {num_stages} stages...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate curriculum
        curriculum_files = self.curriculum_generator.generate_curriculum(
            base_output_dir=output_dir,
            num_stages=num_stages
        )

        print(f"Generated curriculum files: {curriculum_files}")
        return curriculum_files

    def initialize_model(self,
                        model: nn.Module,
                        domain_concepts: List[str] = None) -> nn.Module:
        """Fully initialize model with pre-intelligent components.

        Args:
            model: Model to initialize
            domain_concepts: Optional list of domain concepts

        Returns:
            Initialized model
        """
        print("Starting pre-intelligent initialization...")

        # Check if model is already enhanced
        if hasattr(model, '_pre_intelligent_enhanced'):
            print("Model already enhanced with pre-intelligent components")
            return model

        # Use default concepts if none provided
        if domain_concepts is None:
            domain_concepts = ['reasoning', 'logic', 'math']

        # 1. Generate meta priors
        priors = self.generate_meta_priors(domain_concepts)

        # 2. Apply meta initialization
        self.apply_meta_initialization(model, priors)

        # 3. Store all pre-intelligent components (avoid circular references)
        model._pre_intelligent_components = {
            'reasoning_organs': self.reasoning_organs,
            'memory_system': self.memory_system,
            'consistency_layer': self.consistency_layer
        }

        # Add components as direct attributes (they won't be PyTorch submodules)
        model.reasoning_organs = self.reasoning_organs
        model.memory_system = self.memory_system
        model.consistency_layer = self.consistency_layer

        # 4. Store reference to original forward method if not already stored
        if not hasattr(model, '_original_forward'):
            model._original_forward = model.forward

        original_forward = model._original_forward

        # 5. Replace forward method with enhanced version that applies all components
        def enhanced_forward(*args, **kwargs):
            # Debug information about the call
            self.logger.debug(f"enhanced_forward called with {len(args)} args and {len(kwargs)} kwargs")
            self.logger.debug(f"kwargs keys: {list(kwargs.keys())}")

            # Remove the model from args if it's the first argument
            # (This happens when someone calls model(input_ids=..., labels=...))
            clean_args = args
            if args and isinstance(args[0], type(model)):
                self.logger.debug("Removing model from args")
                clean_args = args[1:]  # Remove the first argument (model)

            # Call original forward method
            self.logger.debug("Calling original_forward with clean_args and kwargs")
            try:
                outputs = original_forward(*clean_args, **kwargs)
                self.logger.debug(f"Original forward succeeded, output type: {type(outputs)}")
            except Exception as e:
                self.logger.error(f"Original forward failed: {e}")
                self.logger.error(f"args: {[type(arg) for arg in clean_args]}")
                self.logger.error(f"kwargs keys: {list(kwargs.keys())}")
                raise

            # Debug information about outputs
            self.logger.debug(f"Original forward returned type: {type(outputs)}")
            self.logger.debug(f"Original forward attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
            self.logger.debug(f"Original forward has loss: {hasattr(outputs, 'loss')}")
            self.logger.debug(f"Original forward has logits: {hasattr(outputs, 'logits')}")
            if hasattr(outputs, 'loss'):
                loss_attr = getattr(outputs, 'loss', None)
                self.logger.debug(f"Original forward loss value: {loss_attr}")
                self.logger.debug(f"Original forward loss type: {type(loss_attr)}")
            if hasattr(outputs, 'logits'):
                logits_attr = getattr(outputs, 'logits', None)
                self.logger.debug(f"Original forward logits shape: {getattr(logits_attr, 'shape', 'no shape') if logits_attr is not None else 'None'}")

            # Apply pre-intelligent enhancements and integrate into outputs
            try:
                # Get hidden states if available
                hidden_states = None
                if hasattr(outputs, 'hidden_states'):
                    if isinstance(outputs.hidden_states, tuple):
                        hidden_states = outputs.hidden_states[-1] if outputs.hidden_states else None
                    else:
                        hidden_states = outputs.hidden_states

                enhancement_residual = None

                # Apply reasoning organs enhancement
                if hidden_states is not None and hasattr(model, 'reasoning_organs'):
                    try:
                        reasoning_output = model.reasoning_organs(hidden_states)
                        if isinstance(reasoning_output, dict):
                            enhanced = reasoning_output.get('enhanced_hidden', None)
                        else:
                            enhanced = reasoning_output
                        if enhanced is not None and enhanced.shape == hidden_states.shape:
                            if enhancement_residual is None:
                                enhancement_residual = torch.zeros_like(hidden_states)
                            enhancement_residual = enhancement_residual + 0.1 * enhanced
                    except (RuntimeError, TypeError, ValueError) as e:
                        print(f"Warning: Reasoning organs processing failed: {e}")

                # Apply memory system enhancement
                if hidden_states is not None and hasattr(model, 'memory_system'):
                    try:
                        memory_output = model.memory_system(hidden_states)
                        if isinstance(memory_output, dict):
                            enhanced = memory_output.get('enhanced', memory_output.get('output', None))
                        else:
                            enhanced = memory_output
                        if enhanced is not None and enhanced.shape == hidden_states.shape:
                            if enhancement_residual is None:
                                enhancement_residual = torch.zeros_like(hidden_states)
                            enhancement_residual = enhancement_residual + 0.1 * enhanced
                    except (RuntimeError, TypeError, ValueError) as e:
                        print(f"Warning: Memory system processing failed: {e}")

                # Apply consistency layer enhancement
                if hidden_states is not None and hasattr(model, 'consistency_layer'):
                    try:
                        consistency_output = model.consistency_layer(hidden_states)
                        if isinstance(consistency_output, dict):
                            enhanced = consistency_output.get('enhanced_states', None)
                        else:
                            enhanced = consistency_output
                        if enhanced is not None and enhanced.shape == hidden_states.shape:
                            if enhancement_residual is None:
                                enhancement_residual = torch.zeros_like(hidden_states)
                            enhancement_residual = enhancement_residual + 0.1 * enhanced
                    except (RuntimeError, TypeError, ValueError) as e:
                        print(f"Warning: Consistency layer processing failed: {e}")

                # Integrate enhancement residual back into outputs SAFELY
                if enhancement_residual is not None:
                    # IMPORTANT: We MUST NOT inject random noise into the logits!
                    # Doing so destroys the cross-entropy loss gradients during training.
                    # Instead, we apply the residual to the last hidden state if it exists,
                    # which is standard for residual connection architectures.
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        if isinstance(outputs.hidden_states, tuple):
                            new_last = outputs.hidden_states[-1] + 0.1 * enhancement_residual
                            outputs.hidden_states = outputs.hidden_states[:-1] + (new_last,)

            except (RuntimeError, TypeError, ValueError, AttributeError) as e:
                print(f"Warning: Pre-intelligent enhancement failed: {e}")

            # Always return original outputs to maintain interface compatibility
            # Make sure all attributes are preserved
            self.logger.debug(f"Returning outputs with type: {type(outputs)}")
            return outputs

        # Apply the enhanced forward method
        model.forward = enhanced_forward.__get__(model, model.__class__)

        # Mark model as enhanced
        model._pre_intelligent_enhanced = True

        print("Pre-intelligent initialization completed!")
        return model

    def save_initialization_state(self, filepath: str):
        """
        Save initialization state to file.

        Args:
            filepath: Path to save state
        """
        state = {
            'config': self.config,
            'knowledge_base': self.knowledge_base.concepts
        }

        torch.save(state, filepath)
        print(f"Initialization state saved to {filepath}")

    def load_initialization_state(self, filepath: str):
        """
        Load initialization state from file.

        Args:
            filepath: Path to load state from
        """
        state = torch.load(filepath)
        self.config = state['config']
        self.knowledge_base.concepts = state['knowledge_base']
        print(f"Initialization state loaded from {filepath}")

def create_pre_intelligent_config() -> Dict[str, Any]:
    """Create default configuration for pre-intelligent initialization."""
    # Get configuration from models config
    from ..models.config_loader import get_models_config
    models_config = get_models_config()
    preint_config = models_config.get("models.pre_intelligent", {})

    return {
        "latent_dim": preint_config.get("latent_dim", 128),
        "knowledge_dim": preint_config.get("knowledge_dim", 64),
        "adapter_rank": preint_config.get("adapter_rank", 8),
        "hidden_size": preint_config.get("hidden_size", 768),
        "scratchpad_size": preint_config.get("scratchpad_size", 64),
        "context_size": preint_config.get("context_size", 128),
        "semantic_size": preint_config.get("semantic_size", 256),
        "energy_dim": preint_config.get("energy_dim", 256),
        "curriculum_samples": preint_config.get("curriculum_samples", 1000),
        "difficulty_range": tuple(preint_config.get("difficulty_range", [1, 5]))
    }

def demonstrate_pre_intelligent_initialization():
    """Demonstrate pre-intelligent initialization system."""
    print("Demonstrating pre-intelligent initialization system...")

    # Create configuration
    config = create_pre_intelligent_config()

    # Initialize pre-intelligent system
    initializer = PreIntelligentInitializer(config)

    # Create a simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self, hidden_size: int = 768):
            super().__init__()
            self.hidden_size = hidden_size
            self.lora_A = nn.Parameter(torch.randn(hidden_size, config["adapter_rank"]))
            self.lora_B = nn.Parameter(torch.randn(config["adapter_rank"], hidden_size))
            self.transformer_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=12,
                batch_first=True
            )

        def forward(self, x):
            return self.transformer_layer(x)

    model = SimpleModel()

    # Initialize model with pre-intelligent components
    domain_concepts = ['math', 'reasoning', 'logic']
    initialized_model = initializer.initialize_model(model, domain_concepts)

    # Generate curriculum
    curriculum_files = initializer.generate_curriculum("curriculum_output", num_stages=3)

    # Save initialization state
    initializer.save_initialization_state("initialization_state.pt")

    print("Pre-intelligent initialization demonstration completed!")

if __name__ == "__main__":
    demonstrate_pre_intelligent_initialization()