"""
Models module for Bangkong LLM Training System

This module handles model creation, training, and management.
"""

from .trainer import DynamicTrainer
from .packager import ModelPackager
from .training_manager import TrainingManager
from .peft import apply_peft_to_model, PEFTAdapter
from .multimodal import create_multimodal_model, MultimodalGPT2Model
from .distillation import create_distiller, ModelDistiller, DistillationLoss
from .quantization import apply_quantization_to_model, quantize_model_weights, QuantizationController
from .pruning import apply_pruning_to_model, PruningController, MagnitudePruning, StructuredPruning
from .regional import load_region_specific_model, create_multilingual_tokenizer, RegionSpecificModelLoader
from .curriculum import create_curriculum_controller, create_adaptive_sampler, CurriculumLearning, AdaptiveBatchSampler
from .specialized import create_specialized_model, CodeGPT2Model, MathGPT2Model, ScientificGPT2Model
from .efficient_attention import create_efficient_attention, EfficientAttentionController, SparseAttention, LongformerAttention
from .intelligent_init import create_intelligent_initializer, apply_intelligent_initialization, IntelligentInitializer
from .cosine_clustered_embeddings import CosineClusteredEmbeddings
from .attention_specialization import AttentionHeadSpecializer

__all__ = [
    "DynamicTrainer",
    "ModelPackager",
    "TrainingManager",
    "apply_peft_to_model",
    "PEFTAdapter",
    "create_multimodal_model",
    "MultimodalGPT2Model",
    "create_distiller",
    "ModelDistiller",
    "DistillationLoss",
    "apply_quantization_to_model",
    "quantize_model_weights",
    "QuantizationController",
    "apply_pruning_to_model",
    "PruningController",
    "MagnitudePruning",
    "StructuredPruning",
    "load_region_specific_model",
    "create_multilingual_tokenizer",
    "RegionSpecificModelLoader",
    "create_curriculum_controller",
    "create_adaptive_sampler",
    "CurriculumLearning",
    "AdaptiveBatchSampler",
    "create_specialized_model",
    "CodeGPT2Model",
    "MathGPT2Model",
    "ScientificGPT2Model",
    "create_efficient_attention",
    "EfficientAttentionController",
    "SparseAttention",
    "LongformerAttention",
    "create_intelligent_initializer",
    "apply_intelligent_initialization",
    "IntelligentInitializer",
    "CosineClusteredEmbeddings",
    "AttentionHeadSpecializer"
]