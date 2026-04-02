"""
Bangkong LLM Training System

A production-ready, cloud-native system for training and deploying large language models
with support for multiple formats, quantization, and task-specific adapters.
"""

__version__ = "0.1.0"
__author__ = "Bangkong AI Team"
__email__ = "ai@bangkong.com"

# Import main components for easy access
from .config.loader import ConfigLoader
from .hardware.detector import HardwareDetector
from .hardware.allocator import ResourceAllocator
from .models.cosine_clustered_embeddings import CosineClusteredEmbeddings
from .models.attention_specialization import AttentionHeadSpecializer
from .validation.scaling_law_validator import ScalingLawValidator

# Define what gets imported with "from bangkong import *"
__all__ = [
    "ConfigLoader",
    "HardwareDetector",
    "ResourceAllocator",
    "CosineClusteredEmbeddings",
    "AttentionHeadSpecializer",
    "ScalingLawValidator",
]