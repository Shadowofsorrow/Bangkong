"""
Configuration schemas for Bangkong LLM Training System
"""

from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Union
from enum import Enum


class ModelSize(str, Enum):
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"


class HardwareBool(str, Enum):
    AUTO = "auto"
    TRUE = "true"
    FALSE = "false"


class ModalityType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


class ModelConfig(BaseModel):
    name: str = "bangkong-llm"
    architecture: str = "gpt2"
    size: ModelSize = ModelSize.SMALL
    modality: ModalityType = ModalityType.TEXT
    cross_modalities: List[str] = ["text"]
    domain: str = "general"
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    sequence_length: int = 1024
    max_sequence_length: int = 2000000  # Support for up to 2M tokens
    attention_block_size: int = 64
    attention_window_size: int = 512
    num_global_blocks: int = 2
    attention_dropout: float = 0.1
    image_size: Optional[int] = None
    audio_sample_rate: Optional[int] = None
    video_fps: Optional[int] = None
    primary_language: str = "en"
    supported_languages: List[str] = ["en"]
    region: str = "global"
    # New fields for intelligent initialization
    pretrained_from: Optional[str] = None
    initialization_strategy: str = "random"  # random, pretrained, distilled, structured, pre_intelligent, xavier, kaiming
    prior_knowledge: Optional[str] = None  # reasoning, math, code, etc.
    
    # Pre-intelligent initialization parameters
    preint_cosine_clustering: bool = True
    preint_attention_specialization: bool = True
    preint_factorial_scaling: bool = False  # Advanced feature
    preint_cluster_groups: Optional[Dict[str, float]] = None
    preint_attention_patterns: Optional[Dict[str, float]] = None


class TrainingConfig(BaseModel):
    max_epochs: int = 3
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    batch_size: Union[int, str] = "auto"  # Can be int or "auto"
    early_stopping_patience: int = 3
    progress_leave: bool = False
    progress_ncols: Optional[int] = None
    epoch_progress_leave: bool = True
    epoch_progress_ncols: Optional[int] = None
    peft_type: str = "none"  # Parameter-efficient fine-tuning type: none, lora
    lora_rank: int = 8
    lora_alpha: int = 1
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5
    pruning_type: str = "none"  # Pruning type: none, magnitude, structured
    sparsity_ratio: float = 0.0  # Ratio of weights/neurons to prune (0.0 to 1.0)
    curriculum_type: str = "none"  # Curriculum type: none, sequence_length, complexity, topic
    curriculum_stages: int = 1
    stage_samples_threshold: int = 10000
    
    # Scaling law validation parameters
    preint_reduction_factor: float = 0.4
    scaling_law_alpha: float = 0.8
    scaling_law_N_ref: float = 0.35e9
    scaling_law_T_ref: float = 1.0e9
    scaling_gpu_hours_ref: float = 1000.0
    
    # Experimentation parameters
    experiment_tracking_enabled: bool = False
    ablation_strategies: List[str] = ["random", "xavier", "kaiming", "pre_intelligent"]

    @validator("batch_size")
    def validate_batch_size(cls, v):
        if isinstance(v, str) and v != "auto":
            raise ValueError("batch_size must be an integer or 'auto'")
        return v


class DataPreprocessingConfig(BaseModel):
    min_text_length: int = 50
    max_text_length: int = 10000
    deduplicate: bool = True
    filter_low_quality: bool = True


class DataConfig(BaseModel):
    preprocessing: DataPreprocessingConfig = DataPreprocessingConfig()
    paths: Dict[str, str] = {
        "raw": "./data/raw",
        "processed": "./data/processed",
        "tokenized": "./data/tokenized",
        "external": "./data/external"
    }


class EvaluationConfig(BaseModel):
    metrics: List[str] = ["perplexity", "accuracy"]
    validation_split: float = 0.1


class HardwareConfig(BaseModel):
    use_gpu: HardwareBool = HardwareBool.AUTO
    use_tpu: HardwareBool = HardwareBool.AUTO
    fp16: HardwareBool = HardwareBool.AUTO
    num_workers: Union[int, str] = "auto"  # Can be int or "auto"
    max_memory_gb: Union[float, str] = "auto"  # Can be float or "auto"
    batch_size: Union[int, str] = "auto"  # Can be int or "auto"

    @validator("num_workers", "max_memory_gb", "batch_size")
    def validate_auto_values(cls, v):
        if isinstance(v, str):
            if v == "auto":
                return v
            # Try to convert string numbers to int/float
            try:
                if '.' in v:
                    return float(v)
                else:
                    return int(v)
            except ValueError:
                raise ValueError("Value must be a number or 'auto'")
        return v


class MonitoringBackend(str, Enum):
    NONE = "none"
    WANDB = "wandb"
    MLFLOW = "mlflow"
    TENSORBOARD = "tensorboard"


class MonitoringConfig(BaseModel):
    backend: MonitoringBackend = MonitoringBackend.NONE
    log_level: str = "INFO"
    log_file: str = "./logs/bangkong.log"


class PackagingConfig(BaseModel):
    default_formats: List[str] = ["pytorch"]
    supported_formats: List[str] = ["pytorch", "onnx", "safetensors"]
    quantization: Dict[str, Union[str, List[str]]] = {
        "default_precision": "none",
        "supported_precisions": ["none", "int8", "int4"]
    }


class DeploymentConfig(BaseModel):
    default_target: str = "local"
    supported_targets: List[str] = ["local", "cloud", "hybrid"]
    api: Dict[str, Union[str, int]] = {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1
    }


class PathsConfig(BaseModel):
    models: str = "./models"
    logs: str = "./logs"
    checkpoints: str = "./checkpoints"
    outputs: str = "./outputs"


class BangkongConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    hardware: HardwareConfig = HardwareConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    packaging: PackagingConfig = PackagingConfig()
    deployment: DeploymentConfig = DeploymentConfig()
    paths: PathsConfig = PathsConfig()