# Bangkong Configuration Guide

## Overview

The Bangkong LLM Training System uses YAML configuration files to control all aspects of the system behavior. This guide explains how to configure the system for different environments and use cases.

## Configuration Files

The system supports multiple configuration files for different environments:

- `configs/default.yaml`: Base configuration for all environments
- `configs/development.yaml`: Development environment configuration
- `configs/production.yaml`: Production environment configuration
- `configs/testing.yaml`: Testing environment configuration
- `bangkong/models/config.yaml`: Models-specific configuration for embeddings, attention, and other model components
- `bangkong/pre_intelligent/curriculum/config.yaml`: Curriculum learning configuration for synthetic reasoning traces

## Module-Specific Configuration

### Models Configuration
The models module has its own configuration system for detailed control over model components:

```yaml
# bangkong/models/config.yaml
models:
  cosine_clustered_embeddings:
    factorial_cap: 11
    default_prototypes_per_group: 8
    # ... other parameters
```

### Curriculum Learning Configuration
The curriculum learning system has extensive configuration options:

```yaml
# bangkong/pre_intelligent/curriculum/config.yaml
curriculum:
  default_num_samples: 1000
  task_distributions:
    default:
      arithmetic_chain: 0.25
      logic_chain: 0.20
      # ... other task distributions
```

## Configuration Loading

The system loads configuration in the following order:

1. Base configuration (`configs/default.yaml`)
2. Environment-specific configuration (if specified)
3. Environment variable overrides

### Environment Variables

All configuration values can be overridden using environment variables. The environment variable name is derived from the configuration path by:

1. Converting to uppercase
2. Replacing dots with underscores
3. Prefixing with `BANGKONG_`

For example:
- `model.size` becomes `BANGKONG_MODEL_SIZE`
- `training.learning_rate` becomes `BANGKONG_TRAINING_LEARNING_RATE`

## Configuration Sections

### Model Configuration

```yaml
model:
  name: "bangkong-llm"
  architecture: "gpt2"
  size: "small"  # tiny, small, medium, large, xlarge
  vocab_size: 50257
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  sequence_length: 1024
  # Intelligent initialization options
  initialization_strategy: "random"  # random, pretrained, structured, xavier, kaiming
  prior_knowledge: null  # reasoning, math, code (used with structured initialization)
  pretrained_from: null  # model name or path for pretrained initialization
```

For detailed information about intelligent initialization strategies, see [Intelligent Initialization Documentation](intelligent_initialization.md).

### Training Configuration

```yaml
training:
  max_epochs: 3
  learning_rate: 5e-5
  warmup_steps: 500
  save_steps: 1000
  logging_steps: 100
  gradient_accumulation_steps: 1
  weight_decay: 0.01
  adam_epsilon: 1e-8
  max_grad_norm: 1.0
  batch_size: "auto"  # Can be integer or "auto"
```

### Data Configuration

```yaml
data:
  preprocessing:
    min_text_length: 50
    max_text_length: 10000
    deduplicate: true
    filter_low_quality: true
  paths:
    raw: "${BANGKONG_DATA_PATH:-./data/raw}"
    processed: "${BANGKONG_PROCESSED_PATH:-./data/processed}"
    tokenized: "${BANGKONG_TOKENIZED_PATH:-./data/tokenized}"
    external: "${BANGKONG_EXTERNAL_PATH:-./data/external}"
```

### Hardware Configuration

```yaml
hardware:
  use_gpu: "auto"  # auto, true, false
  use_tpu: "auto"  # auto, true, false
  fp16: "auto"     # auto, true, false
  num_workers: "auto"  # auto or specific number
  max_memory_gb: "auto"  # auto or specific value
  batch_size: "auto"  # auto or specific value
```

### Monitoring Configuration

```yaml
monitoring:
  backend: "none"  # none, wandb, mlflow
  log_level: "INFO"
  log_file: "${BANGKONG_LOG_PATH:-./logs/bangkong.log}"
```

### Packaging Configuration

```yaml
packaging:
  default_formats:
    - "pytorch"
  supported_formats:
    - "pytorch"
    - "onnx"
    - "safetensors"
  quantization:
    default_precision: "none"  # none, int8, int4
    supported_precisions:
      - "none"
      - "int8"
      - "int4"
```

### Deployment Configuration

```yaml
deployment:
  default_target: "local"
  supported_targets:
    - "local"
    - "cloud"
    - "hybrid"
  api:
    host: "0.0.0.0"
    port: 8000
    workers: 1
```

### Paths Configuration

```yaml
paths:
  models: "${BANGKONG_MODELS_PATH:-./models}"
  logs: "${BANGKONG_LOGS_PATH:-./logs}"
  checkpoints: "${BANGKONG_CHECKPOINTS_PATH:-./checkpoints}"
  outputs: "${BANGKONG_OUTPUTS_PATH:-./outputs}"
```

## Environment-Specific Configurations

### Development Configuration

The development configuration is optimized for local development:

```yaml
defaults:
  - default

model:
  size: "tiny"  # Use smallest model for development
  sequence_length: 256

training:
  max_epochs: 1
  batch_size: "auto"
  gradient_accumulation_steps: 32

hardware:
  use_gpu: "auto"
  fp16: "auto"
  num_workers: "auto"
  max_memory_gb: 8

monitoring:
  backend: "none"
  log_level: "DEBUG"

data:
  preprocessing:
    max_text_length: 1000  # Smaller for faster processing
```

### Production Configuration

The production configuration is optimized for performance:

```yaml
defaults:
  - default

model:
  size: "large"
  sequence_length: 1024

training:
  max_epochs: 3
  batch_size: "auto"
  gradient_accumulation_steps: 1

hardware:
  use_gpu: "auto"
  fp16: "auto"
  num_workers: "auto"
  max_memory_gb: "auto"

monitoring:
  backend: "wandb"
  log_level: "INFO"

packaging:
  default_formats:
    - "pytorch"
    - "onnx"
    - "safetensors"
  quantization:
    default_precision: "int8"
```

### Testing Configuration

The testing configuration is optimized for fast execution:

```yaml
defaults:
  - default

model:
  size: "tiny"  # Use smallest model for testing
  sequence_length: 128

training:
  max_epochs: 1
  batch_size: 1
  gradient_accumulation_steps: 1

hardware:
  use_gpu: false
  fp16: false
  num_workers: 0
  max_memory_gb: 2

monitoring:
  backend: "none"
  log_level: "DEBUG"

data:
  preprocessing:
    max_text_length: 500  # Smaller for faster testing
```

## Configuration Validation

The system uses Pydantic to validate configuration files. Invalid configurations will raise validation errors with descriptive messages.

## Best Practices

1. **Use Environment Variables for Sensitive Data**: Never hardcode sensitive information in configuration files
2. **Use "auto" Values**: Let the system automatically determine optimal values when possible
3. **Environment-Specific Configurations**: Use different configurations for development, testing, and production
4. **Version Control**: Keep configuration files in version control, but exclude environment-specific values
5. **Documentation**: Document custom configuration values and their purpose