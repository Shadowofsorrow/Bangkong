# Bangkong API Reference

## Overview

This document provides detailed API reference documentation for the Bangkong LLM Training System. The system exposes both a REST API for deployment scenarios and a comprehensive Python API for programmatic use.

## Python API

### Configuration Module

#### ConfigLoader
```python
from bangkong.config.loader import ConfigLoader

# Load configuration from file
config_loader = ConfigLoader("configs/development.yaml")
config = config_loader.config

# Get specific configuration values
model_size = config_loader.get("model.size", "small")
learning_rate = config_loader.get("training.learning_rate", 5e-5)
```

#### Methods
- `ConfigLoader(config_path: Optional[str] = None)` - Initialize configuration loader
- `get(key_path: str, default: Any = None)` - Get configuration value using dot notation
- `reload()` - Reload configuration from file

### Hardware Module

#### HardwareDetector
```python
from bangkong.hardware.detector import HardwareDetector

# Get available system resources
resources = HardwareDetector.get_available_resources()
print(resources['cpu'])
print(resources['gpu'])

# Check if GPU is available
gpu_available = HardwareDetector.is_gpu_available()
```

#### Methods
- `get_system_info()` - Get system information
- `get_cpu_info()` - Get CPU information
- `get_gpu_info()` - Get GPU information
- `get_available_resources()` - Get all available resources
- `is_gpu_available()` - Check if GPU is available
- `is_tpu_available()` - Check if TPU is available

### Models Module

#### CosineClusteredEmbeddings
```python
from bangkong.models.cosine_clustered_embeddings import CosineClusteredEmbeddings
from bangkong.config.loader import ConfigLoader

# Initialize with configuration
config_loader = ConfigLoader("configs/development.yaml")
embeddings = CosineClusteredEmbeddings(config_loader.config)

# Build cosine-clustered embeddings
groups = [('reasoning', 150), ('logic', 150), ('general', 700)]
embeddings_tensor, metadata = embeddings.build_cosine_clustered_embeddings(groups)
```

#### Methods
- `CosineClusteredEmbeddings(config)` - Initialize embeddings generator
- `build_cosine_clustered_embeddings(groups, prototypes_per_group=8)` - Build embeddings with clustering
- `initialize_embeddings(embedding_layer)` - Initialize embedding layer

#### AttentionHeadSpecializer
```python
from bangkong.models.attention_specialization import AttentionHeadSpecializer

# Initialize with configuration
specializer = AttentionHeadSpecializer(config_loader.config)

# Specialize attention heads in model
specialized_model = specializer.specialize_attention_heads(model)
```

#### Methods
- `AttentionHeadSpecializer(config)` - Initialize attention specializer
- `specialize_attention_heads(model)` - Specialize attention heads in model

### Curriculum Learning Module

#### ReasoningTraceGenerator
```python
from bangkong.pre_intelligent.curriculum.reasoning_curriculum import ReasoningTraceGenerator

# Initialize generator
generator = ReasoningTraceGenerator(num_samples=1000, difficulty_range=(1, 5))

# Generate curriculum
curriculum_files = generator.generate_curriculum("curriculum_output", num_stages=5)
```

#### Methods
- `ReasoningTraceGenerator(num_samples=1000, difficulty_range=(1, 5))` - Initialize generator
- `generate_traces(output_file, curriculum_scheduler=None)` - Generate reasoning traces
- `generate_curriculum(base_output_dir, num_stages=5)` - Generate curriculum with stages

#### CurriculumScheduler
```python
from bangkong.pre_intelligent.curriculum.reasoning_curriculum import CurriculumScheduler

# Initialize scheduler
scheduler = CurriculumScheduler(
    initial_difficulty=1,
    max_difficulty=5,
    difficulty_increment=0.1,
    competence_threshold=0.8
)

# Update performance and adjust difficulty
scheduler.update_performance(0.85)
current_difficulty = scheduler.get_current_difficulty()
```

#### Methods
- `CurriculumScheduler(initial_difficulty=1, max_difficulty=5, difficulty_increment=0.1, competence_threshold=0.8)` - Initialize scheduler
- `update_performance(accuracy)` - Update performance history and adjust difficulty
- `get_current_difficulty()` - Get current difficulty level
- `get_task_weights()` - Get task weights based on current curriculum stage

### Trainer Module

#### DynamicTrainer
```python
from bangkong.models.trainer import DynamicTrainer

# Initialize trainer
trainer = DynamicTrainer(model, config_loader.config)

# Train model
trainer.train(train_dataloader, val_dataloader)

# Save checkpoint
trainer.save_checkpoint("checkpoint.pt")

# Load checkpoint
trainer.load_checkpoint("checkpoint.pt")
```

#### Methods
- `DynamicTrainer(model, config)` - Initialize dynamic trainer
- `train(train_dataloader, val_dataloader=None, start_epoch=None, additional_epochs=None)` - Train the model
- `train_step(batch)` - Perform a single training step
- `validate(dataloader)` - Validate the model
- `save_checkpoint(path)` - Save a training checkpoint
- `load_checkpoint(path)` - Load a training checkpoint

## REST API

For detailed REST API documentation, see the [OpenAPI Specification](api/openapi.yaml).

### Endpoints

#### /health
Check if the API server is running.

#### /generate
Generate text based on a prompt.

#### /complete
Complete code based on a prefix.

#### /classify
Classify text into categories.

#### /embed
Generate embeddings for text.

#### /train
Start model training with specified configuration.

#### /train/status
Get the status of a training job.

#### /curriculum/generate
Generate synthetic curriculum for pre-intelligent training.

#### /models/initialize
Initialize a model with pre-intelligent features.

## Best Practices

### Configuration Management
1. Use environment variables for sensitive data
2. Keep configuration files in version control
3. Use different configurations for different environments
4. Document custom configuration values

### Hardware Detection
1. Always check hardware availability before initialization
2. Use automatic resource allocation when possible
3. Handle hardware constraints gracefully

### Model Training
1. Monitor training progress and performance
2. Use appropriate initialization strategies for your domain
3. Implement proper error handling and checkpointing
4. Validate models with appropriate test datasets

## Error Handling

The API provides consistent error handling across all modules:

```python
try:
    # API call
    result = some_api_call()
except ValueError as e:
    # Handle value errors
    print(f"Invalid value: {e}")
except FileNotFoundError as e:
    # Handle file not found errors
    print(f"File not found: {e}")
except Exception as e:
    # Handle other errors
    print(f"An error occurred: {e}")
```

## Performance Considerations

### Memory Management
1. Use gradient checkpointing for long sequences
2. Implement proper batch sizing based on available memory
3. Use mixed precision training when supported

### Parallel Processing
1. Utilize multiple CPU cores for data processing
2. Leverage GPU acceleration when available
3. Implement proper multiprocessing for data loading

## Security

### Data Protection
1. Never hardcode sensitive information in configuration files
2. Use environment variables for API keys and secrets
3. Implement proper access controls for sensitive operations

### Input Validation
1. Validate all input data before processing
2. Sanitize user inputs to prevent injection attacks
3. Implement rate limiting for API endpoints

## Troubleshooting

### Common Issues
1. **Configuration Errors**: Check configuration file syntax and values
2. **Hardware Compatibility**: Verify hardware requirements and availability
3. **Memory Constraints**: Reduce batch size or model complexity
4. **Training Divergence**: Adjust learning rate or optimizer settings

### Debugging Tips
1. Enable verbose logging for detailed error information
2. Use debugging tools to inspect model state during training
3. Monitor system resources during intensive operations
4. Validate data quality before training

## Future Enhancements

### Short-term
1. Enhanced API documentation with interactive examples
2. Additional Python API utilities for common tasks
3. Improved error handling and recovery mechanisms

### Long-term
1. GraphQL API for more flexible data querying
2. gRPC support for high-performance communication
3. Advanced monitoring and observability features