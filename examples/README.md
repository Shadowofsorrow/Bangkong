# Bangkong Examples

This directory contains example scripts demonstrating various features of the Bangkong LLM Training System.

## Pre-Intelligent Small Model Example

The `pre_intelligent_small_model_example.py` script demonstrates how to use the pre-intelligent small model configuration.

### Running the Example

```bash
python examples/pre_intelligent_small_model_example.py
```

This will display information about the pre-intelligent small model configuration, including:
- Model specifications
- Pre-intelligent features
- Domain groups and attention patterns
- Training configuration
- Hardware requirements
- Expected benefits

### Using the Configuration

To use the pre-intelligent small model configuration for actual training:

```bash
bangkong-train --config configs/pre_intelligent_small_model.yaml
```

### Key Features Demonstrated

1. **Cosine-Clustered Embeddings**
   - Domain-focused token grouping
   - Semantic neighborhood creation

2. **Attention Head Specialization**
   - Pattern-specific attention heads
   - Reasoning-optimized attention mechanisms

3. **Resource Efficiency**
   - Small model size (~50M parameters)
   - CPU-only execution
   - Minimal memory requirements

This example showcases how pre-intelligent initialization can create efficient, reasoning-focused models that are "born to be Einstein" - with enhanced capabilities from initialization rather than through extensive training.