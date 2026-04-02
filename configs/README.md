# Bangkong Configuration Files

This directory contains configuration files for different training scenarios.

## Initialization Strategies

The Bangkong system supports several model initialization strategies:

1. **random** (default): Uses PyTorch's default initialization
2. **pretrained**: Initializes from a pretrained model (requires `pretrained_from` parameter)
3. **distilled**: Initializes using knowledge distillation (typically during training)
4. **structured**: Initializes with domain-specific patterns based on `prior_knowledge`
5. **pre_intelligent**: Initializes with advanced pre-intelligent components including meta-learning priors, reasoning organs, and hierarchical memory
6. **xavier**: Uses Xavier/Glorot uniform initialization
7. **kaiming**: Uses Kaiming/He normal initialization

## Prior Knowledge Domains

When using `structured` or `pre_intelligent` initialization, you can specify one of these domains:

1. **reasoning**: Optimized for logical reasoning tasks
2. **math**: Optimized for mathematical operations and problem solving
3. **code**: Optimized for code understanding and generation

## Example Configurations

- `default.yaml`: Basic configuration with default settings
- `development.yaml`: Settings optimized for development and testing
- `production.yaml`: Settings optimized for production training
- `pretrained_chatbot.yaml`: Configuration for fine-tuning a pretrained model
- `reasoning_llm_enhanced.yaml`: Configuration optimized for reasoning tasks
- `math_llm_enhanced.yaml`: Configuration optimized for mathematical tasks
- `code_llm_enhanced.yaml`: Configuration optimized for code-related tasks
- `pre_intelligent_reasoning.yaml`: Configuration with pre-intelligent initialization for reasoning
- `pre_intelligent_math.yaml`: Configuration with pre-intelligent initialization for math
- `pre_intelligent_code.yaml`: Configuration with pre-intelligent initialization for code

## Usage

To use a specific configuration:

```bash
python scripts/train.py --config configs/reasoning_llm_enhanced.yaml
```