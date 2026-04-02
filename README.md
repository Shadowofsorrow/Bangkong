# Bangkong LLM Training System

**A production-ready, cloud-native system for training and deploying large language models with Pre-Intelligent Initialization.**

---

## 📄 Research Paper

This project is based on **published research**:

> **Bangkong: Pre-Intelligent LLM Training System for Resources-Efficient Large Language Models**  
> **Author:** Soni Nugraha  
> **DOI:** [10.5281/zenodo.19387331](https://doi.org/10.5281/zenodo.19387331)  
> **Published:** April 2, 2026  
> **License:** [CC-BY 4.0 International](https://creativecommons.org/licenses/by/4.0/)

**Citation:**
```
Nugraha, S. (2026). Bangkong: Pre-Intelligent LLM Training System for 
Resources-Efficient Large Language Model. Zenodo. 
https://doi.org/10.5281/zenodo.19387331
```

**Key Findings:**
- ✅ **40% reduction** in training tokens required
- ✅ **Validated on consumer hardware** (Intel Q8400, 8GB RAM, CPU-only)
- ✅ **$334K-50M savings** per model (scaling law analysis)
- ✅ **Pre-Intelligent Initialization** - models "born to be Einstein"

---

## Features

- **Environment Agnostic**: Works seamlessly across local development and cloud deployment
- **Hardware Adaptive**: Automatically adjusts to available hardware resources
- **Multi-Format Data Processing**: Handles text, images, audio, video, and documents
- **Flexible Model Training**: Supports pre-training and fine-tuning with various architectures
- **Comprehensive Model Packaging**: Converts models to multiple formats with quantization
- **Multiple Deployment Options**: Local, cloud, and hybrid deployment scenarios
- **Monitoring & Evaluation**: Real-time resource monitoring and performance tracking
- **Data Processing Pipeline**: Automated data categorization, cleaning, and preprocessing
- **Pre-Intelligent Initialization**: Models "born to be Einstein" with enhanced reasoning capabilities
- **Scaling Law Validation**: Quantifiable benefits with 30-50% token reduction

## Windows Launchers

For easier use on Windows systems, we provide interactive launcher scripts:

- `bangkong.bat` - Batch file launcher with menu-driven interface
- `bangkong.ps1` - PowerShell launcher with enhanced formatting

See [LAUNCHERS.md](LAUNCHERS.md) for detailed usage instructions.

**Note**: Recent fixes have been applied to improve progress tracking visibility and fix a missing method issue when resuming training from checkpoints. The batch script now uses unbuffered output to ensure proper display of progress bars.

## Data Processing Workflow

The system includes a comprehensive data processing pipeline:

1. **Raw Data Ingestion**: Place raw data in `data/raw` directory
2. **File Categorization**: Automatic categorization by file type (text, code, documents, etc.)
3. **Data Cleaning**: Automated cleaning and preprocessing
4. **Sample Dataset Creation**: Generation of training samples for model training
5. **Processed Data Output**: Cleaned data ready for training in `data/processed` and `data/sample`

## Project Structure

```
bangkong/
├── bangkong/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── schemas.py
│   ├── hardware/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   └── allocator.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   └── processors/
│   │       ├── __init__.py
│   │       ├── base_processor.py
│   │       ├── text_processor.py
│   │       ├── image_processor.py
│   │       ├── audio_processor.py
│   │       └── document_processor.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── packager.py
│   │   └── training_manager.py
│   ├── deployment/
│   │   ├── __init__.py
│   │   └── manager.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── tracker.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── path_manager.py
│   │   └── dynamic_importer.py
│   └── exceptions/
│       ├── __init__.py
│       └── resource_errors.py
├── configs/
│   ├── default.yaml
│   ├── development.yaml
│   ├── production.yaml
│   └── test.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── example_data.jsonl
│   │   └── test_data.jsonl
│   └── external/
├── docs/
│   └── TRAINING_MODES.md
├── notebooks/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── convert.py
│   └── deploy.py
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_hardware.py
│   ├── test_packager.py
│   ├── test_pipeline.py
│   ├── test_trainer.py
│   ├── test_processors.py
│   └── run_tests.py
├── .github/
├── .devcontainer/
├── .dockerignore
├── .gitignore
├── .env.example
├── .env
├── README.md
├── requirements.txt
├── setup.py
├── pyproject.toml
├── Makefile
├── CHANGELOG.md
├── test_system.py
├── test_training_manager.py
└── LICENSE
```

## Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shadowofsorrow/bangkong.git
   cd bangkong
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

### Quick Start

### 1. Prepare Training Data

Create a JSONL file with your training data in the `data/processed` directory. Each line should be a JSON object with a "text" field:

```json
{"text": "This is the first training example."}
{"text": "This is the second training example."}
```

### 2. Configure Your Model

Edit or create a configuration file in the `configs/` directory:

```yaml
model:
  name: "my-awesome-model"
  architecture: "gpt2"
  size: "small"  # tiny, small, medium, large, xlarge
  vocab_size: 50257
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  sequence_length: 1024

training:
  max_epochs: 3
  learning_rate: 5e-5
  batch_size: 8
  gradient_accumulation_steps: 1
  warmup_steps: 500

hardware:
  use_gpu: auto  # auto, true, false
  fp16: auto     # auto, true, false
  num_workers: 0

data:
  preprocessing:
    min_text_length: 50
    max_text_length: 10000
```

### 3. Train a Model

```bash
# Train with a specific config (fresh training)
python scripts/train.py --config configs/development.yaml

# Train with custom data and output paths
python scripts/train.py --config configs/development.yaml --data-path ./data/processed/my_data.jsonl --output-path ./models/my_model

# Train with default config (auto-detects paths)
python scripts/train.py

# Continue training a completed model (add more epochs)
python scripts/train.py --training-mode continue --model-path ./models/my_model

# Resume training from a checkpoint (continue from interruption)
python scripts/train.py --training-mode resume --model-path ./models/my_model --checkpoint-path ./models/my_model/checkpoints/checkpoint_epoch_5.pt

# Fine-tune an existing model on new data
python scripts/train.py --training-mode fine-tune --model-path ./models/my_model

# Interactive training mode (menu-driven)
python scripts/train.py
```

### 4. Package the Model

Convert your trained model to different formats:

```bash
# Convert to all default formats (PyTorch, SafeTensors)
python scripts/convert.py --model-path ./models/my_model

# Convert to specific formats
python scripts/convert.py --model-path ./models/my_model --formats onnx safetensors gguf

# Convert with custom output path
python scripts/convert.py --model-path ./models/my_model --output-path ./converted_models --formats onnx
```

### 5. Evaluate the Model

```bash
# Evaluate with test data
python scripts/evaluate.py --model-path ./models/my_model --data-path ./data/test/test_data.jsonl

# Evaluate with configuration
python scripts/evaluate.py --config configs/development.yaml --model-path ./models/my_model
```

### 6. Deploy the Model

```bash
# Deploy locally
python scripts/deploy.py --model-path ./models/my_model --target local

# Deploy to cloud (if configured)
python scripts/deploy.py --model-path ./models/my_model --target cloud

# Deploy with configuration
python scripts/deploy.py --config configs/production.yaml --model-path ./models/my_model
```

## Pre-Intelligent Initialization

The Bangkong system now includes advanced pre-intelligent initialization features that create models "born to be Einstein" - with enhanced reasoning capabilities, reduced training requirements, and improved efficiency right from initialization.

### Key Features

1. **Cosine-Clustered Embeddings**: Initialize embeddings with semantic neighborhoods based on domain knowledge
2. **Attention Head Specialization**: Specialize attention heads for different reasoning patterns
3. **Scaling Law Validation**: Quantifiable benefits with typically 40% token reduction
4. **Controlled Experimentation Framework**: Systematic validation of approaches

### Benefits

- **30-50% reduction** in required pretraining tokens
- **Billions of tokens saved** at scale (compute/energy efficiency)
- **10-30% faster convergence** to target performance
- **Built-in reasoning capabilities** from initialization
- **Better generalization** from fewer examples

### Configuration

Enable pre-intelligent initialization in your configuration:

```yaml
model:
  initialization_strategy: "pre_intelligent"
  prior_knowledge: "reasoning"  # or "math", "code", "general"
  preint_cosine_clustering: true
  preint_attention_specialization: true

training:
  preint_reduction_factor: 0.4  # 40% token reduction
```

### Validation Tools

```bash
# Run scaling law validation
python scripts/validate_scaling_law.py --config configs/development.yaml

# Run controlled experiments
python scripts/run_pre_intelligent_experiment.py --config configs/development.yaml

# Demo the features
python scripts/demo_pre_intelligent.py
```

For detailed documentation, see [docs/PRE_INTELLIGENT_INITIALIZATION.md](docs/PRE_INTELLIGENT_INITIALIZATION.md).

## Curriculum Learning

The Bangkong system includes an advanced curriculum learning system that progressively increases the difficulty and complexity of training data to improve model performance and convergence speed.

### Key Features

1. **Multiple Curriculum Types**: Sequence length, complexity, and topic-based curricula
2. **Adaptive Scheduling**: Automatically adjusts difficulty based on model performance
3. **Synthetic Reasoning Traces**: Generates domain-specific reasoning problems for pre-intelligent models
4. **Progressive Task Complexity**: Starts with simple tasks and gradually introduces more complex ones

### Configuration

Enable curriculum learning in your configuration:

```yaml
training:
  curriculum_type: "complexity"  # sequence_length, complexity, topic, or none
  curriculum_stages: 5
  stage_samples_threshold: 10000
```

### Usage

```bash
# Train with curriculum learning
python scripts/train.py --config configs/curriculum_learning.yaml

# Generate synthetic curriculum for pre-intelligent training
python scripts/generate_curriculum.py --domain reasoning --output-dir ./curriculum_reasoning
```

For detailed documentation, see [docs/curriculum_learning.md](docs/curriculum_learning.md).

## Dynamic Configuration System

The Bangkong system now features a comprehensive dynamic configuration system that makes the entire system environment-agnostic and fully customizable.

### Key Features

1. **YAML-Based Configuration**: All system parameters are configurable through YAML files
2. **Environment Variables**: Override any configuration value with environment variables
3. **Module-Specific Configurations**: Separate configuration files for different system modules
4. **Runtime Reloading**: Configuration can be reloaded without restarting the system

### Configuration Files

- `configs/default.yaml`: Base configuration for all environments
- `configs/development.yaml`: Development environment configuration
- `configs/production.yaml`: Production environment configuration
- `configs/testing.yaml`: Testing environment configuration
- `bangkong/models/config.yaml`: Models-specific configuration
- `bangkong/pre_intelligent/curriculum/config.yaml`: Curriculum learning configuration

For detailed documentation, see [docs/configuration.md](docs/configuration.md) and [docs/models.md](docs/models.md).

## Development

### Setting Up Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/bangkong.git
   cd bangkong
   ```

3. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### Code Style

We follow these coding standards:
- PEP 8 for Python code
- Black for code formatting
- Flake8 for linting
- Type hints for all functions and classes

Run these commands before committing:
```bash
black .
flake8 .
mypy .
```

### Testing

The project includes comprehensive tests:

```bash
# Run all tests
python -m pytest tests/

# Run specific test files
python -m pytest tests/test_config.py

# Run tests with verbose output
python -m pytest -v tests/

# Run all tests with the test runner
python tests/run_tests.py
```

## Contributing

We welcome contributions to the Bangkong LLM Training System!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Run tests and linting
7. Commit and push
8. Create a pull request

Please follow our Code of Conduct in all interactions.

---

## 📚 Research & Citation

This project implements research from the paper:

**Bangkong: Pre-Intelligent LLM Training System for Resources-Efficient Large Language Model**  
*Author: Soni Nugraha*  
**DOI:** [10.5281/zenodo.19387331](https://doi.org/10.5281/zenodo.19387331)  
*Published: April 2, 2026*  
*License: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)*

### Cite This Work

**BibTeX:**
```bibtex
@misc{nugraha2026bangkong,
  author = {Nugraha, Soni},
  title = {Bangkong: Pre-Intelligent LLM Training System for Resources-Efficient Large Language Model},
  doi = {10.5281/zenodo.19387331},
  url = {https://doi.org/10.5281/zenodo.19387331},
  publisher = {Zenodo},
  year = {2026},
  month = {April},
  license = {CC-BY-4.0}
}
```

### Key Research Contributions

- **Pre-Intelligent Initialization**: Structured knowledge embedding at model creation
- **Cosine-Clustered Embeddings**: Semantic neighborhoods in embedding space
- **Attention Head Specialization**: Pattern-specific attention initialization
- **40% Token Reduction**: Validated efficiency improvement
- **Consumer Hardware Training**: Full pipeline on 8GB RAM CPU

---

## 💼 Available for Hire & Consulting

**Looking for custom LLM solutions? I can help!**

I offer professional services for:

### Services

- **Custom LLM Training** - Train models on your business data
- **Domain Fine-tuning** - Adapt existing models to your domain
- **Low-Resource Optimization** - Optimize models for limited hardware
- **Deployment & Setup** - Production deployment of LLM systems
- **Training Workshops** - Hands-on LLM training for your team
- **Consulting** - Expert advice on LLM strategy

**💰 Typical project range: $5K-50K** (depending on scope and requirements)

### ☕ Support This Project

If you find this project useful but aren't ready for a full consulting engagement, you can support development:

[![Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/bilbobangkong)

Every coffee helps keep the Q8400 running! 😂

### Why Work With Me?

✅ **Published Research** - DOI: 10.5281/zenodo.19387331  
✅ **Proven Efficiency** - 40% reduction in training tokens  
✅ **Resource-Constrained Expert** - Trained on 8GB RAM (your costs: 90% less)  
✅ **Production-Ready** - Full pipeline from data to deployment  
✅ **Flexible Engagement** - From quick consulting to full projects  

### Get In Touch

📧 **Email:** bilbobangkong@gmail.com  
💼 **LinkedIn:** www.linkedin.com/in/soni-nugraha-467a1766  
📄 **GitHub:** [This Repository]  
☕ **Ko-fi:** https://ko-fi.com/bilbobangkong  

**Response Time:** Usually within 24-48 hours

### What to Expect

1. **Initial Contact** - Email me about your project needs
2. **Discovery Call** - 30-min call to understand requirements
3. **Custom Proposal** - Quote based on your specific scope
4. **50% Deposit** - To begin work
5. **Delivery** - Model/training completed
6. **Final Payment** - Before handoff

**Flexible payment plans available for startups and researchers.**

---

## ⚠️ Hardware Notes

**Tested On:** Intel Core 2 Quad Q8400 (2008), 8GB RAM, CPU-only

**What Works:**
- ✅ Training with `configs/8gb_ram_test.yaml`
- ✅ Ultra-small models (2 layers, 64 hidden)
- ✅ Data processing (text, code files)
- ✅ CPU-only training

**What Doesn't Work on 8GB RAM:**
- ❌ Default configs (will OOM - I learned this the hard way! 😂)
- ❌ Large models (768+ hidden)
- ❌ Long sequences (512+ tokens)
- ❌ Batch size > 1

**For Production Use:**
- Recommend 16GB+ RAM for larger models
- GPU acceleration supported (CUDA 11.8+)
- Cloud deployment options available

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.