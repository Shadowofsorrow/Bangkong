# Bangkong LLM Training System

**A training system for causal language models with Pre-Intelligent Initialization — enabling resource-efficient training on constrained hardware.**

---

## Overview

Bangkong is an LLM training system that introduces **Pre-Intelligent Initialization** — a method for embedding structured knowledge into model weights at creation time, before training begins. Models initialized with domain-aware structured priors require fewer training tokens to reach target performance.

The system is designed to run efficiently on constrained hardware, validated on consumer-grade CPUs with limited memory.

**Key Results:**
- Validated on Intel Core 2 Quad Q8400 (2008), 8 GB RAM, CPU-only
- Published research: [DOI 10.5281/zenodo.19387331](https://doi.org/10.5281/zenodo.19387331) (CC-BY 4.0)

---

## Research

This project implements research from:

> **Bangkong: Pre-Intelligent LLM Training System for Resource-Efficient Large Language Model**
> Author: Soni Nugraha
> DOI: [10.5281/zenodo.19387331](https://doi.org/10.5281/zenodo.19387331)
> Published: April 2, 2026
> License: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)

**Citation:**
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

---

## Architecture

The system consists of three layers:

### 1. Base Model — GPT-2 Transformer
Standard causal language model architecture. The system supports GPT-2, GPT-Neo, GPT-J, and compatible Hugging Face models.

### 2. Pre-Intelligent Initialization
Components that initialize model weights with structured knowledge before training:

| Component | What It Does |
|-----------|-------------|
| Cosine-Clustered Embeddings | Initializes token embeddings with domain-aware prototype grouping |
| Attention Head Specialization | Creates pattern-specific attention bias tensors for different reasoning patterns |
| Hierarchical Memory | Three-tier memory system (scratchpad, context, semantic) for persistent knowledge |
| Meta-Learning Priors | Uses MAML/Reptile to learn initialization weights for fast task adaptation |
| Energy-Based Consistency | Validates and regularizes hidden state consistency during forward pass |

### 3. Training System
Full training pipeline with data processing, curriculum learning, model packaging, and evaluation.

---

## Quick Start

### Prerequisites
- Python 3.8+
- 8 GB RAM minimum (tested on Intel Core 2 Quad Q8400, 2008)

### Installation
```bash
git clone https://github.com/shadowofsorrow/bangkong.git
cd bangkong
pip install -r requirements.txt
```

### Train a Model
```bash
# Prepare training data (JSONL with "text" field in data/processed/)
# Configure model in configs/development.yaml

# Train
python scripts/train.py --config configs/development.yaml

# Or use interactive mode
python scripts/train.py
```

### Configuration
Configuration files are in `configs/`. Key options:

```yaml
model:
  architecture: "gpt2"
  hidden_size: 768
  num_layers: 12
  initialization_strategy: "pre_intelligent"
  prior_knowledge: "reasoning"

training:
  max_epochs: 3
  batch_size: 1
  learning_rate: 5e-5

hardware:
  use_gpu: auto
  fp16: auto
```

See `configs/development.yaml` for a working example.

---

## Pre-Intelligent Initialization Details

### How It Works

1. **Cosine-Clustered Embeddings**: Token embeddings are grouped by domain (math, code, reasoning, general) and initialized with prototype vectors on the unit sphere. Tokens in the same domain start closer together in embedding space.

2. **Attention Head Specialization**: Fixed bias tensors are created for different reasoning patterns (causal, sequential, numerical, etc.) and applied to attention outputs via forward hooks. Each head gets a pattern-specific bias at initialization.

3. **Hierarchical Memory**: A three-tier differentiable memory system is attached to the model:
   - **Scratchpad** (64 slots): Immediate computation context
   - **Context** (128 slots): Session/topic information
   - **Semantic** (256 slots): Persistent knowledge

4. **Meta-Learning Priors**: MAML/Reptile is used to learn initialization weights that enable fast adaptation to new tasks. The prior generator produces LoRA adapter weights from knowledge concept embeddings.

### Why It Helps

By embedding structured knowledge at creation time, the model does not start from random weights. It begins with:
- Semantic neighborhoods in embedding space (tokens from the same domain are closer)
- Pattern-specific attention biases (heads pre-configured for different reasoning types)
- Persistent memory slots ready to store and retrieve knowledge
- Meta-learned initialization weights for fast task adaptation

This reduces the number of training tokens needed to reach target performance — the paper reports ~40% reduction on validated benchmarks.

---

## Hardware Notes

**Tested On**: Intel Core 2 Quad Q8400 (2008), 8 GB RAM, CPU-only

| Config | Works | Notes |
|--------|-------|-------|
| `configs/development.yaml` | Yes | Small model, batch size 1 |
| Default production configs | No | Will OOM on 8 GB RAM |

**Requirements for Larger Models**:
- 16 GB+ RAM recommended for medium models
- GPU (CUDA 11.8+) supported and recommended for production
- Multi-GPU distributed training supported

---

## Project Structure

```
bangkong/
├── bangkong/                 # Core Python package
│   ├── config/               # Configuration schemas and loaders
│   ├── data/                 # Data processing pipeline
│   ├── hardware/             # Hardware detection and resource allocation
│   ├── models/               # Model training, packaging, and specialized modules
│   ├── pre_intelligent/      # Core innovation — initialization components
│   │   ├── curriculum/       # Curriculum learning
│   │   ├── energy_layer/     # Energy-based consistency
│   │   ├── hypernetwork/     # Prior knowledge generation
│   │   ├── memory/           # Hierarchical memory system
│   │   ├── meta_learning/    # MAML/Reptile
│   │   └── reasoning_organs/ # Specialized reasoning heads
│   └── utils/                # Utilities
├── configs/                  # YAML configurations
├── docs/                     # Documentation
├── examples/                 # Demo scripts
├── scripts/                  # CLI tools (train, evaluate, convert, etc.)
└── tests/                    # Test suite
```

---

## Development

### Running Tests
```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Code Style
This project follows PEP 8. Before contributing:
```bash
black .
flake8 .
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

---

## Acknowledgments

This project builds upon:
- **GPT-2** (OpenAI, 2019) — [github.com/openai/gpt-2](https://github.com/openai/gpt-2)
- **Attention Is All You Need** (Vaswani et al., 2017) — [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- **Hugging Face Transformers** — [github.com/huggingface/transformers](https://github.com/huggingface/transformers)

Bangkong is an **enhancement framework** for GPT-2, not a novel architecture. The contribution is the Pre-Intelligent Initialization method and the training system.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
