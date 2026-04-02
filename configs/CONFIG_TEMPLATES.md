# Bangkong Configuration Templates

**Ready-to-use configs for different use cases**

---

## 📁 Model Size Configs

### Tiny (Testing/Education)

**File:** `configs/tiny.yaml`

```yaml
# Bangkong Tiny - For quick testing and education
# Training time: ~5 minutes on CPU for small dataset
# VRAM/RAM: 4-6 GB

model:
  name: "bangkong-tiny"
  architecture: "gpt2"
  vocab_size: 50257
  hidden_size: 512
  num_layers: 6
  num_heads: 8
  sequence_length: 256
  dropout: 0.1
  
  # Pre-Intelligent Initialization
  initialization_strategy: "pre_intelligent"
  prior_knowledge: "general"
  preint_cosine_clustering: true
  preint_attention_specialization: true
  preint_cluster_groups:
    reasoning: 0.10
    logic: 0.10
    general: 0.80

training:
  max_epochs: 3
  learning_rate: 1e-4
  batch_size: 4
  gradient_accumulation_steps: 8  # Effective: 32
  warmup_steps: 100
  weight_decay: 0.01
  max_grad_norm: 1.0
  
  # Pre-Intelligent parameters
  preint_reduction_factor: 0.4
  scaling_law_alpha: 0.8

hardware:
  use_gpu: "auto"
  fp16: "auto"
  num_workers: 0
  max_memory_gb: 6
  pin_memory: false

logging:
  log_interval: 10
  save_interval: 100
  enable_tensorboard: false
```

**Best for:**
- ✅ Quick testing
- ✅ Learning/tutorial
- ✅ CI/CD pipelines
- ✅ Proof of concept

**Not for:**
- ❌ Production use
- ❌ High-quality generation

---

### Small (CPU-Friendly)

**File:** `configs/small.yaml`

```yaml
# Bangkong Small - Balanced for CPU training
# Training time: ~2-4 hours on CPU (Q8400)
# RAM: 8 GB minimum

model:
  name: "bangkong-small"
  architecture: "gpt2"
  vocab_size: 50257
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  sequence_length: 512
  dropout: 0.1
  
  # Pre-Intelligent Initialization
  initialization_strategy: "pre_intelligent"
  prior_knowledge: "reasoning"
  preint_cosine_clustering: true
  preint_attention_specialization: true
  preint_cluster_groups:
    reasoning: 0.15
    logic: 0.15
    general: 0.70
  preint_attention_patterns:
    causal: 0.30
    logical: 0.25
    temporal: 0.20
    spatial: 0.15
    general: 0.10

training:
  max_epochs: 5
  learning_rate: 5e-5
  batch_size: 2
  gradient_accumulation_steps: 16  # Effective: 32
  warmup_steps: 500
  weight_decay: 0.01
  max_grad_norm: 1.0
  early_stopping_patience: 3
  
  # Pre-Intelligent parameters
  preint_reduction_factor: 0.4
  scaling_law_alpha: 0.8
  scaling_law_N_ref: 350000000.0
  scaling_law_T_ref: 1000000000.0

hardware:
  use_gpu: "auto"
  fp16: "auto"
  num_workers: 2
  max_memory_gb: 8
  pin_memory: false

logging:
  log_interval: 50
  save_interval: 500
  enable_tensorboard: true
  enable_wandb: false
```

**Best for:**
- ✅ CPU training
- ✅ Limited RAM (8GB)
- ✅ Custom datasets
- ✅ Prototyping

---

### Medium (GPU-Recommended)

**File:** `configs/medium.yaml`

```yaml
# Bangkong Medium - Production-ready
# Training time: ~6-12 hours on GPU (RTX 3090)
# VRAM: 16 GB recommended

model:
  name: "bangkong-medium"
  architecture: "gpt2"
  vocab_size: 50257
  hidden_size: 1024
  num_layers: 24
  num_heads: 16
  sequence_length: 1024
  dropout: 0.1
  
  # Pre-Intelligent Initialization
  initialization_strategy: "pre_intelligent"
  prior_knowledge: "code"
  preint_cosine_clustering: true
  preint_attention_specialization: true
  preint_cluster_groups:
    code: 0.25
    reasoning: 0.15
    logic: 0.15
    general: 0.45
  preint_attention_patterns:
    causal: 0.30
    logical: 0.30
    temporal: 0.15
    spatial: 0.15
    general: 0.10

training:
  max_epochs: 10
  learning_rate: 3e-5
  batch_size: 8
  gradient_accumulation_steps: 8  # Effective: 64
  warmup_steps: 1000
  weight_decay: 0.01
  max_grad_norm: 1.0
  early_stopping_patience: 5
  
  # Pre-Intelligent parameters
  preint_reduction_factor: 0.4
  scaling_law_alpha: 0.8
  
  # Optional: Curriculum learning
  curriculum_type: "sequence_length"
  curriculum_stages: 3

hardware:
  use_gpu: true
  fp16: true
  num_workers: 4
  max_memory_gb: 16
  pin_memory: true

logging:
  log_interval: 20
  save_interval: 200
  enable_tensorboard: true
  enable_wandb: true
```

**Best for:**
- ✅ Production use
- ✅ Code generation
- ✅ Custom applications
- ✅ Fine-tuning base

---

### Large (Research/Enterprise)

**File:** `configs/large.yaml`

```yaml
# Bangkong Large - Research/Enterprise scale
# Training time: ~1-2 days on multi-GPU
# VRAM: 32+ GB, multi-GPU recommended

model:
  name: "bangkong-large"
  architecture: "gpt2"
  vocab_size: 50257
  hidden_size: 2048
  num_layers: 32
  num_heads: 32
  sequence_length: 2048
  dropout: 0.1
  
  # Pre-Intelligent Initialization
  initialization_strategy: "pre_intelligent"
  prior_knowledge: "multi-domain"
  preint_cosine_clustering: true
  preint_attention_specialization: true
  preint_cluster_groups:
    reasoning: 0.15
    math: 0.10
    code: 0.15
    science: 0.10
    general: 0.50
  preint_attention_patterns:
    causal: 0.25
    logical: 0.25
    temporal: 0.20
    spatial: 0.15
    mathematical: 0.10
    general: 0.05

training:
  max_epochs: 20
  learning_rate: 2e-5
  batch_size: 16
  gradient_accumulation_steps: 4  # Effective: 64
  warmup_steps: 2000
  weight_decay: 0.01
  max_grad_norm: 0.5
  early_stopping_patience: 7
  
  # Pre-Intelligent parameters
  preint_reduction_factor: 0.4
  scaling_law_alpha: 0.8
  
  # Advanced training
  curriculum_type: "complexity"
  curriculum_stages: 5
  use_deepspeed: true
  deepspeed_stage: 2

hardware:
  use_gpu: true
  fp16: true
  num_workers: 8
  max_memory_gb: 32
  pin_memory: true
  distributed: true
```

**Best for:**
- ✅ Research
- ✅ Enterprise applications
- ✅ High-quality generation
- ✅ Multi-domain expertise

---

## 🎯 Task-Specific Configs

### Code Generation

**File:** `configs/code_specialist.yaml`

```yaml
model:
  name: "bangkong-code"
  architecture: "gpt2"
  vocab_size: 50257
  hidden_size: 1024
  num_layers: 24
  num_heads: 16
  
  # Specialized for code
  initialization_strategy: "pre_intelligent"
  prior_knowledge: "code"
  preint_cosine_clustering: true
  preint_cluster_groups:
    code: 0.40
    reasoning: 0.20
    logic: 0.20
    general: 0.20
  preint_attention_patterns:
    causal: 0.30
    logical: 0.40
    temporal: 0.15
    spatial: 0.10
    general: 0.05

training:
  max_epochs: 15
  learning_rate: 3e-5
  batch_size: 8
  
  # Code-specific
  data_format: "code"
  syntax_validation: true
```

---

### Math/Reasoning

**File:** `configs/math_specialist.yaml`

```yaml
model:
  name: "bangkong-math"
  architecture: "gpt2"
  hidden_size: 1024
  num_layers: 24
  num_heads: 16
  
  # Specialized for math
  initialization_strategy: "pre_intelligent"
  prior_knowledge: "math"
  preint_cluster_groups:
    math: 0.35
    reasoning: 0.25
    logic: 0.20
    general: 0.20
  preint_attention_patterns:
    logical: 0.40
    causal: 0.20
    temporal: 0.15
    mathematical: 0.20
    general: 0.05

training:
  max_epochs: 20
  learning_rate: 2e-5
  
  # Math-specific
  data_format: "math"
  include_formulas: true
```

---

### Multi-Modal

**File:** `configs/multi_modal.yaml`

```yaml
model:
  name: "bangkong-multimodal"
  architecture: "gpt2"
  hidden_size: 1024
  num_layers: 24
  num_heads: 16
  
  # Multi-domain
  initialization_strategy: "pre_intelligent"
  prior_knowledge: "general"
  preint_cluster_groups:
    text: 0.25
    code: 0.15
    vision: 0.12
    audio: 0.10
    video: 0.10
    science: 0.08
    sensor_3d: 0.08
    geo: 0.07
    container: 0.05

training:
  max_epochs: 15
  batch_size: 8
  
  # Multi-modal specific
  domain_balanced_sampling: true
  domain_weights:
    text: 0.25
    code: 0.15
    vision: 0.12
    audio: 0.10
    video: 0.10
    science: 0.08
    sensor_3d: 0.08
    geo: 0.07
    container: 0.05
```

---

## 🖥️ Hardware-Specific Configs

### Low-End CPU (4GB RAM)

**File:** `configs/hardware/low_end_cpu.yaml`

```yaml
model:
  hidden_size: 512
  num_layers: 6
  num_heads: 8
  sequence_length: 128

training:
  batch_size: 1
  gradient_accumulation_steps: 32  # Effective: 32
  learning_rate: 1e-4

hardware:
  use_gpu: false
  num_workers: 0
  max_memory_gb: 4
  pin_memory: false
  
  # Memory optimizations
  gradient_checkpointing: true
  use_reentrant: false
```

---

### Mid-Range GPU (8GB VRAM)

**File:** `configs/hardware/mid_gpu.yaml`

```yaml
model:
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  sequence_length: 512

training:
  batch_size: 4
  gradient_accumulation_steps: 4  # Effective: 16
  learning_rate: 5e-5

hardware:
  use_gpu: true
  fp16: true
  num_workers: 2
  max_memory_gb: 8
  pin_memory: true
```

---

### High-End GPU (24GB+ VRAM)

**File:** `configs/hardware/high_gpu.yaml`

```yaml
model:
  hidden_size: 2048
  num_layers: 32
  num_heads: 32
  sequence_length: 2048

training:
  batch_size: 16
  gradient_accumulation_steps: 2  # Effective: 32
  learning_rate: 2e-5

hardware:
  use_gpu: true
  fp16: true
  bf16: true  # If supported
  num_workers: 8
  max_memory_gb: 24
  pin_memory: true
  
  # Multi-GPU
  distributed: true
  use_deepspeed: true
  deepspeed_stage: 2
```

---

## 📝 How to Use

### Command Line

```bash
# Use any config
python scripts/train.py \
  --config configs/small.yaml \
  --data-path data/processed \
  --output-path models/my_model
```

### Override Config Values

```bash
# Override specific values
python scripts/train.py \
  --config configs/small.yaml \
  --overrides "training.max_epochs=10" \
  --overrides "training.learning_rate=1e-4"
```

### Create Custom Config

```bash
# Copy template
cp configs/small.yaml configs/my_custom.yaml

# Edit my_custom.yaml
# Then use it
python scripts/train.py --config configs/my_custom.yaml
```

---

## 🔧 Configuration Reference

### Model Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `vocab_size` | Vocabulary size | 50257 (GPT-2) |
| `hidden_size` | Model dimension | 512-4096 |
| `num_layers` | Transformer layers | 6-48 |
| `num_heads` | Attention heads | 8-64 |
| `sequence_length` | Max context | 256-4096 |
| `dropout` | Dropout rate | 0.1-0.3 |

### Training Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `learning_rate` | Initial LR | 1e-4 to 2e-5 |
| `batch_size` | Per-device batch | 1-32 |
| `gradient_accumulation_steps` | Accumulate gradients | 1-32 |
| `warmup_steps` | LR warmup | 100-2000 |
| `max_epochs` | Training epochs | 3-50 |
| `weight_decay` | L2 regularization | 0.01-0.1 |

### Hardware Parameters

| Parameter | Description | Values |
|-----------|-------------|--------|
| `use_gpu` | GPU usage | true/false/auto |
| `fp16` | Mixed precision | true/false/auto |
| `num_workers` | Data loading workers | 0-16 |
| `max_memory_gb` | Memory limit | 4-80 |

---

**Need help?** See [docs/configuration.md](docs/configuration.md) for full reference.
