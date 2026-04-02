# 🚀 Bangkong Quick Start Guide

**Get your first LLM trained in 15 minutes**

---

## Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space
- Windows 10/11, Linux, or macOS

**No GPU required!** Bangkong works on CPU.

---

## Step 1: Installation (3 minutes)

### Option A: Standard Installation

```bash
# Clone the repository
git clone https://github.com/shadowofsorrow/bangkong.git
cd bangkong

# Create virtual environment (recommended)
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install Bangkong
pip install -e .

# Verify installation
python -c "import bangkong; print('✅ Bangkong installed!')"
```

### Option B: With All Features

```bash
# Install with all optional dependencies
pip install -e ".[all]"

# This includes:
# - ONNX export
# - GGUF quantization
# - SafeTensors support
# - Monitoring tools
```

### Troubleshooting Installation

**Error: "No module named 'torch'"**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Error: "Permission denied"**
```bash
# Use --user flag
pip install -e . --user
```

---

## Step 2: Download Sample Data (1 minute)

### Option A: Use Included Sample Data

```bash
# Sample data is already in data/raw/
# Skip to Step 3
```

### Option B: Download Example Dataset

```bash
# Download tiny Shakespeare dataset (1MB)
curl https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/raw/shakespeare.txt

# Or download from Hugging Face
python scripts/download_data.py --dataset tinyshakespeare
```

### Option C: Use Your Own Data

```bash
# Create JSONL file (one sample per line)
echo '{"text": "Your first training sample"}' > data/raw/custom.jsonl
echo '{"text": "Your second training sample"}' >> data/raw/custom.jsonl
echo '{"text": "Add at least 100-1000 samples for good results"}' >> data/raw/custom.jsonl

# Supported formats: .txt, .jsonl, .json, .csv, .md
```

---

## Step 3: Process Data (2 minutes)

```bash
# Process raw data into training format
python scripts/process_data.py \
  --input-dir data/raw \
  --output-dir data/processed

# You should see:
# ✅ Scanned raw data directory
# ✅ Found X files to process
# ✅ Processed data saved to data/processed
```

**What this does:**
- Cleans and normalizes text
- Tokenizes data
- Creates training samples
- Splits into train/validation sets

---

## Step 4: Train Your First Model (10 minutes)

### For CPU (8GB RAM)

```bash
# Small model optimized for CPU
python scripts/train.py \
  --config configs/small_cpu.yaml \
  --data-path data/processed \
  --output-path models/my_first_model

# Expected output:
# - Using device: cpu
# - Pre-Intelligent Initialization enabled
# - Epoch 1/5: Loss decreasing...
# - Checkpoint saved
```

### For GPU (if available)

```bash
# Medium model with GPU
python scripts/train.py \
  --config configs/medium_gpu.yaml \
  --data-path data/processed \
  --output-path models/my_first_model

# Expected output:
# - Using device: cuda
# - Mixed precision enabled
# - Faster training...
```

### Monitor Training

Training will show progress like:
```
Epoch 1/5: 100%|████████████| 100/100 [02:15<00:00, 0.74it/s]
Training Loss: 7.57 | Validation Loss: 7.82
Checkpoint saved to models/my_first_model/checkpoints/epoch_1.pt
```

**Training Time:**
- CPU (Q8400): ~10 minutes for small test dataset
- GPU (RTX 3090): ~2 minutes for same dataset

---

## Step 5: Generate Text (1 minute)

```bash
# Generate text from your trained model
python scripts/generate.py \
  --model-path models/my_first_model \
  --prompt "Once upon a time" \
  --max-length 100 \
  --temperature 0.7

# Output example:
# Prompt: "Once upon a time"
# Generated: "Once upon a time, there was a great kingdom..."
```

### Interactive Mode

```bash
# Chat with your model
python scripts/chat.py --model-path models/my_first_model

# You: Hello!
# Model: Hello! How can I help you today?
# You: Tell me a story
# Model: Once upon a time...
```

---

## 🎯 Next Steps

### 1. Train on Your Custom Data

```bash
# Prepare your dataset
# Minimum 1,000 samples for decent results
# 10,000+ samples for good results

# Process
python scripts/process_data.py \
  --input-dir /path/to/your/data \
  --output-dir data/processed

# Train
python scripts/train.py \
  --config configs/small.yaml \
  --data-path data/processed \
  --output-path models/custom_model \
  --epochs 10
```

### 2. Export Your Model

```bash
# Convert to ONNX (for deployment)
python scripts/convert.py \
  --model-path models/my_first_model \
  --formats onnx \
  --output-path exported_models/

# Convert to GGUF (for llama.cpp)
python scripts/convert.py \
  --model-path models/my_first_model \
  --formats gguf \
  --output-path exported_models/
```

### 3. Deploy as API

```bash
# Start API server
python scripts/serve.py \
  --model-path models/my_first_model \
  --port 8000

# Test API
curl http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_length": 50}'
```

---

## 📋 Configuration Quick Reference

### Small Model (CPU Testing)

```yaml
# configs/small_cpu.yaml
model:
  hidden_size: 768
  num_layers: 12
  num_heads: 12

training:
  batch_size: 2
  gradient_accumulation_steps: 16
  max_epochs: 5

hardware:
  use_gpu: auto
  max_memory_gb: 8
```

**Use when:** Testing, learning, small datasets

---

### Medium Model (GPU Production)

```yaml
# configs/medium_gpu.yaml
model:
  hidden_size: 1024
  num_layers: 24
  num_heads: 16

training:
  batch_size: 8
  gradient_accumulation_steps: 8
  max_epochs: 10

hardware:
  use_gpu: true
  fp16: true
```

**Use when:** Production use, custom applications

---

### Large Model (Research)

```yaml
# configs/large.yaml
model:
  hidden_size: 2048
  num_layers: 32
  num_heads: 32

training:
  batch_size: 16
  gradient_accumulation_steps: 4
  max_epochs: 20

hardware:
  use_gpu: true
  fp16: true
  num_workers: 4
```

**Use when:** Research, large-scale experiments

---

## ❓ Common Questions

### "How much data do I need?"

| Goal | Minimum Samples | Recommended |
|------|----------------|-------------|
| Testing | 100 | 500 |
| Demo | 1,000 | 5,000 |
| Production | 10,000 | 100,000+ |
| Research | 100,000 | 1M+ |

### "How long does training take?"

| Hardware | Small Model | Medium Model |
|----------|-------------|--------------|
| CPU (Q8400) | 10 min | 1-2 hours |
| GPU (RTX 3090) | 2 min | 20 min |
| GPU (A100) | 30 sec | 10 min |

*For 1,000 training samples*

### "Can I resume interrupted training?"

Yes!
```bash
python scripts/train.py \
  --mode resume \
  --checkpoint-path models/my_model/checkpoints/latest.pt
```

### "How do I fine-tune an existing model?"

```bash
python scripts/train.py \
  --mode fine-tune \
  --model-path models/pretrained_model \
  --data-path data/custom_data
```

---

## 🆘 Troubleshooting

### Problem: "CUDA out of memory"

**Solution:**
```yaml
# Reduce batch size
training:
  batch_size: 1  # or 2

# Or use gradient accumulation
training:
  batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch: 16
```

### Problem: "Training is too slow on CPU"

**Solutions:**
1. Use smaller model (configs/tiny.yaml)
2. Reduce sequence length
3. Use fewer epochs for testing
4. Consider cloud GPU (Colab, RunPod, etc.)

### Problem: "Loss is not decreasing"

**Check:**
1. Learning rate (try 1e-4 to 5e-5)
2. Data quality (ensure clean text)
3. Batch size (too small = unstable)
4. Training steps (may need more epochs)

### Problem: "Generated text is gibberish"

**Solutions:**
1. Train longer (more epochs)
2. Lower temperature (0.5-0.7)
3. Check data quality
4. Increase model size

---

## 📚 Learn More

- [Full Documentation](docs/)
- [Configuration Guide](docs/configuration.md)
- [Training Modes](docs/TRAINING_MODES.md)
- [Research Paper](docs/papers/bangkong_paper.md)
- [GitHub Issues](https://github.com/shadowofsorrow/bangkong/issues)
- [GitHub Discussions](https://github.com/shadowofsorrow/bangkong/discussions)

---

## 🎉 You Did It!

You've successfully trained your first LLM with Bangkong!

**Next:**
- Share your model on Hugging Face
- Join our Discord community
- Check out advanced tutorials
- Read the research paper

**Happy Training! 🦍**
