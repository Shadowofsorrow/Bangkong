# 🔧 Bangkong Troubleshooting Guide

**Common issues and solutions**

---

## Installation Issues

### Error: "No module named 'torch'"

**Problem:** PyTorch not installed or not in PATH

**Solution:**
```bash
# Install PyTorch CPU version
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or install with CUDA support (if you have GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

### Error: "Permission denied" during installation

**Problem:** Insufficient permissions

**Solution:**
```bash
# Use --user flag
pip install -e . --user

# Or use sudo (Linux/Mac)
sudo pip install -e .

# Or activate virtual environment first
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -e .
```

---

### Error: "CUDA version mismatch"

**Problem:** PyTorch CUDA version doesn't match system CUDA

**Solution:**
```bash
# Check your CUDA version
nvidia-smi  # Look for CUDA Version

# Install matching PyTorch version
# CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Or use CPU-only (works fine!)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Training Issues

### Problem: Training is extremely slow on CPU

**Symptoms:** < 1 sample/sec, very long epochs

**Solutions:**

1. **Reduce model size:**
```yaml
# configs/tiny.yaml
model:
  hidden_size: 512
  num_layers: 6
  num_heads: 8
  sequence_length: 128
```

2. **Reduce batch size:**
```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 32
```

3. **Enable memory optimizations:**
```yaml
hardware:
  gradient_checkpointing: true
```

4. **Use fewer workers:**
```yaml
hardware:
  num_workers: 0  # or 1
```

---

### Problem: "CUDA out of memory"

**Symptoms:** Training crashes with OOM error

**Solutions:**

1. **Reduce batch size:**
```yaml
training:
  batch_size: 1  # or 2
```

2. **Use gradient accumulation:**
```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 8  # Effective: 32
```

3. **Enable mixed precision:**
```yaml
hardware:
  fp16: true
```

4. **Reduce sequence length:**
```yaml
model:
  sequence_length: 512  # instead of 1024 or 2048
```

5. **Use smaller model:**
```yaml
model:
  hidden_size: 512
  num_layers: 6
```

---

### Problem: Loss is not decreasing

**Symptoms:** Loss stays constant or increases

**Checklist:**

1. **Learning rate too high:**
```yaml
training:
  learning_rate: 1e-5  # Try lower (was 1e-4)
```

2. **Learning rate too low:**
```yaml
training:
  learning_rate: 5e-5  # Try higher (was 1e-6)
```

3. **Batch size too small:**
```yaml
training:
  batch_size: 4  # Increase from 1 or 2
  gradient_accumulation_steps: 8
```

4. **Data quality issues:**
```bash
# Check your data
head data/processed/train.jsonl
# Ensure text is clean and readable
```

5. **Not enough training steps:**
```yaml
training:
  max_epochs: 10  # Increase from 3 or 5
```

6. **Gradient explosion:**
```yaml
training:
  max_grad_norm: 0.5  # Reduce from 1.0
```

---

### Problem: Loss is NaN or Inf

**Symptoms:** Loss becomes NaN or very large

**Solutions:**

1. **Reduce learning rate:**
```yaml
training:
  learning_rate: 1e-5
```

2. **Clip gradients:**
```yaml
training:
  max_grad_norm: 0.5
```

3. **Reduce model size:**
```yaml
model:
  hidden_size: 512
  num_layers: 6
```

4. **Check data for issues:**
```bash
# Look for very long samples
python scripts/check_data.py --input data/processed
```

5. **Disable Pre-Intelligent Init (temporarily):**
```yaml
model:
  initialization_strategy: "random"
```

---

### Problem: Training crashes mid-epoch

**Symptoms:** Training stops unexpectedly

**Solutions:**

1. **Resume from checkpoint:**
```bash
python scripts/train.py \
  --mode resume \
  --checkpoint-path models/my_model/checkpoints/latest.pt
```

2. **Reduce memory usage:**
```yaml
hardware:
  max_memory_gb: 6  # Reduce limit
```

3. **Check disk space:**
```bash
# Ensure enough space for checkpoints
df -h  # Linux/Mac
dir    # Windows
```

4. **Increase save interval:**
```yaml
logging:
  save_interval: 1000  # Instead of 100
```

---

## Data Processing Issues

### Problem: "No files found in data directory"

**Solutions:**

1. **Check directory structure:**
```
data/
└── raw/
    ├── sample1.txt
    └── sample2.jsonl
```

2. **Ensure files have correct extensions:**
```bash
# Supported: .txt, .jsonl, .json, .csv, .md
```

3. **Check file permissions:**
```bash
# Linux/Mac
chmod 644 data/raw/*

# Windows: Right-click → Properties → Security
```

---

### Problem: JSONL parsing errors

**Symptoms:** "Extra data" or "Invalid JSON" errors

**Solutions:**

1. **Check JSONL format:**
```bash
# Each line must be valid JSON
head -n 5 data/raw/data.jsonl
```

2. **Fix malformed lines:**
```python
# Script to validate JSONL
import json
with open('data/raw/data.jsonl', 'r') as f:
    for i, line in enumerate(f):
        try:
            json.loads(line)
        except:
            print(f"Line {i} is invalid: {line}")
```

3. **Ensure UTF-8 encoding:**
```bash
# Convert to UTF-8
iconv -f latin1 -t utf-8 input.jsonl > output.jsonl
```

---

### Problem: Data processing is slow

**Solutions:**

1. **Reduce max text length:**
```yaml
data:
  preprocessing:
    max_text_length: 5000  # Instead of 10000
```

2. **Use fewer workers:**
```yaml
hardware:
  num_workers: 0  # For CPU
```

3. **Process in batches:**
```bash
# Split large files
split -l 10000 large_data.jsonl chunk_
```

---

## Generation Issues

### Problem: Generated text is gibberish

**Symptoms:** Random words, no coherence

**Solutions:**

1. **Lower temperature:**
```bash
python scripts/generate.py \
  --temperature 0.5  # Instead of 0.8 or 1.0
```

2. **Use top-k sampling:**
```bash
python scripts/generate.py \
  --top-k 40 \
  --top-p 0.9
```

3. **Train longer:**
```yaml
training:
  max_epochs: 20  # Increase from 5
```

4. **Check data quality:**
```bash
# Ensure training data is coherent
head data/processed/train.jsonl
```

---

### Problem: Generated text is too repetitive

**Solutions:**

1. **Increase temperature:**
```bash
python scripts/generate.py \
  --temperature 0.8  # Instead of 0.5
```

2. **Use repetition penalty:**
```bash
python scripts/generate.py \
  --repetition-penalty 1.2
```

3. **Reduce sequence length:**
```yaml
model:
  sequence_length: 256  # For generation
```

---

### Problem: Model generates very short text

**Solutions:**

1. **Increase max length:**
```bash
python scripts/generate.py \
  --max-length 200  # Instead of 50
```

2. **Adjust EOS token:**
```yaml
generation:
  min_length: 50
  max_length: 200
```

3. **Use no_repeat_ngram_size:**
```bash
python scripts/generate.py \
  --no-repeat-ngram-size 3
```

---

## Hardware Issues

### Problem: GPU not detected

**Symptoms:** "Using device: cpu" even with GPU

**Solutions:**

1. **Check CUDA availability:**
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

2. **Install CUDA-compatible PyTorch:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

3. **Check GPU drivers:**
```bash
# Linux
nvidia-smi

# Windows
# Device Manager → Display adapters
```

4. **Force GPU usage:**
```yaml
hardware:
  use_gpu: true  # Instead of "auto"
```

---

### Problem: cuDNN error (GTX 650Ti and older)

**Symptoms:** "CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH"

**Solution:** Use CPU (your GPU compute capability is too low)
```yaml
hardware:
  use_gpu: false
```

Bangkong is optimized for CPU training and works great on Q8400!

---

### Problem: Multi-GPU not working

**Solutions:**

1. **Enable distributed training:**
```yaml
hardware:
  distributed: true
  use_deepspeed: true
```

2. **Check GPU visibility:**
```python
import torch
print(f"GPUs available: {torch.cuda.device_count()}")
```

3. **Use torchrun:**
```bash
torchrun --nproc_per_node=2 scripts/train.py --config configs/large.yaml
```

---

## Performance Issues

### Problem: High memory usage

**Solutions:**

1. **Reduce batch size:**
```yaml
training:
  batch_size: 1
```

2. **Enable gradient checkpointing:**
```yaml
hardware:
  gradient_checkpointing: true
```

3. **Use 8-bit optimizer:**
```yaml
training:
  use_8bit_adam: true
```

---

### Problem: Low GPU utilization

**Solutions:**

1. **Increase batch size:**
```yaml
training:
  batch_size: 16  # Instead of 4
```

2. **Increase workers:**
```yaml
hardware:
  num_workers: 8
```

3. **Pin memory:**
```yaml
hardware:
  pin_memory: true
```

4. **Use persistent workers:**
```yaml
hardware:
  persistent_workers: true
```

---

## Checkpoint Issues

### Problem: Can't resume from checkpoint

**Solutions:**

1. **Check checkpoint exists:**
```bash
ls models/my_model/checkpoints/
```

2. **Use correct mode:**
```bash
python scripts/train.py \
  --mode resume \
  --checkpoint-path models/my_model/checkpoints/latest.pt
```

3. **Ensure config matches:**
```bash
# Use same config as original training
python scripts/train.py \
  --config configs/small.yaml \
  --mode resume \
  --checkpoint-path models/my_model/checkpoints/latest.pt
```

---

### Problem: Checkpoint files too large

**Solutions:**

1. **Save less frequently:**
```yaml
logging:
  save_interval: 1000  # Instead of 100
```

2. **Keep only best checkpoints:**
```yaml
logging:
  keep_last_n_checkpoints: 3
```

3. **Use quantization:**
```bash
python scripts/convert.py \
  --model-path models/my_model \
  --quantize int8
```

---

## Still Having Issues?

### Get Help:

1. **Check logs:**
```bash
# Look for error messages
tail -n 100 logs/bangkong.log
```

2. **Run diagnostics:**
```bash
python scripts/diagnostics.py
```

3. **Create minimal reproduction:**
```bash
# Test with tiny config and small dataset
python scripts/train.py \
  --config configs/tiny.yaml \
  --data-path data/test
```

4. **Open GitHub issue:**
- Include error message
- Include config file
- Include hardware specs
- Include steps to reproduce

5. **Join Discord:**
- Real-time help from community
- Share screens for debugging

---

## Quick Reference

| Symptom | First Thing to Try |
|---------|-------------------|
| Slow training | Reduce model size |
| OOM error | Reduce batch size |
| Loss not decreasing | Adjust learning rate |
| NaN loss | Reduce learning rate, clip gradients |
| GPU not used | Check CUDA, use GPU config |
| Bad generation | Lower temperature, train longer |
| Repetitive text | Increase temperature, use penalty |

---

**Need more help?** See [docs/FAQ.md](FAQ.md) for frequently asked questions.
