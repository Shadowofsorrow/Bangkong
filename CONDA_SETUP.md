# CONDA SETUP GUIDE - Bangkong LLM Training System

## Quick Start with Conda

### Option 1: Use bangkong_conda.bat (Easiest)

1. **Edit `bangkong_conda.bat`** and set your conda path:
   ```batch
   set "CONDA_PATH=C:\Users\%USERNAME%\miniconda3"
   set "CONDA_ENV_NAME=webui"
   ```

2. **Double-click `bangkong_conda.bat`**
   - It will activate your conda environment automatically
   - Then launch the Bangkong menu

---

### Option 2: Manual Conda Activation

1. **Open Anaconda Prompt or CMD**

2. **Activate your environment:**
   ```bash
   conda activate webui
   ```

3. **Navigate to bangkong-github folder:**
   ```bash
   cd "C:\odin\alice\bangkong\others sytem train\bangkong-github"
   ```

4. **Run the training script:**
   ```bash
   python scripts/train.py --config configs/test_config.yaml --data-path data/test/test_samples.jsonl
   ```

   **OR run the interactive launcher:**
   ```bash
   python -m bangkong
   ```

   **OR run the batch file:**
   ```bash
   bangkong.bat
   ```

---

### Option 3: Create Dedicated Bangkong Environment

If you want a separate environment for Bangkong:

```bash
# Create new environment
conda create -n bangkong python=3.10 -y

# Activate it
conda activate bangkong

# Install PyTorch (CPU version)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# OR Install PyTorch (CUDA 11.8 version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install transformers pydantic pyyaml python-dotenv tqdm

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

Then use `bangkong_conda.bat` with:
```batch
set "CONDA_ENV_NAME=bangkong"
```

---

## Verify Your Setup

### Test 1: Check Python
```bash
conda activate webui
python --version
# Should show: Python 3.x
```

### Test 2: Check PyTorch
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Test 3: Check Transformers
```bash
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
```

### Test 4: Run Quick Test
```bash
cd "C:\odin\alice\bangkong\others sytem train\bangkong-github"
python quick_test.py
```

Expected output:
```
✓ Model created: 6,837,888 parameters
✓ Tokenizer loaded
✓ Data loaded: 2 samples
✓ Tokenization works: 18 tokens
✓ Forward pass works
✓ Loss calculation works: 10.86

ALL TESTS PASSED!
```

---

## Common Issues

### Issue 1: "conda is not recognized"

**Solution:** Add conda to PATH or use full path:
```bash
C:\Users\%USERNAME%\miniconda3\Scripts\activate.bat
conda activate webui
```

### Issue 2: "ModuleNotFoundError: No module named 'torch'"

**Solution:** Install PyTorch in your environment:
```bash
conda activate webui
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### Issue 3: "Environment 'webui' does not exist"

**Solution:** List available environments:
```bash
conda env list
```

Then use an existing one or create new:
```bash
conda create -n webui python=3.10
```

### Issue 4: Bangkong.bat doesn't start

**Solution:** Run with Python directly:
```bash
conda activate webui
python scripts/train.py --config configs/test_config.yaml
```

---

## Required Packages

Minimum requirements:
```
python>=3.8
torch>=1.9.0
transformers>=4.20.0
pydantic>=1.8.0
pyyaml>=5.4.0
python-dotenv>=0.19.0
tqdm>=4.62.0
```

Install with:
```bash
conda activate webui
pip install torch transformers pydantic pyyaml python-dotenv tqdm
```

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `conda activate webui` | Activate webui environment |
| `conda env list` | List all environments |
| `conda create -n NAME python=3.10` | Create new environment |
| `python scripts/train.py --config CONFIG` | Train model |
| `python quick_test.py` | Run sanity check |
| `bangkok.bat` | Interactive launcher |
| `bangkong_conda.bat` | Launcher with conda activation |

---

## Next Steps

1. **Activate conda environment:**
   ```bash
   conda activate webui
   ```

2. **Navigate to folder:**
   ```bash
   cd "C:\odin\alice\bangkong\others sytem train\bangkong-github"
   ```

3. **Run quick test:**
   ```bash
   python quick_test.py
   ```

4. **Start training (optional):**
   ```bash
   python scripts/train.py --config configs/test_config.yaml --data-path data/test/test_samples.jsonl
   ```

5. **OR use interactive launcher:**
   ```bash
   bangkong.bat
   ```

---

**Ready to go!** 🚀
