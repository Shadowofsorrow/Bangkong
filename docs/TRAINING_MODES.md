# Bangkong Training Modes

The Bangkong LLM Training System supports four distinct training modes to accommodate different scenarios:

## 1. Fresh Training

Start training a new model from scratch with randomly initialized weights.

**Use case**: Creating a new model from your own data or starting with a clean slate.

**Command**:
```bash
python scripts/train.py --training-mode fresh
```

**Features**:
- Creates a new model with random weights
- Starts training from epoch 1
- Saves checkpoints periodically
- Ideal for pre-training on domain-specific data

## 2. Resume Training

Continue training from a saved checkpoint when training was interrupted.

**Use case**: Recovering from system crashes, power outages, or intentional stops during long training sessions.

**Command**:
```bash
python scripts/train.py --training-mode resume --model-path ./models/my_model --checkpoint-path ./models/my_model/checkpoints/checkpoint_epoch_5.pt
```

**Features**:
- Loads model state, optimizer state, and scheduler state from checkpoint
- Continues from the exact epoch and step where training was interrupted
- Preserves learning rate schedule and optimizer momentum
- Maintains all training statistics and configurations

## 3. Continue Training

Add more epochs to a completed model to improve performance.

**Use case**: When initial training wasn't sufficient and you want to train longer without losing progress.

**Command**:
```bash
python scripts/train.py --training-mode continue --model-path ./models/my_model
```

**Features**:
- Loads a fully trained model
- Starts from epoch 1 with the existing trained weights
- Uses the same configuration as the original training
- Allows for additional epochs beyond the original training limit

## 4. Fine-tune Model

Adapt a pre-trained model to new data or a specific task.

**Use case**: Specializing a general-purpose model for a specific domain, task, or dataset.

**Command**:
```bash
python scripts/train.py --training-mode fine-tune --model-path ./models/my_model
```

**Features**:
- Loads a pre-trained model
- May adjust learning rates for better fine-tuning
- Can use different data than the original training data
- Preserves general knowledge while learning new patterns

## Interactive Mode

The system also supports an interactive menu-driven approach for selecting training modes:

```bash
python scripts/train.py
```

This will present a menu system that guides you through:
1. Selecting training mode
2. Choosing from available models
3. Selecting checkpoints (for resume mode)

## Checkpoint Management

The system automatically saves checkpoints during training:

- **Periodic checkpoints**: Saved every N steps as configured
- **Final checkpoints**: Saved when training completes
- **Manual checkpoints**: Can be saved at any time using trainer.save_checkpoint()

Checkpoints contain:
- Model state dictionary
- Optimizer state dictionary
- Scheduler state dictionary
- Current epoch and step numbers
- Configuration parameters

## Model Management

The training manager can list and manage available models:

```bash
# List available models
python test_training_manager.py
```

Models are categorized as:
- **Trained models**: Fully trained models saved during training
- **Packaged models**: Models converted to different formats
- **Checkpoint files**: Intermediate training states