"""
Model trainer for Bangkong LLM Training System
"""

import torch
from torch.cuda.amp import autocast, GradScaler
from typing import Any, Dict, Optional
import logging
from ..config.schemas import BangkongConfig
from ..hardware.allocator import ResourceAllocator
from ..hardware.detector import HardwareDetector
from .peft import apply_peft_to_model
from .pruning import apply_pruning_to_model
from .curriculum import create_curriculum_controller
from .efficient_attention import create_efficient_attention

# Try to import sparse attention libraries
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

try:
    from transformers.models.gpt2.modeling_gpt2 import GPT2Model
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class DynamicTrainer:
    """Dynamically adapts training to available hardware and configuration."""
    
    def __init__(self, model: torch.nn.Module, config: BangkongConfig):
        """
        Initialize the dynamic trainer.
        
        Args:
            model: PyTorch model to train.
            config: Bangkong configuration.
        """
        self.model = model
        self.config = config
        self.hardware_info = HardwareDetector.get_available_resources()
        self.resource_allocator = ResourceAllocator(config)
        self.resources = self.resource_allocator.allocate_training_resources()
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Initialize device
        self.device = self.resources['device']
        self.logger.debug(f"Using device: {self.device}")
        
        # Enable gradient checkpointing for memory efficiency with long sequences
        if (hasattr(self.config.model, 'sequence_length') and 
            self.config.model.sequence_length > 10000) or \
           (hasattr(self.config.model, 'max_sequence_length') and
            self.config.model.max_sequence_length > 10000):
            try:
                self.model.gradient_checkpointing_enable()
                self.logger.debug("Gradient checkpointing enabled for memory efficiency")
            except Exception as e:
                self.logger.warning(f"Failed to enable gradient checkpointing: {e}")
        
        # Apply parameter-efficient fine-tuning if configured
        peft_type = getattr(self.config.training, 'peft_type', 'none')
        if peft_type != 'none':
            try:
                self.model = apply_peft_to_model(self.model, self.config)
                self.logger.debug(f"Applied {peft_type} parameter-efficient fine-tuning")
            except Exception as e:
                self.logger.warning(f"Failed to apply {peft_type} PEFT: {e}")
        
        # Apply pruning if configured
        pruning_type = getattr(self.config.training, 'pruning_type', 'none')
        sparsity_ratio = getattr(self.config.training, 'sparsity_ratio', 0.0)
        if pruning_type != 'none' and sparsity_ratio > 0.0:
            try:
                self.model = apply_pruning_to_model(self.model, self.config)
                self.logger.debug(f"Applied {pruning_type} pruning with {sparsity_ratio} sparsity ratio")
            except Exception as e:
                self.logger.warning(f"Failed to apply {pruning_type} pruning: {e}")
        
        # Initialize curriculum learning controller
        curriculum_type = getattr(self.config.training, 'curriculum_type', 'none')
        if curriculum_type != 'none':
            try:
                self.curriculum_controller = create_curriculum_controller(self.config)
                self.logger.debug(f"Initialized curriculum learning with type: {curriculum_type}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize curriculum learning: {e}")
                self.curriculum_controller = None
        else:
            self.curriculum_controller = None
        
        # Initialize efficient attention controller for long sequences
        sequence_length = getattr(self.config.model, 'sequence_length', 1024)
        max_sequence_length = getattr(self.config.model, 'max_sequence_length', 2000000)
        if sequence_length > 10000 or max_sequence_length > 10000:  # 10K+ tokens
            try:
                self.efficient_attention = create_efficient_attention(self.config)
                self.logger.debug("Initialized efficient attention controller for long sequences")
            except Exception as e:
                self.logger.warning(f"Failed to initialize efficient attention controller: {e}")
                self.efficient_attention = None
        else:
            self.efficient_attention = None
        
        # Apply pruning if configured
        pruning_type = getattr(self.config.training, 'pruning_type', 'none')
        sparsity_ratio = getattr(self.config.training, 'sparsity_ratio', 0.0)
        if pruning_type != 'none' and sparsity_ratio > 0.0:
            try:
                self.model = apply_pruning_to_model(self.model, self.config)
                self.logger.debug(f"Applied {pruning_type} pruning with {sparsity_ratio} sparsity ratio")
            except Exception as e:
                self.logger.warning(f"Failed to apply {pruning_type} pruning: {e}")
        
        try:
            self.model.to(self.device)
        except Exception as e:
            self.logger.warning(f"Failed to move model to {self.device}: {e}")
            self.logger.debug("Falling back to CPU...")
            self.device = torch.device("cpu")
            self.model.to(self.device)
        
        # Initialize mixed precision
        self.use_mixed_precision = self.resources['mixed_precision']
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            self.logger.debug("Mixed precision training enabled")
        else:
            self.logger.debug("Using full precision training")
        
        # Initialize optimizer with dynamic learning rate
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        # Monitoring integration
        self.monitoring_backend = self._setup_monitoring()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        # Create a unique logger name to avoid conflicts
        logger_name = f"bangkong.trainer.{id(self)}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, self.config.monitoring.log_level.upper(), logging.INFO))
        logger.propagate = False  # Prevent propagation to root logger
        
        # Check if logger already has handlers to prevent duplicates
        if not logger.handlers:
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Create console handler with tqdm-safe output
            try:
                from tqdm import tqdm
                class TqdmLoggingHandler(logging.Handler):
                    def emit(self, record):
                        try:
                            msg = self.format(record)
                            tqdm.write(msg)
                            self.flush()
                        except Exception:
                            self.handleError(record)
                
                console_handler = TqdmLoggingHandler()
            except ImportError:
                # Fallback to regular console handler if tqdm is not available
                console_handler = logging.StreamHandler()
            
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # Create file handler if log file is specified
            if self.config.monitoring.log_file:
                try:
                    file_handler = logging.FileHandler(self.config.monitoring.log_file)
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
                except Exception as e:
                    logger.warning(f"Could not create file handler for logging: {e}")
        
        return logger
    
    def _setup_monitoring(self):
        """Set up monitoring backend based on configuration."""
        backend = self.config.monitoring.backend
        
        if backend == "wandb":
            try:
                import wandb
                # Initialize wandb run
                wandb.init(
                    project="bangkong-llm",
                    config=self.config.dict()
                )
                self.logger.debug("Weights & Biases monitoring enabled")
                return wandb
            except ImportError:
                self.logger.warning("Weights & Biases not installed. Falling back to no monitoring.")
                return None
            except Exception as e:
                self.logger.warning(f"Failed to initialize Weights & Biases: {e}")
                return None
        elif backend == "mlflow":
            try:
                import mlflow
                # Start MLflow run
                mlflow.start_run()
                mlflow.log_params(self.config.dict())
                self.logger.debug("MLflow monitoring enabled")
                return mlflow
            except ImportError:
                self.logger.warning("MLflow not installed. Falling back to no monitoring.")
                return None
            except Exception as e:
                self.logger.warning(f"Failed to initialize MLflow: {e}")
                return None
        elif backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                import os
                # Create logs directory if it doesn't exist
                log_dir = os.path.join(self.config.paths.logs, "tensorboard")
                os.makedirs(log_dir, exist_ok=True)
                writer = SummaryWriter(log_dir=log_dir)
                self.logger.debug(f"TensorBoard monitoring enabled. Logs will be saved to {log_dir}")
                return writer
            except ImportError:
                self.logger.warning("TensorBoard not installed. Falling back to no monitoring.")
                return None
            except Exception as e:
                self.logger.warning(f"Failed to initialize TensorBoard: {e}")
                return None
        else:
            self.logger.debug("No external monitoring backend configured")
            return None
    
    def _create_optimizer(self):
        """Create optimizer with hardware-adaptive parameters."""
        # Adjust learning rate based on batch size
        effective_batch_size = self.resources['batch_size'] * self.config.training.gradient_accumulation_steps
        adjusted_lr = self.config.training.learning_rate * (effective_batch_size / 32)  # Scale LR with batch size
        
        try:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=adjusted_lr,
                weight_decay=self.config.training.weight_decay,
                eps=self.config.training.adam_epsilon
            )
            self.logger.debug(f"Created AdamW optimizer with learning rate: {adjusted_lr}")
            return optimizer
        except Exception as e:
            self.logger.warning(f"Failed to create AdamW optimizer: {e}")
            self.logger.debug("Falling back to SGD optimizer...")
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=adjusted_lr,
                weight_decay=self.config.training.weight_decay
            )
            return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        # For now, we'll use a simple linear scheduler
        # In a full implementation, this would be more sophisticated
        try:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.config.training.max_epochs
            )
            self.logger.debug("Created LinearLR scheduler")
            return scheduler
        except Exception as e:
            self.logger.warning(f"Failed to create LinearLR scheduler: {e}")
            self.logger.debug("Using StepLR scheduler instead...")
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=max(1, self.config.training.max_epochs // 3),
                gamma=0.1
            )
            return scheduler
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of training data.
            
        Returns:
            Loss value for the batch.
        """
        self.model.train()
        self.logger.debug(f"train_step - Starting train_step with batch keys: {list(batch.keys())}")
        
        # Debug batch contents
        for key, value in batch.items():
            self.logger.debug(f"train_step - Batch {key}: shape={value.shape}, dtype={value.dtype}")
        
        # Move batch to device
        try:
            batch = {k: v.to(self.device) for k, v in batch.items()}
        except Exception as e:
            self.logger.warning(f"Failed to move batch to {self.device}: {e}")
            return 100.0  # Return a large finite number instead of infinity
        
        # Debug information
        if batch['input_ids'].numel() > 0:
            max_token_id = torch.max(batch['input_ids']).item()
            min_token_id = torch.min(batch['input_ids']).item()
            vocab_size = self.config.model.vocab_size
            
            self.logger.debug(f"Batch token ID range: {min_token_id} to {max_token_id}, model vocab size: {vocab_size}")
            
            if max_token_id >= vocab_size:
                self.logger.error(f"Token ID {max_token_id} is out of bounds for model vocab size {vocab_size}")
                self.logger.debug(f"Batch input_ids: {batch['input_ids']}")
                return 100.0
        
        self.logger.debug("Starting forward pass")
        
        if self.use_mixed_precision:
            try:
                # Log the batch information before calling the model
                self.logger.debug(f"Mixed precision - Calling model with batch keys: {list(batch.keys())}")
                for key, value in batch.items():
                    self.logger.debug(f"  Mixed precision - {key}: shape={value.shape}, dtype={value.dtype}")
                
                with autocast():
                    outputs = self.model(**batch)
                    
                    # Log detailed information about outputs
                    self.logger.debug(f"Mixed precision - Model outputs type: {type(outputs)}")
                    self.logger.debug(f"Mixed precision - Model outputs attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
                    self.logger.debug(f"Mixed precision - Model outputs has loss: {hasattr(outputs, 'loss')}")
                    if hasattr(outputs, 'loss'):
                        loss_attr = getattr(outputs, 'loss', None)
                        self.logger.debug(f"Mixed precision - Model outputs loss value: {loss_attr}")
                        self.logger.debug(f"Mixed precision - Model outputs loss type: {type(loss_attr)}")
                    self.logger.debug(f"Mixed precision - Model outputs has logits: {hasattr(outputs, 'logits')}")
                    if hasattr(outputs, 'logits'):
                        logits_attr = getattr(outputs, 'logits', None)
                        self.logger.debug(f"Mixed precision - Model outputs logits shape: {getattr(logits_attr, 'shape', 'no shape') if logits_attr is not None else 'None'}")
                        self.logger.debug(f"Mixed precision - Model outputs logits type: {type(logits_attr) if logits_attr is not None else 'None'}")
                    
                    # Check if outputs have loss attribute and it's not None
                    loss = None
                    if hasattr(outputs, 'loss'):
                        loss_attr = getattr(outputs, 'loss', None)
                        if loss_attr is not None:
                            loss = loss_attr
                            self.logger.debug(f"Mixed precision - Using model's loss: {loss.item()}")
                        else:
                            self.logger.warning("Mixed precision - Model outputs loss attribute is None")
                    else:
                        self.logger.warning("Mixed precision - Model outputs missing loss attribute")
                    
                    # If loss is None, try to compute it manually
                    if loss is None:
                        self.logger.warning("Mixed precision - Model outputs do not contain a valid loss attribute")
                        # Debug information about what we have
                        self.logger.debug(f"Mixed precision - Batch keys: {list(batch.keys())}")
                        if 'labels' in batch:
                            self.logger.debug(f"Mixed precision - Labels shape: {batch['labels'].shape}")
                            self.logger.debug(f"Mixed precision - Labels dtype: {batch['labels'].dtype}")
                        
                        logits = None
                        if hasattr(outputs, 'logits'):
                            logits_attr = getattr(outputs, 'logits', None)
                            if logits_attr is not None:
                                self.logger.debug(f"Mixed precision - Logits shape: {logits_attr.shape}")
                                self.logger.debug(f"Mixed precision - Logits dtype: {logits_attr.dtype}")
                                logits = logits_attr
                            else:
                                self.logger.debug("Mixed precision - Logits attribute is None")
                        else:
                            self.logger.debug("Mixed precision - Model outputs missing logits attribute")
                        
                        # Try to compute loss manually if needed
                        if logits is not None and 'labels' in batch:
                            labels = batch['labels']
                            self.logger.debug(f"Mixed precision - Computing loss manually with logits shape: {logits.shape} and labels shape: {labels.shape}")
                            
                            # Shift so that tokens < n predict n
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                            self.logger.debug(f"Mixed precision - Shifted logits shape: {shift_logits.shape}")
                            self.logger.debug(f"Mixed precision - Shifted labels shape: {shift_labels.shape}")
                            
                            # Flatten the tokens
                            loss_fct = torch.nn.CrossEntropyLoss()
                            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                            self.logger.debug(f"Mixed precision - Manually computed loss: {loss.item()}")
                        else:
                            self.logger.error("Mixed precision - Cannot compute loss: missing logits or labels")
                            # Log more detailed information
                            if logits is None:
                                self.logger.error("Mixed precision - Model outputs missing logits attribute or logits is None")
                            if 'labels' not in batch:
                                self.logger.error("Mixed precision - Batch missing labels")
                            return 100.0
                    else:
                        self.logger.debug(f"Mixed precision - Using model's loss: {loss.item()}")
                        
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                final_loss = loss.item() if not torch.isnan(loss) and not torch.isinf(loss) else 100.0
                self.logger.debug(f"Mixed precision - Final loss: {final_loss}")
                return final_loss
            except Exception as e:
                self.logger.warning(f"Mixed precision training step failed: {e}")
                import traceback
                self.logger.warning(f"Mixed precision - Traceback: {traceback.format_exc()}")
                # Fallback to full precision
                try:
                    # Log the batch information before calling the model
                    self.logger.debug(f"Fallback - Calling model with batch keys: {list(batch.keys())}")
                    for key, value in batch.items():
                        self.logger.debug(f"  Fallback - {key}: shape={value.shape}, dtype={value.dtype}")
                    
                    outputs = self.model(**batch)
                    
                    # Log detailed information about outputs
                    self.logger.debug(f"Fallback - Model outputs type: {type(outputs)}")
                    self.logger.debug(f"Fallback - Model outputs attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
                    self.logger.debug(f"Fallback - Model outputs has loss: {hasattr(outputs, 'loss')}")
                    if hasattr(outputs, 'loss'):
                        loss_attr = getattr(outputs, 'loss', None)
                        self.logger.debug(f"Fallback - Model outputs loss value: {loss_attr}")
                        self.logger.debug(f"Fallback - Model outputs loss type: {type(loss_attr)}")
                    self.logger.debug(f"Fallback - Model outputs has logits: {hasattr(outputs, 'logits')}")
                    if hasattr(outputs, 'logits'):
                        logits_attr = getattr(outputs, 'logits', None)
                        self.logger.debug(f"Fallback - Model outputs logits shape: {getattr(logits_attr, 'shape', 'no shape') if logits_attr is not None else 'None'}")
                        self.logger.debug(f"Fallback - Model outputs logits type: {type(logits_attr) if logits_attr is not None else 'None'}")
                    
                    # Check if outputs have loss attribute and it's not None
                    loss = None
                    if hasattr(outputs, 'loss'):
                        loss_attr = getattr(outputs, 'loss', None)
                        if loss_attr is not None:
                            loss = loss_attr
                            self.logger.debug(f"Fallback - Using model's loss: {loss.item()}")
                        else:
                            self.logger.warning("Fallback - Model outputs loss attribute is None")
                    else:
                        self.logger.warning("Fallback - Model outputs missing loss attribute")
                    
                    # If loss is None, try to compute it manually
                    if loss is None:
                        self.logger.warning("Fallback - Model outputs do not contain a valid loss attribute")
                        # Debug information about what we have
                        self.logger.debug(f"Fallback - Batch keys: {list(batch.keys())}")
                        if 'labels' in batch:
                            self.logger.debug(f"Fallback - Labels shape: {batch['labels'].shape}")
                            self.logger.debug(f"Fallback - Labels dtype: {batch['labels'].dtype}")
                        
                        logits = None
                        if hasattr(outputs, 'logits'):
                            logits_attr = getattr(outputs, 'logits', None)
                            if logits_attr is not None:
                                self.logger.debug(f"Fallback - Logits shape: {logits_attr.shape}")
                                self.logger.debug(f"Fallback - Logits dtype: {logits_attr.dtype}")
                                logits = logits_attr
                            else:
                                self.logger.debug("Fallback - Logits attribute is None")
                        else:
                            self.logger.debug("Fallback - Model outputs missing logits attribute")
                        
                        # Try to compute loss manually if needed
                        if logits is not None and 'labels' in batch:
                            labels = batch['labels']
                            self.logger.debug(f"Fallback - Computing loss manually with logits shape: {logits.shape} and labels shape: {labels.shape}")
                            
                            # Shift so that tokens < n predict n
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                            self.logger.debug(f"Fallback - Shifted logits shape: {shift_logits.shape}")
                            self.logger.debug(f"Fallback - Shifted labels shape: {shift_labels.shape}")
                            
                            # Flatten the tokens
                            loss_fct = torch.nn.CrossEntropyLoss()
                            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                            self.logger.debug(f"Fallback - Manually computed loss: {loss.item()}")
                        else:
                            self.logger.error("Fallback - Cannot compute loss: missing logits or labels")
                            # Log more detailed information
                            if logits is None:
                                self.logger.error("Fallback - Model outputs missing logits attribute or logits is None")
                            if 'labels' not in batch:
                                self.logger.error("Fallback - Batch missing labels")
                            return 100.0
                    else:
                        self.logger.debug(f"Fallback - Using model's loss: {loss.item()}")
                        
                    loss.backward()
                    
                    # Gradient clipping
                    if self.config.training.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.max_grad_norm
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    final_loss = loss.item() if not torch.isnan(loss) and not torch.isinf(loss) else 100.0
                    self.logger.debug(f"Fallback - Final loss: {final_loss}")
                    return final_loss
                except Exception as e2:
                    self.logger.warning(f"Fallback training step failed: {e2}")
                    import traceback
                    self.logger.warning(f"Fallback - Traceback: {traceback.format_exc()}")
                    return 100.0
        else:
            try:
                # Log the batch information before calling the model
                self.logger.debug(f"Calling model with batch keys: {list(batch.keys())}")
                for key, value in batch.items():
                    self.logger.debug(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                
                outputs = self.model(**batch)
                
                # Log detailed information about outputs
                self.logger.debug(f"Model outputs type: {type(outputs)}")
                self.logger.debug(f"Model outputs attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
                self.logger.debug(f"Model outputs has loss: {hasattr(outputs, 'loss')}")
                if hasattr(outputs, 'loss'):
                    loss_attr = getattr(outputs, 'loss', None)
                    self.logger.debug(f"Model outputs loss value: {loss_attr}")
                    self.logger.debug(f"Model outputs loss type: {type(loss_attr)}")
                self.logger.debug(f"Model outputs has logits: {hasattr(outputs, 'logits')}")
                if hasattr(outputs, 'logits'):
                    logits_attr = getattr(outputs, 'logits', None)
                    self.logger.debug(f"Model outputs logits shape: {getattr(logits_attr, 'shape', 'no shape') if logits_attr is not None else 'None'}")
                    self.logger.debug(f"Model outputs logits type: {type(logits_attr) if logits_attr is not None else 'None'}")
                
                # Check if outputs have loss attribute and it's not None
                loss = None
                if hasattr(outputs, 'loss'):
                    loss_attr = getattr(outputs, 'loss', None)
                    if loss_attr is not None:
                        loss = loss_attr
                        self.logger.debug(f"Using model's loss: {loss.item()}")
                    else:
                        self.logger.warning("Model outputs loss attribute is None")
                else:
                    self.logger.warning("Model outputs missing loss attribute")
                
                # If loss is None, try to compute it manually
                if loss is None:
                    self.logger.warning("Model outputs do not contain a valid loss attribute")
                    # Debug information about what we have
                    self.logger.debug(f"Batch keys: {list(batch.keys())}")
                    if 'labels' in batch:
                        self.logger.debug(f"Labels shape: {batch['labels'].shape}")
                        self.logger.debug(f"Labels dtype: {batch['labels'].dtype}")
                    
                    logits = None
                    if hasattr(outputs, 'logits'):
                        logits_attr = getattr(outputs, 'logits', None)
                        if logits_attr is not None:
                            self.logger.debug(f"Logits shape: {logits_attr.shape}")
                            self.logger.debug(f"Logits dtype: {logits_attr.dtype}")
                            logits = logits_attr
                        else:
                            self.logger.debug("Logits attribute is None")
                    else:
                        self.logger.debug("Model outputs missing logits attribute")
                    
                    # Try to compute loss manually if needed
                    if logits is not None and 'labels' in batch:
                        labels = batch['labels']
                        self.logger.debug(f"Computing loss manually with logits shape: {logits.shape} and labels shape: {labels.shape}")
                        
                        # Shift so that tokens < n predict n
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        self.logger.debug(f"Shifted logits shape: {shift_logits.shape}")
                        self.logger.debug(f"Shifted labels shape: {shift_labels.shape}")
                        
                        # Flatten the tokens
                        loss_fct = torch.nn.CrossEntropyLoss()
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        self.logger.debug(f"Manually computed loss: {loss.item()}")
                    else:
                        self.logger.error("Cannot compute loss: missing logits or labels")
                        # Log more detailed information
                        if logits is None:
                            self.logger.error("Model outputs missing logits attribute or logits is None")
                        if 'labels' not in batch:
                            self.logger.error("Batch missing labels")
                        return 100.0
                else:
                    self.logger.debug(f"Using model's loss: {loss.item()}")
                    
                loss.backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                final_loss = loss.item() if not torch.isnan(loss) and not torch.isinf(loss) else 100.0
                self.logger.debug(f"Final loss: {final_loss}")
                return final_loss
            except Exception as e:
                self.logger.warning(f"Training step failed: {e}")
                import traceback
                self.logger.warning(f"Traceback: {traceback.format_exc()}")
                return 100.0  # Return a large finite number instead of infinity
        
        try:
            self.scheduler.step()
            self.logger.debug("Scheduler step completed")
        except Exception as e:
            self.logger.warning(f"Scheduler step failed: {e}")
        
        self.optimizer.zero_grad()
        self.logger.debug("Optimizer gradients zeroed")
        
        self.global_step += 1
        self.logger.debug(f"Incremented global_step to: {self.global_step}")
        
        # Log metrics periodically
        if self.global_step % self.config.training.logging_steps == 0:
            self.logger.debug(f"Step {self.global_step}: Loss = {loss.item():.4f}")
            
            # Log to monitoring backend if available
            if self.monitoring_backend:
                try:
                    if hasattr(self.monitoring_backend, 'log'):
                        self.monitoring_backend.log({
                            'train/loss': loss.item(),
                            'train/step': self.global_step
                        })
                    elif hasattr(self.monitoring_backend, 'log_metric'):
                        self.monitoring_backend.log_metric('train_loss', loss.item(), self.global_step)
                    elif hasattr(self.monitoring_backend, 'add_scalar'):
                        # TensorBoard
                        self.monitoring_backend.add_scalar('train/loss', loss.item(), self.global_step)
                except Exception as e:
                    self.logger.warning(f"Failed to log to monitoring backend: {e}")
        
        final_loss = loss.item() if not torch.isnan(loss) and not torch.isinf(loss) else 0.0
        self.logger.debug(f"Completed train_step with final loss: {final_loss}")
        return final_loss
    
    def train_epoch(self, dataloader) -> float:
        """
        Train for one epoch (deprecated - use train_epoch_with_progress instead).
        
        Args:
            dataloader: DataLoader for training data.
            
        Returns:
            Average loss for the epoch.
        """
        # Call the new method with no progress tracking for backward compatibility
        return self.train_epoch_with_progress(dataloader, None, 0)
    
    def train_epoch_with_progress(self, dataloader, progress_bar=None, global_batch_offset: int = 0) -> float:
        """
        Train for one epoch with optional progress tracking.
        
        Args:
            dataloader: DataLoader for training data.
            progress_bar: Optional tqdm progress bar for tracking progress.
            global_batch_offset: Offset for global batch counting in unified progress tracking.
            
        Returns:
            Average loss for the epoch.
        """
        total_loss = 0.0
        num_batches = 0
        
        batch_count = 0
        
        # Process batches and update the provided progress bar
        for batch_idx, batch in enumerate(dataloader):
            batch_count += 1
            
            try:
                loss = self.train_step(batch)
                
                if not torch.isnan(torch.tensor(loss)) and not torch.isinf(torch.tensor(loss)):
                    total_loss += loss
                    num_batches += 1
                    
                # Update the provided progress bar if available
                if progress_bar is not None:
                    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
                    progress_bar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'avg_loss': f'{avg_loss:.4f}'
                    })
                    progress_bar.update(1)
                    
                # Log additional metrics periodically but less frequently to avoid progress bar interference
                if self.global_step % self.config.training.logging_steps == 0 and batch_idx > 0 and batch_idx % 10 == 0:
                    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
                    self.logger.debug(f"Epoch {self.current_epoch + 1}, Step {self.global_step}: Avg Loss = {avg_loss:.4f}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to train on batch {batch_idx}: {e}")
                # Update progress bar even if batch fails
                if progress_bar is not None:
                    progress_bar.update(1)
                # Continue to next batch instead of returning
                continue
        
        self.current_epoch += 1
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return avg_loss
    
    def train(self, train_dataloader, val_dataloader=None, start_epoch: int = None, 
              additional_epochs: int = None):
        """
        Train the model for the configured number of epochs.
        
        Args:
            train_dataloader: DataLoader for training data.
            val_dataloader: Optional DataLoader for validation data.
            start_epoch: Starting epoch number (for resuming training).
            additional_epochs: Additional epochs to train (for continuing training).
        """
        # Debug information
        self.logger.debug(f"Starting training with dataloader of length: {len(train_dataloader) if hasattr(train_dataloader, '__len__') else 'unknown'}")
        self.logger.debug(f"Training configuration - max_epochs: {self.config.training.max_epochs}, batch_size: {getattr(self.config.training, 'batch_size', 'not set')}")
        
        # Log more details about the training setup
        self.logger.debug(f"Dataloader batch size: {getattr(train_dataloader, 'batch_size', 'unknown')}")
        self.logger.debug(f"Device: {self.device}")
        self.logger.debug(f"Model type: {type(self.model)}")
        
        # Determine starting epoch
        if start_epoch is not None:
            starting_epoch = start_epoch
        else:
            starting_epoch = self.current_epoch
        
        # Determine total epochs
        if additional_epochs is not None:
            total_epochs = starting_epoch + additional_epochs
        else:
            total_epochs = self.config.training.max_epochs
        
        self.logger.debug(f"Training parameters - starting_epoch: {starting_epoch}, total_epochs: {total_epochs}, additional_epochs: {additional_epochs}")
        
        # If we're already at or beyond the target epochs, add more epochs for resume/continue
        if starting_epoch >= total_epochs:
            if additional_epochs is not None:
                total_epochs = starting_epoch + additional_epochs
            else:
                # Default to adding 2 more epochs for resume/continue
                total_epochs = starting_epoch + 2
        
        # Adjust scheduler if we're continuing training
        if starting_epoch > 0 and hasattr(self.scheduler, 'last_epoch'):
            self.scheduler.last_epoch = starting_epoch - 1
        
        self.logger.debug(f"Training from epoch {starting_epoch + 1} to {total_epochs}")
        
        # Track overall training metrics
        training_start_time = None
        try:
            import time
            training_start_time = time.time()
        except:
            pass
        
        best_val_loss = float('inf')
        no_improvement_count = 0
        patience = getattr(self.config.training, 'early_stopping_patience', 3)  # Early stopping patience
        self.logger.debug(f"Early stopping patience set to: {patience}")
        
        epoch_count = 0
        global_batch_count = self.global_step
        
        # Import tqdm
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
            self.logger.debug("tqdm not available, using simple iteration")
        
        for epoch in range(starting_epoch, total_epochs):
            epoch_count += 1
            current_epoch = epoch + 1
            global_epoch_count = epoch + 1
            
            # Print a blank line to ensure clean separation between epochs
            print()
            
            # Print epoch header
            print(f"📅 Epoch {current_epoch}/{total_epochs} (Global: {global_epoch_count})")
            
            # Small delay to allow any warning messages to be printed before progress bar starts
            try:
                import time
                time.sleep(0.1)
            except:
                pass
            
            # Training phase
            train_start_time = None
            try:
                import time
                train_start_time = time.time()
            except:
                pass
            
            # Initialize training progress bar
            train_pbar = None
            
            # Create training progress bar
            train_dataloader_len = len(train_dataloader) if hasattr(train_dataloader, '__len__') else None
            if tqdm and train_dataloader_len:
                train_pbar = tqdm(
                    total=train_dataloader_len,
                    desc="Training",
                    leave=False,
                    ncols=100
                )
            else:
                if train_start_time and train_dataloader_len:
                    print(f"Training {train_dataloader_len} batches...")
                elif train_start_time:
                    print("Training batches (unknown count)...")
            
            total_train_loss = 0.0
            num_train_batches = 0
            
            # Train for one epoch
            for batch_idx, batch in enumerate(train_dataloader):
                try:
                    loss = self.train_step(batch)
                    global_batch_count += 1
                    
                    if not torch.isnan(torch.tensor(loss)) and not torch.isinf(torch.tensor(loss)):
                        total_train_loss += loss
                        num_train_batches += 1
                    
                    # Update training progress bar
                    if train_pbar is not None:
                        avg_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0
                        train_pbar.set_postfix({
                            'loss': f'{loss:.4f}',
                            'avg_loss': f'{avg_loss:.4f}'
                        })
                        train_pbar.update(1)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to train on batch {batch_idx}: {e}")
                    # Update progress bar even if batch fails
                    if train_pbar is not None:
                        train_pbar.update(1)
                    continue
            
            # Close training progress bar
            if train_pbar is not None:
                train_pbar.close()
            
            # Calculate training metrics
            avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0
            
            # Print training completion
            if train_start_time:
                try:
                    import time
                    train_duration = time.time() - train_start_time
                    print(f"   Train Loss: {avg_train_loss:.4f}")
                    print(f"   Time: {train_duration:.2f}s")
                except:
                    pass
            
            # Validation phase (if provided)
            if val_dataloader:
                val_start_time = None
                try:
                    import time
                    val_start_time = time.time()
                except:
                    pass
                
                # Initialize validation progress bar
                val_pbar = None
                
                # Create validation progress bar
                val_dataloader_len = len(val_dataloader) if hasattr(val_dataloader, '__len__') else None
                if tqdm and val_dataloader_len:
                    val_pbar = tqdm(
                        total=val_dataloader_len,
                        desc="Validating",
                        leave=False,
                        ncols=100
                    )
                else:
                    if val_start_time and val_dataloader_len:
                        print(f"Validating {val_dataloader_len} batches...")
                    elif val_start_time:
                        print("Validating batches (unknown count)...")
                
                total_val_loss = 0.0
                num_val_batches = 0
                
                # Validation loop
                self.model.eval()
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_dataloader):
                        try:
                            batch = {k: v.to(self.device) for k, v in batch.items()}
                            outputs = self.model(**batch)
                            # Check if outputs have loss attribute
                            if hasattr(outputs, 'loss') and outputs.loss is not None:
                                if not torch.isnan(outputs.loss) and not torch.isinf(outputs.loss):
                                    total_val_loss += outputs.loss.item()
                                    num_val_batches += 1
                            else:
                                # Try to compute loss manually if needed
                                if hasattr(outputs, 'logits') and 'labels' in batch:
                                    logits = outputs.logits
                                    labels = batch['labels']
                                    # Shift so that tokens < n predict n
                                    shift_logits = logits[..., :-1, :].contiguous()
                                    shift_labels = labels[..., 1:].contiguous()
                                    # Flatten the tokens
                                    loss_fct = torch.nn.CrossEntropyLoss()
                                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                                    if not torch.isnan(loss) and not torch.isinf(loss):
                                        total_val_loss += loss.item()
                                        num_val_batches += 1
                                else:
                                    # Log information about what's missing
                                    self.logger.debug(f"Validation batch {batch_idx} missing logits or labels")
                                    if not hasattr(outputs, 'logits'):
                                        self.logger.debug("  Missing logits in outputs")
                                    if 'labels' not in batch:
                                        self.logger.debug("  Missing labels in batch")
                        except Exception as e:
                            self.logger.warning(f"Failed to validate on batch {batch_idx}: {e}")
                            # Update progress bar even if batch fails
                            if val_pbar is not None:
                                val_pbar.update(1)
                            continue
                
                # Close validation progress bar
                if val_pbar is not None:
                    val_pbar.close()
                
                # Calculate validation metrics
                avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
                
                # Print validation completion
                print(f"   Validation Loss: {avg_val_loss:.4f}")
                
                if val_start_time:
                    try:
                        import time
                        val_duration = time.time() - val_start_time
                        print(f"   Validation Time: {val_duration:.2f}s")
                    except:
                        pass
                
                # Check if this is the best model so far
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    no_improvement_count = 0
                    print("💾 Model saved to outputs/alice_distillation_temp_best.pt")
                    print("   🏆 New best model saved!")
                else:
                    no_improvement_count += 1
            
            # Save checkpoint
            try:
                import time
                timestamp = int(time.time())
                checkpoint_path = f"outputs/alice_distillation_temp_epoch_{current_epoch}.pt"
                # Create outputs directory if it doesn't exist
                import os
                os.makedirs("outputs", exist_ok=True)
                self.save_checkpoint(checkpoint_path)
                print(f"💾 Checkpoint saved to {checkpoint_path}")
                print(f"   💾 Epoch checkpoint saved: {checkpoint_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save checkpoint: {e}")
            
            # Early stopping check
            if no_improvement_count >= patience:
                self.logger.debug(f"No improvement for {patience} epochs. Early stopping...")
                print("⏹️ Early stopping triggered")
                break
        
        self.logger.debug(f"Completed training loop with {epoch_count} epochs processed")
        
        # Display overall training metrics
        if training_start_time:
            try:
                import time
                training_duration = time.time() - training_start_time
                hours = int(training_duration // 3600)
                minutes = int((training_duration % 3600) // 60)
                seconds = int(training_duration % 60)
                print(f"🏁 Training completed in {hours:02d}:{minutes:02d}:{seconds:02d}")
            except:
                print("🏁 Training completed.")
        else:
            print("🏁 Training completed.")
        
        # Close monitoring backend
        self.close_monitoring()
    
    def validate(self, dataloader) -> float:
        """
        Validate the model.
        
        Args:
            dataloader: DataLoader for validation data.
            
        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                try:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    # Check if outputs have loss attribute
                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        if not torch.isnan(outputs.loss) and not torch.isinf(outputs.loss):
                            total_loss += outputs.loss.item()
                            num_batches += 1
                    else:
                        # Try to compute loss manually if needed
                        if hasattr(outputs, 'logits') and 'labels' in batch:
                            logits = outputs.logits
                            labels = batch['labels']
                            # Shift so that tokens < n predict n
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                            # Flatten the tokens
                            loss_fct = torch.nn.CrossEntropyLoss()
                            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                            if not torch.isnan(loss) and not torch.isinf(loss):
                                total_loss += loss.item()
                                num_batches += 1
                        else:
                            # Log information about what's missing
                            self.logger.debug(f"Validation batch {batch_idx} missing logits or labels")
                            if not hasattr(outputs, 'logits'):
                                self.logger.debug("  Missing logits in outputs")
                            if 'labels' not in batch:
                                self.logger.debug("  Missing labels in batch")
                except Exception as e:
                    self.logger.warning(f"Failed to validate on batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def save_checkpoint(self, path: str):
        """
        Save a training checkpoint.
        
        Args:
            path: Path to save the checkpoint.
        """
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'config': self.config.dict(),
                'device': str(self.device)
            }
            
            torch.save(checkpoint, path)
            self.logger.debug(f"Checkpoint saved to {path}")
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, path: str):
        """
        Load a training checkpoint.
        
        Args:
            path: Path to the checkpoint file.
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)
            
            self.logger.debug(f"Checkpoint loaded from {path}")
            self.logger.debug(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def close_monitoring(self):
        """Close monitoring backend connections."""
        if self.monitoring_backend:
            try:
                if hasattr(self.monitoring_backend, 'finish'):
                    self.monitoring_backend.finish()
                elif hasattr(self.monitoring_backend, 'end_run'):
                    self.monitoring_backend.end_run()
                elif hasattr(self.monitoring_backend, 'close'):
                    self.monitoring_backend.close()
                self.logger.debug("Monitoring backend closed successfully")
            except Exception as e:
                self.logger.warning(f"Failed to close monitoring backend: {e}")