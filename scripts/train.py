#!/usr/bin/env python3
"""
Training script for Bangkong LLM Training System
"""

import argparse
import sys
import os
import json
import warnings
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
def setup_logging(log_file_path=None, log_level="INFO"):
    """Set up logging configuration."""
    # Create log directory if it doesn't exist
    if log_file_path:
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a specific logger for the main script
    logger = logging.getLogger("bangkong.main")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear any existing handlers to prevent duplicates
    logger.handlers.clear()
    
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
        console_handler = logging.StreamHandler(sys.stdout)
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log file path is provided
    if log_file_path:
        try:
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create file handler: {e}")
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    return logger
    
    return logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from bangkong.config.loader import ConfigLoader
from bangkong.hardware.detector import HardwareDetector
from bangkong.data.pipeline import DataPipeline
from bangkong.models.trainer import DynamicTrainer
from bangkong.models.training_manager import TrainingManager
from bangkong.models.multimodal import create_multimodal_model, MultimodalGPT2Model
from bangkong.models.regional import load_region_specific_model, create_multilingual_tokenizer
from bangkong.models.specialized import create_specialized_model, CodeGPT2Model, MathGPT2Model, ScientificGPT2Model
from bangkong.models.intelligent_init import apply_intelligent_initialization
from bangkong.monitoring.tracker import ResourceTracker
from bangkong.utils.path_manager import PathManager
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config


# Suppress PyTorch warnings about old GPUs
warnings.filterwarnings("ignore", message=".*Found GPU.*which is of cuda capability.*")

# Suppress Transformers warnings about unrecognized config parameters
warnings.filterwarnings("ignore", message=".*loss_type=None.*")
warnings.filterwarnings("ignore", message=".*unrecognized.*")

# Suppress tqdm-related warnings
warnings.filterwarnings("ignore", message=".*tqdm.*")

# Additional filter for the specific warning we're seeing
warnings.filterwarnings("ignore", message="`loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.")

# Even more specific filter
import transformers
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


def find_default_data_path():
    """Find default data path based on common locations."""
    path_manager = PathManager()
    
    # Check common data paths
    common_paths = [
        "./data/train",
        "./data/training",
        "./data",
        "../data",
        "~/data/bangkong",
        os.environ.get("BANGKONG_DATA_PATH", "")
    ]
    
    for path_str in common_paths:
        if path_str:
            try:
                path = path_manager.resolve_path(path_str)
                if path.exists():
                    return str(path)
            except:
                continue
    
    return None


def find_default_output_path():
    """Find default output path based on common locations."""
    path_manager = PathManager()
    
    # Check common output paths
    common_paths = [
        "./models",
        "./outputs",
        "../models",
        "~/models/bangkong",
        os.environ.get("BANGKONG_MODELS_PATH", "")
    ]
    
    for path_str in common_paths:
        if path_str:
            try:
                path = path_manager.resolve_path(path_str)
                # Create directory if it doesn't exist
                path.mkdir(parents=True, exist_ok=True)
                return str(path)
            except:
                continue
    
    # Default to current directory
    return "./models"


def load_jsonl_data(file_path):
    """
    Load data from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of data samples
    """
    samples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        continue
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error reading file {file_path}: {e}")
        return []
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loaded {len(samples)} samples from {file_path}")
    return samples


def _create_fallback_model(config):
    """Create a fallback model when specific model loading fails."""
    # Create model config without unrecognized parameters
    model_kwargs = {
        'vocab_size': config.model.vocab_size,
        'n_positions': min(config.model.sequence_length, config.model.max_sequence_length),
        'n_embd': config.model.hidden_size,
        'n_layer': config.model.num_layers,
        'n_head': config.model.num_heads
    }
    
    # Filter out any None values or unrecognized parameters
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
    
    # Additional filtering to prevent unrecognized parameter warnings
    recognized_params = {
        'activation_function', 'architectures', 'attn_pdrop', 'bos_token_id',
        'dtype', 'embd_pdrop', 'eos_token_id', 'initializer_range',
        'layer_norm_epsilon', 'model_type', 'n_embd', 'n_head', 'n_inner',
        'n_layer', 'n_positions', 'reorder_and_upcast_attn', 'resid_pdrop',
        'scale_attn_by_inverse_layer_idx', 'scale_attn_weights',
        'summary_activation', 'summary_first_dropout', 'summary_proj_to_labels',
        'summary_type', 'summary_use_proj', 'transformers_version', 'use_cache',
        'vocab_size'
    }
    
    # Filter to only recognized parameters
    filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in recognized_params}
    
    model_config = GPT2Config(**filtered_kwargs)
    model = GPT2LMHeadModel(model_config)
    
    # Explicitly set the loss type to prevent warnings
    try:
        model.config.loss_type = "ForCausalLMLoss"
    except:
        pass  # If it can't be set, that's fine
    
    return model


class TextDataset(Dataset):
    """Custom dataset for text data."""
    
    def __init__(self, texts, tokenizer, max_length=1024):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text samples
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Encoded sample as a dictionary
        """
        text = self.texts[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # For language modeling, labels are the same as input_ids
        # Return the input_ids, attention_mask, and labels
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()  # Labels for language modeling
        }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train a language model with Bangkong")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--data-path", type=str, help="Path to training data (JSONL file)")
    parser.add_argument("--output-path", type=str, help="Path to save trained model")
    parser.add_argument("--training-mode", type=str, choices=["fresh", "resume", "continue", "fine-tune"], 
                       help="Training mode: fresh, resume, continue, or fine-tune")
    parser.add_argument("--model-path", type=str, help="Path to existing model for resume/continue/fine-tune")
    parser.add_argument("--checkpoint-path", type=str, help="Path to checkpoint for resume mode")
    
    args = parser.parse_args()
    
    # Initialize training manager
    training_manager = TrainingManager(os.environ.get("BANGKONG_MODELS_PATH", "./models"))
    
    # Determine training mode
    if args.training_mode:
        training_mode = args.training_mode
        selected_model = {"path": args.model_path} if args.model_path else None
        selected_checkpoint = {"path": args.checkpoint_path} if args.checkpoint_path else None
    else:
        # Interactive mode
        # Set up logging first
        log_file = os.environ.get("BANGKONG_LOG_PATH", "./logs/bangkong.log")
        log_level = os.environ.get("BANGKONG_LOG_LEVEL", "INFO")
        logger = setup_logging(log_file, log_level)
        logger.info("Welcome to Bangkong LLM Training System!")
        training_mode, selected_model, selected_checkpoint = training_manager.interactive_training_menu()
        logger.info(f"Selected mode: {training_mode}")
        if selected_model:
            logger.info(f"Selected model: {selected_model['name']}")
        if selected_checkpoint:
            logger.info(f"Selected checkpoint: {selected_checkpoint['name']}")
    
    # Set up logging (always do this regardless of mode)
    log_file = os.environ.get("BANGKONG_LOG_PATH", "./logs/bangkong.log")
    log_level = os.environ.get("BANGKONG_LOG_LEVEL", "INFO")
    logger = setup_logging(log_file, log_level)
    logger.info("Starting Bangkong LLM Training System")
    logger.info(f"Training mode: {training_mode}")
    
    # Load configuration
    try:
        config_loader = ConfigLoader(args.config)
        config = config_loader.config
        logger.info(f"Configuration loaded successfully from {args.config or 'auto-detected path'}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Detect hardware
    try:
        hardware_info = HardwareDetector.get_available_resources()
        logger.info(f"Available hardware: {hardware_info}")
    except Exception as e:
        logger.warning(f"Could not detect hardware: {e}")
        hardware_info = {}
    
    # Initialize resource tracker
    try:
        tracker = ResourceTracker()
    except Exception as e:
        logger.warning(f"Could not initialize resource tracker: {e}")
        tracker = None
    
    # Load training data
    data_samples = []
    if args.data_path:
        if os.path.exists(args.data_path):
            if args.data_path.endswith('.jsonl'):
                logger.info(f"Loading training data from JSONL file: {args.data_path}")
                data_samples = load_jsonl_data(args.data_path)
                if not data_samples:
                    logger.warning("No valid data samples found in JSONL file")
            else:
                logger.warning(f"Unsupported file format: {args.data_path}")
                logger.info("Please provide a JSONL file for training")
        else:
            logger.warning(f"Data file not found: {args.data_path}")
    else:
        logger.info("No data path provided. Looking for JSONL files in processed data directory...")
        # Look for JSONL files in data/processed
        processed_dir = Path("./data/processed")
        if processed_dir.exists():
            jsonl_files = list(processed_dir.glob("*.jsonl"))
            if jsonl_files:
                # Use the first JSONL file found
                first_file = jsonl_files[0]
                logger.info(f"Using training data from: {first_file}")
                data_samples = load_jsonl_data(str(first_file))
            else:
                logger.warning("No JSONL files found in data/processed directory")
        else:
            logger.warning("Processed data directory not found")
    
    if not data_samples:
        logger.warning("No training data available. Proceeding with demo mode.")
        # Create dummy data for demonstration
        data_samples = [
            {"text": "This is a sample text for training."},
            {"text": "Another example sentence for the model to learn from."},
            {"text": "The quick brown fox jumps over the lazy dog."}
        ]
        logger.info("Using dummy data for demonstration.")
    
    # Extract text from samples (assuming they have a 'text' field)
    texts = [sample["text"] for sample in data_samples if "text" in sample]
    if not texts:
        logger.error("No valid text data found in samples")
        sys.exit(1)
    
    logger.info(f"Loaded {len(texts)} text samples for training")
    
    # Initialize tokenizer
    try:
        if training_mode in ["resume", "continue", "fine-tune"] and selected_model:
            # Try to load tokenizer from existing model
            try:
                tokenizer = GPT2Tokenizer.from_pretrained(selected_model["path"])
                logger.info(f"Loaded tokenizer from existing model: {selected_model['path']}")
            except:
                # Fall back to default tokenizer
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                logger.info("Using default GPT-2 tokenizer")
        else:
            # Use appropriate tokenizer for fresh training
            # Check if we need a regional/multilingual tokenizer
            if config.model.region != "global" or config.model.primary_language != "en":
                try:
                    tokenizer = create_multilingual_tokenizer(config)
                    logger.info("Using multilingual tokenizer")
                except Exception as tokenizer_error:
                    logger.warning(f"Failed to create multilingual tokenizer: {tokenizer_error}")
                    # Fallback to default tokenizer
                    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                    logger.info("Using default GPT-2 tokenizer")
            else:
                # Use default tokenizer for fresh training
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                logger.info("Using default GPT-2 tokenizer for fresh training")
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Error initializing tokenizer: {e}")
        sys.exit(1)
    
    # Initialize model
    model = None
    try:
        if training_mode in ["resume", "continue", "fine-tune"] and selected_model:
            # Load existing model
            try:
                model = GPT2LMHeadModel.from_pretrained(selected_model["path"])
                logger.info(f"Loaded model from existing model: {selected_model['path']}")
            except Exception as load_error:
                logger.warning(f"Could not load model from {selected_model['path']}: {load_error}")
                logger.info("Creating new model instead...")
                # Check if we need a regional model
                if config.model.region != "global" or config.model.name in ["qwen", "glm", "chatglm", "baichuan", "broda", "zecha"]:
                    try:
                        model, _ = load_region_specific_model(config)
                        logger.info(f"Loaded region-specific model: {config.model.name}")
                    except Exception as region_error:
                        logger.warning(f"Failed to load region-specific model: {region_error}")
                        # Fallback to other model types
                        model = _create_fallback_model(config)
                # Check if we need a specialized model
                elif config.model.domain != "general":
                    try:
                        model = create_specialized_model(config)
                        logger.info(f"Created specialized model for domain: {config.model.domain}")
                    except Exception as specialized_error:
                        logger.warning(f"Failed to create specialized model: {specialized_error}")
                        # Fallback to other model types
                        model = _create_fallback_model(config)
                # Check if we need a multimodal model
                elif config.model.modality == "multimodal":
                    model = create_multimodal_model(config)
                    logger.info("Created multimodal model")
                else:
                    model = _create_fallback_model(config)
        else:
            # Create new model for fresh training
            # Check if we need a regional model
            if config.model.region != "global" or config.model.name in ["qwen", "glm", "chatglm", "baichuan", "broda", "zecha"]:
                try:
                    model, _ = load_region_specific_model(config)
                    logger.info(f"Loaded region-specific model: {config.model.name}")
                except Exception as region_error:
                    logger.warning(f"Failed to load region-specific model: {region_error}")
                    # Fallback to other model types
                    model = _create_fallback_model(config)
            # Check if we need a specialized model
            elif config.model.domain != "general":
                try:
                    model = create_specialized_model(config)
                    logger.info(f"Created specialized model for domain: {config.model.domain}")
                except Exception as specialized_error:
                    logger.warning(f"Failed to create specialized model: {specialized_error}")
                    # Fallback to other model types
                    model = _create_fallback_model(config)
            # Check if we need a multimodal model
            elif config.model.modality == "multimodal":
                model = create_multimodal_model(config)
                logger.info("Created multimodal model")
            else:
                model = _create_fallback_model(config)
        
        # Apply intelligent initialization if specified
        initialization_strategy = getattr(config.model, 'initialization_strategy', 'random')
        if initialization_strategy != 'random':
            try:
                model = apply_intelligent_initialization(model, config)
                logger.info(f"Applied intelligent initialization: {initialization_strategy}")
            except Exception as init_error:
                logger.warning(f"Failed to apply intelligent initialization: {init_error}")
        
        logger.info(f"Initialized model with architecture: {config.model.architecture}")
        logger.info(f"Model modality: {config.model.modality}")
        logger.info(f"Model region: {config.model.region}")
        logger.info(f"Model domain: {config.model.domain}")
        logger.info(f"Initialization strategy: {initialization_strategy}")
        logger.debug(f"Model config: {model.config}")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Initialize trainer
    try:
        trainer = DynamicTrainer(model, config)
        
        # Load checkpoint if resuming
        if training_mode == "resume" and selected_checkpoint:
            try:
                trainer.load_checkpoint(selected_checkpoint["path"])
                logger.info(f"Resumed training from checkpoint: {selected_checkpoint['path']}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
    except Exception as e:
        logger.error(f"Error initializing trainer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create dataset
    # Use the model's sequence length if available, otherwise use config
    if training_mode in ["resume", "continue", "fine-tune"] and hasattr(model.config, 'n_positions'):
        sequence_length = model.config.n_positions
        logger.info(f"Using model sequence length: {sequence_length}")
    else:
        sequence_length = config.model.sequence_length
        logger.info(f"Using config sequence length: {sequence_length}")
    
    dataset = TextDataset(
        texts, 
        tokenizer, 
        max_length=sequence_length
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Create data loader
    batch_size = trainer.resources['batch_size']  # Use the batch size from resource allocator
    logger.info(f"Using batch size: {batch_size}")
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    logger.info(f"Number of batches in dataloader: {len(dataloader)}")
    
    # Debug: Check the first batch
    try:
        first_batch = next(iter(dataloader))
        logger.info(f"First batch keys: {list(first_batch.keys())}")
        for key, value in first_batch.items():
            logger.info(f"  {key}: shape = {value.shape}")
    except Exception as e:
        logger.warning(f"Error examining first batch: {e}")
    
    # Determine output path
    output_path = args.output_path
    if not output_path:
        logger.info("No output path provided, attempting to auto-detect...")
        output_path = find_default_output_path()
        logger.info(f"Using auto-detected output path: {output_path}")
    
    # Create output directory
    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)
    
    # Train the model
    logger.info("Starting training...")
    logger.info(f"Dataloader length: {len(dataloader)}")
    logger.info(f"Dataloader batch size: {dataloader.batch_size}")
    
    try:
        trainer.train(dataloader)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save the trained model
    try:
        # Create a unique output directory to avoid file locking issues
        import time
        timestamp = int(time.time())
        model_save_path = output_path_obj / f"trained_model_{timestamp}"
        model_save_path.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
        # Save checkpoint
        checkpoint_path = model_save_path / "checkpoints" / "final_checkpoint.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(checkpoint_path))
        logger.info(f"Final checkpoint saved to {checkpoint_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
    
    # Print resource summary
    if tracker:
        try:
            summary = tracker.get_resource_summary()
            logger.info(f"Resource usage summary: {summary}")
        except Exception as e:
            logger.warning(f"Could not get resource summary: {e}")
    else:
        logger.info("Resource usage summary: Resource tracker not available")


if __name__ == "__main__":
    main()