#!/usr/bin/env python3
"""
Model conversion script for Bangkong LLM Training System
"""

import argparse
import sys
import os
import time
from pathlib import Path
import torch
from dotenv import load_dotenv
from requests.exceptions import HTTPError

# Load environment variables from .env file
load_dotenv()

# Set HF endpoint to mirror to avoid rate limiting
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def safe_from_pretrained(from_pretrained_func, *args, **kwargs):
    """
    Safely call a from_pretrained method with retry on HTTP 429 errors.
    """
    max_retries = 5
    base_delay = 1  # second
    for i in range(max_retries):
        try:
            return from_pretrained_func(*args, **kwargs)
        except Exception as e:
            # Check if it's a 429 error (rate limit)
            is_429 = False
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                if e.response.status_code == 429:
                    is_429 = True
            elif "429" in str(e) and ("Too Many Requests" in str(e) or "rate limit" in str(e).lower()):
                is_429 = True

            if is_429:
                if i == max_retries - 1:
                    raise  # Re-raise the last exception if we've exhausted retries
                delay = base_delay * (2 ** i)  # Exponential backoff
                print(f"Encountered 429 rate limit. Retrying in {delay} seconds... (attempt {i+1}/{max_retries})")
                time.sleep(delay)
            else:
                # Re-raise if it's not a 429 error
                raise

sys.path.insert(0, str(Path(__file__).parent.parent))

from bangkong.config.loader import ConfigLoader
from bangkong.models.packager import ModelPackager
from bangkong.utils.path_manager import PathManager
from transformers import AutoModel, AutoTokenizer


def find_default_model_path():
    """Find default model path based on common locations."""
    path_manager = PathManager()
    
    # Check common model paths
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
                # Look for model files in the directory
                if path.exists():
                    # Check for common model file patterns
                    for model_file in path.glob("*"):
                        if model_file.is_dir() or model_file.suffix in ['.pt', '.pth', '.bin']:
                            return str(path)
            except Exception:
                continue
    
    return None


def find_default_output_path(model_path):
    """Find default output path based on model path."""
    if model_path:
        # Create converted_models directory next to the model
        base_path = Path(model_path).parent
        converted_path = base_path / "converted_models"
        converted_path.mkdir(parents=True, exist_ok=True)
        return str(converted_path)
    
    # Fallback to default
    path_manager = PathManager()
    common_paths = [
        "./converted_models",
        "./outputs/converted",
        "../converted_models",
        "~/models/bangkong_converted"
    ]
    
    for path_str in common_paths:
        if path_str:
            try:
                path = path_manager.resolve_path(path_str)
                path.mkdir(parents=True, exist_ok=True)
                return str(path)
            except Exception:
                continue
    
    return "./converted_models"


def load_model_and_tokenizer(model_path):
    """
    Load model and tokenizer from a path.

    Args:
        model_path: Path to the model directory

    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        # Try to load as a Hugging Face model
        model = safe_from_pretrained(AutoModel.from_pretrained, model_path)
        tokenizer = safe_from_pretrained(AutoTokenizer.from_pretrained, model_path)
        return model, tokenizer
    except Exception as e:
        print(f"Warning: Could not load model as Hugging Face model: {e}")

        # Try to load as a PyTorch model
        try:
            # Load model state dict
            model_state_path = Path(model_path) / "model.pt"
            if model_state_path.exists():
                # For this example, we'll assume GPT-2 architecture
                from transformers import GPT2Model, GPT2Config
                config = GPT2Config()
                model = GPT2Model(config)
                model.load_state_dict(torch.load(model_state_path))

                # Try to load tokenizer
                try:
                    tokenizer = safe_from_pretrained(AutoTokenizer.from_pretrained, model_path)
                except Exception:
                    # Create a default GPT-2 tokenizer
                    tokenizer = safe_from_pretrained(AutoTokenizer.from_pretrained, 'gpt2')
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token

                return model, tokenizer
        except Exception as e2:
            print(f"Error loading model: {e2}")
            return None, None

    return None, None


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description="Convert model to different formats with Bangkong")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model-path", type=str, help="Path to the model to convert")
    parser.add_argument("--output-path", type=str, help="Path to save converted model")
    parser.add_argument("--formats", type=str, nargs="+", 
                       help="Formats to convert to (e.g., onnx safetensors)")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config_loader = ConfigLoader(args.config)
        config = config_loader.config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Initialize packager
    packager = ModelPackager(config)
    
    # Determine model path
    model_path = args.model_path
    if not model_path:
        print("No model path provided, attempting to auto-detect...")
        model_path = find_default_model_path()
        if model_path:
            print(f"Using auto-detected model path: {model_path}")
        else:
            print("Error: No model path found. Please provide a model path.")
            sys.exit(1)
    
    # Check if model path exists
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        print(f"Error: Model path does not exist: {model_path}")
        sys.exit(1)
    
    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    if model is None:
        print("Error: Could not load model")
        sys.exit(1)
    
    print(f"Successfully loaded model of type: {type(model).__name__}")
    
    # Determine output path
    output_path = args.output_path
    if not output_path:
        print("No output path provided, attempting to auto-detect...")
        output_path = find_default_output_path(model_path)
        print(f"Using auto-detected output path: {output_path}")
    
    # Determine formats
    formats = args.formats or config.packaging.default_formats
    if not formats:
        formats = ["pytorch"]  # Default fallback
    
    # Convert model
    print(f"Converting model from {model_path} to {output_path}")
    print(f"Target formats: {formats}")
    
    try:
        packager.package_model(model, tokenizer, output_path, formats)
        print("Model conversion completed successfully")
    except Exception as e:
        print(f"Error during model conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()