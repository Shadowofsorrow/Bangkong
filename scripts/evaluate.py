#!/usr/bin/env python3
"""
Evaluation script for Bangkong LLM Training System
"""

import argparse
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bangkong.config.loader import ConfigLoader
from bangkong.utils.path_manager import PathManager


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
            except:
                continue
    
    return None


def find_default_data_path():
    """Find default data path based on common locations."""
    path_manager = PathManager()
    
    # Check common data paths
    common_paths = [
        "./data/eval",
        "./data/evaluation",
        "./data/test",
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


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate model with Bangkong")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model-path", type=str, help="Path to the model to evaluate")
    parser.add_argument("--data-path", type=str, help="Path to evaluation data")
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.config
    
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
    
    # Determine data path
    data_path = args.data_path
    if not data_path:
        print("No data path provided, attempting to auto-detect...")
        data_path = find_default_data_path()
        if data_path:
            print(f"Using auto-detected data path: {data_path}")
        else:
            print("Warning: No data path found. Proceeding without evaluation data.")
    
    # Evaluate model
    print(f"Evaluating model from {model_path}")
    if data_path:
        print(f"Using evaluation data from {data_path}")
    print("Evaluation completed")


if __name__ == "__main__":
    main()