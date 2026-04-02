#!/usr/bin/env python3
"""
Deployment script for Bangkong LLM Training System
"""

import argparse
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bangkong.config.loader import ConfigLoader
from bangkong.deployment.manager import DeploymentManager
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


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy model with Bangkong")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model-path", type=str, help="Path to the model to deploy")
    parser.add_argument("--target", type=str, default="local", 
                       help="Deployment target (local, cloud, hybrid)")
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.config
    
    # Initialize deployment manager
    deployment_manager = DeploymentManager(config)
    
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
    
    # Deploy model
    print(f"Deploying model from {model_path} to {args.target}")
    
    try:
        success = deployment_manager.deploy(model_path, args.target)
        if success:
            print("Model deployment completed successfully")
        else:
            print("Model deployment failed")
            sys.exit(1)
    except Exception as e:
        print(f"Deployment error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()