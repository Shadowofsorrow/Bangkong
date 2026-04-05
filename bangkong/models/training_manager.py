"""
Training management system for Bangkong LLM Training System
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import torch
from datetime import datetime


class TrainingManager:
    """Manages different training modes: fresh training, resume, continue, fine-tune."""
    
    def __init__(self, base_models_path: str = "./models"):
        """
        Initialize the training manager.
        
        Args:
            base_models_path: Base path where models are stored
        """
        self.base_models_path = Path(base_models_path)
        self.base_models_path.mkdir(parents=True, exist_ok=True)
    
    def list_available_models(self) -> List[Dict[str, str]]:
        """
        List all available trained models.
        
        Returns:
            List of dictionaries containing model information
        """
        models = []
        
        # Look for model directories
        for item in self.base_models_path.iterdir():
            if item.is_dir():
                # Check if it's a trained model (has config and model files)
                if (item / "config.json").exists() or (item / "pytorch_model.bin").exists():
                    model_info = {
                        'name': item.name,
                        'path': str(item),
                        'type': 'trained_model',
                        'timestamp': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    }
                    models.append(model_info)
                elif (item / "model_metadata.json").exists():
                    model_info = {
                        'name': item.name,
                        'path': str(item),
                        'type': 'packaged_model',
                        'timestamp': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    }
                    models.append(model_info)
        
        return sorted(models, key=lambda x: x['timestamp'], reverse=True)
    
    def list_checkpoints(self, model_path: str) -> List[Dict[str, str]]:
        """
        List all checkpoints for a specific model.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            List of dictionaries containing checkpoint information
        """
        checkpoints = []
        model_dir = Path(model_path)
        checkpoints_dir = model_dir / "checkpoints"
        
        # Check main model directory for checkpoints
        if checkpoints_dir.exists():
            for item in checkpoints_dir.iterdir():
                if item.suffix == '.pt' or item.suffix == '.pth':
                    checkpoint_info = {
                        'name': item.name,
                        'path': str(item),
                        'timestamp': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    }
                    checkpoints.append(checkpoint_info)
        
        # Also check model directory directly
        for item in model_dir.iterdir():
            if (item.suffix == '.pt' or item.suffix == '.pth') and 'checkpoint' in item.name.lower():
                checkpoint_info = {
                    'name': item.name,
                    'path': str(item),
                    'timestamp': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                }
                # Avoid duplicates
                if not any(cp['path'] == str(item) for cp in checkpoints):
                    checkpoints.append(checkpoint_info)
        
        return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)
    
    def get_training_mode_menu(self) -> str:
        """
        Get the training mode selection menu.
        
        Returns:
            Menu string for training mode selection
        """
        menu = """
=====================================
Bangkong LLM Training Manager
=====================================

Select training mode:
1. Fresh Training (Start new model from scratch)
2. Resume Training (Continue from checkpoint)
3. Continue Training (Add more epochs to completed model)
4. Fine-tune Model (Adapt existing model to new data)

Enter your choice (1-4): """
        return menu
    
    def get_model_selection_menu(self, models: List[Dict[str, str]]) -> str:
        """
        Get the model selection menu.
        
        Args:
            models: List of available models
            
        Returns:
            Menu string for model selection
        """
        if not models:
            return "No models available.\n"
        
        menu = "\nAvailable Models:\n"
        menu += "-" * 50 + "\n"
        for i, model in enumerate(models, 1):
            menu += f"{i}. {model['name']} ({model['type']})\n"
            menu += f"   Path: {model['path']}\n"
            menu += f"   Last modified: {model['timestamp']}\n\n"
        
        menu += f"Select model (1-{len(models)}) or 0 to go back: "
        return menu
    
    def get_checkpoint_selection_menu(self, checkpoints: List[Dict[str, str]]) -> str:
        """
        Get the checkpoint selection menu.
        
        Args:
            checkpoints: List of available checkpoints
            
        Returns:
            Menu string for checkpoint selection
        """
        if not checkpoints:
            return "No checkpoints available.\n"
        
        menu = "\nAvailable Checkpoints:\n"
        menu += "-" * 50 + "\n"
        for i, checkpoint in enumerate(checkpoints, 1):
            menu += f"{i}. {checkpoint['name']}\n"
            menu += f"   Path: {checkpoint['path']}\n"
            menu += f"   Created: {checkpoint['timestamp']}\n\n"
        
        menu += f"Select checkpoint (1-{len(checkpoints)}) or 0 to go back: "
        return menu
    
    def interactive_training_menu(self):
        """
        Run the interactive training menu system.
        
        Returns:
            Tuple of (mode, selected_model, selected_checkpoint)
        """
        while True:
            # Get training mode
            print(self.get_training_mode_menu())
            try:
                choice = input().strip()
                mode_choice = int(choice)
                
                if mode_choice == 1:
                    mode = "fresh"
                    return mode, None, None
                elif mode_choice in [2, 3, 4]:
                    mode = ["resume", "continue", "fine-tune"][mode_choice - 2]
                    break
                else:
                    print("Invalid choice. Please select 1-4.\n")
                    continue
            except ValueError:
                print("Invalid input. Please enter a number.\n")
                continue
        
        # Get model selection for non-fresh training
        models = self.list_available_models()
        if not models:
            print("No trained models available. Starting fresh training instead.")
            return "fresh", None, None
        
        while True:
            print(self.get_model_selection_menu(models))
            try:
                choice = input().strip()
                model_choice = int(choice)
                
                if model_choice == 0:
                    # Go back to mode selection
                    return self.interactive_training_menu()
                elif 1 <= model_choice <= len(models):
                    selected_model = models[model_choice - 1]
                    break
                else:
                    print(f"Invalid choice. Please select 0-{len(models)}.\n")
                    continue
            except ValueError:
                print("Invalid input. Please enter a number.\n")
                continue
        
        # For resume training, get checkpoint selection
        if mode == "resume":
            checkpoints = self.list_checkpoints(selected_model['path'])
            if not checkpoints:
                print("No checkpoints available for this model. Starting fresh training instead.")
                return "fresh", None, None
            
            while True:
                print(self.get_checkpoint_selection_menu(checkpoints))
                try:
                    choice = input().strip()
                    checkpoint_choice = int(choice)
                    
                    if checkpoint_choice == 0:
                        # Go back to model selection
                        return self.interactive_training_menu()
                    elif 1 <= checkpoint_choice <= len(checkpoints):
                        selected_checkpoint = checkpoints[checkpoint_choice - 1]
                        return mode, selected_model, selected_checkpoint
                    else:
                        print(f"Invalid choice. Please select 0-{len(checkpoints)}.\n")
                        continue
                except ValueError:
                    print("Invalid input. Please enter a number.\n")
                    continue
        else:
            # For continue and fine-tune, no checkpoint needed
            return mode, selected_model, None
    
    def prepare_model_for_training(self, mode: str, model_info: Optional[Dict] = None, 
                                 checkpoint_info: Optional[Dict] = None):
        """
        Prepare model based on training mode.
        
        Args:
            mode: Training mode (fresh, resume, continue, fine-tune)
            model_info: Selected model information
            checkpoint_info: Selected checkpoint information
            
        Returns:
            Tuple of (model, tokenizer, start_epoch)
        """
        if mode == "fresh":
            # Create new model and tokenizer
            return self._create_fresh_model()
        elif mode == "resume":
            # Load from checkpoint
            return self._load_from_checkpoint(model_info, checkpoint_info)
        elif mode == "continue":
            # Load completed model and continue training
            return self._load_completed_model(model_info)
        elif mode == "fine-tune":
            # Load completed model for fine-tuning
            return self._load_for_fine_tuning(model_info)
    
    def _create_fresh_model(self):
        """Create a fresh model and tokenizer."""
        # This would be implemented based on the specific model type
        # For now, we'll return None to indicate it should be handled elsewhere
        return None, None, 0
    
    def _load_from_checkpoint(self, model_info: Dict, checkpoint_info: Dict):
        """Load model from checkpoint for resuming training."""
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_info['path'])
            
            # Extract information
            start_epoch = checkpoint.get('epoch', 0)
            
            # The model and tokenizer would be loaded by the calling code
            return None, None, start_epoch
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None, None, 0
    
    def _load_completed_model(self, model_info: Dict):
        """Load completed model for continuing training."""
        try:
            # For continuing training, we start from epoch 0 but with a pre-trained model
            return None, None, 0
        except Exception as e:
            print(f"Error loading completed model: {e}")
            return None, None, 0
    
    def _load_for_fine_tuning(self, model_info: Dict):
        """Load model for fine-tuning."""
        try:
            # For fine-tuning, we start from epoch 0 but with a pre-trained model
            # The learning rate and other parameters might be adjusted
            return None, None, 0
        except Exception as e:
            print(f"Error loading model for fine-tuning: {e}")
            return None, None, 0