"""
Image processor for Bangkong LLM Training System
"""

import torch
import numpy as np
from typing import Any, List, Union
from PIL import Image
from .base_processor import DataProcessor
from ...config.schemas import BangkongConfig


class ImageProcessor(DataProcessor):
    """Processor for image data."""
    
    def load(self, path: str) -> Image.Image:
        """
        Load image data from a file.
        
        Args:
            path: Path to the image file.
            
        Returns:
            PIL Image object.
        """
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image from {path}: {e}")
    
    def preprocess(self, data: Image.Image) -> torch.Tensor:
        """
        Preprocess image data.
        
        Args:
            data: PIL Image object.
            
        Returns:
            Preprocessed image tensor.
        """
        try:
            # Resize image to model input size
            target_size = (224, 224)  # Default for many vision models
            data = data.resize(target_size)
            
            # Convert to numpy array
            img_array = np.array(data, dtype=np.float32)
            
            # Normalize to [0, 1]
            img_array = img_array / 255.0
            
            # Convert to PyTorch tensor
            tensor = torch.from_numpy(img_array)
            
            # Rearrange dimensions to (C, H, W)
            tensor = tensor.permute(2, 0, 1)
            
            return tensor
        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {e}")
    
    def validate(self, data: torch.Tensor) -> bool:
        """
        Validate image data.
        
        Args:
            data: Image tensor to validate.
            
        Returns:
            True if data is valid, False otherwise.
        """
        if not isinstance(data, torch.Tensor):
            return False
            
        # Check if tensor has 3 dimensions (C, H, W)
        if len(data.shape) != 3:
            return False
            
        # Check if first dimension is 3 (RGB channels)
        if data.shape[0] != 3:
            return False
            
        return True