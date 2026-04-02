"""
Video processor for Bangkong LLM Training System
"""

import cv2
import numpy as np
from typing import List, Union, Tuple
from .base_processor import DataProcessor
from ...config.schemas import BangkongConfig


class VideoProcessor(DataProcessor):
    """Processor for video data."""
    
    def load(self, path: str) -> List[np.ndarray]:
        """
        Load video data from a file.
        
        Args:
            path: Path to the video file.
            
        Returns:
            List of video frames.
        """
        try:
            # Open video file
            cap = cv2.VideoCapture(path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            return frames
        except Exception as e:
            raise ValueError(f"Failed to load video file {path}: {e}")
    
    def preprocess(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """
        Preprocess video data.
        
        Args:
            data: List of video frames.
            
        Returns:
            Preprocessed video frames.
        """
        preprocessed_frames = []
        
        # Target frame size from config
        target_size = getattr(self.config.model, 'image_size', None)
        target_fps = getattr(self.config.model, 'video_fps', None)
        
        for i, frame in enumerate(data):
            # Resize frame if needed
            if target_size:
                frame = cv2.resize(frame, (target_size, target_size))
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            preprocessed_frames.append(frame)
        
        # If we need to adjust FPS, we can sample frames accordingly
        if target_fps and len(data) > 0:
            # This is a simplified approach - in practice, more sophisticated methods would be used
            pass
        
        return preprocessed_frames
    
    def validate(self, data: List[np.ndarray]) -> bool:
        """
        Validate video data.
        
        Args:
            data: Video data to validate.
            
        Returns:
            True if data is valid, False otherwise.
        """
        if not isinstance(data, list):
            return False
            
        if len(data) == 0:
            return False
            
        for frame in data:
            if not isinstance(frame, np.ndarray):
                return False
                
            # Check if frame has valid shape (height, width, channels)
            if len(frame.shape) != 3:
                return False
                
            # Check if frame is not empty
            if frame.size == 0:
                return False
                
        return True