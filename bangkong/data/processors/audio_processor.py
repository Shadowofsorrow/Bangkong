"""
Audio processor for Bangkong LLM Training System
"""

import librosa
import numpy as np
from typing import List, Union, Tuple
from .base_processor import DataProcessor
from ...config.schemas import BangkongConfig


class AudioProcessor(DataProcessor):
    """Processor for audio data."""
    
    def load(self, path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio data from a file.
        
        Args:
            path: Path to the audio file.
            
        Returns:
            Tuple of (audio_data, sample_rate).
        """
        try:
            # Load audio file
            audio_data, sample_rate = librosa.load(path, sr=None)
            return audio_data, sample_rate
        except Exception as e:
            raise ValueError(f"Failed to load audio file {path}: {e}")
    
    def preprocess(self, data: Tuple[np.ndarray, int]) -> Tuple[np.ndarray, int]:
        """
        Preprocess audio data.
        
        Args:
            data: Tuple of (audio_data, sample_rate).
            
        Returns:
            Preprocessed audio data.
        """
        audio_data, sample_rate = data
        
        # Resample if needed
        target_sample_rate = getattr(self.config.model, 'audio_sample_rate', sample_rate)
        if target_sample_rate and sample_rate != target_sample_rate:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
            sample_rate = target_sample_rate
        
        # Normalize audio
        audio_data = librosa.util.normalize(audio_data)
        
        # Apply noise reduction if configured
        # This is a simple noise reduction - in practice, more sophisticated methods would be used
        if hasattr(self.config.data.preprocessing, 'reduce_noise') and self.config.data.preprocessing.reduce_noise:
            # Simple noise reduction by applying a noise gate
            threshold = np.mean(np.abs(audio_data)) * 0.1  # 10% of mean amplitude
            audio_data[np.abs(audio_data) < threshold] = 0
        
        return audio_data, sample_rate
    
    def validate(self, data: Tuple[np.ndarray, int]) -> bool:
        """
        Validate audio data.
        
        Args:
            data: Audio data to validate.
            
        Returns:
            True if data is valid, False otherwise.
        """
        if not isinstance(data, tuple) or len(data) != 2:
            return False
            
        audio_data, sample_rate = data
        
        if not isinstance(audio_data, np.ndarray) or not isinstance(sample_rate, int):
            return False
            
        # Check if audio data is not empty
        if audio_data.size == 0:
            return False
            
        # Check if sample rate is positive
        if sample_rate <= 0:
            return False
            
        return True