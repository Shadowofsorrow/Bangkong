"""
Cross-modal dataset processor for Bangkong LLM Training System
"""

import json
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path
from ...config.schemas import BangkongConfig
from ...data.processors.base_processor import DataProcessor


class CrossModalDatasetProcessor(DataProcessor):
    """Processor for cross-modal datasets (text-image, text-audio, etc.)."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize cross-modal dataset processor.
        
        Args:
            config: Bangkong configuration
        """
        super().__init__(config)
        self.modalities = getattr(config.model, 'cross_modalities', ['text', 'image'])
    
    def load(self, path: str) -> List[Dict[str, Any]]:
        """
        Load cross-modal data from a directory or JSON file.
        
        Args:
            path: Path to the cross-modal dataset.
            
        Returns:
            List of cross-modal data samples.
        """
        path_obj = Path(path)
        
        # If it's a JSON file, load it directly
        if path_obj.is_file() and path_obj.suffix == '.json':
            return self._load_from_json(path_obj)
        
        # If it's a directory, look for data files
        if path_obj.is_dir():
            return self._load_from_directory(path_obj)
        
        raise ValueError(f"Unsupported path type: {path}")
    
    def _load_from_json(self, path: Path) -> List[Dict[str, Any]]:
        """
        Load cross-modal data from a JSON file.
        
        Args:
            path: Path to the JSON file.
            
        Returns:
            List of cross-modal data samples.
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # If it's a list of samples, return as-is
        if isinstance(data, list):
            return data
        
        # If it's a dictionary, assume it has a 'samples' key
        if isinstance(data, dict) and 'samples' in data:
            return data['samples']
        
        raise ValueError(f"Unsupported JSON format in {path}")
    
    def _load_from_directory(self, path: Path) -> List[Dict[str, Any]]:
        """
        Load cross-modal data from a directory.
        
        Args:
            path: Path to the directory.
            
        Returns:
            List of cross-modal data samples.
        """
        samples = []
        
        # Look for a dataset manifest file
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            
            # Load samples based on manifest
            for item in manifest.get('samples', []):
                sample = self._load_sample_from_manifest_item(path, item)
                if sample:
                    samples.append(sample)
        else:
            # Try to infer structure from directory contents
            samples = self._infer_structure_from_directory(path)
        
        return samples
    
    def _load_sample_from_manifest_item(self, base_path: Path, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load a single sample from a manifest item.
        
        Args:
            base_path: Base path for the dataset.
            item: Manifest item describing the sample.
            
        Returns:
            Cross-modal data sample.
        """
        sample = {}
        
        # Load text data
        if 'text' in item:
            text_path = base_path / item['text']
            if text_path.exists():
                with open(text_path, 'r', encoding='utf-8') as f:
                    sample['text'] = f.read().strip()
        
        # Load image data
        if 'image' in item:
            image_path = base_path / item['image']
            if image_path.exists():
                sample['image_path'] = str(image_path)
        
        # Load audio data
        if 'audio' in item:
            audio_path = base_path / item['audio']
            if audio_path.exists():
                sample['audio_path'] = str(audio_path)
        
        # Load video data
        if 'video' in item:
            video_path = base_path / item['video']
            if video_path.exists():
                sample['video_path'] = str(video_path)
        
        # Add any additional metadata
        for key, value in item.items():
            if key not in ['text', 'image', 'audio', 'video']:
                sample[key] = value
        
        return sample if sample else None
    
    def _infer_structure_from_directory(self, path: Path) -> List[Dict[str, Any]]:
        """
        Infer dataset structure from directory contents.
        
        Args:
            path: Path to the directory.
            
        Returns:
            List of cross-modal data samples.
        """
        samples = []
        
        # Look for common patterns
        text_files = list(path.glob("*.txt")) + list(path.glob("*.md"))
        image_files = list(path.glob("*.jpg")) + list(path.glob("*.png")) + list(path.glob("*.jpeg"))
        audio_files = list(path.glob("*.wav")) + list(path.glob("*.mp3"))
        video_files = list(path.glob("*.mp4")) + list(path.glob("*.avi"))
        
        # Try to match files by name patterns
        # This is a simplified approach - in practice, more sophisticated matching would be used
        for text_file in text_files:
            sample = {'text_path': str(text_file)}
            
            # Look for matching media files
            base_name = text_file.stem
            for image_file in image_files:
                if base_name in image_file.stem:
                    sample['image_path'] = str(image_file)
                    break
            
            for audio_file in audio_files:
                if base_name in audio_file.stem:
                    sample['audio_path'] = str(audio_file)
                    break
            
            for video_file in video_files:
                if base_name in video_file.stem:
                    sample['video_path'] = str(video_file)
                    break
            
            samples.append(sample)
        
        return samples
    
    def preprocess(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess cross-modal data.
        
        Args:
            data: List of cross-modal data samples.
            
        Returns:
            Preprocessed cross-modal data.
        """
        preprocessed = []
        
        for sample in data:
            processed_sample = {}
            
            # Process text data
            if 'text' in sample:
                processed_sample['text'] = self._preprocess_text(sample['text'])
            elif 'text_path' in sample:
                with open(sample['text_path'], 'r', encoding='utf-8') as f:
                    text = f.read()
                processed_sample['text'] = self._preprocess_text(text)
            
            # Process image data (paths are preserved, actual processing happens during training)
            if 'image_path' in sample:
                processed_sample['image_path'] = sample['image_path']
            
            # Process audio data
            if 'audio_path' in sample:
                processed_sample['audio_path'] = sample['audio_path']
            
            # Process video data
            if 'video_path' in sample:
                processed_sample['video_path'] = sample['video_path']
            
            # Add any other fields
            for key, value in sample.items():
                if key not in ['text', 'text_path', 'image_path', 'audio_path', 'video_path']:
                    processed_sample[key] = value
            
            preprocessed.append(processed_sample)
        
        return preprocessed
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text data.
        
        Args:
            text: Text to preprocess.
            
        Returns:
            Preprocessed text.
        """
        # Apply basic text cleaning
        import re
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        return text
    
    def validate(self, data: List[Dict[str, Any]]) -> bool:
        """
        Validate cross-modal data.
        
        Args:
            data: Cross-modal data to validate.
            
        Returns:
            True if data is valid, False otherwise.
        """
        if not isinstance(data, list):
            return False
        
        for sample in data:
            if not isinstance(sample, dict):
                return False
            
            # Check that at least one modality is present
            if not any(key in sample for key in ['text', 'text_path', 'image_path', 'audio_path', 'video_path']):
                return False
        
        return True


def create_cross_modal_processor(config: BangkongConfig) -> CrossModalDatasetProcessor:
    """
    Create a cross-modal dataset processor.
    
    Args:
        config: Bangkong configuration
        
    Returns:
        Cross-modal dataset processor
    """
    return CrossModalDatasetProcessor(config)