"""
Text processor for Bangkong LLM Training System
"""

import re
from typing import List, Union
from .base_processor import DataProcessor
from ...config.schemas import BangkongConfig


class TextProcessor(DataProcessor):
    """Processor for text data."""
    
    def load(self, path: str) -> List[str]:
        """
        Load text data from a file.
        
        Args:
            path: Path to the text file.
            
        Returns:
            List of text lines.
        """
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def preprocess(self, data: List[str]) -> List[str]:
        """
        Preprocess text data.
        
        Args:
            data: List of text strings.
            
        Returns:
            Preprocessed text data.
        """
        preprocessed = []
        min_length = self.config.data.preprocessing.min_text_length
        max_length = self.config.data.preprocessing.max_text_length
        
        for text in data:
            # Filter by length
            if min_length <= len(text) <= max_length:
                # Apply text cleaning
                processed_text = self._clean_text(text)
                if processed_text:  # Only add non-empty texts
                    preprocessed.append(processed_text)
        
        return preprocessed
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing unnecessary characters.
        
        Args:
            text: Text to clean.
            
        Returns:
            Cleaned text.
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def validate(self, data: List[str]) -> bool:
        """
        Validate text data.
        
        Args:
            data: Text data to validate.
            
        Returns:
            True if data is valid, False otherwise.
        """
        if not isinstance(data, list):
            return False
            
        for item in data:
            if not isinstance(item, str):
                return False
                
            # Check if text is not empty
            if not item.strip():
                return False
                
        return True