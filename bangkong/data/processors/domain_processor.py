"""
Domain-specific data processors for Bangkong LLM Training System
"""

import re
from typing import List, Dict, Any
from ...config.schemas import BangkongConfig
from .text_processor import TextProcessor


class CodeProcessor(TextProcessor):
    """Processor for code data."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize code processor.
        
        Args:
            config: Bangkong configuration
        """
        super().__init__(config)
    
    def preprocess(self, data: List[str]) -> List[str]:
        """
        Preprocess code data.
        
        Args:
            data: List of code strings.
            
        Returns:
            Preprocessed code data.
        """
        preprocessed = []
        min_length = self.config.data.preprocessing.min_text_length
        max_length = self.config.data.preprocessing.max_text_length
        
        for code in data:
            # Filter by length
            if min_length <= len(code) <= max_length:
                # Apply code-specific cleaning
                processed_code = self._clean_code(code)
                if processed_code:  # Only add non-empty code
                    preprocessed.append(processed_code)
        
        return preprocessed
    
    def _clean_code(self, code: str) -> str:
        """
        Clean code by removing unnecessary elements.
        
        Args:
            code: Code to clean.
            
        Returns:
            Cleaned code.
        """
        # Remove extra whitespace while preserving indentation
        lines = code.split('\n')
        cleaned_lines = [line.rstrip() for line in lines if line.strip()]
        cleaned_code = '\n'.join(cleaned_lines)
        
        return cleaned_code
    
    def validate(self, data: List[str]) -> bool:
        """
        Validate code data.
        
        Args:
            data: Code data to validate.
            
        Returns:
            True if data is valid, False otherwise.
        """
        if not isinstance(data, list):
            return False
            
        for item in data:
            if not isinstance(item, str):
                return False
                
            # Check if code is not empty
            if not item.strip():
                return False
                
        return True


class MathProcessor(TextProcessor):
    """Processor for mathematical text data."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize math processor.
        
        Args:
            config: Bangkong configuration
        """
        super().__init__(config)
    
    def preprocess(self, data: List[str]) -> List[str]:
        """
        Preprocess mathematical text data.
        
        Args:
            data: List of mathematical text strings.
            
        Returns:
            Preprocessed mathematical text data.
        """
        preprocessed = []
        min_length = self.config.data.preprocessing.min_text_length
        max_length = self.config.data.preprocessing.max_text_length
        
        for text in data:
            # Filter by length
            if min_length <= len(text) <= max_length:
                # Apply math-specific cleaning
                processed_text = self._clean_math_text(text)
                if processed_text:  # Only add non-empty texts
                    preprocessed.append(processed_text)
        
        return preprocessed
    
    def _clean_math_text(self, text: str) -> str:
        """
        Clean mathematical text.
        
        Args:
            text: Mathematical text to clean.
            
        Returns:
            Cleaned mathematical text.
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize mathematical symbols
        # This is a simplified approach - in practice, more sophisticated normalization would be used
        text = re.sub(r'\\times', '×', text)
        text = re.sub(r'\\div', '÷', text)
        text = re.sub(r'\\pm', '±', text)
        
        return text
    
    def validate(self, data: List[str]) -> bool:
        """
        Validate mathematical text data.
        
        Args:
            data: Mathematical text data to validate.
            
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


class ScientificProcessor(TextProcessor):
    """Processor for scientific text data."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize scientific processor.
        
        Args:
            config: Bangkong configuration
        """
        super().__init__(config)
    
    def preprocess(self, data: List[str]) -> List[str]:
        """
        Preprocess scientific text data.
        
        Args:
            data: List of scientific text strings.
            
        Returns:
            Preprocessed scientific text data.
        """
        preprocessed = []
        min_length = self.config.data.preprocessing.min_text_length
        max_length = self.config.data.preprocessing.max_text_length
        
        for text in data:
            # Filter by length
            if min_length <= len(text) <= max_length:
                # Apply scientific-specific cleaning
                processed_text = self._clean_scientific_text(text)
                if processed_text:  # Only add non-empty texts
                    preprocessed.append(processed_text)
        
        return preprocessed
    
    def _clean_scientific_text(self, text: str) -> str:
        """
        Clean scientific text.
        
        Args:
            text: Scientific text to clean.
            
        Returns:
            Cleaned scientific text.
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize scientific notation
        text = re.sub(r'(\d+)e(-?\d+)', r'\1×10^\2', text)
        
        return text
    
    def validate(self, data: List[str]) -> bool:
        """
        Validate scientific text data.
        
        Args:
            data: Scientific text data to validate.
            
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


def create_domain_processor(domain: str, config: BangkongConfig) -> TextProcessor:
    """
    Create a domain-specific processor.
    
    Args:
        domain: Domain type ('code', 'math', 'scientific', etc.)
        config: Bangkong configuration
        
    Returns:
        Domain-specific processor
    """
    if domain == 'code':
        return CodeProcessor(config)
    elif domain == 'math':
        return MathProcessor(config)
    elif domain == 'scientific':
        return ScientificProcessor(config)
    else:
        # Default to base text processor
        return TextProcessor(config)