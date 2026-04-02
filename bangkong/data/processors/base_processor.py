"""
Base data processor for Bangkong LLM Training System
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict
from ...config.schemas import BangkongConfig


class DataProcessor(ABC):
    """Abstract base class for data processors."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize the data processor.
        
        Args:
            config: Bangkong configuration.
        """
        self.config = config
    
    @abstractmethod
    def load(self, path: str) -> Any:
        """
        Load data from a file.
        
        Args:
            path: Path to the data file.
            
        Returns:
            Loaded data.
        """
        pass
    
    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """
        Preprocess the data.
        
        Args:
            data: Raw data to preprocess.
            
        Returns:
            Preprocessed data.
        """
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        Validate the data.
        
        Args:
            data: Data to validate.
            
        Returns:
            True if data is valid, False otherwise.
        """
        pass