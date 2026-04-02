"""
Data pipeline for Bangkong LLM Training System
"""

from typing import Any, Dict, List, Optional
from ..config.schemas import BangkongConfig
from ..utils.path_manager import PathManager
from ..utils.dynamic_importer import DynamicImporter


class DataPipeline:
    """Manages the data processing pipeline for training."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize the data pipeline.
        
        Args:
            config: Bangkong configuration.
        """
        self.config = config
        self.path_manager = PathManager()
        self.processors = {}
        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize data processors based on configuration."""
        # Processors are loaded dynamically when needed
        pass
    
    def load_data(self, data_path: str, data_type: str = "text") -> Any:
        """
        Load data from a specified path.
        
        Args:
            data_path: Path to the data.
            data_type: Type of data to load.
            
        Returns:
            Loaded data.
        """
        # Resolve the path
        resolved_path = self.path_manager.resolve_path(data_path)
        
        # Load data based on type
        if data_type == "text":
            return self._load_text_data(resolved_path)
        elif data_type == "json":
            return self._load_json_data(resolved_path)
        else:
            # Try to dynamically load a processor for this data type
            processor = self._get_processor(data_type)
            if processor:
                return processor.load(str(resolved_path))
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
    
    def _load_text_data(self, path) -> List[str]:
        """
        Load text data from a file.
        
        Args:
            path: Path to the text file.
            
        Returns:
            List of text lines.
        """
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def _load_json_data(self, path) -> List[Dict]:
        """
        Load JSON data from a file.
        
        Args:
            path: Path to the JSON file.
            
        Returns:
            List of JSON objects.
        """
        import json
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    
    def _get_processor(self, data_type: str):
        """
        Get a processor for a specific data type.
        
        Args:
            data_type: Type of data to process.
            
        Returns:
            Data processor or None if not found.
        """
        if data_type in self.processors:
            return self.processors[data_type]
        
        # Handle domain-specific processors
        if data_type.startswith('domain_'):
            domain = data_type[7:]  # Remove 'domain_' prefix
            try:
                from .processors.domain_processor import create_domain_processor
                processor = create_domain_processor(domain, self.config)
                self.processors[data_type] = processor
                return processor
            except Exception as e:
                self.logger.warning(f"Failed to create domain processor for {domain}: {e}")
        
        # Try to dynamically import a processor
        processor_class = DynamicImporter.import_class(
            f"bangkong.data.processors.{data_type}_processor",
            f"{data_type.capitalize()}Processor"
        )
        
        if processor_class:
            processor = processor_class(self.config)
            self.processors[data_type] = processor
            return processor
        
        return None
    
    def preprocess(self, data: Any, data_type: str = "text") -> Any:
        """
        Preprocess data.
        
        Args:
            data: Data to preprocess.
            data_type: Type of data to preprocess.
            
        Returns:
            Preprocessed data.
        """
        if data_type == "text":
            return self._preprocess_text_data(data)
        else:
            processor = self._get_processor(data_type)
            if processor:
                return processor.preprocess(data)
            else:
                # Return data as-is if no processor found
                return data
    
    def _preprocess_text_data(self, data: List[str]) -> List[str]:
        """
        Preprocess text data with regional language support.
        
        Args:
            data: List of text strings.
            
        Returns:
            Preprocessed text data.
        """
        # Import regional processor here to avoid circular imports
        from .processors.regional_processor import create_regional_data_processor
        
        # Create appropriate processor based on configuration
        processor = create_regional_data_processor(self.config)
        
        # Preprocess data using the regional processor
        return processor.preprocess(data)