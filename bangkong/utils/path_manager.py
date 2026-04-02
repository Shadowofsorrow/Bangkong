"""
Path manager utility for Bangkong LLM Training System
"""

import os
from pathlib import Path
from typing import Union


class PathManager:
    """Manages paths dynamically across different environments."""
    
    def __init__(self, base_path: Union[str, Path] = None):
        """
        Initialize the path manager.
        
        Args:
            base_path: Base path for relative path resolution. Defaults to current working directory.
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
    
    def resolve_path(self, path_str: str, create_parents: bool = True) -> Path:
        """
        Resolve a path dynamically based on environment and create parent directories if needed.
        
        Args:
            path_str: Path string to resolve.
            create_parents: Whether to create parent directories if they don't exist.
            
        Returns:
            Resolved Path object.
        """
        # Expand environment variables
        path_str = os.path.expandvars(path_str)
        
        # Convert to Path object
        path = Path(path_str)
        
        # If path is relative, resolve against base path
        if not path.is_absolute():
            path = self.base_path / path
        
        # Create parent directories if they don't exist
        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        return path
    
    def get_data_paths(self, data_config) -> dict:
        """
        Get all data paths from configuration.
        
        Args:
            data_config: Data configuration object.
            
        Returns:
            Dictionary of resolved data paths.
        """
        data_paths = {}
        
        if hasattr(data_config, 'paths'):
            for name, path_str in data_config.paths.items():
                data_paths[name] = self.resolve_path(path_str)
        
        return data_paths
    
    def get_model_path(self, model_name: str, models_base_path: str) -> Path:
        """
        Get path for a specific model.
        
        Args:
            model_name: Name of the model.
            models_base_path: Base path for models.
            
        Returns:
            Path to the model directory.
        """
        base_path = self.resolve_path(models_base_path)
        return base_path / model_name