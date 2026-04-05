#!/usr/bin/env python3
"""
Configuration loader for models system
"""

import yaml
import os
from typing import Dict, Any, Optional


class ModelsConfig:
    """Configuration loader for models system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration file. If None, will look for default locations.
        """
        if config_path is None:
            # Look for config file in current directory or package directory
            possible_paths = [
                "models_config.yaml",
                os.path.join(os.path.dirname(__file__), "config.yaml"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "config.yaml"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "bangkong", "models", "config.yaml")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            else:
                raise FileNotFoundError("Could not find models configuration file")
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary.
        """
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., "models.cosine_clustered_embeddings.factorial_cap").
            default: Default value to return if key is not found.
            
        Returns:
            Configuration value or default.
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()


# Global configuration instance
_config: Optional[ModelsConfig] = None


def get_models_config(config_path: Optional[str] = None) -> ModelsConfig:
    """
    Get singleton configuration instance.
    
    Args:
        config_path: Path to configuration file. Only used on first call.
        
    Returns:
        Configuration instance.
    """
    global _config
    if _config is None:
        _config = ModelsConfig(config_path)
    return _config