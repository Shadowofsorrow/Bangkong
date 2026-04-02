"""
Dynamic configuration loader for Bangkong LLM Training System
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from .schemas import BangkongConfig


class ConfigLoader:
    """Dynamic configuration loader that supports environment variables and multiple config files."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the configuration file. If None, will search for config files.
        """
        self.config_path = config_path or self._find_config()
        self.config = self._load_config()
    
    def _find_config(self) -> Path:
        """
        Dynamically locate configuration file.
        
        Returns:
            Path to the configuration file.
        """
        # Check environment variable first
        if 'BANGKONG_CONFIG' in os.environ:
            config_path = Path(os.environ['BANGKONG_CONFIG'])
            if config_path.exists():
                return config_path
            else:
                raise FileNotFoundError(f"Configuration file specified in BANGKONG_CONFIG not found: {config_path}")
        
        # Check common locations
        possible_paths = [
            Path.cwd() / 'config.yaml',
            Path.cwd() / 'configs' / 'default.yaml',
            Path.cwd() / 'configs' / 'development.yaml',
            Path.home() / '.bangkong' / 'config.yaml'
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # If no config found, raise an exception
        raise FileNotFoundError(
            "No configuration file found. Please provide a config file or set BANGKONG_CONFIG environment variable."
        )
    
    def _load_config(self) -> BangkongConfig:
        """
        Load configuration with environment variable substitution.
        
        Returns:
            Loaded configuration as a BangkongConfig object.
        """
        # Read the config file content
        with open(self.config_path, 'r') as f:
            content = f.read()
        
        # Substitute environment variables
        content = os.path.expandvars(content)
        
        # Parse YAML
        config_dict = yaml.safe_load(content)
        
        # Handle inheritance from other config files
        if 'defaults' in config_dict:
            config_dict = self._merge_with_defaults(config_dict)
        
        # Recursively substitute environment variables in the final config
        config_dict = self._substitute_env_vars(config_dict)
        
        # Validate and return as Pydantic model
        return BangkongConfig(**config_dict)
    
    def _substitute_env_vars(self, obj):
        """
        Recursively substitute environment variables in a nested structure.
        
        Args:
            obj: Object to process (dict, list, or value)
            
        Returns:
            Object with environment variables substituted
        """
        if isinstance(obj, dict):
            return {key: self._substitute_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Handle default values in environment variables (format: ${VAR:-default})
            import re
            def replace_var(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) else ""
                return os.environ.get(var_name, default_value)
            
            # Replace ${VAR:-default} and ${VAR} patterns
            obj = re.sub(r'\$\{([^}:-]+):-([^}]*)\}', replace_var, obj)
            obj = re.sub(r'\$\{([^}:-]+)\}', replace_var, obj)
            return obj
        else:
            return obj
    
    def _merge_with_defaults(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configuration with default configurations.
        
        Args:
            config_dict: The configuration dictionary to merge.
            
        Returns:
            Merged configuration dictionary.
        """
        defaults = config_dict.pop('defaults', [])
        merged_config = {}
        
        # Load default configurations
        for default in defaults:
            default_path = Path.cwd() / 'configs' / f'{default}.yaml'
            if default_path.exists():
                with open(default_path, 'r') as f:
                    default_content = os.path.expandvars(f.read())
                    default_dict = yaml.safe_load(default_content)
                    merged_config = self._deep_merge(merged_config, default_dict)
        
        # Merge with current config (current config takes precedence)
        merged_config = self._deep_merge(merged_config, config_dict)
        
        return merged_config
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            dict1: First dictionary.
            dict2: Second dictionary.
            
        Returns:
            Merged dictionary.
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., "model.size").
            default: Default value to return if key is not found.
            
        Returns:
            Configuration value or default.
        """
        keys = key_path.split('.')
        value = self.config.dict()
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def reload(self) -> None:
        """Reload the configuration from file."""
        self.config = self._load_config()