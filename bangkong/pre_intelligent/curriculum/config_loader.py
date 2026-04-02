#!/usr/bin/env python3
"""
Configuration loader for curriculum learning system
"""

import yaml
import os
from typing import Dict, Any, Optional


class CurriculumConfig:
    """Configuration loader for curriculum learning system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration file. If None, will look for default locations.
        """
        if config_path is None:
            # Look for config file in current directory or package directory
            possible_paths = [
                "config.yaml",
                os.path.join(os.path.dirname(__file__), "config.yaml"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "curriculum", "config.yaml"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pre_intelligent", "curriculum", "config.yaml")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            else:
                raise FileNotFoundError("Could not find curriculum configuration file")
        
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
            key_path: Dot-separated path to configuration value (e.g., "curriculum.difficulty.initial").
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
    
    def get_task_distribution(self, stage: str = "default") -> Dict[str, float]:
        """
        Get task distribution for a specific stage.
        
        Args:
            stage: Stage name ("default", "early_stage", "middle_stage", "advanced_stage").
            
        Returns:
            Task distribution dictionary.
        """
        return self.get(f"curriculum.task_distributions.{stage}", {})
    
    def get_task_data(self, task_type: str) -> Dict[str, Any]:
        """
        Get task-specific data.
        
        Args:
            task_type: Type of task.
            
        Returns:
            Task data dictionary.
        """
        return self.get(f"curriculum.task_data.{task_type}", {})
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()


# Global configuration instance
_config: Optional[CurriculumConfig] = None


def get_config(config_path: Optional[str] = None) -> CurriculumConfig:
    """
    Get singleton configuration instance.
    
    Args:
        config_path: Path to configuration file. Only used on first call.
        
    Returns:
        Configuration instance.
    """
    global _config
    if _config is None:
        _config = CurriculumConfig(config_path)
    return _config