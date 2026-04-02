"""
Dynamic importer utility for Bangkong LLM Training System
"""

import importlib
import importlib.util
from typing import Any, Optional


class DynamicImporter:
    """Dynamically imports modules and classes with error handling."""
    
    @staticmethod
    def safe_import(module_name: str, feature_name: str) -> Any:
        """
        Safely import a module with error handling.
        
        Args:
            module_name: Name of the module to import.
            feature_name: Name of the feature for error messaging.
            
        Returns:
            Imported module or None if import fails.
        """
        try:
            return importlib.import_module(module_name)
        except ImportError as e:
            print(f"Warning: {feature_name} requires {module_name} which is not installed: {e}")
            return None
    
    @staticmethod
    def import_class(module_name: str, class_name: str) -> Any:
        """
        Dynamically import a class from a module.
        
        Args:
            module_name: Name of the module.
            class_name: Name of the class to import.
            
        Returns:
            Imported class or None if import fails.
        """
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not import {class_name} from {module_name}: {e}")
            return None
    
    @staticmethod
    def is_module_available(module_name: str) -> bool:
        """
        Check if a module is available for import.
        
        Args:
            module_name: Name of the module to check.
            
        Returns:
            True if module is available, False otherwise.
        """
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False