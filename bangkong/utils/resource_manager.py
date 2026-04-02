"""
Resource manager utility for Bangkong LLM Training System
"""

import psutil
import torch
from typing import Dict, Any


class ResourceManager:
    """Manages system resources and provides resource monitoring."""
    
    @staticmethod
    def get_current_memory_usage() -> Dict[str, float]:
        """
        Get current memory usage.
        
        Returns:
            Dictionary with memory usage information.
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        usage = {
            "process_memory_mb": memory_info.rss / 1024 / 1024,
            "process_memory_percent": process.memory_percent(),
            "system_memory_available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "system_memory_total_mb": psutil.virtual_memory().total / 1024 / 1024
        }
        
        if torch.cuda.is_available():
            usage["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            usage["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return usage
    
    @staticmethod
    def get_cpu_usage() -> float:
        """
        Get current CPU usage percentage.
        
        Returns:
            CPU usage percentage.
        """
        return psutil.cpu_percent(interval=1)
    
    @staticmethod
    def check_resource_constraints(max_memory_gb: float = None) -> Dict[str, Any]:
        """
        Check if system meets resource constraints.
        
        Args:
            max_memory_gb: Maximum memory constraint in GB.
            
        Returns:
            Dictionary with constraint check results.
        """
        system_memory_gb = psutil.virtual_memory().total / (1024**3)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        results = {
            "system_memory_gb": system_memory_gb,
            "available_memory_gb": available_memory_gb,
            "meets_memory_requirement": True
        }
        
        if max_memory_gb is not None:
            results["required_memory_gb"] = max_memory_gb
            results["meets_memory_requirement"] = available_memory_gb >= max_memory_gb
        
        return results