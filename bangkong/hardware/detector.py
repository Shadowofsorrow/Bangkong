"""
Hardware detection for Bangkong LLM Training System
"""

import psutil
import torch
import platform
import os
from typing import Dict, List, Any


class HardwareDetector:
    """Detects available hardware and system information."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Get system information.
        
        Returns:
            Dictionary with system information.
        """
        return {
            'os': platform.system(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'node': platform.node()
        }
    
    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """
        Get CPU information.
        
        Returns:
            Dictionary with CPU information.
        """
        try:
            cpu_freq = psutil.cpu_freq()
            return {
                'count': psutil.cpu_count(),
                'count_logical': psutil.cpu_count(logical=True),
                'frequency': cpu_freq._asdict() if cpu_freq else None,
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3)
            }
        except Exception as e:
            # Fallback if psutil fails
            return {
                'count': 1,
                'count_logical': 1,
                'frequency': None,
                'memory_total_gb': 0,
                'memory_available_gb': 0
            }
    
    @staticmethod
    def get_gpu_info() -> List[Dict[str, Any]]:
        """
        Get GPU information.
        
        Returns:
            List of dictionaries with GPU information.
        """
        gpu_info = []
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        gpu_props = torch.cuda.get_device_properties(i)
                        # Check if GPU is supported (compute capability >= 3.7)
                        compute_capability = torch.cuda.get_device_capability(i)
                        is_supported = compute_capability[0] >= 3 and compute_capability[1] >= 7
                        
                        gpu_info.append({
                            'id': i,
                            'name': torch.cuda.get_device_name(i),
                            'memory_total_gb': gpu_props.total_memory / (1024**3),
                            'memory_allocated_gb': torch.cuda.memory_allocated(i) / (1024**3),
                            'memory_reserved_gb': torch.cuda.memory_reserved(i) / (1024**3),
                            'compute_capability': compute_capability,
                            'supported': is_supported
                        })
                    except Exception:
                        # Skip GPU if we can't get its properties
                        continue
        except Exception:
            # If CUDA is not available or fails, return empty list
            pass
        return gpu_info
    
    @staticmethod
    def get_available_resources() -> Dict[str, Any]:
        """
        Get all available resources.
        
        Returns:
            Dictionary with all available resources.
        """
        return {
            'cpu': HardwareDetector.get_cpu_info(),
            'gpu': HardwareDetector.get_gpu_info(),
            'system': HardwareDetector.get_system_info(),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3) if psutil else 0
        }
    
    @staticmethod
    def is_gpu_available() -> bool:
        """
        Check if GPU is available and supported.
        
        Returns:
            True if GPU is available and supported, False otherwise.
        """
        try:
            if not torch.cuda.is_available():
                return False
            
            # Check if any GPU is supported
            gpu_info = HardwareDetector.get_gpu_info()
            return any(gpu.get('supported', False) for gpu in gpu_info)
        except Exception:
            return False
    
    @staticmethod
    def is_tpu_available() -> bool:
        """
        Check if TPU is available.
        
        Returns:
            True if TPU is available, False otherwise.
        """
        try:
            import torch_xla
            return True
        except ImportError:
            return False