"""
Resource tracker for Bangkong LLM Training System
"""

import psutil
import torch
import time
import platform
from typing import Dict, List, Any
from ..hardware.detector import HardwareDetector


class ResourceTracker:
    """Tracks system resources during training and inference."""
    
    def __init__(self):
        """Initialize the resource tracker."""
        self.metrics = []
        self.start_time = time.time()
        self.initial_resources = self._get_current_resources()
    
    def _get_current_resources(self) -> Dict[str, Any]:
        """
        Get current system resources.
        
        Returns:
            Dictionary with current resource usage.
        """
        resources = {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / (1024**3)
        }
        
        # Handle disk usage in a cross-platform way
        try:
            if platform.system() == "Windows":
                # Use current directory for disk usage on Windows
                disk_usage = psutil.disk_usage(".")
            else:
                # Use root directory for disk usage on Unix-like systems
                disk_usage = psutil.disk_usage("/")
            resources["disk_usage_percent"] = disk_usage.percent
        except Exception:
            # If disk usage fails, set to 0
            resources["disk_usage_percent"] = 0
        
        if torch.cuda.is_available():
            try:
                resources["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
                resources["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
                # Only try to get utilization if the method exists
                if hasattr(torch.cuda, 'utilization'):
                    resources["gpu_utilization"] = torch.cuda.utilization()
                else:
                    resources["gpu_utilization"] = 0
            except Exception:
                # If GPU monitoring fails, set GPU resources to 0
                resources["gpu_memory_allocated_gb"] = 0
                resources["gpu_memory_reserved_gb"] = 0
                resources["gpu_utilization"] = 0
        
        return resources
    
    def log_resources(self) -> Dict[str, Any]:
        """
        Log current resource usage.
        
        Returns:
            Dictionary with current resource usage.
        """
        resources = self._get_current_resources()
        self.metrics.append(resources)
        return resources
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """
        Get a summary of resource usage.
        
        Returns:
            Dictionary with resource usage summary.
        """
        if not self.metrics:
            return {}
        
        # Calculate averages
        summary = {
            "duration_seconds": time.time() - self.start_time,
            "cpu_avg_percent": sum(m["cpu_percent"] for m in self.metrics) / len(self.metrics),
            "memory_avg_percent": sum(m["memory_percent"] for m in self.metrics) / len(self.metrics),
            "peak_memory_gb": max(m["memory_available_gb"] for m in self.metrics),
        }
        
        # Add GPU summary if GPU was available
        gpu_metrics = [m for m in self.metrics if "gpu_memory_allocated_gb" in m]
        if gpu_metrics:
            summary["gpu_avg_memory_gb"] = sum(
                m.get("gpu_memory_allocated_gb", 0) for m in gpu_metrics
            ) / len(gpu_metrics)
        
        return summary
    
    def reset(self):
        """Reset the tracker."""
        self.metrics = []
        self.start_time = time.time()
        self.initial_resources = self._get_current_resources()