"""
Resource allocator for Bangkong LLM Training System
"""

import torch
import platform
import os
from typing import Dict, Any, Union
from .detector import HardwareDetector
from ..config.schemas import BangkongConfig, HardwareBool


class ResourceAllocator:
    """Dynamically allocates resources based on hardware and configuration."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize the resource allocator.
        
        Args:
            config: Bangkong configuration.
        """
        self.config = config
        self.hardware_info = HardwareDetector.get_available_resources()
        self.allocated_resources = {}
    
    def allocate_training_resources(self) -> Dict[str, Any]:
        """
        Dynamically allocate resources for training.
        
        Returns:
            Dictionary with allocated training resources.
        """
        # Determine optimal batch size based on available memory
        batch_size = self._calculate_optimal_batch_size()
        
        # Determine device placement
        device = self._determine_optimal_device()
        
        # Configure mixed precision if available
        use_mixed_precision = self._should_use_mixed_precision()
        
        # Determine number of workers
        num_workers = self._determine_num_workers()
        
        return {
            'batch_size': batch_size,
            'device': device,
            'mixed_precision': use_mixed_precision,
            'num_workers': num_workers
        }
    
    def _calculate_optimal_batch_size(self) -> int:
        """
        Calculate optimal batch size based on available memory.
        
        Returns:
            Optimal batch size.
        """
        # Get configured batch size if not auto
        if isinstance(self.config.training.batch_size, int):
            return self.config.training.batch_size
        
        # For auto mode, start with a reasonable default
        if self.config.training.batch_size == "auto":
            # Calculate based on available memory
            available_memory_gb = self.hardware_info['memory_available_gb']
            
            # Adjust for GPU memory if available and supported
            supported_gpus = [gpu for gpu in self.hardware_info['gpu'] if gpu.get('supported', False)]
            if supported_gpus:
                gpu_memory_gb = sum(gpu['memory_total_gb'] for gpu in supported_gpus)
                available_memory_gb = min(available_memory_gb, gpu_memory_gb * 0.8)  # Use 80% of GPU memory
            
            # Adjust for sequence length - longer sequences need more memory
            sequence_length = getattr(self.config.model, 'sequence_length', 1024)
            max_sequence_length = getattr(self.config.model, 'max_sequence_length', 2000000)
            
            # Scale memory requirements based on sequence length
            # For very long sequences, we need to reduce batch size significantly
            if sequence_length > 1000000:  # 1M+ tokens
                base_memory_per_sample_gb = 0.1  # Much higher memory per sample
            elif sequence_length > 100000:  # 100K+ tokens
                base_memory_per_sample_gb = 0.05
            elif sequence_length > 10000:  # 10K+ tokens
                base_memory_per_sample_gb = 0.01
            else:
                base_memory_per_sample_gb = 0.001  # Original estimate
            
            max_batch_size = int(available_memory_gb / base_memory_per_sample_gb)
            
            # Apply configuration limits
            if isinstance(self.config.hardware.max_memory_gb, (int, float)) and self.config.hardware.max_memory_gb != "auto":
                max_memory_limit = self.config.hardware.max_memory_gb
                max_batch_size_limit = int(max_memory_limit / base_memory_per_sample_gb)
                max_batch_size = min(max_batch_size, max_batch_size_limit)
            
            # Set a reasonable maximum to prevent memory issues
            # For very long sequences, cap at much lower values
            if sequence_length > 1000000:
                max_batch_size = min(max_batch_size, 1)  # Cap at 1 for safety
            elif sequence_length > 100000:
                max_batch_size = min(max_batch_size, 2)  # Cap at 2 for safety
            elif sequence_length > 10000:
                max_batch_size = min(max_batch_size, 4)  # Cap at 4 for safety
            else:
                max_batch_size = min(max_batch_size, 8)  # Cap at 8 for safety on consumer hardware
            
            # Ensure minimum batch size of 1
            return max(1, max_batch_size)
        
        # Default fallback
        return 1
    
    def _determine_optimal_device(self) -> torch.device:
        """
        Determine optimal device for training.
        
        Returns:
            Optimal torch device.
        """
        # Check if GPU is explicitly enabled/disabled
        if self.config.hardware.use_gpu == HardwareBool.TRUE:
            if HardwareDetector.is_gpu_available():
                return torch.device("cuda")
            else:
                print("Warning: GPU requested but not available or supported. Falling back to CPU.")
                return torch.device("cpu")
        elif self.config.hardware.use_gpu == HardwareBool.FALSE:
            return torch.device("cpu")
        
        # Auto-detect - use GPU only if available and supported
        if HardwareDetector.is_gpu_available():
            return torch.device("cuda")
        else:
            print("Info: No supported GPU available. Using CPU for training.")
            return torch.device("cpu")
    
    def _should_use_mixed_precision(self) -> bool:
        """
        Determine if mixed precision should be used.
        
        Returns:
            True if mixed precision should be used, False otherwise.
        """
        # Check if FP16 is explicitly enabled/disabled
        if self.config.hardware.fp16 == HardwareBool.TRUE:
            if not HardwareDetector.is_gpu_available():
                print("Warning: FP16 requested but no supported GPU available. Disabling mixed precision.")
                return False
            # Check if GPU supports mixed precision (compute capability >= 7.0)
            supported_gpus = [gpu for gpu in self.hardware_info['gpu'] if gpu.get('supported', False)]
            if supported_gpus and supported_gpus[0]['compute_capability'][0] >= 7:
                return True
            else:
                print("Warning: FP16 requested but GPU does not support it. Disabling mixed precision.")
                return False
        elif self.config.hardware.fp16 == HardwareBool.FALSE:
            return False
        
        # Auto-detect
        # Use mixed precision if FP16 is enabled and GPU is available with compute capability >= 7
        supported_gpus = [gpu for gpu in self.hardware_info['gpu'] if gpu.get('supported', False)]
        if (supported_gpus and 
            supported_gpus[0]['compute_capability'][0] >= 7):
            return True
        return False
    
    def _determine_num_workers(self) -> int:
        """
        Determine optimal number of data loading workers.
        
        Returns:
            Optimal number of workers.
        """
        # Check if explicitly set
        if isinstance(self.config.hardware.num_workers, int):
            return self.config.hardware.num_workers
        
        # For Windows, 0 is often better due to multiprocessing limitations
        if platform.system() == "Windows":
            return 0
        
        # For other systems, use CPU count or 0
        cpu_count = self.hardware_info['cpu']['count_logical']
        return cpu_count if cpu_count is not None else 0