"""
Resource-related exceptions for Bangkong LLM Training System
"""

class ResourceError(Exception):
    """Base exception for resource-related errors."""
    
    def __init__(self, message: str, resource_type: str = None):
        super().__init__(message)
        self.resource_type = resource_type


class InsufficientMemoryError(ResourceError):
    """Raised when there is insufficient memory for an operation."""
    
    def __init__(self, message: str, required_gb: float, available_gb: float):
        super().__init__(message, "memory")
        self.required_gb = required_gb
        self.available_gb = available_gb
    
    def suggest_solution(self) -> str:
        """Suggest a solution for the memory error."""
        return "Try reducing batch size, using model quantization, or freeing up system memory."


class InsufficientGPUMemoryError(InsufficientMemoryError):
    """Raised when there is insufficient GPU memory for an operation."""
    
    def suggest_solution(self) -> str:
        """Suggest a solution for the GPU memory error."""
        return "Try reducing batch size, using CPU training, or freeing up GPU memory."


class HardwareNotFoundError(ResourceError):
    """Raised when required hardware is not found."""
    
    def __init__(self, message: str, hardware_type: str):
        super().__init__(message, hardware_type)
        self.hardware_type = hardware_type
    
    def suggest_solution(self) -> str:
        """Suggest a solution for the hardware error."""
        if self.hardware_type == "gpu":
            return "Install CUDA-compatible GPU drivers or use CPU training."
        elif self.hardware_type == "tpu":
            return "Ensure TPU runtime is properly configured or use GPU/CPU training."
        return "Check hardware requirements and installation."