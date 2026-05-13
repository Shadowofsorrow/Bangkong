#!/usr/bin/env python3
"""
Meta-learning initialization for pre-intelligent LLMs using MAML/Reptile approach
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod
import copy

class MetaLearningTask(ABC):
    """Abstract base class for meta-learning tasks."""

    @abstractmethod
    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of data for this task."""
        raise NotImplementedError("Subclasses must implement sample_batch method")

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Get a model suitable for this task."""
        raise NotImplementedError("Subclasses must implement get_model method")

class ArithmeticTask(MetaLearningTask):
    """Simple arithmetic task for meta-learning."""

    def __init__(self, operation: str = "add", hidden_size: int = 768):
        self.operation = operation
        self.input_size = hidden_size  # Use consistent input size
        self.output_size = hidden_size  # Use consistent output size
        self.dtype = torch.float32  # Ensure consistent dtype

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of arithmetic problems."""
        # Generate random numbers with consistent size
        x = torch.randn(batch_size, self.input_size, dtype=self.dtype)

        # Compute the result based on operation
        if self.operation == "add":
            # Simple addition: first half + second half
            y = x[:, :min(self.input_size//2, self.input_size)] + x[:, self.input_size//2:self.input_size]
        elif self.operation == "sub":
            # Simple subtraction: first half - second half
            y = x[:, :min(self.input_size//2, self.input_size)] - x[:, self.input_size//2:self.input_size]
        elif self.operation == "mul":
            # Simple multiplication: first half * second half
            y = x[:, :min(self.input_size//2, self.input_size)] * x[:, self.input_size//2:self.input_size]
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

        # Ensure output has the same size as input
        if y.shape[1] < self.input_size:
            y = torch.cat([y, torch.zeros(batch_size, self.input_size - y.shape[1])], dim=1)
        return x, y

    def get_model(self) -> nn.Module:
        """Get a simple model for arithmetic tasks."""
        model = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_size)
        )
        # Ensure all parameters have consistent dtype
        for param in model.parameters():
            param.data = param.data.to(self.dtype)
        return model

class LogicTask(MetaLearningTask):
    """Simple logic task for meta-learning."""

    def __init__(self, hidden_size: int = 768):
        self.input_size = hidden_size  # Use consistent input size
        self.output_size = hidden_size  # Use consistent output size
        self.dtype = torch.float32  # Ensure consistent dtype

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of logic problems."""
        # Generate random boolean values with consistent size
        x = torch.rand(batch_size, self.input_size, dtype=self.dtype)
        # Convert to boolean-like values (0 or 1) with consistent size
        x = (x > 0.5).float()

        # Compute AND operation on first half of the data
        half_size = self.input_size // 2
        if half_size > 0:
            y = (x[:, :half_size] * x[:, half_size:half_size + half_size]).float()
        else:
            y = x.clone()
        
        # Ensure output has the same size as input
        if y.shape[1] < self.input_size:
            y = torch.cat([y, torch.zeros(batch_size, self.input_size - y.shape[1])], dim=1)
        elif y.shape[1] > self.input_size:
            y = y[:, :self.input_size]
        return x, y

    def get_model(self) -> nn.Module:
        """Get a simple model for logic tasks."""
        model = nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.output_size),
            nn.Sigmoid()
        )
        # Ensure all parameters have consistent dtype
        for param in model.parameters():
            param.data = param.data.to(self.dtype)
        return model

class TextGenerationTask(MetaLearningTask):
    """Text generation task for meta-learning."""

    def __init__(self, model_config: Dict[str, Any] = None):
        self.model_config = model_config or {}
        self.hidden_size = self.model_config.get('hidden_size', 768)
        self.vocab_size = self.model_config.get('vocab_size', 50257)
        self.sequence_length = self.model_config.get('sequence_length', 512)
        self.dtype = torch.float32  # Ensure consistent dtype

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of text generation data."""
        # Generate synthetic text data that mimics real language patterns
        # For compatibility with the model, we flatten the 3D tensor to 2D
        x = torch.randn(batch_size, self.hidden_size, dtype=self.dtype)
        # For text generation, target is the same as input (shifted)
        y = x.clone()
        return x, y

    def get_model(self) -> nn.Module:
        """Get a model for text generation tasks."""
        model = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        # Ensure all parameters have consistent dtype
        for param in model.parameters():
            param.data = param.data.to(self.dtype)
        return model

class CodeCompletionTask(MetaLearningTask):
    """Code completion task for meta-learning."""

    def __init__(self, model_config: Dict[str, Any] = None):
        self.model_config = model_config or {}
        self.hidden_size = self.model_config.get('hidden_size', 768)
        self.vocab_size = self.model_config.get('vocab_size', 50257)
        self.sequence_length = self.model_config.get('sequence_length', 512)
        self.dtype = torch.float32  # Ensure consistent dtype

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of code completion data."""
        # Generate synthetic code data with common programming patterns
        # For compatibility with the model, we create 2D tensors
        x = torch.randn(batch_size, self.hidden_size, dtype=self.dtype)
        # For code completion, target is the same as input (shifted)
        y = x.clone()
        return x, y

    def get_model(self) -> nn.Module:
        """Get a model for code completion tasks."""
        model = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        # Ensure all parameters have consistent dtype
        for param in model.parameters():
            param.data = param.data.to(self.dtype)
        return model

class LanguageModelingTask(MetaLearningTask):
    """Language modeling task for meta-learning with larger models."""

    def __init__(self, model_config: Dict[str, Any] = None):
        self.model_config = model_config or {}
        self.hidden_size = self.model_config.get('hidden_size', 768)
        self.vocab_size = self.model_config.get('vocab_size', 50257)
        self.sequence_length = self.model_config.get('sequence_length', 512)
        self.dtype = torch.float32  # Ensure consistent dtype

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of language modeling data."""
        # Generate random embeddings (not tokens) to match the model's hidden size
        # For compatibility with the model, we create 2D tensors
        x = torch.randn(batch_size, self.hidden_size, dtype=self.dtype)
        # For language modeling, target is the same as input (shifted)
        y = x.clone()
        return x, y

    def get_model(self) -> nn.Module:
        """Get a model for language modeling tasks."""
        # For language modeling, we would typically use the actual model
        # But for meta-learning, we use a simplified representation
        model = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        # Ensure all parameters have consistent dtype
        for param in model.parameters():
            param.data = param.data.to(self.dtype)
        return model

class MAML:
    """Model-Agnostic Meta-Learning implementation."""

    def __init__(self, model: nn.Module, lr_inner: float = 0.01, lr_meta: float = 0.001):
        """
        Initialize MAML.

        Args:
            model: Base model to use for meta-learning
            lr_inner: Learning rate for inner loop (task-specific adaptation)
            lr_meta: Learning rate for meta loop (meta-learning update)
        """
        self.model = model
        self.lr_inner = lr_inner
        self.lr_meta = lr_meta
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=lr_meta)
        self.dtype = torch.float32  # Ensure consistent dtype

    def inner_update(self, task: MetaLearningTask, batch_size: int = 32) -> nn.Module:
        """
        Perform inner loop update for a specific task.

        Args:
            task: Task to adapt to
            batch_size: Batch size for adaptation

        Returns:
            Adapted model
        """
        # Create a copy of the model for this task
        task_model = copy.deepcopy(self.model)
        # Ensure consistent dtype for task_model parameters
        for param in task_model.parameters():
            param.data = param.data.to(self.dtype)
        task_optimizer = optim.SGD(task_model.parameters(), lr=self.lr_inner)
        criterion = nn.MSELoss()

        # Sample data for this task
        x, y = task.sample_batch(batch_size)

        # Perform inner loop update
        task_optimizer.zero_grad()
        outputs = task_model(x)
        loss = criterion(outputs, y)
        loss.backward()
        task_optimizer.step()

        return task_model

    def meta_update(self, tasks: List[MetaLearningTask], batch_size: int = 32, num_inner_steps: int = 5) -> float:
        """
        Perform meta update across multiple tasks.

        Args:
            tasks: List of tasks to meta-learn from
            batch_size: Batch size for each task
            num_inner_steps: Number of inner loop steps per task

        Returns:
            Average meta-loss
        """
        self.meta_optimizer.zero_grad()

        total_loss = 0.0
        criterion = nn.MSELoss()

        for task in tasks:
            # Get task model and adapt it
            task_model = self.inner_update(task, batch_size)

            # Sample new data for meta-update
            x_meta, y_meta = task.sample_batch(batch_size)

            # Compute meta-loss
            outputs = task_model(x_meta)
            loss = criterion(outputs, y_meta)
            total_loss += loss.item()

            # Accumulate gradients
            loss.backward()

        # Average the loss and update meta-parameters
        avg_loss = total_loss / len(tasks)
        self.meta_optimizer.step()

        return avg_loss

class Reptile:
    """Reptile meta-learning implementation."""

    def __init__(self, model: nn.Module, lr_inner: float = 0.01, lr_meta: float = 0.1):
        """
        Initialize Reptile.

        Args:
            model: Base model to use for meta-learning
            lr_inner: Learning rate for inner loop (task-specific adaptation)
            lr_meta: Learning rate for meta loop (meta-learning update)
        """
        self.model = model
        self.lr_inner = lr_inner
        self.lr_meta = lr_meta
        self.dtype = torch.float32  # Ensure consistent dtype

    def inner_update(self, task: MetaLearningTask, batch_size: int = 32, num_steps: int = 5) -> Tuple[nn.Module, float]:
        """
        Perform inner loop update for a specific task.

        Args:
            task: Task to adapt to
            batch_size: Batch size for adaptation
            num_steps: Number of inner loop steps

        Returns:
            Adapted model and final loss
        """
        # Create a copy of the model for this task
        task_model = copy.deepcopy(self.model)
        # Ensure consistent dtype for task_model parameters
        for param in task_model.parameters():
            param.data = param.data.to(self.dtype)
        task_optimizer = optim.SGD(task_model.parameters(), lr=self.lr_inner)
        criterion = nn.MSELoss()

        # Perform inner loop updates
        for _ in range(num_steps):
            x, y = task.sample_batch(batch_size)
            task_optimizer.zero_grad()
            outputs = task_model(x)
            loss = criterion(outputs, y)
            loss.backward()
            task_optimizer.step()

        return task_model, loss.item()

    def meta_update(self, tasks: List[MetaLearningTask], batch_size: int = 32, num_inner_steps: int = 5) -> float:
        """
        Perform meta update across multiple tasks using Reptile.

        Args:
            tasks: List of tasks to meta-learn from
            batch_size: Batch size for each task
            num_inner_steps: Number of inner loop steps per task

        Returns:
            Average loss across tasks
        """
        # Store original model parameters
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}

        total_loss = 0.0
        criterion = nn.MSELoss()

        for task in tasks:
            # Create a copy of the model for this task
            task_model = copy.deepcopy(self.model)
            # Ensure consistent dtype for task_model parameters
            for param in task_model.parameters():
                param.data = param.data.to(self.dtype)
            task_optimizer = optim.SGD(task_model.parameters(), lr=self.lr_inner)

            # Perform inner loop updates
            for _ in range(num_inner_steps):
                x, y = task.sample_batch(batch_size)
                task_optimizer.zero_grad()
                outputs = task_model(x)
                loss = criterion(outputs, y)
                loss.backward()
                task_optimizer.step()
                total_loss += loss.item()

            # Update original model parameters using Reptile formula
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    task_param = dict(task_model.named_parameters())[name]
                    # Ensure consistent dtype
                    task_param_data = task_param.data.to(self.dtype)
                    param_data = param.data.to(self.dtype)
                    param.data = param_data + self.lr_meta * (task_param_data - param_data)

        avg_loss = total_loss / (len(tasks) * num_inner_steps)
        return avg_loss

def create_meta_learning_tasks(model_config: Dict[str, Any] = None) -> List[MetaLearningTask]:
    """Create a set of tasks for meta-learning."""
    # Use the hidden size from the model config if available, otherwise default to 768
    hidden_size = model_config.get('hidden_size', 768) if model_config else 768
    
    tasks = [
        ArithmeticTask("add", hidden_size),
        ArithmeticTask("sub", hidden_size),
        ArithmeticTask("mul", hidden_size),
        LogicTask(hidden_size)
    ]

    # Add language modeling task if model_config is provided
    if model_config:
        tasks.append(LanguageModelingTask(model_config))
        tasks.append(TextGenerationTask(model_config))
        tasks.append(CodeCompletionTask(model_config))

    return tasks

def train_meta_initializer(num_epochs: int = 100, method: str = "reptile", model_config: Dict[str, Any] = None, hidden_size: int = 768) -> nn.Module:
    """
    Train a meta-initializer using MAML or Reptile for larger models.

    Args:
        num_epochs: Number of meta-training epochs
        method: Meta-learning method ("maml" or "reptile")
        model_config: Configuration for the model (for integration with larger models)
        hidden_size: Hidden size for the model (default: 768 for GPT-2 base)

    Returns:
        Meta-trained model
    """
    # Create tasks
    tasks = create_meta_learning_tasks(model_config)

    # Create base model (we'll use the first task's model as template)
    base_model = tasks[0].get_model()

    # For larger models, we might want to create a more complex model
    # But for meta-initialization, we focus on the smaller tasks first
    # and then apply the learned initialization to the larger model

    # Initialize meta-learner
    if method == "maml":
        meta_learner = MAML(base_model, lr_inner=0.01, lr_meta=0.001)
    else:  # reptile
        meta_learner = Reptile(base_model, lr_inner=0.01, lr_meta=0.1)

    # Meta-training loop
    print(f"Training meta-initializer using {method.upper()}...")
    for epoch in range(num_epochs):
        if isinstance(meta_learner, MAML):
            loss = meta_learner.meta_update(tasks, batch_size=32)
        else:  # Reptile
            loss = meta_learner.meta_update(tasks, batch_size=32)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

    print("Meta-training completed!")
    return meta_learner.model

def map_meta_initialization_to_model(model: nn.Module, meta_init: Dict[str, torch.Tensor]) -> None:
    """
    Map meta-initialization patterns to the target model with proper shape handling.
    
    Args:
        model: Target model to initialize
        meta_init: Meta-initialization patterns
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Try to find matching parameter in meta-initialization
            if name in meta_init:
                meta_param = meta_init[name]
                # Ensure shape compatibility
                if meta_param.shape == param.shape:
                    # Use meta-initialization as a guide for parameter initialization
                    param.copy_(meta_param)
                else:
                    # Handle shape mismatch by interpolation
                    if len(meta_param.shape) == len(param.shape):
                        # Same number of dimensions, interpolate
                        if meta_param.numel() <= param.numel():
                            # Expand smaller tensor to fit larger one
                            # This is a simplified approach - in practice, more sophisticated interpolation would be used
                            param.data = torch.nn.functional.interpolate(
                                meta_param.view(1, -1), 
                                size=param.numel(), 
                                mode='linear', 
                                align_corners=True
                            ).view(param.shape)
            else:
                # Handle parameter name mismatch
                # Try to find parameters with similar structure
                for meta_name, meta_param in meta_init.items():
                    if meta_param.shape == param.shape:
                        param.copy_(meta_param)
                        break

def extract_meta_initialization(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract meta-initialization parameters from a trained model.

    Args:
        model: Meta-trained model

    Returns:
        Dictionary of parameter names and values
    """
    meta_init = {}
    for name, param in model.named_parameters():
        meta_init[name] = param.data.clone()

    return meta_init


def create_meta_initialization_patterns(model_config: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:
    """
    Create meta-initialization patterns for a target model.
    
    Args:
        model_config: Configuration for the target model
        
    Returns:
        Dictionary of meta-initialization patterns
    """
    # Create a simple model to extract patterns from
    model = nn.Sequential(
        nn.Linear(768, 768),
        nn.ReLU(),
        nn.Linear(768, 768)
    )
    
    # Extract meta-initialization patterns from the model
    meta_init = extract_meta_initialization(model)
    
    return meta_init


if __name__ == "__main__":
    # Train meta-initializer
    model_config = {
        'hidden_size': 768,
        'vocab_size': 50257,
        'sequence_length': 1024
    }
    meta_model = train_meta_initializer(num_epochs=100, method="reptile", model_config=model_config)

    # Extract meta-initialization
    meta_init = extract_meta_initialization(meta_model)

    # Create initialization patterns
    init_patterns = create_meta_initialization_patterns(model_config)

    print(f"Extracted {len(meta_init)} parameter tensors for meta-initialization")
    print("Meta-initialization ready for use in large model initialization!")