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
        pass
    
    @abstractmethod
    def get_model(self) -> nn.Module:
        """Get a model suitable for this task."""
        pass

class ArithmeticTask(MetaLearningTask):
    """Simple arithmetic task for meta-learning."""
    
    def __init__(self, operation: str = "add"):
        self.operation = operation
        self.input_size = 2  # Two numbers
        self.output_size = 1  # One result
    
    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of arithmetic problems."""
        # Generate random numbers between 0 and 100
        x = torch.randint(0, 100, (batch_size, self.input_size)).float()
        
        # Compute the result based on operation
        if self.operation == "add":
            y = x[:, 0:1] + x[:, 1:2]
        elif self.operation == "sub":
            y = x[:, 0:1] - x[:, 1:2]
        elif self.operation == "mul":
            y = x[:, 0:1] * x[:, 1:2]
        else:
            raise ValueError(f"Unknown operation: {self.operation}")
        
        return x, y
    
    def get_model(self) -> nn.Module:
        """Get a simple model for arithmetic tasks."""
        return nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_size)
        )

class LogicTask(MetaLearningTask):
    """Simple logic task for meta-learning."""
    
    def __init__(self):
        self.input_size = 2  # Two boolean values
        self.output_size = 1  # One boolean result
    
    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of logic problems."""
        # Generate random boolean values
        x = torch.randint(0, 2, (batch_size, self.input_size)).float()
        
        # Compute AND operation
        y = (x[:, 0:1] * x[:, 1:2]).float()
        
        return x, y
    
    def get_model(self) -> nn.Module:
        """Get a simple model for logic tasks."""
        return nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.output_size),
            nn.Sigmoid()
        )

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
                    param.data = param.data + self.lr_meta * (task_param.data - param.data)
        
        avg_loss = total_loss / (len(tasks) * num_inner_steps)
        return avg_loss

def create_meta_learning_tasks() -> List[MetaLearningTask]:
    """Create a set of tasks for meta-learning."""
    tasks = [
        ArithmeticTask("add"),
        ArithmeticTask("sub"),
        ArithmeticTask("mul"),
        LogicTask()
    ]
    return tasks

def train_meta_initializer(num_epochs: int = 100, method: str = "reptile") -> nn.Module:
    """
    Train a meta-initializer using MAML or Reptile.
    
    Args:
        num_epochs: Number of meta-training epochs
        method: Meta-learning method ("maml" or "reptile")
        
    Returns:
        Meta-trained model
    """
    # Create tasks
    tasks = create_meta_learning_tasks()
    
    # Create base model (we'll use the first task's model as template)
    base_model = tasks[0].get_model()
    
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

if __name__ == "__main__":
    # Train meta-initializer
    meta_model = train_meta_initializer(num_epochs=100, method="reptile")
    
    # Extract meta-initialization
    meta_init = extract_meta_initialization(meta_model)
    
    print(f"Extracted {len(meta_init)} parameter tensors for meta-initialization")
    print("Meta-initialization ready for use in large model initialization!")