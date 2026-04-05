"""
Model distillation for Bangkong LLM Training System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from ..config.schemas import BangkongConfig


class DistillationLoss(nn.Module):
    """Distillation loss function that combines hard and soft targets."""
    
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        """
        Initialize distillation loss.
        
        Args:
            temperature: Temperature for softening probability distributions
            alpha: Weight for combining hard and soft losses
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            labels: Ground truth labels
            
        Returns:
            Combined distillation loss
        """
        # Compute hard target loss (standard cross-entropy)
        hard_loss = self.ce_loss(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
        
        # Compute soft target loss (KL divergence between softened distributions)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
        
        # Combine losses
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return total_loss


class ModelDistiller:
    """Model distillation trainer that transfers knowledge from teacher to student."""
    
    def __init__(
        self, 
        teacher_model: nn.Module, 
        student_model: nn.Module, 
        config: BangkongConfig
    ):
        """
        Initialize model distiller.
        
        Args:
            teacher_model: Teacher model (pre-trained, larger model)
            student_model: Student model (smaller model to be trained)
            config: Bangkong configuration
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        
        # Set models to evaluation/train mode
        self.teacher_model.eval()
        self.student_model.train()
        
        # Initialize distillation loss
        self.distillation_loss = DistillationLoss(
            temperature=getattr(config.training, 'distillation_temperature', 2.0),
            alpha=getattr(config.training, 'distillation_alpha', 0.5)
        )
        
        # Move models to device if specified
        if hasattr(config, 'device'):
            self.teacher_model.to(config.device)
            self.student_model.to(config.device)
    
    def distill_step(
        self, 
        inputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Perform a single distillation step.
        
        Args:
            inputs: Input tensors for both models
            labels: Ground truth labels
            
        Returns:
            Dictionary with loss and other metrics
        """
        # Get teacher predictions (no gradient computation)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Get student predictions
        student_outputs = self.student_model(**inputs)
        student_logits = student_outputs.logits
        
        # Compute distillation loss
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        
        return {
            "loss": loss,
            "student_logits": student_logits,
            "teacher_logits": teacher_logits
        }
    
    def distill_epoch(
        self, 
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> float:
        """
        Distill for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            optimizer: Optimizer for student model
            scheduler: Learning rate scheduler (optional)
            
        Returns:
            Average loss for the epoch
        """
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Move batch to device
            if hasattr(self.config, 'device'):
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
            
            # Extract inputs and labels
            inputs = {k: v for k, v in batch.items() if k != 'labels'}
            labels = batch['labels']
            
            # Perform distillation step
            outputs = self.distill_step(inputs, labels)
            loss = outputs['loss']
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update scheduler if provided
            if scheduler:
                scheduler.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0


def create_distiller(
    teacher_model: nn.Module, 
    student_model: nn.Module, 
    config: BangkongConfig
) -> ModelDistiller:
    """
    Create a model distiller.
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
        config: Bangkong configuration
        
    Returns:
        Model distiller instance
    """
    return ModelDistiller(teacher_model, student_model, config)