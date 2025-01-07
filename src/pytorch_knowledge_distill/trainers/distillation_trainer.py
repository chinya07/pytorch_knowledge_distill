"""Knowledge Distillation trainer implementation."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer


class DistillationTrainer(BaseTrainer):
    """Trainer implementing knowledge distillation."""
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        temperature: float = 3.0,
        alpha: float = 0.1,
        learning_rate: float = 0.001,
        device: Optional[torch.device] = None
    ) -> None:
        """Initialize the distillation trainer.
        
        Args:
            student_model: Student model to train
            teacher_model: Teacher model to learn from
            train_loader: Training data loader
            test_loader: Test data loader
            temperature: Temperature for softening probability distributions
            alpha: Weight for balancing soft and hard targets
            learning_rate: Learning rate for optimization
            device: Device to run on (defaults to CUDA if available)
        """
        super().__init__(student_model, train_loader, test_loader, learning_rate, device)
        self.teacher_model = teacher_model
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        
        self.temperature = temperature
        self.alpha = alpha

    def train_epoch(self) -> float:
        """Train for one epoch using knowledge distillation.
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        
        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Get soft targets from teacher
            with torch.no_grad():
                teacher_outputs = self.teacher_model(inputs)
                soft_targets = F.softmax(teacher_outputs / self.temperature, dim=1)
            
            # Student forward pass
            self.optimizer.zero_grad()
            student_outputs = self.model(inputs)
            
            # Calculate soft and hard losses
            soft_loss = F.kl_div(
                F.log_softmax(student_outputs / self.temperature, dim=1),
                soft_targets,
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            hard_loss = self.criterion(student_outputs, labels)
            
            # Combined loss
            loss = (1 - self.alpha) * soft_loss + self.alpha * hard_loss
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
        return running_loss / len(self.train_loader) 