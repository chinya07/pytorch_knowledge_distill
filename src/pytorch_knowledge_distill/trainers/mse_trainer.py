"""MSE-based feature distillation trainer implementation."""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer


class MSEDistillationTrainer(BaseTrainer):
    """Trainer implementing MSE-based feature distillation."""
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        feature_weight: float = 0.1,
        learning_rate: float = 0.001,
        device: Optional[torch.device] = None
    ) -> None:
        """Initialize the MSE distillation trainer.
        
        Args:
            student_model: Student model to train
            teacher_model: Teacher model to learn from
            train_loader: Training data loader
            test_loader: Test data loader
            feature_weight: Weight for feature matching loss
            learning_rate: Learning rate for optimization
            device: Device to run on (defaults to CUDA if available)
        """
        super().__init__(student_model, train_loader, test_loader, learning_rate, device)
        self.teacher_model = teacher_model
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        
        self.feature_weight = feature_weight
        self.mse_loss = nn.MSELoss()

    def train_epoch(self) -> float:
        """Train for one epoch using MSE feature distillation.
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        
        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Get teacher features
            with torch.no_grad():
                teacher_outputs, teacher_features = self.teacher_model(inputs)
            
            # Student forward pass
            self.optimizer.zero_grad()
            student_outputs, student_features = self.model(inputs)
            
            # Calculate classification and feature matching losses
            cls_loss = self.criterion(student_outputs, labels)
            feature_loss = self.mse_loss(student_features, teacher_features)
            
            # Combined loss
            loss = cls_loss + self.feature_weight * feature_loss
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
        return running_loss / len(self.train_loader) 