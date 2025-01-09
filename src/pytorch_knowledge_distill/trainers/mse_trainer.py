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
        ce_loss_weight: float = 0.1,
        feature_map_weight: float = 0.1,
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
        self.student_model = student_model
        self.student_model.to(self.device)
        self.learning_rate = learning_rate
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.student_model.parameters(), lr=learning_rate)    

        self.ce_loss_weight = ce_loss_weight
        self.feature_map_weight = feature_map_weight

    def train_epoch(self) -> float:
        """Train for one epoch using MSE feature distillation.
        
        Returns:
            Average loss for the epoch
        """
        self.teacher_model.eval()
        self.student_model.train()
        running_loss = 0.0
        
        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
    
            # Student forward pass
            self.optimizer.zero_grad()

            with torch.no_grad():
                _, teacher_feature_map = self.teacher_model(inputs)

            student_logits, regressor_feature_map = self.student_model(inputs)
            
            # Calculate the loss
            hidden_rep_loss = self.mse_loss(regressor_feature_map, teacher_feature_map)

            # Calculate classification and feature matching losses
            label_loss = self.ce_loss(student_logits, labels)
        
            
            # Combined loss
            loss = self.feature_map_weight * hidden_rep_loss + self.ce_loss_weight * label_loss
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
        return running_loss / len(self.train_loader) 