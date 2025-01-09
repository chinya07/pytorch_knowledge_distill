"""SSIM-based structural knowledge distillation trainer implementation."""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from kornia.losses import ssim_loss

from .base_trainer import BaseTrainer


class SSIMDistillationTrainer(BaseTrainer):
    """Trainer implementing SSIM-based structural knowledge distillation.
    
    This implements the structural KD approach from:
    "Structural Knowledge Distillation for Object Detection"
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        feature_weight: float = 0.1,
        window_size: int = 11,
        learning_rate: float = 0.001,
        device: Optional[torch.device] = None
    ) -> None:
        """Initialize the SSIM distillation trainer.
        
        Args:
            student_model: Student model to train
            teacher_model: Teacher model to learn from
            train_loader: Training data loader
            test_loader: Test data loader
            feature_weight: Weight for SSIM feature matching loss
            window_size: Window size for SSIM computation
            learning_rate: Learning rate for optimization
            device: Device to run on (defaults to CUDA if available)
        """
        super().__init__(student_model, train_loader, test_loader, learning_rate, device)
        self.teacher_model = teacher_model
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        
        self.feature_weight = feature_weight
        self.window_size = window_size

    def compute_ssim_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute SSIM-based feature matching loss.
        
        Args:
            student_features: Features from student model [B, C, H, W]
            teacher_features: Features from teacher model [B, C, H, W]
            
        Returns:
            SSIM loss between feature maps
        """
        return ssim_loss(
            student_features,
            teacher_features,
            window_size=self.window_size
        )

    def train_epoch(self) -> float:
        """Train for one epoch using SSIM-based feature distillation.
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        
        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()

            # Get teacher features
            with torch.no_grad():
                teacher_outputs, teacher_features = self.teacher_model(inputs)
            
            # Student forward pass
            # 1x1 Convolution to match channels

            
            student_outputs, student_features = self.model(inputs)
            adaptation_layer = nn.Conv2d(in_channels=student_features.shape[1], out_channels=teacher_features.shape[1], kernel_size=1)
            student_features = adaptation_layer(student_features)
                        
            # Calculate classification and SSIM feature matching losses
            cls_loss = self.criterion(student_outputs, labels)
            ssim_loss = self.compute_ssim_loss(student_features, teacher_features)
            
            # Combined loss
            loss = cls_loss + self.feature_weight * ssim_loss
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
        return running_loss / len(self.train_loader)