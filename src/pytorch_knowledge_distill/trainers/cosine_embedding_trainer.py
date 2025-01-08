"""Cosine Embedding-based feature distillation trainer implementation."""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from .base_trainer import BaseTrainer


class CosineEmbeddingTrainer(BaseTrainer):
    """Trainer implementing Cosine Embedding-based feature distillation."""
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        cosine_embedding_weight: float = 0.1,
        ce_loss_weight: float = 0.1,
        learning_rate: float = 0.001,
        device: Optional[torch.device] = None
    ) -> None:
        """Initialize the Cosine Embedding trainer.
        
        Args:
            student_model: Student model to train
            teacher_model: Teacher model to learn from
            train_loader: Training data loader
            test_loader: Test data loader
            cosine_embedding_weight: Weight for feature matching loss
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
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.optimizer = torch.optim.Adam(self.student_model.parameters(), lr=learning_rate)    
        
        self.cosine_embedding_weight = cosine_embedding_weight
        self.ce_loss_weight = ce_loss_weight


    def train_epoch(self) -> float:
        """Train for one epoch using Cosine Embedding feature distillation.
        
        Returns:
            Average loss for the epoch
        """
        self.teacher_model.eval()
        self.student_model.train()


        running_loss = 0.0
        # model_weights = torch.load('/Users/chinya07/Documents/KD-pypi/teacher_model.pth')

        # # Load the weights into the model
        # self.teacher_model.load_state_dict(model_weights)
        
        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            # Get teacher features
            with torch.no_grad():
                _, teacher_hidden_representation = self.teacher_model(inputs)
            
            # Student forward pass
            
            student_logits, student_hidden_representation = self.student_model(inputs)
            
            # Calculate the cosine loss. Target is a vector of ones. From the loss formula above we can see that is the case where loss minimization leads to cosine similarity increase.
            hidden_rep_loss = self.cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(inputs.size(0)).to(self.device))


            # Calculate classification and feature matching losses
            label_loss = self.ce_loss(student_logits, labels)
            feature_loss = self.cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(inputs.size(0)).to(self.device))
            
            # Combined loss
            loss = self.cosine_embedding_weight * hidden_rep_loss + self.ce_loss_weight * label_loss
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
        return running_loss / len(self.train_loader) 