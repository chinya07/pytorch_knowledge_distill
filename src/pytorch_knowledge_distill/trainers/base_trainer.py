"""Base trainer implementation.

This module provides the base trainer class that implements common functionality
for model training and evaluation. It handles device management, optimization,
and basic training loops.

The BaseTrainer class can be extended to implement specific training strategies
like knowledge distillation or feature matching.

Example:
    >>> model = MyModel()
    >>> trainer = BaseTrainer(
    ...     model=model,
    ...     train_loader=train_loader,
    ...     test_loader=test_loader,
    ...     learning_rate=0.001
    ... )
    >>> loss = trainer.train_epoch()
    >>> accuracy = trainer.evaluate()
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseTrainer:
    """Base trainer class implementing common functionality.
    
    This class provides the basic infrastructure for training neural networks:
    - Device management (CPU/GPU)
    - Optimization setup
    - Training and evaluation loops
    
    Attributes:
        model: The neural network to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test/validation data
        learning_rate: Learning rate for optimization
        device: Device to run the model on
        criterion: Loss function (defaults to CrossEntropyLoss)
        optimizer: Optimizer (defaults to Adam)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        learning_rate: float = 0.001,
        device: Optional[torch.device] = None
    ) -> None:
        """Initialize the trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            test_loader: Test data loader
            learning_rate: Learning rate for optimization
            device: Device to run on (defaults to CUDA if available)
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        self.model.to(self.device)

    def train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        
        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
        return running_loss / len(self.train_loader)

    def evaluate(self) -> float:
        """Evaluate the model.
        
        Returns:
            Accuracy score
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                _, predicted = torch.max(logits.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        return 100 * correct / total 