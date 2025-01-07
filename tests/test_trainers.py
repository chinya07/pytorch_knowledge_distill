"""Unit tests for trainer implementations."""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from pytorch_knowledge_distill.trainers import (
    BaseTrainer,
    DistillationTrainer,
    MSEDistillationTrainer
)


class DummyModel(nn.Module):
    """Dummy model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.linear(x)


class TestTrainers(unittest.TestCase):
    """Test cases for trainer implementations."""

    def setUp(self):
        """Set up test cases."""
        # Create dummy data
        self.batch_size = 4
        self.input_size = 10
        self.num_classes = 2
        
        x = torch.randn(20, self.input_size)
        y = torch.randint(0, self.num_classes, (20,))
        dataset = TensorDataset(x, y)
        
        self.train_loader = DataLoader(dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(dataset, batch_size=self.batch_size)
        
        self.model = DummyModel()
        self.teacher_model = DummyModel()

    def test_base_trainer(self):
        """Test BaseTrainer functionality."""
        trainer = BaseTrainer(
            self.model,
            self.train_loader,
            self.test_loader
        )
        
        loss = trainer.train_epoch()
        accuracy = trainer.evaluate()
        
        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)
        self.assertTrue(0 <= accuracy <= 100)

    def test_distillation_trainer(self):
        """Test DistillationTrainer functionality."""
        trainer = DistillationTrainer(
            self.model,
            self.teacher_model,
            self.train_loader,
            self.test_loader
        )
        
        loss = trainer.train_epoch()
        accuracy = trainer.evaluate()
        
        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)
        self.assertTrue(0 <= accuracy <= 100)


if __name__ == '__main__':
    unittest.main() 