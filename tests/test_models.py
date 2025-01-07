"""Unit tests for model implementations."""

import unittest
import torch

from pytorch_knowledge_distill.models import DeepNN, LightNN


class TestModels(unittest.TestCase):
    """Test cases for neural network models."""

    def setUp(self):
        """Set up test cases."""
        self.batch_size = 4
        self.channels = 3
        self.height = 32
        self.width = 32
        self.num_classes = 10
        
        self.input_tensor = torch.randn(
            self.batch_size, self.channels, self.height, self.width
        )

    def test_deep_nn(self):
        """Test DeepNN forward pass."""
        model = DeepNN(num_classes=self.num_classes)
        output = model(self.input_tensor)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertTrue(torch.isfinite(output).all())

    def test_light_nn(self):
        """Test LightNN forward pass."""
        model = LightNN(num_classes=self.num_classes)
        output = model(self.input_tensor)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main() 