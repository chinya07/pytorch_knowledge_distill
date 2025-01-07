"""PyTorch Knowledge Distillation package.

This package provides a simple implementation of Knowledge Distillation using PyTorch.
Knowledge Distillation is a model compression technique where a small model (student)
is trained to mimic a larger model (teacher) while maintaining similar performance.

Key Components:
    - Models: Deep and Light neural network implementations
    - Trainers: Various training strategies including standard KD and MSE-based KD
    - Utils: Data loading and preprocessing utilities

Example:
    >>> from pytorch_knowledge_distill.models import DeepNN, LightNN
    >>> from pytorch_knowledge_distill.trainers import DistillationTrainer
    >>> from pytorch_knowledge_distill.utils.data import get_cifar10_loaders
    >>>
    >>> # Get data loaders
    >>> train_loader, test_loader = get_cifar10_loaders()
    >>>
    >>> # Initialize models
    >>> teacher = DeepNN(num_classes=10)
    >>> student = LightNN(num_classes=10)
    >>>
    >>> # Create trainer
    >>> trainer = DistillationTrainer(
    ...     student_model=student,
    ...     teacher_model=teacher,
    ...     train_loader=train_loader,
    ...     test_loader=test_loader
    ... )
    >>>
    >>> # Train for one epoch
    >>> loss = trainer.train_epoch()
"""

__version__ = "0.1.0" 