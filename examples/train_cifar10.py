import torch
import argparse
import os
from torch.utils.data import Subset
from pytorch_knowledge_distill.models import (
    DeepNN, 
    LightNN, 
    CosineEmbeddingDeepNN, 
    CosineEmbeddingLightNN, 
    ModifiedLightRegressorNN, 
    ModifiedDeepRegressorNN
)
from pytorch_knowledge_distill.trainers import (
    BaseTrainer, 
    DistillationTrainer, 
    SSIMDistillationTrainer, 
    CosineEmbeddingTrainer, 
    MSEDistillationTrainer
)
from pytorch_knowledge_distill.utils.data import get_cifar10_loaders

def parse_args():
    parser = argparse.ArgumentParser(description='Knowledge Distillation Training Script')
    parser.add_argument(
        '--kd_type',
        type=str,
        choices=['kl_div', 'cosine', 'ssim', 'mse'],
        required=True,
        help='Type of knowledge distillation to use'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate for training'
    )
    parser.add_argument(
        '--teacher_epochs',
        type=int,
        default=5,
        help='Number of epochs to train teacher'
    )
    parser.add_argument(
        '--student_epochs',
        type=int,
        default=10,
        help='Number of epochs to train student'
    )
    return parser.parse_args()

def get_models_and_trainer(args, train_loader, test_loader):
    """Configure models and trainer based on KD type."""
    
    if args.kd_type == 'kl_div':
        teacher = DeepNN(num_classes=10)
        student = LightNN(num_classes=10)
        trainer = DistillationTrainer(
            student_model=student,
            teacher_model=teacher,
            train_loader=train_loader,
            test_loader=test_loader,
            temperature=3.0,
            alpha=0.1,
            learning_rate=args.learning_rate
        )
    
    elif args.kd_type == 'cosine':
        teacher = CosineEmbeddingDeepNN(num_classes=10)
        student = CosineEmbeddingLightNN(num_classes=10)
        trainer = CosineEmbeddingTrainer(
            student_model=student,
            teacher_model=teacher,
            train_loader=train_loader,
            test_loader=test_loader,
            learning_rate=args.learning_rate
        )
    
    elif args.kd_type == 'ssim':
        teacher = DeepNN(num_classes=10)
        student = LightNN(num_classes=10)
        trainer = SSIMDistillationTrainer(
            student_model=student,
            teacher_model=teacher,
            train_loader=train_loader,
            test_loader=test_loader,
            feature_weight=0.1,
            window_size=11,
            learning_rate=args.learning_rate
        )
    
    elif args.kd_type == 'mse':
        teacher = ModifiedDeepRegressorNN(num_classes=10)
        student = ModifiedLightRegressorNN(num_classes=10)
        trainer = MSEDistillationTrainer(
            student_model=student,
            teacher_model=teacher,
            train_loader=train_loader,
            test_loader=test_loader,
            learning_rate=args.learning_rate
        )
    
    return teacher, student, trainer

def main():
    args = parse_args()
    
    # Get CIFAR-10 data
    train_loader, test_loader = get_cifar10_loaders(batch_size=64)

    # Create smaller subset for faster testing
    train_subset = Subset(train_loader.dataset, range(100))
    test_subset = Subset(test_loader.dataset, range(100))
    
    # Create new data loaders with subsets
    small_train_loader = torch.utils.data.DataLoader(
        train_subset, 
        batch_size=64,
        shuffle=True
    )
    small_test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=64,
        shuffle=False
    )

    # Get appropriate models and trainer
    teacher, student, trainer = get_models_and_trainer(args, small_train_loader, small_test_loader)

    # Train teacher model first
    teacher_trainer = BaseTrainer(
        model=teacher,
        train_loader=small_train_loader,
        test_loader=small_test_loader,
        learning_rate=args.learning_rate
    )

    print(f"Training teacher model using {args.kd_type} approach...")
    for epoch in range(args.teacher_epochs):
        loss = teacher_trainer.train_epoch()
        accuracy = teacher_trainer.evaluate()
        print(f"Teacher Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")

    # Train student using knowledge distillation
    print(f"\nTraining student model using {args.kd_type} distillation...")
    for epoch in range(args.student_epochs):
        loss = trainer.train_epoch()
        accuracy = trainer.evaluate()
        print(f"Student Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")
        
    # Save the trained models
    torch.save(teacher.state_dict(), f'teacher_model_{args.kd_type}.pth')
    torch.save(student.state_dict(), f'student_model_{args.kd_type}.pth')

if __name__ == "__main__":
    main() 