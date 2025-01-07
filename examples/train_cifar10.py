import torch
from pytorch_knowledge_distill.models import DeepNN, LightNN
from pytorch_knowledge_distill.trainers import BaseTrainer, DistillationTrainer
from pytorch_knowledge_distill.utils.data import get_cifar10_loaders

def main():
    # Get CIFAR-10 data
    train_loader, test_loader = get_cifar10_loaders(batch_size=64)
    
    # Initialize teacher model and train it first
    teacher = DeepNN(num_classes=10)
    teacher_trainer = BaseTrainer(
        model=teacher,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=0.001
    )
    
    # Train teacher for a few epochs
    print("Training teacher model...")
    for epoch in range(5):
        loss = teacher_trainer.train_epoch()
        accuracy = teacher_trainer.evaluate()
        print(f"Teacher Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")
    
    # Initialize student model
    student = LightNN(num_classes=10)
    
    # Create distillation trainer
    distill_trainer = DistillationTrainer(
        student_model=student,
        teacher_model=teacher,
        train_loader=train_loader,
        test_loader=test_loader,
        temperature=3.0,
        alpha=0.1,
        learning_rate=0.001
    )
    
    # Train student using knowledge distillation
    print("\nTraining student model...")
    for epoch in range(10):
        loss = distill_trainer.train_epoch()
        accuracy = distill_trainer.evaluate()
        print(f"Student Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")
        
    # Save the trained models if needed
    torch.save(teacher.state_dict(), 'teacher_model.pth')
    torch.save(student.state_dict(), 'student_model.pth')

if __name__ == "__main__":
    main() 