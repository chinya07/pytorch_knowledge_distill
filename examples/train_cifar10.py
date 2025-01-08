import torch
import os
from torch.utils.data import Subset
from pytorch_knowledge_distill.models import DeepNN, LightNN, ModifiedDeepNN, ModifiedLightNN
from pytorch_knowledge_distill.trainers import BaseTrainer, DistillationTrainer, SSIMDistillationTrainer, CosineEmbeddingTrainer    
from pytorch_knowledge_distill.utils.data import get_cifar10_loaders

def main():
    # Get CIFAR-10 data
    train_loader, test_loader = get_cifar10_loaders(batch_size=64)

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
    # if not os.path.exists("/Users/chinya07/Documents/KD-pypi/teacher_model.pth"):
    # Initialize teacher model and train it first
    # teacher = DeepNN(num_classes=10)
    teacher = ModifiedDeepNN(num_classes=10)
    teacher_trainer = BaseTrainer(
        model=teacher,
        train_loader=small_train_loader,
        test_loader=small_test_loader,
        learning_rate=0.001
    )

    # Train teacher for a few epochs
    print("Training teacher model...")
    for epoch in range(5):
        loss = teacher_trainer.train_epoch()
        accuracy = teacher_trainer.evaluate()
        print(f"Teacher Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")
        
    # else:
        # teacher = ModifiedDeepNN(num_classes=10)
        # model_weights = torch.load("/Users/chinya07/Documents/KD-pypi/teacher_model.pth")
        # teacher.load_state_dict(model_weights)
    
    # Initialize student model
    student = ModifiedLightNN(num_classes=10)
    
    # ssim_trainer = SSIMDistillationTrainer(
    #     student_model=student,
    #     teacher_model=teacher,
    #     train_loader=small_train_loader,
    #     test_loader=small_test_loader,
    #     feature_weight=0.1,
    #     window_size=11
    # )


    # # Create distillation trainer
    # distill_trainer = DistillationTrainer(
    #     student_model=student,
    #     teacher_model=teacher,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     temperature=3.0,
    #     alpha=0.1,
    #     learning_rate=0.001
    # )

    cosine_embedding_trainer = CosineEmbeddingTrainer(
        student_model=student,
        teacher_model=teacher,
        train_loader=small_train_loader,
        test_loader=small_test_loader,
        learning_rate=0.001
    )
    
    # Train student using knowledge distillation
    print("\nTraining student model...")
    for epoch in range(10):
        loss = cosine_embedding_trainer.train_epoch()
        accuracy = cosine_embedding_trainer.evaluate()
        print(f"Student Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")
        
    # Save the trained models if needed
    torch.save(teacher.state_dict(), 'teacher_model.pth')
    torch.save(student.state_dict(), 'student_model.pth')

if __name__ == "__main__":
    main() 