import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging
import datetime
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # First block
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Third block
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # Reduced from 64 to 32
        self.bn3 = nn.BatchNorm2d(32)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(32 * 3 * 3, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 10)
        )
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # 28x28 -> 14x14
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)  # 14x14 -> 7x7
        
        
        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)  # 7x7 -> 3x3
        
        # Classifier
        x = x.view(-1, 32 * 3 * 3)
        x = self.classifier(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def display_augmented_samples(dataset, num_samples=5):
    """Display original and augmented images side by side"""
    plt.figure(figsize=(2*num_samples, 4))
    
    # Get some random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx, img_idx in enumerate(indices):
        # Original image (before transforms)
        orig_img = dataset.data[img_idx].numpy()
        
        # Augmented image
        img, _ = dataset[img_idx]
        aug_img = img.squeeze().numpy()
        
        # Plot original
        plt.subplot(2, num_samples, idx + 1)
        plt.imshow(orig_img, cmap='gray')
        plt.axis('off')
        if idx == 0:
            plt.title('Original')
        
        # Plot augmented
        plt.subplot(2, num_samples, num_samples + idx + 1)
        plt.imshow(aug_img, cmap='gray')
        plt.axis('off')
        if idx == 0:
            plt.title('Augmented')
    
    plt.tight_layout()
    plt.savefig('augmented_samples.png')
    logging.info("Saved augmented samples visualization to 'augmented_samples.png'")
    plt.close()

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Optimized transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        # Reduced rotation angle
        transforms.RandomRotation((-2, 2), fill=0),
        # Very minimal translation
        transforms.RandomAffine(
            degrees=0,
            translate=(0.02, 0.02),
            fill=0
        ),
        # Added random erasing with small patches
        transforms.RandomErasing(
            p=0.1,
            scale=(0.01, 0.02),
            ratio=(0.3, 3.3),
            value=0
        )
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Display augmented samples
    display_augmented_samples(train_dataset, num_samples=5)
    
    # Increased batch size for better stability with augmentations
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = MNISTModel().to(device)
    num_params = count_parameters(model)
    logging.info(f"Number of trainable parameters: {num_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    # Slightly increased learning rate to compensate for augmentations
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.004,  # Increased max_lr
        steps_per_epoch=len(train_loader),
        epochs=1,
        pct_start=0.1,  # Faster warmup
        div_factor=10.0,
        final_div_factor=50.0
    )
    
    # Training loop
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    best_acc = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        running_loss += loss.item()
        
        current_acc = 100 * correct / total
        
        # Save best model
        if current_acc > best_acc and batch_idx > 50:
            best_acc = current_acc
            best_state = model.state_dict()
            
        if (batch_idx + 1) % 50 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            lr = optimizer.param_groups[0]['lr']
            logging.info(f'Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {avg_loss:.4f} - Accuracy: {current_acc:.2f}% - LR: {lr:.6f}')
    
    # Load best model state if it's better
    final_acc = max(current_acc, best_acc)
    if best_acc > current_acc:
        model.load_state_dict(best_state)
    
    logging.info(f"Final Training Accuracy: {final_acc:.2f}%")
    
    # Save model with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'mnist_model_{timestamp}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_accuracy': final_acc
    }, save_path)
    logging.info(f"Model saved to {save_path}")
    
    return model, final_acc

if __name__ == "__main__":
    model, train_acc = train_model()
    assert train_acc > 95, f"Training accuracy {train_acc:.2f}% is less than 95%"
