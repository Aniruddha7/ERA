import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pytest
from MNIST import MNISTModel
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_architecture():
    model = MNISTModel()
    
    # Test number of parameters
    num_params = count_parameters(model)
    assert num_params < 25000, f"Model has {num_params} parameters, should be less than 25000"
    logging.info(f"Model parameter count test passed: {num_params:,} parameters")

def test_training_accuracy():
    # Load latest model and its training accuracy
    import glob
    latest_model = max(glob.glob('mnist_model_*.pth'))
    checkpoint = torch.load(latest_model)
    train_accuracy = checkpoint['train_accuracy']
    
    logging.info(f"Training Accuracy: {train_accuracy:.2f}%")
    assert train_accuracy > 95, f"Training accuracy {train_accuracy:.2f}% is less than required 95%"

def test_model_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    model = MNISTModel().to(device)
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    logging.info("Test dataset loaded")
    
    # Load latest model
    import glob
    latest_model = max(glob.glob('mnist_model_*.pth'))
    checkpoint = torch.load(latest_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Loaded model from {latest_model}")
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            if (batch_idx + 1) % 2 == 0:
                logging.info(f'Testing Progress: [{batch_idx + 1}/{len(test_loader)}]')
    
    accuracy = 100 * correct / total
    logging.info(f"Test Accuracy: {accuracy:.2f}%")
    assert accuracy > 95, f"Model accuracy is {accuracy:.2f}%, should be > 95%"

if __name__ == "__main__":
    pytest.main([__file__]) 