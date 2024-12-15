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

if __name__ == "__main__":
    pytest.main([__file__]) 
