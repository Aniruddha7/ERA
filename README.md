# MNIST Classification with CI/CD Pipeline

A lightweight CNN implementation for MNIST digit classification with automated testing and deployment pipeline. This project demonstrates how to build an efficient model that achieves >95% accuracy in just one epoch while maintaining a small parameter footprint (<25,000 parameters).

## 🌟 Features

- Efficient CNN architecture with only ~23K parameters
- Achieves >95% accuracy in single epoch training
- Complete CI/CD pipeline with GitHub Actions
- Automated testing and model validation
- Model artifact storage

## 🏗️ Architecture

The model uses a simple but effective architecture:
- 3 Convolutional blocks with BatchNorm and MaxPooling
- Progressive channel expansion (16→32→32)
- Efficient classifier with minimal parameters
- No dropout layers to maximize single-epoch performance

## 📊 Model Performance

- Parameters: 23,850
- Training Accuracy: 95.51%
- Training Accuracy (with augmentation):  96.46% (single epoch)
- Training Time: ~2-3 minutes on CPU

## 🚀 Quick Start

1. Clone the repository:
