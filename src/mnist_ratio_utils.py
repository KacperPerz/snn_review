import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import time
from sinabs.from_torch import from_model
from torchvision import datasets
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from architecture.ann_architecture import *


def conversion_ann_snn(ann_model):
    result = nn.Sequential()
    for name, layer in ann_model.encoder.named_children():
        result.add_module(f"encoder_{name}", layer)
    for name, layer in ann_model.decoder.named_children():
        result.add_module(f"decoder_{name}", layer)
    return result

def convert_to_snn(a, num_steps=100):
    a = conversion_ann_snn(a)
    snn = from_model(
        a,
        input_shape=(784,),
        add_spiking_output=True, # ?
        num_timesteps=num_steps
    )
    return snn

def detect_anomalies_spiking(snn_model, val_loader_reduced, device, num_examples=500):
    reconstructions = []
    originals = []
    labels = []
    errors = []
    
    with torch.no_grad():
        # Get first batch of examples
        for data, label in val_loader_reduced:
            if len(originals) >= num_examples:
                break
                
            for example, l in zip(data, label):
                if len(originals) >= num_examples:
                    break
                    
                # Create input spikes
                example = example.to(device)
                input_spikes = (torch.rand(100, 1, 784, device=device) < example).float()
                
                # Get reconstruction using spiking model
                output_spikes = snn_model.spiking_model(input_spikes.reshape(-1, 784))
                output_spikes = output_spikes.reshape(100, -1, 784)
                
                # Average spikes over time dimension to get reconstruction
                reconstruction = output_spikes.mean(dim=0).squeeze()
                
                # Calculate reconstruction error
                error = torch.mean((reconstruction - example) ** 2)
                
                # Store results
                reconstructions.append(reconstruction.cpu())
                originals.append(example.cpu())
                labels.append(l.cpu())
                errors.append(error.cpu())
    return errors, reconstructions, originals, labels

def check_thresholds(errors, labels):
    f1_scores = []

    # Sort the errors and get corresponding labels
    errors = torch.tensor(errors)
    sorted_errors, indices = torch.sort(errors, descending=True)
    sorted_labels = [labels[i] for i in indices.cpu()]

    thresholds = torch.linspace(0, 1, 20)
    # Calculate metrics at different thresholds
    for threshold in thresholds:
        # Determine predictions based on threshold
        predictions = sorted_errors >= torch.quantile(errors, threshold)
        
        # Convert labels to binary (0 for anomaly, 1 for normal)
        binary_labels = torch.tensor([1 if label != 0 else 0 for label in sorted_labels])
        
        # Calculate confusion matrix values
        tp = torch.sum((predictions == True) & (binary_labels == 0))
        fp = torch.sum((predictions == True) & (binary_labels == 1))
        tn = torch.sum((predictions == False) & (binary_labels == 1))
        fn = torch.sum((predictions == False) & (binary_labels == 0))
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.)
        recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.)
        
        f1_scores.append(f1)

    # find the best threshold according to f1 score
    best_threshold_index = torch.argmax(torch.tensor(f1_scores))
    best_threshold = thresholds[best_threshold_index]
    best_f1_score = f1_scores[best_threshold_index]

    return best_threshold, best_f1_score, indices, sorted_labels, sorted_errors

def load_models():
  models_dir = '../models/mnist/ann'
  model_names = ['SmallAutoencoder_32.pth', 'SmallAutoencoder_16.pth', 'SmallAutoencoder_8.pth',
           'BigAutoencoder_32.pth', 'BigAutoencoder_16.pth', 'BigAutoencoder_8.pth',
           'Autoencoder_32.pth', 'Autoencoder_16.pth', 'Autoencoder_8.pth']
  
  # First check all model files exist
  for model_name in model_names:
    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path):
      raise FileNotFoundError(f"Model file not found: {model_path}")
  
  loaded_models = []
  for model_name in model_names:
    model_path = os.path.join(models_dir, model_name)
    # Create an empty model instance based on the model name
    if 'SmallAutoencoder_32' in model_name:
      model = SmallAutoencoder()
    elif 'SmallAutoencoder_16' in model_name:
      model = SmallAutoencoder_16()
    elif 'SmallAutoencoder_8' in model_name:
      model = SmallAutoencoder_8()
    elif 'BigAutoencoder_32' in model_name:
      model = BigAutoencoder()
    elif 'BigAutoencoder_16' in model_name:
      model = BigAutoencoder_16()
    elif 'BigAutoencoder_8' in model_name:
      model = BigAutoencoder_8()
    elif 'Autoencoder_32' in model_name:
      model = Autoencoder()
    elif 'Autoencoder_16' in model_name:
      model = Autoencoder_16()
    elif 'Autoencoder_8' in model_name:
      model = Autoencoder_8()
    else:
      raise ValueError(f"Unknown model name: {model_name}")
      
    # Load the state dictionary
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    loaded_models.append(model)
  
  return loaded_models


class MNIST_ratio(datasets.MNIST):
    def __init__(self, root, train=True, is_spiking=False, time_window=100):
        super().__init__(
            root=root, train=train, download=True, transform=transforms.ToTensor()
        )
        self.is_spiking = is_spiking
        self.time_window = time_window

    def __getitem__(self, index):
        img, target = self.data[index].view(1, -1), self.targets[index]
        # img is now a tensor of 1x784

        if self.is_spiking:
            img = (torch.rand(self.time_window, *img.shape) < img).float()

        return img, target
    

def get_mnist_ratio_dataloaders(batch_size=128, data_root='../data'):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = MNIST_ratio(root=data_root, train=True, is_spiking=False, time_window=100)
    test_dataset = MNIST_ratio(root=data_root, train=False, is_spiking=False, time_window=100)

    # split train dataset for train and validation
    train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])

    # Filter out zeros from training data
    train_data = []
    train_labels = []
    for data, label in train_dataset:
        if label != 0:
            train_data.append(data.view(-1))
            train_labels.append(label)

    # Keep all training data
    train_data = torch.stack(train_data)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    # Keep all validation data
    val_data = torch.stack([data.view(-1) for data, _ in val_set])
    val_labels = torch.tensor([label for _, label in val_set])
    val_loader = DataLoader(TensorDataset(val_data, val_labels), batch_size=128)

    # Keep all test data
    test_data = torch.stack([data.view(-1) for data, _ in test_dataset])
    test_labels = torch.tensor([label for _, label in test_dataset])
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=128)

    indices_1_9 = (val_labels != 0).nonzero(as_tuple=True)[0]
    indices_1_9 = val_labels != 0
    indices_0 = val_labels == 0

    # Randomly choose 1000 examples from val_data[indices_1_9]
    num_samples = 1000
    random_indices = torch.randperm(len(val_data[indices_1_9]))[:num_samples]
    val_data_reduced = val_data[indices_1_9][random_indices]
    val_labels_reduced = val_labels[indices_1_9][random_indices]

    val_data_reduced = torch.cat((val_data_reduced, val_data[indices_0]), dim=0)
    val_labels_reduced = torch.cat((val_labels_reduced, val_labels[indices_0]), dim=0)

    val_loader_reduced = DataLoader(TensorDataset(val_data_reduced, val_labels_reduced), batch_size=128, shuffle=True)

    return train_loader, val_loader_reduced, test_loader
