import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os
import time
from tqdm import tqdm

# Add the project root directory to sys.path
# to allow for absolute imports from 'architecture' and 'src'
# when this script is run directly.
_CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT_DIR = os.path.abspath(os.path.join(_CURRENT_SCRIPT_DIR, '..'))
if _PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_DIR)

from src.mnist_snn_loading import get_snn_autoencoder_dataloaders
from architecture.snn_architecture import SmallSNNAutoencoder, MediumSNNAutoencoder, BigSNNAutoencoder


torch.manual_seed(0)
np.random.seed(0)
batch_size_ae = 128
data_root_ae = '../data'

print("Loading data...")
train_loader_ae, val_loader_ae, test_loader_ae = get_snn_autoencoder_dataloaders(
    batch_size=batch_size_ae, 
    data_root=data_root_ae
)

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
learning_rate_ae = 2e-3
beta_snn = 0.9

# Define all model configurations
model_configs = [
    {"class": SmallSNNAutoencoder, "name": "SmallSNNAutoencoder", "latent_size": 8, "epochs": 15},
    {"class": SmallSNNAutoencoder, "name": "SmallSNNAutoencoder", "latent_size": 16, "epochs": 15},
    {"class": SmallSNNAutoencoder, "name": "SmallSNNAutoencoder", "latent_size": 32, "epochs": 15},
    {"class": MediumSNNAutoencoder, "name": "MediumSNNAutoencoder", "latent_size": 8, "epochs": 20},
    {"class": MediumSNNAutoencoder, "name": "MediumSNNAutoencoder", "latent_size": 16, "epochs": 20},
    {"class": MediumSNNAutoencoder, "name": "MediumSNNAutoencoder", "latent_size": 32, "epochs": 20},
    {"class": BigSNNAutoencoder, "name": "BigSNNAutoencoder", "latent_size": 8, "epochs": 25},
    {"class": BigSNNAutoencoder, "name": "BigSNNAutoencoder", "latent_size": 16, "epochs": 25},
    {"class": BigSNNAutoencoder, "name": "BigSNNAutoencoder", "latent_size": 32, "epochs": 25}
]

# Create directories for saving models and plots
os.makedirs("../models/mnist/snn", exist_ok=True)
os.makedirs("../plots/snn_training", exist_ok=True)

def train_snn_autoencoder(model, train_loader, val_loader, epochs, model_name, device):
    """Train a single SNN autoencoder model"""
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_ae)
    
    train_losses = []
    val_losses = []
    
    print(f"\nTraining {model_name}...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for spikes_batch, images_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Move data to device
            spikes_batch = spikes_batch.to(device)
            images_batch = images_batch.to(device)
            # labels_batch is not needed for autoencoder training but available if needed
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(spikes_batch)
            loss = criterion(outputs, images_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for spikes_batch, images_batch, labels_batch in val_loader:
                spikes_batch = spikes_batch.to(device)
                images_batch = images_batch.to(device)
                
                outputs = model(spikes_batch)
                loss = criterion(outputs, images_batch)
                
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate average losses
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    return train_losses, val_losses

def plot_training_curves(train_losses, val_losses, model_name, save_path):
    """Plot and save training curves"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title(f'Training Curves for {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Dictionary to store all training results
all_results = {}

# # train only one model
# model = SmallSNNAutoencoder(latent_size=8, beta=beta_snn)
# model_name = "SmallSNNAutoencoder_8"
# train_losses, val_losses = train_snn_autoencoder(
#     model, train_loader_ae, val_loader_ae, 
#     15, model_name, device
# )

# plot_training_curves(train_losses, val_losses, model_name, f"../plots/snn_training/{model_name}_training_curves.png")

# # save it temporarily
# # Save model
# model_save_path = f"../models/mnist/snn/temporal/{model_name}.pth"
# torch.save(model.state_dict(), model_save_path)
# print(f"Model saved to: {model_save_path}")

# # Save training results
# results = {
#     "train_loss": train_losses,
#     "val_loss": val_losses
# }
# results_save_path = f"../models/mnist/snn/temporal/{model_name}_losses.json"
# with open(results_save_path, 'w') as f:
#     json.dump(results, f)
# print(f"Training results saved to: {results_save_path}")


# Train all models
for i, config in enumerate(model_configs):
    start_time = time.time()
    
    # Create model
    model = config["class"](latent_size=config["latent_size"], beta=beta_snn)
    model_name = f"{config['name']}_{config['latent_size']}"
    
    print(f"\n{'='*60}")
    print(f"Training Model {i+1}/9: {model_name}")
    print(f"Architecture: {config['name']}")
    print(f"Latent Size: {config['latent_size']}")
    print(f"Epochs: {config['epochs']}")
    print(f"{'='*60}")
    
    # Train the model
    train_losses, val_losses = train_snn_autoencoder(
        model, train_loader_ae, val_loader_ae, 
        config["epochs"], model_name, device
    )
    
    # Save model
    model_save_path = f"../models/mnist/snn/{model_name}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    # Plot and save training curves
    plot_save_path = f"../plots/snn_training/{model_name}_training_curves.png"
    plot_training_curves(train_losses, val_losses, model_name, plot_save_path)
    print(f"Training curves saved to: {plot_save_path}")
    
    # Store results
    training_time = time.time() - start_time
    all_results[model_name] = {
        "architecture": config["name"],
        "latent_size": config["latent_size"],
        "epochs": config["epochs"],
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "min_val_loss": min(val_losses),
        "training_time_seconds": training_time,
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final Train Loss: {train_losses[-1]:.6f}")
    print(f"Final Val Loss: {val_losses[-1]:.6f}")
    print(f"Min Val Loss: {min(val_losses):.6f}")

# Save all results
results_save_path = "../results/snn_training_results.json"
os.makedirs("../results", exist_ok=True)

with open(results_save_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*60}")
print("ALL TRAINING COMPLETED!")
print(f"{'='*60}")
print(f"All results saved to: {results_save_path}")

# Print summary table
print("\nTRAINING SUMMARY:")
print("-" * 80)
print(f"{'Model':<25} {'Latent':<8} {'Epochs':<8} {'Final Train':<12} {'Final Val':<12} {'Min Val':<12} {'Time (s)':<10}")
print("-" * 80)

for model_name, results in all_results.items():
    print(f"{model_name:<25} {results['latent_size']:<8} {results['epochs']:<8} "
          f"{results['final_train_loss']:<12.6f} {results['final_val_loss']:<12.6f} "
          f"{results['min_val_loss']:<12.6f} {results['training_time_seconds']:<10.1f}")

print("-" * 80)
total_time = sum(results['training_time_seconds'] for results in all_results.values())
print(f"Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

# Create comparison plot
plt.figure(figsize=(15, 10))

# Plot 1: Final validation losses by architecture and latent size
plt.subplot(2, 2, 1)
architectures = ["SmallSNNAutoencoder", "MediumSNNAutoencoder", "BigSNNAutoencoder"]
latent_sizes = [8, 16, 32]

for arch in architectures:
    val_losses = []
    for latent in latent_sizes:
        model_name = f"{arch}_{latent}"
        val_losses.append(all_results[model_name]['final_val_loss'])
    plt.plot(latent_sizes, val_losses, marker='o', label=arch)

plt.xlabel('Latent Size')
plt.ylabel('Final Validation Loss')
plt.title('Final Validation Loss by Architecture and Latent Size')
plt.legend()
plt.grid(True)

# Plot 2: Training time by architecture and latent size
plt.subplot(2, 2, 2)
for arch in architectures:
    times = []
    for latent in latent_sizes:
        model_name = f"{arch}_{latent}"
        times.append(all_results[model_name]['training_time_seconds'])
    plt.plot(latent_sizes, times, marker='s', label=arch)

plt.xlabel('Latent Size')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time by Architecture and Latent Size')
plt.legend()
plt.grid(True)

# Plot 3: Min validation loss comparison
plt.subplot(2, 2, 3)
for arch in architectures:
    min_val_losses = []
    for latent in latent_sizes:
        model_name = f"{arch}_{latent}"
        min_val_losses.append(all_results[model_name]['min_val_loss'])
    plt.plot(latent_sizes, min_val_losses, marker='^', label=arch)

plt.xlabel('Latent Size')
plt.ylabel('Minimum Validation Loss')
plt.title('Minimum Validation Loss by Architecture and Latent Size')
plt.legend()
plt.grid(True)

# Plot 4: Bar chart of all final validation losses
plt.subplot(2, 2, 4)
model_names = list(all_results.keys())
final_val_losses = [all_results[name]['final_val_loss'] for name in model_names]

colors = ['lightblue'] * 3 + ['lightgreen'] * 3 + ['lightcoral'] * 3
bars = plt.bar(range(len(model_names)), final_val_losses, color=colors)
plt.xlabel('Models')
plt.ylabel('Final Validation Loss')
plt.title('Final Validation Loss Comparison')
plt.xticks(range(len(model_names)), [name.replace('SNNAutoencoder', '') for name in model_names], rotation=45)

# Add value labels on bars
for bar, val in zip(bars, final_val_losses):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{val:.4f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("../plots/snn_training/all_models_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"Comparison plots saved to: ../plots/snn_training/all_models_comparison.png")
print("Training script completed successfully!")

