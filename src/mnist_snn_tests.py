import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the project root directory to sys.path
_CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT_DIR = os.path.abspath(os.path.join(_CURRENT_SCRIPT_DIR, '..'))
if _PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_DIR)

from architecture.snn_architecture import SmallSNNAutoencoder
from src.mnist_snn_loading import get_snn_autoencoder_dataloaders

# Load the trained model
model = SmallSNNAutoencoder(latent_size=8, beta=0.9)
model.load_state_dict(torch.load("../models/mnist/snn/temporal/SmallSNNAutoencoder_8.pth"))
model.eval()

# Get test data
_, test_loader, _ = get_snn_autoencoder_dataloaders(batch_size=16, data_root='../data')

# Get a batch of test data
spikes_batch, images_batch, labels_batch = next(iter(test_loader))

# Generate reconstructions
with torch.no_grad():
    reconstructions = model(spikes_batch)

# Create a figure to display original and reconstructed images
plt.figure(figsize=(12, 6))

# Display original images
plt.subplot(1, 2, 1)
plt.imshow(images_batch[0].reshape(28, 28), cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Display reconstructed image
plt.subplot(1, 2, 2)
plt.imshow(reconstructions[0].reshape(28, 28), cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')

plt.tight_layout()
plt.savefig('../plots/snn_training/reconstruction_example.png')
plt.close()

# Create a grid of reconstructions
n_samples = 8
plt.figure(figsize=(15, 5))

for i in range(n_samples):
    # Original
    plt.subplot(2, n_samples, i + 1)
    plt.imshow(images_batch[i].reshape(28, 28), cmap='gray')
    plt.title(f'Original {i+1}')
    plt.axis('off')
    
    # Reconstructed
    plt.subplot(2, n_samples, i + n_samples + 1)
    plt.imshow(reconstructions[i].reshape(28, 28), cmap='gray')
    plt.title(f'Reconstructed {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('../plots/snn_training/reconstruction_grid.png')
plt.close()

# Extract latent representations properly for SNN
print("Extracting latent representations...")
with torch.no_grad():
    # Reset encoder states
    from snntorch import utils
    utils.reset(model.encoder)
    
    # Process through encoder to get latent spikes
    batch_size, num_time_steps, num_features = spikes_batch.shape
    
    # Collect latent spikes over time
    latent_spikes_over_time = []
    for t in range(num_time_steps):
        current_input_slice = spikes_batch[:, t, :]  # Shape: (batch_size, num_features)
        spk_out_encoder_t, _mem_out_encoder_t = model.encoder(current_input_slice)
        latent_spikes_over_time.append(spk_out_encoder_t)
    
    # Stack and average over time to get representative latent features
    latent_spikes_tensor = torch.stack(latent_spikes_over_time, dim=0)  # (time_steps, batch_size, latent_size)
    latent_representations = latent_spikes_tensor.mean(dim=0)  # Average over time: (batch_size, latent_size)

print(f"Latent representations shape: {latent_representations.shape}")

# Plot latent space distribution (only if latent_size >= 2)
if latent_representations.shape[1] >= 2:
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_representations[:, 0].cpu().numpy(), 
                latent_representations[:, 1].cpu().numpy(), alpha=0.7)
    plt.title('Latent Space Distribution (First 2 Dimensions)')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.grid(True)
    plt.savefig('../plots/snn_training/latent_space_2d.png')
    plt.close()
    print("2D latent space plot saved.")

# Plot all latent dimensions
plt.figure(figsize=(15, 10))
latent_size = latent_representations.shape[1]

for i in range(latent_size):
    plt.subplot(2, 4, i + 1)
    plt.hist(latent_representations[:, i].cpu().numpy(), bins=20, alpha=0.7)
    plt.title(f'Latent Dimension {i+1}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)

plt.tight_layout()
plt.savefig('../plots/snn_training/latent_dimensions_histogram.png')
plt.close()

print("All visualizations saved to ../plots/snn_training/")
print("Generated files:")
print("  - reconstruction_example.png")
print("  - reconstruction_grid.png")
if latent_representations.shape[1] >= 2:
    print("  - latent_space_2d.png")
print("  - latent_dimensions_histogram.png")

# Calculate reconstruction errors
reconstruction_errors = []
true_labels = []

print("Calculating reconstruction errors...")
with torch.no_grad():
    for spikes_batch, images_batch, labels_batch in test_loader:
        # spikes_batch is already temporally encoded with shape (batch_size, time_steps, 784)
        # images_batch is the target reconstruction with shape (batch_size, 784)
        # labels_batch contains the true labels with shape (batch_size,)
        
        # Get reconstructions using the model's forward method
        reconstructions = model(spikes_batch)
        
        # Calculate MSE error between target images and reconstructions
        error = torch.mean((images_batch - reconstructions) ** 2, dim=1)
        reconstruction_errors.extend(error.cpu().numpy())
        true_labels.extend(labels_batch.cpu().numpy())

# Convert to numpy arrays
reconstruction_errors = np.array(reconstruction_errors)
true_labels = np.array(true_labels)

# Separate errors for anomalous (0) and normal (1-9) digits
anomalous_errors = reconstruction_errors[true_labels == 0]
normal_errors = reconstruction_errors[true_labels != 0]

print(f"Total samples: {len(reconstruction_errors)}")
print(f"Anomalous samples (digit 0): {len(anomalous_errors)}")
print(f"Normal samples (digits 1-9): {len(normal_errors)}")
print(f"Mean error for anomalous: {anomalous_errors.mean():.6f}")
print(f"Mean error for normal: {normal_errors.mean():.6f}")

# Plot error distribution
plt.figure(figsize=(10, 6))
plt.hist(anomalous_errors, bins=50, alpha=0.6, label=f'Anomalous (0) - {len(anomalous_errors)} samples', density=True)
plt.hist(normal_errors, bins=50, alpha=0.6, label=f'Normal (1-9) - {len(normal_errors)} samples', density=True)
plt.xlabel('Reconstruction Error (MSE)')
plt.ylabel('Density')
plt.title('Distribution of Reconstruction Errors by Class')
plt.legend()
plt.grid(True)
plt.savefig('../plots/snn_training/error_distribution.png')
plt.close()

print("Error distribution plot saved to ../plots/snn_training/error_distribution.png")

