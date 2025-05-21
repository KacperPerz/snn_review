import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os
import time

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


# ------------------- -------------------
# Get DataLoaders for SNN Autoencoder Training
# ------------------- -------------------
batch_size_ae = 128
data_root_ae = '../data'

train_loader_ae, val_loader_reduced_ae, test_loader_ae = get_snn_autoencoder_dataloaders(
    batch_size=batch_size_ae, 
    data_root=data_root_ae
)

# ------------------- -------------------
# SNN Autoencoder Training
# ------------------- -------------------

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
learning_rate_ae = 2e-3
beta_snn = 0.9

# Epochs per model size (base architecture)
epochs_per_base_architecture = {
    "SmallSNNAutoencoder": 15,
    "MediumSNNAutoencoder": 20,
    "BigSNNAutoencoder": 25
}

def train_snn_autoencoder(model_name, model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    print(f"\n--- Training {model_name} ---")
    model.to(device)
    
    train_loss_history = []
    val_loss_history = []
    
    model_train_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        for batch_idx, (spike_data, target_images) in enumerate(train_loader):
            spike_data = spike_data.to(device)
            target_images = target_images.to(device)
            reconstructed_images = model(spike_data)
            loss = criterion(reconstructed_images, target_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], {model_name}, Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for spike_data_val, target_images_val in val_loader:
                spike_data_val = spike_data_val.to(device)
                target_images_val = target_images_val.to(device)
                reconstructed_images_val = model(spike_data_val)
                loss_val = criterion(reconstructed_images_val, target_images_val)
                val_loss += loss_val.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], {model_name}, Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}, Duration: {epoch_duration:.2f}s')
        
    model_train_end_time = time.time()
    model_total_duration = model_train_end_time - model_train_start_time
    print(f"--- Finished Training {model_name} in {model_total_duration:.2f}s ---")
    return model, train_loss_history, val_loss_history

criterion_ae = nn.MSELoss()

# Define the base SNN architectures to iterate over
base_architectures_map = {
    "SmallSNNAutoencoder": SmallSNNAutoencoder,
    "MediumSNNAutoencoder": MediumSNNAutoencoder,
    "BigSNNAutoencoder": BigSNNAutoencoder
}

# Define the latent sizes to iterate over for each architecture
latent_sizes_to_train = [8, 16, 32]

trained_snn_autoencoders = {}
all_loss_histories = {}

# Iterate over each base architecture type
for arch_name, model_class in base_architectures_map.items():
    # Iterate over each latent size for the current architecture
    for latent_dim in latent_sizes_to_train:
        model_name = f"{arch_name}_Latent_{latent_dim}"
        
        # Instantiate the model with the current latent_dim
        snn_ae_model = model_class(latent_size=latent_dim, beta=beta_snn)
        optimizer_ae = torch.optim.Adam(snn_ae_model.parameters(), lr=learning_rate_ae)
        
        # Get the number of epochs for the current base architecture
        current_num_epochs = epochs_per_base_architecture.get(arch_name, 10) # Default to 10
        
        trained_model, t_loss_hist, v_loss_hist = train_snn_autoencoder(
            model_name,
            snn_ae_model,
            train_loader_ae, 
            val_loader_reduced_ae, 
            criterion_ae,
            optimizer_ae,
            current_num_epochs,
            device
        )
        trained_snn_autoencoders[model_name] = trained_model
        all_loss_histories[model_name] = {
            'train_loss': t_loss_hist,
            'val_loss': v_loss_hist
        }

print("\nAll SNN Autoencoder training complete.")

# ------------------- -------------------
# Save Trained Models and Loss Histories
# ------------------- -------------------

models_save_dir = '../models/mnist/snn/temporal'
os.makedirs(models_save_dir, exist_ok=True)

for model_name, model_instance in trained_snn_autoencoders.items():
    # Save model state_dict
    model_save_path = os.path.join(models_save_dir, f"{model_name}.pth")
    torch.save(model_instance.state_dict(), model_save_path)
    print(f"Saved {model_name} model to {model_save_path}")

    # Save loss histories
    if model_name in all_loss_histories:
        losses_save_path = os.path.join(models_save_dir, f"{model_name}_losses.json")
        with open(losses_save_path, 'w') as f:
            json.dump(all_loss_histories[model_name], f, indent=4)
        print(f"Saved {model_name} loss histories to {losses_save_path}")


# Example: You can now access your trained models, e.g.:
# big_snn_ae = trained_snn_autoencoders['SNN_AE_Latent_32']
# medium_snn_ae = trained_snn_autoencoders['SNN_AE_Latent_16']
# small_snn_ae = trained_snn_autoencoders['SNN_AE_Latent_8']

# Further steps could include saving the models, evaluating on test_loader_ae, etc.

# (Commented out placeholder for original classification DataLoaders as they are not the current focus)
# # Optional: If the original filtered classification DataLoaders are still needed, they should be redefined here
# # based on train_subset, val_subset (for val_loader_reduced with labels), and master_test_dataset.
# # For now, this part is omitted to focus on AE data loading. If you need them, let me know.
# # Example: Original train_loader (filtered)
# # train_data_clf = []
# # train_labels_clf = []
# # for s, i, l in train_subset:
# #     if l != 0:
# #         train_data_clf.append(s) # Or i, depending on what that loader was for
# #         train_labels_clf.append(l)
# # train_data_clf = torch.stack(train_data_clf)
# # train_loader_clf = DataLoader(TensorDataset(train_data_clf, torch.tensor(train_labels_clf)), batch_size=128, shuffle=True)

# # Example: Original val_loader_reduced (for classification)
# # val_loader_reduced_clf = DataLoader(TensorDataset(val_spikes_reduced_ae_input, val_labels_for_reduced_set), batch_size=128, shuffle=True)
