import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# Seed for reproducibility within data loading, if desired, though often set in main script
# torch.manual_seed(0)
# np.random.seed(0)

class MNIST_Temporal(datasets.MNIST):
    def __init__(self, root, train=True, download=True, transform=transforms.ToTensor(),
                 time_steps=25, t_max=1.0):
        """
        MNIST dataset with temporal (latency) encoding.
        
        Args:
            root: Root directory of dataset
            train: If True, creates dataset from training set, else from test set
            download: If True, downloads the dataset
            transform: pytorch transforms for preprocessing
            time_steps: Number of time steps for encoding
            t_max: Maximum time value (neurons with 0 intensity will spike at t_max)
        """
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.time_steps = time_steps
        self.t_max = t_max
    
    def __getitem__(self, index):
        """Get temporally encoded sample from dataset"""
        img, target = super().__getitem__(index)
        # Reshape image to 784 pixels (flattened 28x28)
        img_flat = img.view(-1)
        
        # Temporal encoding: Convert pixel intensity to spike timing
        # Higher intensity (closer to 1) = earlier spike (closer to 0)
        # Lower intensity (closer to 0) = later spike (closer to t_max)
        # Pixels with intensity 0 will not spike (set to t_max)
        
        # Create temporal encoding where spike_time = (1-intensity) * t_max
        temporal_code = (1.0 - img_flat) * self.t_max
        
        # Convert continuous time to discrete time steps
        temporal_code = (temporal_code * self.time_steps).long()
        
        # Generate one-hot encoded tensor of size [time_steps, 784]
        spikes = torch.zeros(self.time_steps, 784)
        # For each neuron, set spike to 1 at its specific time step
        for i, t in enumerate(temporal_code):
            if t < self.time_steps:  # Ensure time index is valid
                spikes[t, i] = 1.0
        
        return spikes, img_flat, target # Return spikes, original flat image, and label

def get_snn_autoencoder_dataloaders(batch_size=128, data_root='../data'):
    """
    Loads and preprocesses MNIST data with temporal encoding for SNN autoencoders.

    Returns:
        train_loader_ae: DataLoader for training.
        val_loader_reduced_ae: DataLoader for validation (reduced set with specific class distribution).
        test_loader_ae: DataLoader for testing.
    """
    transform_ae = transforms.Compose([transforms.ToTensor()])

    # Load master datasets once
    master_train_val_dataset = MNIST_Temporal(root=data_root, train=True, download=True, transform=transform_ae)
    master_test_dataset = MNIST_Temporal(root=data_root, train=False, download=True, transform=transform_ae)

    # Perform train/validation split once
    # Ensure consistent splitting if seeds are set globally
    generator = torch.Generator().manual_seed(42) # Use a fixed seed for splitting
    train_subset, val_subset = random_split(master_train_val_dataset, [50000, 10000], generator=generator)

    # Train AE DataLoader (derived from train_subset, filtering out digit '0')
    train_spikes_normal_ae = []
    train_images_normal_ae = []
    for spikes, img_flat, label in train_subset:
        if label != 0: # Filter out digit '0'
            train_spikes_normal_ae.append(spikes)
            train_images_normal_ae.append(img_flat)
    
    if not train_spikes_normal_ae:
        raise ValueError("Training subset for autoencoder is empty after filtering out '0'. Check data or split.")

    train_spikes_ae_tensor = torch.stack(train_spikes_normal_ae)
    train_images_ae_tensor = torch.stack(train_images_normal_ae)
    train_loader_ae = DataLoader(TensorDataset(train_spikes_ae_tensor, train_images_ae_tensor), batch_size=batch_size, shuffle=True)
    
    # Count number of 0s and 1-9s in the original train_subset for verification
    original_train_labels = torch.tensor([label for _, _, label in train_subset])
    num_zeros_in_train = (original_train_labels == 0).sum().item()
    num_non_zeros_in_train = (original_train_labels != 0).sum().item()
    print(f"Original train_subset: {len(train_subset)} samples. Zeros: {num_zeros_in_train}, Non-zeros: {num_non_zeros_in_train}")
    print(f"Filtered train_loader_ae: {len(train_spikes_ae_tensor)} samples (should be non-zeros).")

    # Test AE DataLoader (derived from master_test_dataset)
    test_spikes_ae = torch.stack([s for s, _, _ in master_test_dataset])
    test_images_ae = torch.stack([i for _, i, _ in master_test_dataset])
    test_loader_ae = DataLoader(TensorDataset(test_spikes_ae, test_images_ae), batch_size=batch_size, shuffle=False)

    # Validation AE DataLoader (Reduced, derived from val_subset)
    val_spikes_full = torch.stack([s for s, _, _ in val_subset])
    val_images_full = torch.stack([i for _, i, _ in val_subset])
    val_labels_full = torch.tensor([l for _, _, l in val_subset])

    indices_1_9_mask = val_labels_full != 0
    indices_0_mask = val_labels_full == 0

    # Ensure consistent sampling for val_loader_reduced_ae if seeds are set globally
    # For randperm, if global seed is set, it should be deterministic.
    # If not, and specific reproducibility for this part is needed, a local generator can be used for randperm.

    num_samples_1_9_desired = 1000 
    num_available_1_9 = (indices_1_9_mask).sum().item()
    actual_num_samples_1_9 = min(num_samples_1_9_desired, num_available_1_9)

    random_indices_1_9 = torch.randperm(num_available_1_9)[:actual_num_samples_1_9]

    val_spikes_reduced_1_9 = val_spikes_full[indices_1_9_mask][random_indices_1_9]
    val_spikes_reduced_0 = val_spikes_full[indices_0_mask]
    val_spikes_reduced_ae_input = torch.cat((val_spikes_reduced_1_9, val_spikes_reduced_0), dim=0)

    val_images_reduced_1_9 = val_images_full[indices_1_9_mask][random_indices_1_9]
    val_images_reduced_0 = val_images_full[indices_0_mask]
    target_images_val_reduced_ae = torch.cat((val_images_reduced_1_9, val_images_reduced_0), dim=0)
    
    # val_labels_for_reduced_set = torch.cat((val_labels_full[indices_1_9_mask][random_indices_1_9], val_labels_full[indices_0_mask]), dim=0)
    # print(f"Reduced validation set label counts: {torch.unique(val_labels_for_reduced_set, return_counts=True)}")

    val_loader_reduced_ae = DataLoader(TensorDataset(val_spikes_reduced_ae_input, target_images_val_reduced_ae), batch_size=batch_size, shuffle=False)

    return train_loader_ae, val_loader_reduced_ae, test_loader_ae

if __name__ == '__main__':
    # Example of how to use the function
    # Note: Seeds should be set here if you run this file directly for testing data loading
    torch.manual_seed(42)
    np.random.seed(42)

    print("Testing MNIST SNN data loading...")
    train_loader, val_loader, test_loader = get_snn_autoencoder_dataloaders()
    
    print(f"Number of batches in train_loader_ae: {len(train_loader)}")
    print(f"Number of batches in val_loader_reduced_ae: {len(val_loader)}")
    print(f"Number of batches in test_loader_ae: {len(test_loader)}")

    # Check a batch from train_loader
    spikes_batch, images_batch = next(iter(train_loader))
    print(f"Train spikes batch shape: {spikes_batch.shape}") # Expected: (batch_size, time_steps, 784)
    print(f"Train images batch shape: {images_batch.shape}")   # Expected: (batch_size, 784)

    # Check a batch from val_loader_reduced_ae
    spikes_val_batch, images_val_batch = next(iter(val_loader))
    print(f"Val spikes batch shape: {spikes_val_batch.shape}")
    print(f"Val images batch shape: {images_val_batch.shape}")

    print("Data loading test complete.") 