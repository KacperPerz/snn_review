import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split


ANOMALY_LABEL = 8  # Bag - more visually distinct from clothing items

# fmnist for ann or rate snn
def get_fmnist_ratio_dataloaders(batch_size=128, data_root='../data'):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)

    # split train dataset for train and validation
    train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])

    # Filter out zeros from training data
    train_data = []
    train_labels = []
    for data, label in train_dataset:
        if label != ANOMALY_LABEL:
            train_data.append(data.view(-1))
            train_labels.append(label)

    # Keep all training data
    train_data = torch.stack(train_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Keep all validation data
    val_data = torch.stack([data.view(-1) for data, _ in val_set])
    val_labels = torch.tensor([label for _, label in val_set])
    val_loader = DataLoader(TensorDataset(val_data, val_labels), batch_size=batch_size)

    # Keep all test data
    test_data = torch.stack([data.view(-1) for data, _ in test_dataset])
    test_labels = torch.tensor([label for _, label in test_dataset])
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=batch_size)

    normal_indices = (val_labels != ANOMALY_LABEL).nonzero(as_tuple=True)[0]
    normal_indices = val_labels != ANOMALY_LABEL
    anomaly_indices = val_labels == ANOMALY_LABEL

    # Randomly choose 1000 examples from val_data[indices_1_9]
    num_samples = 1000  
    random_indices = torch.randperm(len(val_data[normal_indices]))[:num_samples]
    val_data_reduced = val_data[normal_indices][random_indices]
    val_labels_reduced = val_labels[normal_indices][random_indices]

    val_data_reduced = torch.cat((val_data_reduced, val_data[anomaly_indices]), dim=0)
    val_labels_reduced = torch.cat((val_labels_reduced, val_labels[anomaly_indices]), dim=0)

    val_loader_reduced = DataLoader(TensorDataset(val_data_reduced, val_labels_reduced), batch_size=batch_size, shuffle=True)

    return train_loader, val_loader_reduced, test_loader


class FashionMNIST_Temporal(datasets.FashionMNIST):
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
        
        # Optimized vectorized spike setting (replaces slow loop)
        # Create mask for valid time indices
        valid_mask = temporal_code < self.time_steps
        valid_times = temporal_code[valid_mask]
        valid_indices = torch.arange(784, device=img_flat.device)[valid_mask]
        
        # Set spikes using advanced indexing - much faster than loop
        if len(valid_times) > 0:  # Only if there are valid spikes
            spikes[valid_times, valid_indices] = 1.0
        
        return spikes, img_flat, target # Return spikes, original flat image, and label

# fmnist for temporal snn
def get_fmnist_temporal_dataloaders(batch_size=128, data_root='../data'):
    """
    Loads and preprocesses MNIST data with temporal encoding for SNN autoencoders.

    Returns:
        train_loader_ae: DataLoader for training - returns (spikes, images, labels).
        val_loader_reduced_ae: DataLoader for validation - returns (spikes, images, labels).
        test_loader_ae: DataLoader for testing - returns (spikes, images, labels).
    """
    transform_ae = transforms.Compose([transforms.ToTensor()])

    # Load master datasets once
    master_train_val_dataset = FashionMNIST_Temporal(root=data_root, train=True, download=True, transform=transform_ae)
    master_test_dataset = FashionMNIST_Temporal(root=data_root, train=False, download=True, transform=transform_ae)

    # Perform train/validation split once
    # Ensure consistent splitting if seeds are set globally
    generator = torch.Generator().manual_seed(42) # Use a fixed seed for splitting
    train_subset, val_subset = random_split(master_train_val_dataset, [50000, 10000], generator=generator)

    # Train AE DataLoader (derived from master_train_val_dataset, filtering out digit '0')
    train_spikes_normal_ae = []
    train_images_normal_ae = []
    train_labels_normal_ae = []
    for spikes, img_flat, label in master_train_val_dataset:
        if label != ANOMALY_LABEL: # Filter out digit '0'
            train_spikes_normal_ae.append(spikes)
            train_images_normal_ae.append(img_flat)
            train_labels_normal_ae.append(label)
    
    if not train_spikes_normal_ae:
        raise ValueError("Training subset for autoencoder is empty after filtering out '0'. Check data or split.")

    train_spikes_ae_tensor = torch.stack(train_spikes_normal_ae)
    train_images_ae_tensor = torch.stack(train_images_normal_ae)
    train_labels_ae_tensor = torch.tensor(train_labels_normal_ae)
    train_loader_ae = DataLoader(TensorDataset(train_spikes_ae_tensor, train_images_ae_tensor, train_labels_ae_tensor), 
                                batch_size=batch_size, shuffle=True)
    
    # Count number of 0s and 1-9s in the original train_subset for verification
    original_train_labels = torch.tensor([label for _, _, label in master_train_val_dataset])
    num_zeros_in_train = (original_train_labels == ANOMALY_LABEL).sum().item()
    num_non_zeros_in_train = (original_train_labels != ANOMALY_LABEL).sum().item()
    print(f"Original master_train_val_dataset: {len(master_train_val_dataset)} samples. Zeros: {num_zeros_in_train}, Non-zeros: {num_non_zeros_in_train}")
    print(f"Filtered train_loader_ae: {len(train_spikes_ae_tensor)} samples (should be non-zeros).")

    # Test AE DataLoader (derived from master_test_dataset)
    test_spikes_ae = torch.stack([s for s, _, _ in master_test_dataset])
    test_images_ae = torch.stack([i for _, i, _ in master_test_dataset])
    test_labels_ae = torch.tensor([l for _, _, l in master_test_dataset])
    test_loader_ae = DataLoader(TensorDataset(test_spikes_ae, test_images_ae, test_labels_ae), 
                               batch_size=batch_size, shuffle=False)

    # Validation AE DataLoader (Reduced, derived from val_subset)
    val_spikes_full = torch.stack([s for s, _, _ in val_subset])
    val_images_full = torch.stack([i for _, i, _ in val_subset])
    val_labels_full = torch.tensor([l for _, _, l in val_subset])

    indices_1_9_mask = val_labels_full != ANOMALY_LABEL
    indices_0_mask = val_labels_full == ANOMALY_LABEL

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
    
    # Include labels for validation set
    val_labels_reduced_1_9 = val_labels_full[indices_1_9_mask][random_indices_1_9]
    val_labels_reduced_0 = val_labels_full[indices_0_mask]
    val_labels_reduced = torch.cat((val_labels_reduced_1_9, val_labels_reduced_0), dim=0)
    
    num_zeros_reduced = (val_labels_reduced == ANOMALY_LABEL).sum().item()
    num_non_zeros_reduced = (val_labels_reduced != ANOMALY_LABEL).sum().item()
    print(f"Reduced validation set: {len(val_labels_reduced)} samples. Zeros: {num_zeros_reduced}, Non-zeros: {num_non_zeros_reduced}")

    val_loader_reduced_ae = DataLoader(TensorDataset(val_spikes_reduced_ae_input, target_images_val_reduced_ae, val_labels_reduced), 
                                      batch_size=batch_size, shuffle=False)

    return train_loader_ae, val_loader_reduced_ae, test_loader_ae