import os
import torch
import numpy as np
import time
import gc
from torchvision import datasets, transforms
from mnist_ratio_utils import load_models, convert_to_snn, get_mnist_ratio_dataloaders

# 1. Liczbę neuronów
# 2. Liczbę synaps
# 3. Czas symulacji per epoka
# 4. Czas symulacji per spike
# 5. Sumaryczną liczbę spike'ów w sieci
# 6. Zajętość pamięci na model
# 7. Moc per spike
# 8. zestawić parametry samego projektu
import torch.nn as nn
import matplotlib.pyplot as plt

# get mnist data
train_loader, val_loader, test_loader = get_mnist_ratio_dataloaders()

# load models
models = load_models()

# convert models to snn
snn_models = [convert_to_snn(model) for model in models]

# calculate time of inference for different batch sizes of anns models and their snns counterparts
# Define batch sizes to test
batch_sizes = [16, 32, 64, 128]

# Initialize lists to store results
ann_times = []
snn_times = []

# Function to measure inference time
def measure_inference_time(model, data_loader, batch_size, is_snn=False):
    model.eval()
    total_time = 0
    num_batches = 0
    
    with torch.no_grad():
        for data, _ in data_loader:
            if num_batches >= 10:  # Measure only first 10 batches for consistency
                break
                
            data = data[:batch_size]  # Take only the specified batch size
            
            if is_snn:
                # For SNN, create input spikes
                input_spikes = (torch.rand(100, batch_size, 784) < data).float()
                start_time = time.time()
                output_spikes = model.spiking_model(input_spikes.reshape(-1, 784))
                # output_spikes = output_spikes.reshape(100, batch_size, 784)
                # reconstruction = output_spikes.mean(dim=0)
            else:
                start_time = time.time()
                reconstruction = model(data)
                
            end_time = time.time()
            total_time += (end_time - start_time)
            num_batches += 1
            
    return total_time / num_batches

# Test each model
for model_idx, (ann_model, snn_model) in enumerate(zip(models, snn_models)):
    print(f"\nTesting Model {model_idx + 1}")
    
    model_ann_times = []
    model_snn_times = []
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        # Measure ANN performance
        gc.collect()
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        ann_time = measure_inference_time(ann_model, val_loader, batch_size)
        model_ann_times.append(ann_time)
        
        # Measure SNN performance
        gc.collect()
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        snn_time = measure_inference_time(snn_model, val_loader, batch_size, is_snn=True)
        model_snn_times.append(snn_time)
    
    ann_times.append(model_ann_times)
    snn_times.append(model_snn_times)

# Calculate average times across all models
avg_ann_times = np.mean(ann_times, axis=0)
avg_snn_times = np.mean(snn_times, axis=0)

# Create single figure for averaged results
fig, ax = plt.subplots(figsize=(10, 6))

# Plot averaged inference times
ax.plot(batch_sizes, avg_ann_times, 'o-', label='ANN (Average)', color='blue')
ax.plot(batch_sizes, avg_snn_times, 's--', label='SNN (Average)', color='red')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Inference Time (s)')
ax.set_title('Average Inference Time vs Batch Size Across All Models')
ax.legend()
ax.grid(True)

# Set x-axis ticks to show batch sizes
ax.set_xticks(batch_sizes)
ax.set_xticklabels(batch_sizes)

plt.tight_layout()
plt.savefig('results/inference_benchmarks_average.png')
plt.close()

# Save numerical results
results = {
    'batch_sizes': batch_sizes,
    'ann_times': ann_times,
    'snn_times': snn_times,
    'avg_ann_times': avg_ann_times,
    'avg_snn_times': avg_snn_times
}

torch.save(results, 'results/inference_benchmarks.pt')

