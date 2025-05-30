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

def run_inference_benchmarks(models, snn_models, val_loader, batch_sizes):
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
                else:
                    start_time = time.time()
                    reconstruction = model(data)
                    
                end_time = time.time()
                total_time += (end_time - start_time)
                num_batches += 1
                
        return total_time / num_batches

    # Initialize lists to store results
    ann_times = []
    snn_times = []

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
    
    return results


def count_spikes_during_inference(snn_models, val_loader):
  """Count total spikes during inference for SNN models with ratio encoding"""
  spike_counts = []
  
  for model_idx, snn_model in enumerate(snn_models):
    print(f"\nCounting spikes for SNN Model {model_idx + 1}")
    snn_model.eval()
    
    total_spikes = 0
    total_samples = 0
    
    with torch.no_grad():
      for batch_idx, (data, _) in enumerate(val_loader):
        if batch_idx >= 10:  # Limit to first 10 batches for consistency
          break
        
        batch_size = data.size(0)
        
        # Process each sample individually
        for sample_idx in range(batch_size):
          sample_data = data[sample_idx:sample_idx+1]  # Keep batch dimension
          # sample_data shape is [1, 784]

          # For ratio encoding, create input spikes based on pixel intensities
          # Higher intensity = higher spike probability
          timesteps = 100
          input_spikes = (torch.rand(timesteps, 1, 784) < sample_data.view(1, -1)).float()
          
          # Count input spikes for this sample
          input_spike_count = input_spikes.sum().item()
          
          # Hook to capture internal spikes
          internal_spike_count = 0
          
          def spike_counter_hook(module, input, output):
            nonlocal internal_spike_count
            if output is not None:
              if isinstance(output, torch.Tensor):
                # Check if this is an IAFSqueeze unit and get its threshold
                if hasattr(module, 'spike_threshold'):
                  threshold = module.spike_threshold
                  # Count spikes based on threshold crossing
                  internal_spike_count += (output >= threshold).sum().item()
          
          # Register hooks on all layers to catch spikes
          hooks = []
          for name, module in snn_model.spiking_model.named_modules():
            # Register hook on all non-container modules
            if len(list(module.children())) == 0:  # Leaf modules only
              hook = module.register_forward_hook(spike_counter_hook)
              hooks.append(hook)
          
          # Forward pass through SNN
          output_spikes = snn_model.spiking_model(input_spikes.reshape(-1, 784))
          output_spike_count = output_spikes.sum().item()
          
          # Remove hooks
          for hook in hooks:
            hook.remove()
          
          # Total spikes for this sample = input spikes + internal spikes + output spikes
          sample_total_spikes = input_spike_count + internal_spike_count + output_spike_count
          total_spikes += sample_total_spikes
          total_samples += 1
          
          if sample_idx == 0 and batch_idx % 5 == 0:
            print(f"Batch {batch_idx}, Sample 0: {sample_total_spikes} spikes")
            print(f"  Input spikes: {input_spike_count}, Internal spikes: {internal_spike_count}, Output spikes: {output_spike_count}")
            print(f"  Output spikes shape: {output_spikes.shape}")
    
    avg_spikes_per_sample = total_spikes / total_samples if total_samples > 0 else 0
    spike_counts.append({
      'model_idx': model_idx,
      'total_spikes': total_spikes,
      'total_samples': total_samples,
      'avg_spikes_per_sample': avg_spikes_per_sample
    })
    
    print(f"Model {model_idx + 1}: {total_spikes} total spikes, {avg_spikes_per_sample:.2f} spikes/sample")
  
  return spike_counts

spikes = count_spikes_during_inference(snn_models, val_loader)
# Print spike statistics
print("\n" + "="*50)
print("SPIKE STATISTICS SUMMARY")
print("="*50)

total_spikes_all_models = sum(spike['total_spikes'] for spike in spikes)
total_samples_all_models = sum(spike['total_samples'] for spike in spikes)
avg_spikes_all_models = total_spikes_all_models / total_samples_all_models if total_samples_all_models > 0 else 0

print(f"Total spikes across all models: {total_spikes_all_models:,}")
print(f"Average spikes per sample across all models: {avg_spikes_all_models:.2f}")
print(f"Total samples processed: {total_samples_all_models}")

print("\nPer-model breakdown:")
for spike in spikes:
  print(f"Model {spike['model_idx'] + 1}: {spike['total_spikes']:,} total spikes, {spike['avg_spikes_per_sample']:.2f} avg spikes/sample")

# Create visualization for spike statistics (moved outside the loop)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Total spikes per model
model_indices = [spike['model_idx'] + 1 for spike in spikes]
total_spikes_per_model = [spike['total_spikes'] for spike in spikes]

ax1.bar(model_indices, total_spikes_per_model, color='skyblue', alpha=0.7)
ax1.set_xlabel('Model Index')
ax1.set_ylabel('Total Spikes')
ax1.set_title('Total Spikes per SNN Model')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(model_indices)

# Add value labels on bars
for i, v in enumerate(total_spikes_per_model):
  ax1.text(i + 1, v + max(total_spikes_per_model) * 0.01, f'{v:,}', 
       ha='center', va='bottom', fontsize=10)

# Plot 2: Average spikes per sample
avg_spikes_per_model = [spike['avg_spikes_per_sample'] for spike in spikes]

ax2.bar(model_indices, avg_spikes_per_model, color='lightcoral', alpha=0.7)
ax2.set_xlabel('Model Index')
ax2.set_ylabel('Average Spikes per Sample')
ax2.set_title('Average Spikes per Sample by SNN Model')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(model_indices)

# Add value labels on bars
for i, v in enumerate(avg_spikes_per_model):
  ax2.text(i + 1, v + max(avg_spikes_per_model) * 0.01, f'{v:.1f}', 
       ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('results/spike_statistics.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a summary comparison plot
fig, ax = plt.subplots(figsize=(12, 8))

# Create a grouped bar chart
x = np.arange(len(model_indices))
width = 0.35

# Normalize values for comparison (scale total spikes to similar range as avg spikes)
normalized_total_spikes = [s / 1000 for s in total_spikes_per_model]  # Scale down by 1000

bars1 = ax.bar(x - width/2, normalized_total_spikes, width, label='Total Spikes (×1000)', 
         color='skyblue', alpha=0.7)
bars2 = ax.bar(x + width/2, avg_spikes_per_model, width, label='Avg Spikes per Sample', 
         color='lightcoral', alpha=0.7)

ax.set_xlabel('Model Index')
ax.set_ylabel('Spike Count')
ax.set_title('SNN Model Spike Activity Comparison')
ax.set_xticks(x)
ax.set_xticklabels([f'Model {i}' for i in model_indices])
ax.legend()
ax.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars1, normalized_total_spikes):
  height = bar.get_height()
  ax.text(bar.get_x() + bar.get_width()/2., height + max(normalized_total_spikes) * 0.01,
      f'{value:.1f}k', ha='center', va='bottom', fontsize=9)

for bar, value in zip(bars2, avg_spikes_per_model):
  height = bar.get_height()
  ax.text(bar.get_x() + bar.get_width()/2., height + max(avg_spikes_per_model) * 0.01,
      f'{value:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/spike_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Save spike statistics
torch.save(spikes, 'results/spike_statistics.pt')



# main
