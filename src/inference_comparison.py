import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import json

# Add the project root directory to sys.path
_CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT_DIR = os.path.abspath(os.path.join(_CURRENT_SCRIPT_DIR, '..'))
if _PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_DIR)

from mnist_ratio_utils import load_models, convert_to_snn, get_mnist_ratio_dataloaders, load_temporal_models
from mnist_snn_loading import get_snn_autoencoder_dataloaders
from architecture.snn_architecture import SmallSNNAutoencoder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders_for_comparison(batch_sizes, data_root='../data'):
    """
    Create dataloaders for all three model types.
    
    Returns:
        ann_loaders: Dict of batch_size -> DataLoader (regular MNIST)
        rate_loaders: Dict of batch_size -> DataLoader (for rate encoding)
        temporal_loaders: Dict of batch_size -> DataLoader (temporal encoding)
    """
    
    
    
    ann_loaders = {}
    rate_loaders = {}  # Same as ANN but we'll process differently
    temporal_loaders = {}
    for batch_size in batch_sizes:
      _, test_set, _ = get_mnist_ratio_dataloaders(batch_size=batch_size, data_root=data_root)
      _, test_set_temporal, _ = get_snn_autoencoder_dataloaders(batch_size=batch_size, data_root=data_root)
      ann_loaders[batch_size] = test_set
      rate_loaders[batch_size] = test_set
      temporal_loaders[batch_size] = test_set_temporal
    
    return ann_loaders, rate_loaders, temporal_loaders

def load_all_models():
    """Load all model types for comparison."""
    
    # Load ANN models
    print("Loading ANN models...")
    ann_models = load_models()
    
    # Convert to rate-encoded SNN models
    print("Converting ANN to SNN models...")
    rate_snn_models = [convert_to_snn(model, num_steps=100) for model in ann_models]
    
    # Load temporal SNN models
    print("Loading temporal SNN models...")
    temporal_snn_models = load_temporal_models()
    
    return ann_models, rate_snn_models, temporal_snn_models

def measure_ann_inference_time(model, dataloader, device, num_batches=10):
    """Measure inference time for ANN models."""
    model = model.to(device)
    model.eval()
    
    times = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            data = data.to(device)
            
            start_time = time.time()
            _ = model(data)
            end_time = time.time()
            
            times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)

def measure_rate_snn_inference_time(model, dataloader, device, num_batches=10, num_timesteps=100):
    """Measure inference time for rate-encoded SNN models."""
    model = model.to(device)
    model.eval()
    
    times = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            data = data.to(device)
            batch_size = data.shape[0]
            
            # Create rate-encoded input following sinabs format
            # data is shape (batch_size, 784)
            # We need to create spikes over time dimension
            input_spikes = (torch.rand(num_timesteps, batch_size, 784, device=device) < data.unsqueeze(0)).float()
            
            start_time = time.time()
            
            # Reset state for sinabs models
            if hasattr(model, 'reset'):
                model.reset()
            elif hasattr(model, 'reset_states'):
                model.reset_states()
            # If no reset method available, the model might be stateless or reset automatically
            
            # Pass input in the correct format for sinabs models
            # Reshape to (num_timesteps * batch_size, 784) as expected by sinabs
            _ = model.spiking_model(input_spikes.reshape(-1, 784))
            end_time = time.time()
            
            times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)

def measure_temporal_snn_inference_time(model, dataloader, device, num_batches=10):
    """Measure inference time for temporal-encoded SNN models."""
    model = model.to(device)
    model.eval()
    
    times = []
    with torch.no_grad():
        for batch_idx, (spikes, _, _) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            spikes = spikes.to(device)
            
            start_time = time.time()
            _ = model(spikes)
            end_time = time.time()
            
            times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)

def compare_inference_times(batch_sizes=[16, 32, 64, 128], num_batches=10, device=None):
    """
    Compare inference times across all model types and batch sizes.
    
    Args:
        batch_sizes: List of batch sizes to test
        num_batches: Number of batches to average over
        device: Device to run inference on
    
    Returns:
        results: Dictionary containing all timing results
    """
    
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else 
                              "cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    ann_loaders, rate_loaders, temporal_loaders = create_dataloaders_for_comparison(batch_sizes)
    
    # Load models
    ann_models, rate_snn_models, temporal_snn_models = load_all_models()
    
    # Results storage
    results = {
        'batch_sizes': batch_sizes,
        'ann_times': {bs: [] for bs in batch_sizes},
        'ann_stds': {bs: [] for bs in batch_sizes},
        'rate_snn_times': {bs: [] for bs in batch_sizes},
        'rate_snn_stds': {bs: [] for bs in batch_sizes},
        'temporal_snn_times': {bs: [] for bs in batch_sizes},
        'temporal_snn_stds': {bs: [] for bs in batch_sizes},
        'model_names': {
            'ann': [model.__class__.__name__ for model in ann_models],
            'rate_snn': [f"Rate_{model.__class__.__name__}" for model in ann_models],
            'temporal_snn': [f"Temporal_{model.__class__.__name__}" for model in temporal_snn_models]
        }
    }
    
    # Test ANN models
    print("\nTesting ANN models...")
    for model_idx, model in enumerate(ann_models):
        model_name = model.__class__.__name__
        print(f"  Testing {model_name}...")
        
        for batch_size in batch_sizes:
            mean_time, std_time = measure_ann_inference_time(
                model, ann_loaders[batch_size], device, num_batches
            )
            results['ann_times'][batch_size].append(mean_time)
            results['ann_stds'][batch_size].append(std_time)
            print(f"    Batch size {batch_size}: {mean_time:.4f}±{std_time:.4f}s")
    
    # Test Rate-encoded SNN models
    print("\nTesting Rate-encoded SNN models...")
    for model_idx, model in enumerate(rate_snn_models):
        model_name = f"Rate_{ann_models[model_idx].__class__.__name__}"
        print(f"  Testing {model_name}...")
        
        for batch_size in batch_sizes:
            mean_time, std_time = measure_rate_snn_inference_time(
                model, rate_loaders[batch_size], device, num_batches
            )
            results['rate_snn_times'][batch_size].append(mean_time)
            results['rate_snn_stds'][batch_size].append(std_time)
            print(f"    Batch size {batch_size}: {mean_time:.4f}±{std_time:.4f}s")
    
    # Test Temporal-encoded SNN models
    print("\nTesting Temporal-encoded SNN models...")
    for model_idx, model in enumerate(temporal_snn_models):
        model_name = f"Temporal_{model.__class__.__name__}"
        print(f"  Testing {model_name}...")
        
        for batch_size in batch_sizes:
            mean_time, std_time = measure_temporal_snn_inference_time(
                model, temporal_loaders[batch_size], device, num_batches
            )
            results['temporal_snn_times'][batch_size].append(mean_time)
            results['temporal_snn_stds'][batch_size].append(std_time)
            print(f"    Batch size {batch_size}: {mean_time:.4f}±{std_time:.4f}s")
    
    return results

def plot_inference_comparison(results, save_dir="../plots"):
    """Create comprehensive plots of inference time comparison."""
    
    os.makedirs(save_dir, exist_ok=True)
    batch_sizes = results['batch_sizes']
    
    # Handle both integer and string keys (from JSON loading)
    def get_results_for_batch_size(results_dict, bs):
        # Try integer key first, then string key
        if bs in results_dict:
            return results_dict[bs]
        else:
            return results_dict[str(bs)]
    
    # Calculate average times across models for each type
    avg_ann_times = [np.mean(get_results_for_batch_size(results['ann_times'], bs)) for bs in batch_sizes]
    avg_rate_snn_times = [np.mean(get_results_for_batch_size(results['rate_snn_times'], bs)) for bs in batch_sizes]
    avg_temporal_snn_times = [np.mean(get_results_for_batch_size(results['temporal_snn_times'], bs)) for bs in batch_sizes]
    
    # Plot 1: Average inference times by batch size
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(batch_sizes, avg_ann_times, 'o-', label='ANN', linewidth=2)
    plt.plot(batch_sizes, avg_rate_snn_times, 's-', label='Rate SNN', linewidth=2)
    plt.plot(batch_sizes, avg_temporal_snn_times, '^-', label='Temporal SNN', linewidth=2)
    plt.xlabel('Batch Size')
    plt.xticks(batch_sizes)
    plt.ylabel('Average Inference Time (s)')
    plt.title('Average Inference Time vs Batch Size')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Plot 2: Inference time per sample
    plt.subplot(2, 2, 2)
    per_sample_ann = [t/bs for t, bs in zip(avg_ann_times, batch_sizes)]
    per_sample_rate = [t/bs for t, bs in zip(avg_rate_snn_times, batch_sizes)]
    per_sample_temporal = [t/bs for t, bs in zip(avg_temporal_snn_times, batch_sizes)]
    
    plt.plot(batch_sizes, per_sample_ann, 'o-', label='ANN', linewidth=2)
    plt.plot(batch_sizes, per_sample_rate, 's-', label='Rate SNN', linewidth=2)
    plt.plot(batch_sizes, per_sample_temporal, '^-', label='Temporal SNN', linewidth=2)
    plt.xlabel('Batch Size')
    plt.xticks(batch_sizes)
    plt.ylabel('Time per Sample (s)')
    plt.title('Inference Time per Sample vs Batch Size')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Plot 3: Speedup/Slowdown relative to ANN
    plt.subplot(2, 2, 3)
    rate_speedup = [ann/rate for ann, rate in zip(avg_ann_times, avg_rate_snn_times)]
    temporal_speedup = [ann/temp for ann, temp in zip(avg_ann_times, avg_temporal_snn_times)]
    
    plt.plot(batch_sizes, rate_speedup, 's-', label='Rate SNN vs ANN', linewidth=2)
    plt.plot(batch_sizes, temporal_speedup, '^-', label='Temporal SNN vs ANN', linewidth=2)
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Same as ANN')
    plt.xlabel('Batch Size')
    plt.xticks(batch_sizes)
    plt.ylabel('Speedup Factor (>1 means faster)')
    plt.title('Speedup Relative to ANN')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Plot 4: Bar chart for specific batch size (64)
    plt.subplot(2, 2, 4)
    if 64 in batch_sizes:
        idx_64 = batch_sizes.index(64)
        times_64 = [avg_ann_times[idx_64], avg_rate_snn_times[idx_64], avg_temporal_snn_times[idx_64]]
        labels = ['ANN', 'Rate SNN', 'Temporal SNN']
        colors = ['blue', 'orange', 'green']
        
        bars = plt.bar(labels, times_64, color=colors, alpha=0.7)
        plt.ylabel('Inference Time (s)')
        plt.title('Inference Time Comparison (Batch Size 64)')
        plt.yscale('log')
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times_64):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f'{time_val:.4f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/inference_time_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {save_dir}/inference_time_comparison.png")

def save_results(results, filename="../results/inference_comparison_results.json"):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            json_results[key] = {str(k): (v.tolist() if isinstance(v, np.ndarray) else v) 
                               for k, v in value.items()}
        else:
            json_results[key] = value.tolist() if isinstance(value, np.ndarray) else value
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {filename}")

# average inference time vs batch size across all models
def plot_average_inference_time(results, save_dir="../plots"):
    """Plot average inference time vs batch size across all models."""
    batch_sizes = results['batch_sizes']
    
    # Handle both integer and string keys (from JSON loading)
    def get_results_for_batch_size(results_dict, bs):
        # Try integer key first, then string key
        if bs in results_dict:
            return results_dict[bs]
        else:
            return results_dict[str(bs)]
    
    avg_ann_times = [np.mean(get_results_for_batch_size(results['ann_times'], bs)) for bs in batch_sizes]
    avg_rate_snn_times = [np.mean(get_results_for_batch_size(results['rate_snn_times'], bs)) for bs in batch_sizes]
    avg_temporal_snn_times = [np.mean(get_results_for_batch_size(results['temporal_snn_times'], bs)) for bs in batch_sizes]

    # Plot 1: Average inference times by batch size
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(batch_sizes, avg_ann_times, 'o-', label='ANN', linewidth=2)
    plt.plot(batch_sizes, avg_rate_snn_times, 's-', label='Rate SNN', linewidth=2)
    plt.plot(batch_sizes, avg_temporal_snn_times, '^-', label='Temporal SNN', linewidth=2)
    plt.xlabel('Batch Size')
    plt.xticks(batch_sizes)
    plt.ylabel('Average Inference Time (s)')
    plt.title('Average Inference Time vs Batch Size')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/average_inference_time_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

# inference time on average depending on architecture type (Small, Medium, Big)
def plot_inference_time_by_architecture(results, save_dir="../plots"):
    """Plot inference time on average depending on architecture type (Small, Medium, Big)."""
    # Group models by architecture type
    architecture_types = ['Small', 'Medium', 'Big']
    
    # Handle both integer and string keys (from JSON loading)
    def get_results_for_batch_size(results_dict, bs):
        # Try integer key first, then string key
        if bs in results_dict:
            return results_dict[bs]
        else:
            return results_dict[str(bs)]
    
    # Create figure with subplots
    plt.figure(figsize=(15, 10))
    
    # Plot for each batch size
    for i, bs in enumerate(results['batch_sizes']):
        plt.subplot(2, 2, i+1)
        
        # Calculate average times for each architecture type
        ann_times = []
        rate_snn_times = []
        temporal_snn_times = []
        
        for arch_type in architecture_types:
            # Get indices for current architecture type
            if arch_type == 'Small':
                indices = [0, 1, 2]  # First 3 models
            elif arch_type == 'Medium':
                indices = [3, 4, 5]  # Middle 3 models
            else:  # Big
                indices = [6, 7, 8]  # Last 3 models
            
            # Get batch size results
            ann_bs_results = get_results_for_batch_size(results['ann_times'], bs)
            rate_snn_bs_results = get_results_for_batch_size(results['rate_snn_times'], bs)
            temporal_snn_bs_results = get_results_for_batch_size(results['temporal_snn_times'], bs)
            
            # Calculate averages for current architecture (only if we have enough models)
            if len(ann_bs_results) > max(indices):
                ann_avg = np.mean([ann_bs_results[idx] for idx in indices])
                rate_snn_avg = np.mean([rate_snn_bs_results[idx] for idx in indices])
                temporal_snn_avg = np.mean([temporal_snn_bs_results[idx] for idx in indices])
            else:
                # Skip if we don't have enough models
                continue
            
            ann_times.append(ann_avg)
            rate_snn_times.append(rate_snn_avg)
            temporal_snn_times.append(temporal_snn_avg)
        
        # Plot bars
        x = np.arange(len(architecture_types))
        width = 0.25
        
        plt.bar(x - width, ann_times, width, label='ANN', color='blue')
        plt.bar(x, rate_snn_times, width, label='Rate SNN', color='red')
        plt.bar(x + width, temporal_snn_times, width, label='Temporal SNN', color='green')
        
        plt.xlabel('Architecture Type')
        plt.ylabel('Average Inference Time (s)')
        plt.title(f'Inference Time by Architecture (Batch Size: {bs})')
        plt.xticks(x, architecture_types)
        plt.legend()
        plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/inference_time_by_architecture.png", dpi=300, bbox_inches='tight')
    plt.close()

# load results from json file
def load_results(filename="../results/inference_comparison_results.json"):
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    print("Starting inference time comparison...")
    
    # Run comparison
    batch_sizes = [16, 32, 64, 128]
    if not os.path.exists("../results/inference_comparison_results.json"):
        results = compare_inference_times(batch_sizes=batch_sizes, num_batches=10)
    else:
        results = load_results()
    
    # Save results
    save_results(results)
    
    # Create plots
    plot_inference_comparison(results)

    # Plot average inference time vs batch size
    plot_average_inference_time(results)

    # Plot inference time by architecture type
    plot_inference_time_by_architecture(results)
    
    # Print summary
    print("\n" + "="*60)
    print("INFERENCE TIME COMPARISON SUMMARY")
    print("="*60)
    
    # Handle both integer and string keys (from JSON loading)
    def get_results_for_batch_size(results_dict, bs):
        # Try integer key first, then string key
        if bs in results_dict:
            return results_dict[bs]
        else:
            return results_dict[str(bs)]
    
    for bs in batch_sizes:
        print(f"\nBatch Size: {bs}")
        print("-" * 30)
        
        ann_times_bs = get_results_for_batch_size(results['ann_times'], bs)
        rate_snn_times_bs = get_results_for_batch_size(results['rate_snn_times'], bs)
        temporal_snn_times_bs = get_results_for_batch_size(results['temporal_snn_times'], bs)
        
        if ann_times_bs:
            avg_ann = np.mean(ann_times_bs)
            print(f"ANN:           {avg_ann:.4f}s")
        
        if rate_snn_times_bs:
            avg_rate = np.mean(rate_snn_times_bs)
            speedup_rate = avg_ann / avg_rate if avg_ann > 0 else 0
            print(f"Rate SNN:      {avg_rate:.4f}s ({speedup_rate:.2f}x)")
        
        if temporal_snn_times_bs:
            avg_temporal = np.mean(temporal_snn_times_bs)
            speedup_temporal = avg_ann / avg_temporal if avg_ann > 0 else 0
            print(f"Temporal SNN:  {avg_temporal:.4f}s ({speedup_temporal:.2f}x)")
    
    print("\nComparison completed successfully!") 