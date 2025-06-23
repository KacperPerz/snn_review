import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# Add the project root directory to sys.path
_CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT_DIR = os.path.abspath(os.path.join(_CURRENT_SCRIPT_DIR, '..'))
if _PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_DIR)

from mnist_ratio_utils import load_temporal_models
from mnist_snn_loading import get_snn_autoencoder_dataloaders


def detect_anomalies_temporal_snn(model, dataloader, device, num_examples=2000):
    """
    Detect anomalies using temporal SNN models.
    
    Args:
        model: Temporal SNN autoencoder model
        dataloader: DataLoader with temporal encoded data (spikes, images, labels)
        device: Device to run inference on
        num_examples: Number of examples to process
    
    Returns:
        errors: List of reconstruction errors
        reconstructions: List of reconstructed images
        originals: List of original images
        labels: List of labels
    """
    model = model.to(device)
    model.eval()
    
    reconstructions = []
    originals = []
    labels = []
    errors = []
    
    with torch.no_grad():
        for spikes, images, batch_labels in dataloader:
            if len(originals) >= num_examples:
                break
                
            for spike_seq, original_img, label in zip(spikes, images, batch_labels):
                if len(originals) >= num_examples:
                    break
                
                # Move to device
                spike_seq = spike_seq.to(device)
                original_img = original_img.to(device)
                
                # Get reconstruction from temporal SNN
                # spike_seq shape: (time_steps, features)
                spike_seq = spike_seq.unsqueeze(0)  # Add batch dimension: (1, time_steps, features)
                
                # Forward pass through the temporal SNN
                reconstruction = model(spike_seq)
                
                # reconstruction is (batch_size, features), remove batch dimension
                reconstruction = reconstruction.squeeze(0)
                
                # Flatten original image for comparison
                original_flat = original_img.view(-1)
                
                # Calculate reconstruction error (MSE)
                error = torch.mean((reconstruction - original_flat) ** 2)
                
                # Store results
                reconstructions.append(reconstruction.cpu())
                originals.append(original_flat.cpu())
                labels.append(label.cpu())
                errors.append(error.cpu())
    
    return errors, reconstructions, originals, labels


def check_thresholds_temporal(errors, labels):
    """
    Check different thresholds for anomaly detection and calculate F1 scores.
    """
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

    # Find the best threshold according to f1 score
    best_threshold_index = torch.argmax(torch.tensor(f1_scores))
    best_threshold = thresholds[best_threshold_index]
    best_f1_score = f1_scores[best_threshold_index]

    return best_threshold, best_f1_score, indices, sorted_labels, sorted_errors


def plot_error_distribution_temporal(errors, labels, threshold, title, save_path=None):
    """
    Plot error distribution for temporal SNN anomaly detection.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to numpy for easier indexing
    errors_np = np.array(errors)
    labels_np = np.array(labels)
    
    # Separate errors for anomalous and normal data
    anomalous_errors = errors_np[labels_np == 0]
    normal_errors = errors_np[labels_np != 0]
    
    # Plot histograms
    ax.hist(anomalous_errors, bins=50, alpha=0.6, label='Anomalous (0)', density=True, color='red')
    ax.hist(normal_errors, bins=50, alpha=0.6, label='Normal (1-9)', density=True, color='blue')
    ax.axvline(np.quantile(errors_np, threshold), color='green', linestyle='--', 
             label=f'Threshold ({threshold:.2f})')
    ax.set_xlabel('Reconstruction Error')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_reconstruction_examples_temporal(originals, reconstructions, labels, errors, model_name, save_path=None):
    """
    Plot reconstruction examples for temporal SNN models.
    """
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle(f'Reconstruction Examples - {model_name}', fontsize=14)
    
    # Get some examples - mix of good and bad reconstructions
    error_indices = np.argsort(errors)
    good_indices = error_indices[:10]  # Best reconstructions
    bad_indices = error_indices[-10:]  # Worst reconstructions
    
    # Select 10 examples (5 good, 5 bad)
    selected_indices = np.concatenate([good_indices[:5], bad_indices[-5:]])
    
    for i, idx in enumerate(selected_indices):
        row = i // 5
        col = i % 5
        
        # Original image
        axes[row*2, col].imshow(originals[idx].reshape(28, 28), cmap='gray')
        axes[row*2, col].set_title(f'Original (Label: {labels[idx]})')
        axes[row*2, col].axis('off')
        
        # Reconstruction
        axes[row*2+1, col].imshow(reconstructions[idx].reshape(28, 28), cmap='gray')
        axes[row*2+1, col].set_title(f'Recon (Error: {errors[idx]:.4f})')
        axes[row*2+1, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def main():
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load temporal SNN models
    print("Loading temporal SNN models...")
    temporal_models = load_temporal_models()
    print(f"Loaded {len(temporal_models)} temporal SNN models")
    
    # Load validation data with temporal encoding
    print("Loading temporal validation data...")
    _, val_loader, _ = get_snn_autoencoder_dataloaders(batch_size=32, data_root='../data')
    
    # Inspect sample batch
    sample_spikes, sample_images, sample_labels = next(iter(val_loader))
    print(f"Spike data shape: {sample_spikes.shape}")
    print(f"Image data shape: {sample_images.shape}")
    print(f"Labels shape: {sample_labels.shape}")
    print(f"Spike data range: min={sample_spikes.min().item():.4f}, max={sample_spikes.max().item():.4f}")
    
    # Create lists to store results for all models
    all_errors = []
    all_reconstructions = []
    all_originals = []
    all_labels = []
    all_metrics = []
    
    # Define model names
    model_names = [
        'SmallSNNAutoencoder_32', 'SmallSNNAutoencoder_16', 'SmallSNNAutoencoder_8',
        'MediumSNNAutoencoder_32', 'MediumSNNAutoencoder_16', 'MediumSNNAutoencoder_8',
        'BigSNNAutoencoder_32', 'BigSNNAutoencoder_16', 'BigSNNAutoencoder_8'
    ]
    
    # Process each temporal SNN model
    for model_idx, model in enumerate(tqdm(temporal_models, desc="Processing temporal SNN models")):
        model_name = model_names[model_idx] if model_idx < len(model_names) else f"temporal_model_{model_idx}"
        print(f"\nProcessing {model_name}")
        
        # Detect anomalies using temporal SNN
        errors, reconstructions, originals, labels = detect_anomalies_temporal_snn(
            model, val_loader, device, num_examples=2000
        )
        
        # Calculate metrics
        best_threshold, best_f1_score, indices, sorted_labels, sorted_errors = check_thresholds_temporal(
            errors, labels
        )
        
        # Calculate additional metrics
        sorted_labels_indices = torch.tensor(sorted_labels)
        zeros_indices = sorted_labels_indices == 0
        other_indices = sorted_labels_indices != 0
        
        positive_class = torch.tensor(sorted_errors) >= torch.quantile(torch.tensor(sorted_errors), best_threshold)
        negative_class = torch.tensor(sorted_errors) < torch.quantile(torch.tensor(sorted_errors), best_threshold)
        
        tp = positive_class[zeros_indices].sum()
        fn = negative_class[zeros_indices].sum()
        tn = negative_class[other_indices].sum()
        fp = positive_class[other_indices].sum()
        
        precision = tp/(tp + fp) if (tp + fp) > 0 else torch.tensor(0.)
        recall = tp/(tp + fn) if (tp + fn) > 0 else torch.tensor(0.)
        
        # Store results
        all_errors.append(sorted_errors)
        all_reconstructions.append(reconstructions)
        all_originals.append(originals)
        all_labels.append(sorted_labels)
        all_metrics.append({
            'threshold': best_threshold,
            'f1_score': best_f1_score,
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fn': fn,
            'tn': tn,
            'fp': fp
        })
        
        print(f"Model metrics:")
        print(f"Best Threshold: {best_threshold:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {best_f1_score:.4f}")
    
    # Save results
    print("\nSaving results...")
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    
    save_path = os.path.join(results_dir, "mnist_temporal_snn_inference_results.pt")
    torch.save({
        'errors': all_errors,
        'reconstructions': all_reconstructions,
        'originals': all_originals,
        'labels': all_labels,
        'metrics': all_metrics,
        'model_names': model_names[:len(temporal_models)]
    }, save_path)
    print(f"Results saved to {save_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plots_dir = os.path.join(results_dir, "temporal_snn_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot error distributions for each model
    for i, (errors, labels, metrics, model_name) in enumerate(zip(all_errors, all_labels, all_metrics, model_names[:len(temporal_models)])):
        # Error distribution plot
        title = f"Temporal SNN Error Distribution - {model_name}"
        error_plot_path = os.path.join(plots_dir, f"error_dist_temporal_{model_name}.png")
        plot_error_distribution_temporal(errors, labels, metrics['threshold'], title, error_plot_path)
        
        # Reconstruction examples plot
        recon_plot_path = os.path.join(plots_dir, f"reconstructions_temporal_{model_name}.png")
        plot_reconstruction_examples_temporal(
            all_originals[i], all_reconstructions[i], all_labels[i], 
            [e.item() for e in all_errors[i]], model_name, recon_plot_path
        )
        
        print(f"Saved plots for {model_name}")
    
    # Create summary comparison plot
    create_summary_comparison_plot(all_metrics, model_names[:len(temporal_models)], plots_dir)
    
    print(f"\nAll visualizations saved to {plots_dir}")
    print("Temporal SNN anomaly detection analysis completed!")


def create_summary_comparison_plot(all_metrics, model_names, plots_dir):
    """
    Create a summary comparison plot of all temporal SNN models.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Temporal SNN Models - Anomaly Detection Performance', fontsize=14)
    
    # Extract metrics
    f1_scores = [m['f1_score'].item() for m in all_metrics]
    precisions = [m['precision'].item() for m in all_metrics]
    recalls = [m['recall'].item() for m in all_metrics]
    thresholds = [m['threshold'].item() for m in all_metrics]
    
    x = range(len(model_names))
    
    # F1 Scores
    ax1.bar(x, f1_scores, color='lightblue', alpha=0.7)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Scores by Model')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Precision vs Recall
    ax2.scatter(recalls, precisions, c=f1_scores, cmap='viridis', s=100)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision vs Recall')
    ax2.grid(True, alpha=0.3)
    for i, name in enumerate(model_names):
        ax2.annotate(name.split('_')[0], (recalls[i], precisions[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Precision and Recall bars
    ax3.bar([i-0.2 for i in x], precisions, width=0.4, label='Precision', alpha=0.7)
    ax3.bar([i+0.2 for i in x], recalls, width=0.4, label='Recall', alpha=0.7)
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Score')
    ax3.set_title('Precision and Recall by Model')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Thresholds
    ax4.plot(x, thresholds, 'o-', color='red', linewidth=2, markersize=8)
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Best Threshold')
    ax4.set_title('Optimal Thresholds by Model')
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(plots_dir, "temporal_snn_summary_comparison.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Summary comparison plot saved to {save_path}")


if __name__ == "__main__":
    main()
