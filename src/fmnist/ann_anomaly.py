import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Add the project root directory to sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from architecture.ann_architecture import SmallAutoencoder, SmallAutoencoder_16, SmallAutoencoder_8, Autoencoder, Autoencoder_16, Autoencoder_8, BigAutoencoder, BigAutoencoder_16, BigAutoencoder_8
from src.fmnist.data_loading import get_fmnist_ratio_dataloaders, ANOMALY_LABEL

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_fmnist_ann_models():
    """
    Load all Fashion-MNIST ANN models from the models/fmnist/ann directory.
    
    Returns:
        models: List of loaded ANN models
        model_names: List of model names
    """
    models = []
    model_names = []
    
    # Define model configurations with specific classes matching the actual file names
    model_configs = [
        {"class": SmallAutoencoder_8, "name": "SmallAutoencoder_8"},
        {"class": SmallAutoencoder_16, "name": "SmallAutoencoder_16"},
        {"class": SmallAutoencoder, "name": "SmallAutoencoder_32"},
        {"class": Autoencoder_8, "name": "Autoencoder_8"},
        {"class": Autoencoder_16, "name": "Autoencoder_16"},
        {"class": Autoencoder, "name": "Autoencoder_32"},
        {"class": BigAutoencoder_8, "name": "BigAutoencoder_8"},
        {"class": BigAutoencoder_16, "name": "BigAutoencoder_16"},
        {"class": BigAutoencoder, "name": "BigAutoencoder_32"}
    ]
    
    models_dir = "models/fmnist/ann"
    
    print("Loading Fashion-MNIST ANN models...")
    
    for config in model_configs:
        model_name = config["name"]
        model_path = os.path.join(models_dir, f"{model_name}.pth")
        
        if os.path.exists(model_path):
            print(f"Loading {model_name}...")
            
            # Initialize model (no parameters needed)
            model = config["class"]()
            
            # Load state dict
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            
            models.append(model)
            model_names.append(model_name)
            
            print(f"✓ Loaded {model_name}")
        else:
            print(f"✗ Model file not found: {model_path}")
    
    print(f"\nSuccessfully loaded {len(models)} Fashion-MNIST ANN models")
    return models, model_names

def compute_reconstruction_errors(models, model_names, val_loader, device):
    """
    Compute reconstruction errors for all models on validation data.
    Following the methodology from the Jupyter notebook exactly.
    
    Args:
        models: List of ANN models
        model_names: List of model names
        val_loader: Validation data loader
        device: Device to run inference on
    
    Returns:
        all_predictions: List of reconstruction errors for each model
        all_labels: List of labels for each model
    """
    criterion = nn.MSELoss()  # Use scalar MSE loss like in Jupyter
    
    all_predictions = [[] for _ in range(len(models))]
    all_labels = [[] for _ in range(len(models))]
    
    print("\nComputing reconstruction errors (individual sample processing)...")
    
    for idx, (model, model_name) in enumerate(zip(models, model_names)):
        print(f"Processing {model_name} ({idx+1}/{len(models)})...")
        
        model.eval()
        with torch.no_grad():
            for val_batch, val_label in tqdm(val_loader, desc=f"Model {idx+1}"):
                # Process each sample individually like in Jupyter
                for example, label in zip(val_batch, val_label):
                    example = example.to(device)
                    out = model(example)
                    # Apply criterion to individual sample (returns scalar)
                    error = criterion(example, out).unsqueeze(0)
                    all_predictions[idx].append(error)
                    all_labels[idx].append(label.unsqueeze(0))
    
    # Concatenate all predictions and labels and convert to tensors
    for idx in range(len(all_predictions)):
        all_predictions[idx] = torch.cat(all_predictions[idx])
        all_labels[idx] = torch.cat(all_labels[idx])
        
        print(f"Model {idx+1}: {len(all_predictions[idx])} samples processed")
        assert len(all_predictions[idx]) == len(all_labels[idx])
    
    return all_predictions, all_labels

def analyze_anomaly_detection(all_predictions, all_labels, model_names):
    """
    Analyze anomaly detection performance following Jupyter notebook methodology exactly.
    
    Args:
        all_predictions: List of reconstruction errors for each model
        all_labels: List of labels for each model
        model_names: List of model names
    
    Returns:
        results: Dictionary containing all results
    """
    thresholds = torch.linspace(0, 1, 40)  # Use same as Jupyter notebook
    
    tprs = [[] for _ in range(len(model_names))]
    fprs = [[] for _ in range(len(model_names))]
    f1_scores = [[] for _ in range(len(model_names))]
    precisions = [[] for _ in range(len(model_names))]
    recalls = [[] for _ in range(len(model_names))]
    sorted_labels = [[] for _ in range(len(model_names))]
    
    results = {}
    
    print("\nAnalyzing anomaly detection performance...")
    
    for idx, model_name in enumerate(model_names):
        print(f"\nAnalyzing {model_name}...")
        
        predictions = all_predictions[idx]
        labels = all_labels[idx]
        
        # Sort predictions and corresponding labels (DESCENDING like Jupyter)
        all_predictions[idx], indices = torch.sort(predictions, descending=True)
        sorted_labels[idx] = [labels[i] for i in indices.cpu()]
        
        # Convert to tensor for easier indexing
        sorted_labels_tensor = torch.tensor(sorted_labels[idx])
        
        # Indices for anomalous and normal classes
        zeros_indices = sorted_labels_tensor == ANOMALY_LABEL  # anomalous
        other_indices = sorted_labels_tensor != ANOMALY_LABEL  # normal
        
        best_f1 = 0
        best_threshold = 0
        best_metrics = {}
        
        # Evaluate different thresholds
        for threshold in thresholds:
            # Calculate threshold value as quantile (same as Jupyter)
            threshold_value = torch.quantile(all_predictions[idx].cpu(), threshold)
            
            # Predictions: 1 if error >= threshold (anomalous), 0 if error < threshold (normal)
            # This matches Jupyter: positive_class = all_predictions[idx] >= torch.quantile(...)
            positive_class = all_predictions[idx] >= threshold_value
            negative_class = all_predictions[idx] < threshold_value
            
            # Calculate confusion matrix values (matching Jupyter logic)
            tp = positive_class[zeros_indices].sum()   # Correctly identified anomalies
            fn = negative_class[zeros_indices].sum()   # Missed anomalies  
            tn = negative_class[other_indices].sum()   # Correctly identified normal
            fp = positive_class[other_indices].sum()   # False alarms
            
            # Calculate metrics
            tpr = tp/(tp + fn) if (tp + fn) > 0 else torch.tensor(0.)
            fpr = fp/(fp + tn) if (fp + tn) > 0 else torch.tensor(0.)
            precision = tp/(tp + fp) if (tp + fp) > 0 else torch.tensor(0.)
            recall = tp/(tp + fn) if (tp + fn) > 0 else torch.tensor(0.)
            f1_score = 2*tp/(2*tp + fp + fn) if (2*tp + fp + fn) > 0 else torch.tensor(0.)
            
            # Store for ROC curve
            fprs[idx].append(fpr.cpu())
            tprs[idx].append(tpr.cpu())
            f1_scores[idx].append(f1_score.cpu())
            precisions[idx].append(precision.cpu())
            recalls[idx].append(recall.cpu())
            
            # Track best F1 score
            if f1_score > best_f1:
                best_f1 = f1_score
                best_threshold = threshold
                best_threshold_value = threshold_value
                best_metrics = {
                    'threshold_percentile': threshold.item(),
                    'threshold': threshold_value.item(),  # Actual error threshold value
                    'f1_score': f1_score.item(),
                    'precision': precision.item(),
                    'recall': recall.item(),
                    'tpr': tpr.item(),
                    'fpr': fpr.item(),
                    'tp': tp.item(),
                    'fp': fp.item(),
                    'tn': tn.item(),
                    'fn': fn.item()
                }
        
        # Print results for this model
        print(f"Best Threshold (Error Value): {best_metrics['threshold']:.6f}")
        print(f"Best Threshold (Percentile): {best_metrics['threshold_percentile']:.4f}")
        print(f"Precision at Best Threshold: {best_metrics['precision']:.4f}")
        print(f"Recall at Best Threshold: {best_metrics['recall']:.4f}")
        print(f"F1 Score at Best Threshold: {best_metrics['f1_score']:.4f}")
        
        # Store results (note: we store the sorted labels tensor for plotting)
        results[model_names[idx]] = {
            'best_metrics': best_metrics,
            'all_f1_scores': f1_scores[idx],
            'all_precisions': precisions[idx],
            'all_recalls': recalls[idx],
            'all_tprs': tprs[idx],
            'all_fprs': fprs[idx],
            'predictions': all_predictions[idx],
            'labels': sorted_labels_tensor  # Use tensor for easier plotting
        }
    
    return results

def plot_error_distributions(results, model_names, save_dir="results/fmnist_ann_anomaly_plots"):
    """
    Plot error distributions grouped by architecture (3 latent sizes per architecture).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nCreating error distribution plots...")
    
    # Reset to default matplotlib style
    plt.style.use('default')
    
    # Group models by architecture
    architectures = ['SmallAutoencoder', 'Autoencoder', 'BigAutoencoder']
    latent_sizes = [8, 16, 32]
    
    for arch in architectures:
        # Create subplot figure for this architecture
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Fashion-MNIST Error Distributions - {arch}', fontsize=16, fontweight='bold')
        
        for i, latent_size in enumerate(latent_sizes):
            model_name = f"{arch}_{latent_size}"
            
            if model_name in results:
                model_results = results[model_name]
                predictions = model_results['predictions'].cpu().numpy()
                labels = model_results['labels'].cpu().numpy()
                threshold_value = model_results['best_metrics']['threshold']  # Actual error threshold
                threshold_percentile = model_results['best_metrics']['threshold_percentile']  # Percentile
                
                # Separate errors for anomalous and normal data
                anomalous_errors = predictions[labels == ANOMALY_LABEL]
                normal_errors = predictions[labels != ANOMALY_LABEL]
                
                # Plot on subplot
                ax = axes[i]
                ax.hist(normal_errors, bins=40, alpha=0.7, label=f'Normal (non-{ANOMALY_LABEL})', density=True)
                ax.hist(anomalous_errors, bins=40, alpha=0.7, label=f'Anomalous ({ANOMALY_LABEL})', density=True)
                ax.axvline(threshold_value, color='red', linestyle='--', linewidth=2,
                          label=f'Threshold ({threshold_percentile:.3f})')
                
                ax.set_xlabel('Reconstruction Error')
                ax.set_ylabel('Density')
                ax.set_title(f'Latent Size: {latent_size}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add F1 score as text annotation
                f1_score = model_results['best_metrics']['f1_score']
                ax.text(0.02, 0.98, f'F1: {f1_score:.3f}', transform=ax.transAxes, 
                       fontsize=12, fontweight='bold', verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(save_dir, f"error_distributions_{arch}.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved error distributions for {arch}")

def plot_roc_curves(results, model_names, save_dir="results/fmnist_ann_anomaly_plots"):
    """
    Plot ROC curves grouped by architecture.
    """
    # Reset to default matplotlib style
    plt.style.use('default')
    
    # Group models by architecture
    architectures = ['SmallAutoencoder', 'Autoencoder', 'BigAutoencoder']
    latent_sizes = [8, 16, 32]
    
    # Create subplot figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Fashion-MNIST ROC Curves - ANN Anomaly Detection', fontsize=16, fontweight='bold')
    
    for i, arch in enumerate(architectures):
        ax = axes[i]
        
        # Plot ROC curves for all latent sizes of this architecture
        for latent_size in latent_sizes:
            model_name = f"{arch}_{latent_size}"
            
            if model_name in results:
                model_results = results[model_name]
                fprs = [fpr.item() for fpr in model_results['all_fprs']]
                tprs = [tpr.item() for tpr in model_results['all_tprs']]
                
                # Calculate AUC - convert to binary labels (0 = anomalous, 1 = normal)
                binary_labels = (model_results['labels'] != ANOMALY_LABEL).int()  # 1 for normal, 0 for anomalous
                auc_score = roc_auc_score(binary_labels.cpu().numpy(), model_results['predictions'].cpu().numpy())
                
                # Plot ROC curve
                ax.plot(fprs, tprs, linewidth=2, 
                       label=f'Latent {latent_size} (AUC: {auc_score:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{arch.replace("Autoencoder", "")}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, "roc_curves_by_architecture.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved ROC curves plot")

def plot_performance_summary(results, model_names, save_dir="results/fmnist_ann_anomaly_plots"):
    """
    Create comprehensive performance summary plots grouped by architecture.
    """
    # Group models by architecture
    architectures = ['SmallAutoencoder', 'Autoencoder', 'BigAutoencoder']
    latent_sizes = [8, 16, 32]
    
    # Extract metrics grouped by architecture
    arch_metrics = {}
    for arch in architectures:
        arch_metrics[arch] = {
            'f1_scores': [],
            'precisions': [],
            'recalls': [],
            'thresholds': []
        }
        for latent_size in latent_sizes:
            model_name = f"{arch}_{latent_size}"
            if model_name in results:
                metrics = results[model_name]['best_metrics']
                arch_metrics[arch]['f1_scores'].append(metrics['f1_score'])
                arch_metrics[arch]['precisions'].append(metrics['precision'])
                arch_metrics[arch]['recalls'].append(metrics['recall'])
                arch_metrics[arch]['thresholds'].append(metrics['threshold'])
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Fashion-MNIST ANN Anomaly Detection Performance Summary', fontsize=16, fontweight='bold')
    
    x_positions = np.arange(len(latent_sizes))
    width = 0.25
    
    # F1 Scores grouped by architecture
    for i, arch in enumerate(architectures):
        arch_short = arch.replace('Autoencoder', '')
        ax1.bar(x_positions + i * width, arch_metrics[arch]['f1_scores'], 
               width, label=arch_short, alpha=0.8)
    
    ax1.set_xlabel('Latent Size')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Scores by Architecture and Latent Size')
    ax1.set_xticks(x_positions + width)
    ax1.set_xticklabels(latent_sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Precision vs Recall scatter plot (colored by architecture)
    colors = plt.cm.tab10(np.linspace(0, 1, len(architectures)))
    for i, arch in enumerate(architectures):
        arch_short = arch.replace('Autoencoder', '')
        ax2.scatter(arch_metrics[arch]['recalls'], arch_metrics[arch]['precisions'], 
                   c=[colors[i]], s=100, alpha=0.8, label=arch_short)
        
        # Add latent size annotations
        for j, latent_size in enumerate(latent_sizes):
            if j < len(arch_metrics[arch]['recalls']):
                ax2.annotate(f'{latent_size}', 
                           (arch_metrics[arch]['recalls'][j], arch_metrics[arch]['precisions'][j]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision vs Recall by Architecture')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Precision and Recall comparison
    for i, arch in enumerate(architectures):
        arch_short = arch.replace('Autoencoder', '')
        ax3.plot(latent_sizes, arch_metrics[arch]['precisions'], 
                marker='o', linewidth=2, label=f'{arch_short} Precision')
        ax3.plot(latent_sizes, arch_metrics[arch]['recalls'], 
                marker='s', linewidth=2, linestyle='--', label=f'{arch_short} Recall')
    
    ax3.set_xlabel('Latent Size')
    ax3.set_ylabel('Score')
    ax3.set_title('Precision and Recall Trends')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(latent_sizes)
    
    # Optimal Thresholds
    for i, arch in enumerate(architectures):
        arch_short = arch.replace('Autoencoder', '')
        ax4.plot(latent_sizes, arch_metrics[arch]['thresholds'], 
                marker='o', linewidth=2, markersize=8, label=arch_short)
    
    ax4.set_xlabel('Latent Size')
    ax4.set_ylabel('Optimal Threshold (Error Value)')
    ax4.set_title('Optimal Thresholds by Architecture')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(latent_sizes)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "performance_summary_by_architecture.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved performance summary plot")

def main():
    """Main execution function"""
    print("="*60)
    print("FASHION-MNIST ANN ANOMALY DETECTION ANALYSIS")
    print("="*60)
    
    # Load Fashion-MNIST ANN models
    models, model_names = load_fmnist_ann_models()
    
    if not models:
        print("No models found! Please check the models directory.")
        return
    
    # Load validation data (using reduced balanced dataset like Jupyter notebook)
    print("\nLoading Fashion-MNIST validation data...")
    _, val_loader, _ = get_fmnist_ratio_dataloaders(batch_size=128)
    
    # Check data composition for debugging
    all_val_labels = []
    for _, labels in val_loader:
        all_val_labels.extend(labels.tolist())
    
    normal_count = sum(1 for label in all_val_labels if label != ANOMALY_LABEL)
    anomaly_count = sum(1 for label in all_val_labels if label == ANOMALY_LABEL)
    
    print(f"Validation dataset composition:")
    print(f"  Normal samples (non-{ANOMALY_LABEL}): {normal_count}")
    print(f"  Anomaly samples ({ANOMALY_LABEL}): {anomaly_count}")
    print(f"  Total samples: {len(all_val_labels)}")
    print(f"  Class balance ratio: {normal_count/anomaly_count:.2f}:1")
    
    # Compute reconstruction errors
    all_predictions, all_labels = compute_reconstruction_errors(models, model_names, val_loader, device)
    
    # Analyze anomaly detection performance
    results = analyze_anomaly_detection(all_predictions, all_labels, model_names)
    
    # Create output directory
    save_dir = "results/fmnist_ann_anomaly_plots"
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate visualizations
    plot_error_distributions(results, model_names, save_dir)
    plot_roc_curves(results, model_names, save_dir)
    plot_performance_summary(results, model_names, save_dir)
    
    # Save results
    results_path = "results/fmnist_ann_anomaly_results.pt"
    torch.save(results, results_path)
    print(f"\nResults saved to: {results_path}")
    
    # Print summary table
    print("\n" + "="*100)
    print("FASHION-MNIST ANOMALY DETECTION PERFORMANCE SUMMARY")
    print("="*100)
    print(f"{'Model':<25} {'Threshold (Error)':<15} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-"*100)
    
    for model_name in model_names:
        metrics = results[model_name]['best_metrics']
        print(f"{model_name:<25} {metrics['threshold']:<15.6f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}")
    
    print("-"*100)
    
    # Find best performing model
    best_model = max(model_names, key=lambda name: results[name]['best_metrics']['f1_score'])
    best_f1 = results[best_model]['best_metrics']['f1_score']
    print(f"Best performing model: {best_model} (F1 Score: {best_f1:.4f})")
    
    print(f"\nAll plots saved to: {save_dir}")
    print("Fashion-MNIST anomaly detection analysis completed successfully!")

if __name__ == "__main__":
    main()
