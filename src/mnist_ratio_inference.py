from mnist_ratio_utils import *
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# Create lists to store results for all models
all_errors = []
all_reconstructions = []
all_originals = []
all_labels = []
all_metrics = []

ALL_MODELS = load_models()

device = ("mps" if torch.backends.mps.is_available() else "cpu")

# load data
train_loader, val_loader, _ = get_mnist_ratio_dataloaders()

# After loading data
sample_batch = next(iter(train_loader))
print(sample_batch)
print(f"Input data range: min={sample_batch.min().item():.4f}, max={sample_batch.max().item():.4f}")
print(f"Input data mean: {sample_batch.mean().item():.4f}, std: {sample_batch.std().item():.4f}")

print("="*60)
print("MNIST RATIO SNN ANOMALY DETECTION ANALYSIS")
print("="*60)

# Iterate through each model in ALL_MODELS with tqdm for progress tracking
for model_idx, model in enumerate(tqdm(ALL_MODELS, desc="Processing models")):
    print(f"\nProcessing model {model_idx + 1}/{len(ALL_MODELS)}")
    
    # Convert ANN to SNN
    snn_model = convert_to_snn(model).to(device)
    
    # Detect anomalies using the SNN
    errors, reconstructions, originals, labels = detect_anomalies_spiking(
        snn_model, val_loader, device, num_examples=2000
    )
    
    # Calculate metrics
    best_threshold, best_f1_score, indices, sorted_labels, sorted_errors = check_thresholds(
        errors, labels
    )
    
    # Calculate additional metrics
    sorted_labels_indices = torch.tensor(sorted_labels)
    zeros_indices = sorted_labels_indices == 0
    other_indices = sorted_labels_indices != 0
    
    threshold_value = torch.quantile(torch.tensor(sorted_errors), best_threshold)
    positive_class = torch.tensor(sorted_errors) >= threshold_value
    negative_class = torch.tensor(sorted_errors) < threshold_value
    
    tp = positive_class[zeros_indices].sum()
    fn = negative_class[zeros_indices].sum()
    tn = negative_class[other_indices].sum()
    fp = positive_class[other_indices].sum()
    
    precision = tp/(tp + fp) if (tp + fp) > 0 else torch.tensor(0.)
    recall = tp/(tp + fn) if (tp + fn) > 0 else torch.tensor(0.)
    
    # Store results with enhanced metrics
    all_errors.append(sorted_errors)
    all_reconstructions.append(reconstructions)
    all_originals.append(originals)
    all_labels.append(sorted_labels)
    all_metrics.append({
        'threshold_percentile': best_threshold.item(),
        'threshold': threshold_value.item(),  # Actual error threshold value
        'f1_score': best_f1_score.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'tp': tp.item(),
        'fn': fn.item(),
        'tn': tn.item(),
        'fp': fp.item()
    })
    
    print(f"Model metrics:")
    print(f"Best Threshold (Error Value): {threshold_value.item():.6f}")
    print(f"Best Threshold (Percentile): {best_threshold.item():.4f}")
    print(f"Precision at Best Threshold: {precision.item():.4f}")
    print(f"Recall at Best Threshold: {recall.item():.4f}")
    print(f"F1 Score: {best_f1_score.item():.4f}")

# Define model names based on their architecture (corrected order)
model_names = []
for i in range(len(ALL_MODELS)):
    if i < 3:
        model_names.append(f"SmallAutoencoder_{32 if i==0 else 16 if i==1 else 8}")
    elif i < 6:
        model_names.append(f"BigAutoencoder_{32 if i==3 else 16 if i==4 else 8}")
    elif i < 9:
        model_names.append(f"Autoencoder_{32 if i==6 else 16 if i==7 else 8}")
    else:
        model_names.append(f"model_{i+1}")

# Save results
save_path = "results/mnist_ratio_inference_results.pt"
results_dir = os.path.dirname(save_path)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Create results dictionary similar to other scripts
results = {}
for i, model_name in enumerate(model_names):
    results[model_name] = {
        'best_metrics': all_metrics[i],
        'predictions': all_errors[i],
        'labels': torch.tensor(all_labels[i])
    }

torch.save({
    'errors': all_errors,
    'reconstructions': all_reconstructions,
    'originals': all_originals,
    'labels': all_labels,
    'metrics': all_metrics,
    'model_names': model_names,
    'results': results
}, save_path)

def plot_error_distributions(results, model_names, save_dir="results/ratio_snn_anomaly_plots"):
    """
    Plot error distributions grouped by architecture (3 latent sizes per architecture).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nCreating error distribution plots...")
    
    # Reset to default matplotlib style
    plt.style.use('default')
    
    # Group models by architecture
    architectures = ['SmallAutoencoder', 'BigAutoencoder', 'Autoencoder']
    latent_sizes = [32, 16, 8]
    
    for arch in architectures:
        # Create subplot figure for this architecture
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Error Distributions - {arch} (Ratio SNN)', fontsize=16, fontweight='bold')
        
        for i, latent_size in enumerate(latent_sizes):
            model_name = f"{arch}_{latent_size}"
            
            if model_name in results:
                model_results = results[model_name]
                predictions = model_results['predictions'].numpy()
                labels = np.array(model_results['labels'])
                threshold_value = model_results['best_metrics']['threshold']  # Actual error threshold
                threshold_percentile = model_results['best_metrics']['threshold_percentile']  # Percentile
                
                # Separate errors for anomalous and normal data
                anomalous_errors = predictions[labels == 0]
                normal_errors = predictions[labels != 0]
                
                # Plot on subplot
                ax = axes[i]
                ax.hist(normal_errors, bins=40, alpha=0.7, label='Normal (1-9)', density=True)
                ax.hist(anomalous_errors, bins=40, alpha=0.7, label='Anomalous (0)', density=True)
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

def plot_roc_curves(results, model_names, save_dir="results/ratio_snn_anomaly_plots"):
    """
    Plot ROC curves grouped by architecture.
    """
    # Reset to default matplotlib style
    plt.style.use('default')
    
    # Group models by architecture
    architectures = ['SmallAutoencoder', 'BigAutoencoder', 'Autoencoder']
    latent_sizes = [32, 16, 8]
    
    # Create subplot figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('ROC Curves - Ratio SNN Anomaly Detection', fontsize=16, fontweight='bold')
    
    for i, arch in enumerate(architectures):
        ax = axes[i]
        
        # Plot ROC curves for all latent sizes of this architecture
        for latent_size in latent_sizes:
            model_name = f"{arch}_{latent_size}"
            
            if model_name in results:
                model_results = results[model_name]
                predictions = model_results['predictions'].numpy()
                labels = model_results['labels'].numpy()
                
                # Calculate ROC curve manually
                thresholds = np.linspace(0, 1, 100)
                fprs = []
                tprs = []
                
                sorted_predictions = np.sort(predictions)
                
                for threshold in thresholds:
                    threshold_value = np.quantile(sorted_predictions, threshold)
                    binary_predictions = (predictions > threshold_value).astype(int)
                    binary_labels = (labels == 0).astype(int)  # 1 for anomalous, 0 for normal
                    
                    tp = np.sum((binary_predictions == 1) & (binary_labels == 1))
                    fp = np.sum((binary_predictions == 1) & (binary_labels == 0))
                    tn = np.sum((binary_predictions == 0) & (binary_labels == 0))
                    fn = np.sum((binary_predictions == 0) & (binary_labels == 1))
                    
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    
                    tprs.append(tpr)
                    fprs.append(fpr)
                
                # Calculate AUC
                binary_labels = (labels == 0).astype(int)
                auc_score = roc_auc_score(binary_labels, predictions)
                
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

def plot_performance_summary(results, model_names, save_dir="results/ratio_snn_anomaly_plots"):
    """
    Create comprehensive performance summary plots grouped by architecture.
    """
    # Group models by architecture
    architectures = ['SmallAutoencoder', 'BigAutoencoder', 'Autoencoder']
    latent_sizes = [32, 16, 8]
    
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
    fig.suptitle('Ratio SNN Anomaly Detection Performance Summary', fontsize=16, fontweight='bold')
    
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

# Create output directory
save_dir = "results/ratio_snn_anomaly_plots"
os.makedirs(save_dir, exist_ok=True)

# Generate comprehensive visualizations
print("\nGenerating comprehensive visualizations...")
plot_error_distributions(results, model_names, save_dir)
plot_roc_curves(results, model_names, save_dir)
plot_performance_summary(results, model_names, save_dir)

# Print summary table
print("\n" + "="*100)
print("RATIO SNN ANOMALY DETECTION PERFORMANCE SUMMARY")
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
print("Analysis completed successfully!")
