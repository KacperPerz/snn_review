from mnist_ratio_utils import *
import os
import numpy as np
from tqdm import tqdm


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

# After loading datadd
sample_batch = next(iter(train_loader))
print(sample_batch)
print(f"Input data range: min={sample_batch.min().item():.4f}, max={sample_batch.max().item():.4f}")
print(f"Input data mean: {sample_batch.mean().item():.4f}, std: {sample_batch.std().item():.4f}")

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
  
  positive_class = torch.tensor(sorted_errors) >= torch.quantile(torch.tensor(sorted_errors), best_threshold)
  negative_class = torch.tensor(sorted_errors) < torch.quantile(torch.tensor(sorted_errors), best_threshold)
  
  tp = positive_class[zeros_indices].sum()
  fn = negative_class[zeros_indices].sum()
  tn = negative_class[other_indices].sum()
  fp = positive_class[other_indices].sum()
  
  precision = tp/(tp + fp)
  recall = tp/(tp + fn)
  
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
  print(f"Precision at Best Threshold: {precision:.4f}")
  print(f"Recall at Best Threshold: {recall:.4f}")
  print(f"F1 Score: {best_f1_score:.4f}")

# save results
# Define model names based on their architecture
model_names = []
for i in range(len(ALL_MODELS)):
  if i < 3:
    model_names.append(f"BigAutoencoder_latent{32 if i==0 else 16 if i==1 else 8}")
  elif i < 6:
    model_names.append(f"Autoencoder_latent{32 if i==3 else 16 if i==4 else 8}")
  elif i < 9:
    model_names.append(f"SmallAutoencoder_latent{32 if i==6 else 16 if i==7 else 8}")
  else:
    model_names.append(f"model_{i+1}")

save_path = "results/mnist_ratio_inference_results.pt"
# Create the directory if it doesn't exist
results_dir = os.path.dirname(save_path)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

torch.save({
  'errors': all_errors,
  'reconstructions': all_reconstructions,
  'originals': all_originals,
  'labels': all_labels,
  'metrics': all_metrics,
  'model_names': model_names
}, save_path)

import matplotlib.pyplot as plt

# Create a function to plot error distributions for a model
def plot_error_distribution(errors, labels, threshold, title):
  fig, ax = plt.subplots(figsize=(10, 6))
  
  # Convert to numpy for easier indexing
  errors_np = np.array(errors)
  labels_np = np.array(labels)
  
  # Separate errors for anomalous and normal data
  anomalous_errors = errors_np[labels_np == 0]
  normal_errors = errors_np[labels_np != 0]
  
  # Plot histograms
  ax.hist(anomalous_errors, bins=50, alpha=0.6, label='Anomalous (0)', density=True)
  ax.hist(normal_errors, bins=50, alpha=0.6, label='Normal (1-9)', density=True)
  ax.axvline(np.quantile(errors_np, threshold), color='r', linestyle='--', 
         label=f'Threshold ({threshold:.2f})')
  ax.set_xlabel('Reconstruction Error')
  ax.set_ylabel('Density')
  ax.set_title(title)
  ax.legend()
  
  return fig

# Create a directory for the plots
plots_dir = "results/error_distributions"
if not os.path.exists(plots_dir):
  os.makedirs(plots_dir)

# Plot error distribution for each model
for i, (errors, labels, metrics, model_name) in enumerate(zip(all_errors, all_labels, all_metrics, model_names)):
  title = f"Distribution of Reconstruction Errors - {model_name}"
  fig = plot_error_distribution(errors, labels, metrics['threshold'], title)
  
  # Save the figure
  save_path = os.path.join(plots_dir, f"error_dist_model_{i}.png")
  fig.savefig(save_path)
  plt.close(fig)
  
  print(f"Saved error distribution plot for {model_name} to {save_path}")
