from mnist_ratio_utils import *
import os
import numpy as np



# Create lists to store results for all models
all_errors = []
all_reconstructions = []
all_originals = []
all_labels = []
all_metrics = []

ALL_MODELS = load_models()
# check what is model in all_models
for i, model in enumerate(ALL_MODELS):
  print(f"Model {i}: {model}")
  
device = ("mps" if torch.backends.mps.is_available() else "cpu")

# load data
train_loader, val_loader, _ = get_mnist_ratio_dataloaders()


# Iterate through each model in ALL_MODELS
for model_idx, model in enumerate(ALL_MODELS):
  print(f"\nProcessing model {model_idx + 1}/{len(ALL_MODELS)}")
  
  # Convert ANN to SNN
  snn_model = convert_to_snn(model).to(device)
  
  # Detect anomalies using the SNN
  errors, reconstructions, originals, labels = detect_anomalies_spiking(
    snn_model, val_loader, device, num_examples=10
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

# Print data shapes for debugging and understanding
print("\nExamining data shapes:")
for i, (errors, reconstructions, originals, labels) in enumerate(zip(all_errors, all_reconstructions, all_originals, all_labels)):
  print(f"\nModel {i} ({model_names[i]}):")
  print(f"  Errors shape: {np.array(errors).shape}")
  if isinstance(reconstructions, torch.Tensor):
    print(f"  Reconstructions shape: {reconstructions.shape}")
  else:
    print(f"  Reconstructions type: {type(reconstructions)}")
  
  if isinstance(originals, torch.Tensor):
    print(f"  Originals shape: {originals.shape}")
  else:
    print(f"  Originals type: {type(originals)}")
  
  print(f"  Labels shape: {np.array(labels).shape}")
  
  # Print some statistics
  print(f"  Errors - min: {min(errors):.4f}, max: {max(errors):.4f}, mean: {np.mean(errors):.4f}")
  print(f"  Label distribution: {np.bincount(np.array(labels))}")