import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import json

# Add project root to path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from architecture.ann_architecture import SmallAutoencoder, SmallAutoencoder_16, SmallAutoencoder_8, Autoencoder, Autoencoder_16, Autoencoder_8, BigAutoencoder, BigAutoencoder_16, BigAutoencoder_8
from src.fmnist.data_loading import get_fmnist_ratio_dataloaders


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def train_autoencoder(model, train_loader, val_loader, num_epochs=50, learning_rate=2e-3, device=device):
    """Train an autoencoder model with validation"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
            data = data.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed = model(data)
            loss = criterion(reconstructed, data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for data, _ in val_loader:  # val_loader returns (data, labels)
                data = data.to(device)
                reconstructed = model(data)
                loss = criterion(reconstructed, data)
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    return model, train_losses, val_losses


def main():
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data - now using validation set as well
    train_loader, val_loader, _ = get_fmnist_ratio_dataloaders(batch_size=128)
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Validation loader: {len(val_loader)} batches")
    
    # Define models to train with their epochs
    models_to_train = [
        # Small architectures - 15 epochs
        (SmallAutoencoder(), "SmallAutoencoder_32", 15),
        (SmallAutoencoder_16(), "SmallAutoencoder_16", 15),
        (SmallAutoencoder_8(), "SmallAutoencoder_8", 15),
        
        # Medium architectures (regular Autoencoder) - 20 epochs
        (Autoencoder(), "Autoencoder_32", 20),
        (Autoencoder_16(), "Autoencoder_16", 20),
        (Autoencoder_8(), "Autoencoder_8", 20),
        
        # Big architectures - 25 epochs
        (BigAutoencoder(), "BigAutoencoder_32", 25),
        (BigAutoencoder_16(), "BigAutoencoder_16", 25),
        (BigAutoencoder_8(), "BigAutoencoder_8", 25),
    ]
    
    # Create models directory
    models_dir = "../models/fmnist/ann"
    os.makedirs(models_dir, exist_ok=True)
    
    # Dictionary to store all losses
    all_training_results = {}
    
    # Train each model
    for model, model_name, epochs in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_name} for {epochs} epochs")
        print(f"{'='*50}")
        
        # Train the model with validation
        trained_model, train_losses, val_losses = train_autoencoder(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            learning_rate=2e-3,
            device=device
        )
        
        # Save the trained model
        model_path = os.path.join(models_dir, f"{model_name}.pth")
        torch.save(trained_model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
        
        # Store both training and validation losses
        all_training_results[model_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'epochs': epochs
        }
    
    # Save all training results to a JSON file
    results_path = os.path.join(models_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_training_results, f, indent=2)
    print(f"Training results saved to: {results_path}")
    
    print(f"\n{'='*50}")
    print("All models trained and saved successfully!")
    print("Training and validation losses tracked for each model.")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
