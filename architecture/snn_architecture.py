import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils # Import utils for reset

class BaseSNNAutoencoder(nn.Module):
    def __init__(self, beta=0.5, spike_grad=surrogate.atan(alpha=2.0)):
        super().__init__()
        self.beta = beta
        self.spike_grad = spike_grad
        self.encoder = None
        self.decoder = None

    def forward(self, x): # x is expected to be (batch_size, num_time_steps, num_features)
        # Reset hidden states at the beginning of a new batch / forward pass
        utils.reset(self.encoder)
        utils.reset(self.decoder)
        
        # Process ALL timesteps while maintaining states
        for t in range(x.shape[1]):
            current_input = x[:, t, :]
            
            # Encoder maintains its state across timesteps
            spk_encoded = self.encoder(current_input)
            
            # Decoder maintains its state across timesteps  
            mem_out = self.decoder(spk_encoded)
        
            # Handle potential tuple output from final SNN layer
            if isinstance(mem_out, tuple):
                mem_out = mem_out[1]  # Get membrane potential from tuple
            else:
                mem_out = mem_out  # Already membrane potential
        
        # Apply sigmoid to constrain to [0, 1] range
        reconstruction = torch.sigmoid(mem_out)
        return reconstruction

# Small SNN Autoencoder (mimicking ANN SmallAutoencoder structure)
class SmallSNNAutoencoder(BaseSNNAutoencoder):
    def __init__(self, latent_size, beta=0.5, spike_grad=surrogate.atan(alpha=2.0)):
        super().__init__(beta, spike_grad)
        input_size = 28*28

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0),
            nn.Linear(128, 64),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0),
            nn.Linear(64, latent_size),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0),
            nn.Linear(64, 128),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0),
            nn.Linear(128, input_size),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, threshold=20000)
        )

# Medium SNN Autoencoder (mimicking ANN Autoencoder structure)
class MediumSNNAutoencoder(BaseSNNAutoencoder):
    def __init__(self, latent_size, beta=0.5, spike_grad=surrogate.atan(alpha=2.0)):
        super().__init__(beta, spike_grad)
        input_size = 28*28

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0),
            nn.Linear(256, 128),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0),
            nn.Linear(128, 64),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0),
            nn.Linear(64, latent_size),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0),
            nn.Linear(64, 128),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0),
            nn.Linear(128, 256),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0),
            nn.Linear(256, input_size),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, threshold=20000)
        )

# Big SNN Autoencoder (mimicking ANN BigAutoencoder structure)
class BigSNNAutoencoder(BaseSNNAutoencoder):
    def __init__(self, latent_size, beta=0.5, spike_grad=surrogate.atan(alpha=2.0)):
        super().__init__(beta, spike_grad)
        input_size = 28*28

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0),
            nn.Linear(512, 256),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0),
            nn.Linear(256, 128),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0),
            nn.Linear(128, 64),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0),
            nn.Linear(64, latent_size),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0),
            nn.Linear(64, 128),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0),
            nn.Linear(128, 256),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0),
            nn.Linear(256, 512),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, threshold=1.0),
            nn.Linear(512, input_size),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, threshold=20000)
        )
