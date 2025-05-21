import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils # Import utils for reset

class BaseSNNAutoencoder(nn.Module):
    def __init__(self, beta=0.9, spike_grad=surrogate.fast_sigmoid()):
        super().__init__()
        self.beta = beta
        self.spike_grad = spike_grad
        self.encoder = None
        self.decoder = None

    def forward(self, x): # x is expected to be (batch_size, num_time_steps, num_features)
        # Reset hidden states at the beginning of a new batch / forward pass
        if self.encoder is not None:
            utils.reset(self.encoder)
        if self.decoder is not None:
            utils.reset(self.decoder)
        
        num_time_steps = x.shape[1] # Get T from input X: (B, T, F)
        
        # Encoder pass
        spk_rec_encoder_over_time = [] 
        for t in range(num_time_steps): # Iterate over time dimension of input x
            current_input_slice = x[:, t, :] # Shape: (batch_size, num_features)
            # Pass current_input_slice to the encoder.
            # The snn.Leaky layers in the encoder handle the temporal dynamics for this one time step.
            spk_out_encoder_t, _mem_out_encoder_t = self.encoder(current_input_slice) 
            spk_rec_encoder_over_time.append(spk_out_encoder_t)
        
        # Stack recorded spikes from encoder: (num_time_steps, batch_size, latent_features)
        spk_rec_encoder_tensor = torch.stack(spk_rec_encoder_over_time, dim=0) 

        # Decoder pass
        mem_rec_decoder_over_time = []
        # The decoder's snn.Leaky layers also have their states evolving over time.
        # We feed the sequence of latent spikes to the decoder one time step at a time.
        for t in range(num_time_steps): # Iterate through the time steps of encoded spikes
            # Input to decoder is spk_rec_encoder_tensor[t, :, :], which is (batch_size, latent_features)
            # Since the decoder's final snn.Leaky layer has output=False, it returns only membrane potential.
            mem_out_decoder_t = self.decoder(spk_rec_encoder_tensor[t, :, :])
            mem_rec_decoder_over_time.append(mem_out_decoder_t)
        
        # Stack recorded membrane potentials from decoder: (num_time_steps, batch_size, output_features)
        mem_rec_decoder_tensor = torch.stack(mem_rec_decoder_over_time, dim=0)
        
        # Return the membrane potential of the output layer at the last time step
        # This is a common choice for reconstruction tasks.
        return mem_rec_decoder_tensor[-1, :, :] # Shape: (batch_size, output_features)

# Small SNN Autoencoder (mimicking ANN SmallAutoencoder structure)
class SmallSNNAutoencoder(BaseSNNAutoencoder):
    def __init__(self, latent_size, beta=0.9, spike_grad=surrogate.fast_sigmoid()):
        super().__init__(beta, spike_grad)
        input_size = 28*28

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(128, 64),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(64, latent_size),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)  # Output spikes
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(64, 128),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(128, input_size),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=False) # Output membrane potential
        )

# Medium SNN Autoencoder (mimicking ANN Autoencoder structure)
class MediumSNNAutoencoder(BaseSNNAutoencoder):
    def __init__(self, latent_size, beta=0.9, spike_grad=surrogate.fast_sigmoid()):
        super().__init__(beta, spike_grad)
        input_size = 28*28

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(256, 128),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(128, 64),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(64, latent_size),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)  # Output spikes
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(64, 128),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(128, 256),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(256, input_size),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=False) # Output membrane potential
        )

# Big SNN Autoencoder (mimicking ANN BigAutoencoder structure)
class BigSNNAutoencoder(BaseSNNAutoencoder):
    def __init__(self, latent_size, beta=0.9, spike_grad=surrogate.fast_sigmoid()):
        super().__init__(beta, spike_grad)
        input_size = 28*28

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(512, 256),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(256, 128),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(128, 64),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(64, latent_size),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)  # Output spikes
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(64, 128),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(128, 256),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(256, 512),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(512, input_size),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=False) # Output membrane potential
        )
