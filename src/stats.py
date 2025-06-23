import os
import torch
import numpy as np
import time
import gc
from torchvision import datasets, transforms
from mnist_ratio_utils import load_models, convert_to_snn, get_mnist_ratio_dataloaders, load_temporal_models
from mnist_snn_loading import get_snn_autoencoder_dataloaders
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Liczbę neuronów
# 2. Liczbę synaps
# 3. Czas symulacji per epoka
# 4. Czas symulacji per spike
# 5. Sumaryczną liczbę spike'ów w sieci
# 6. Zajętość pamięci na model
# 7. Moc per spike
# 8. zestawić parametry samego projektu

batch_sizes = [16, 32, 64, 128]

# load models
ann_models = load_models()

# convert models to snn
ann_to_snn_models = [convert_to_snn(model) for model in ann_models]

# load temporal models
temporal_models = load_temporal_models()
# -----



