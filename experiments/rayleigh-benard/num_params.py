import torch
from sda.mcs import *
from sda.score import *
from sda.utils import *

from utils import *
import json

path = "/home/baris-turan/sda/experiments/rayleigh-benard/runs/None_qxh9vkvh/"
model_path = path + "state.pth"
config_path = path + "config.json"

# with open(config_path, "r") as file:
#     CONFIG = json.load(file)

CONFIG = {
    # Architecture
    'window': 5,
    'embedding': 64,
    'hidden_channels': (64, 128, 256),
    'hidden_blocks': (3, 3, 3),
    'kernel_size': 3,
    'activation': 'SiLU',
    # Data
    'coarsen': 1,            # 512x128 -> 256x64 to keep memory/compute tractable
    't_coarsen': 1,          # temporal downsampling factor (must match prepare.py); 1 = off
    # Latent diffusion: set to a VAE run name (under runs_vae) to train on VAE
    # latents from data_latent/ instead of pixels. None = pixel diffusion (default).
    'vae_run': "vae_1786110754",
    # Training
    'epochs': 4096,
    'batch_size': 2048,
    'optimizer': 'AdamW',
    'learning_rate': 2e-4,
    'weight_decay': 1e-3,
    'scheduler': 'exponential',
    'num_workers': 4,
}

#checkpoint = torch.load(model_path, weights_only=True)


model = make_score(**CONFIG)
#model.load_state_dict(checkpoint)

num_params =  sum(p.numel() for p in model.parameters())
print("Number of parameters:", num_params)