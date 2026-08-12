#!/usr/bin/env python

import wandb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np

from dawgz import job, schedule
from typing import *

from sda.mcs import *
from sda.score import *
from sda.utils import *

from utils import *


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
    'vae_run': "vae_1786432980",
    # Training
    'epochs': 4096,
    'batch_size': 128,
    'optimizer': 'AdamW',
    'learning_rate': 2e-4,
    'weight_decay': 1e-3,
    'scheduler': 'exponential',
    'num_workers': 4,
}


@job(array=1, cpus=4, gpus=3, ram='32GB', time='24:00:00')
def train(i: int):
    # Network
    window = CONFIG['window']
    coarsen = CONFIG['coarsen']
    vae_run = CONFIG.get('vae_run')

    # Latent vs pixel diffusion. With `vae_run` set we diffuse in the VAE latent
    # space: the state is C_z-channel on the small latent grid and the data comes
    # from data_latent/ (built by encode_latents.py). Otherwise it is single-
    # channel buoyancy on the coarsen grid from data/ -- the original behaviour.
    vae = None
    if vae_run:
        from vae import load_vae, _resolve_vae_run
        vae = load_vae(_resolve_vae_run(vae_run), 'cuda')
        CONFIG['latent_channels'] = vae.latent_channels
        CONFIG['latent_height'] = vae.latent_height
        CONFIG['latent_width'] = vae.latent_width
        channels, height, width = vae.latent_channels, vae.latent_height, vae.latent_width
        datadir = PATH / 'data_latent_beta_1e-5'
    else:
        channels, height, width = 1, HEIGHT // coarsen, WIDTH // coarsen
        datadir = PATH / 'data'

    # Data: consolidated, contiguous HDF5 loaded into memory by TrajectoryDataset.
    # Pixel: run `python prepare.py --coarsen N --t-coarsen M` first. Latent: run
    # `python encode_latents.py --vae-run <run>` to build data_latent/ first.
    if not (datadir / 'train.h5').exists():
        hint = (f"python encode_latents.py --vae-run {vae_run}" if vae_run
                else f"python prepare.py --coarsen {coarsen} --t-coarsen {CONFIG['t_coarsen']}")
        raise SystemExit(f"missing {datadir/'train.h5'}; run `{hint}` first")
    with h5py.File(datadir / 'train.h5', 'r') as f:
        _, _, dc, dh, dw = f['x'].shape
    assert (dc, dh, dw) == (channels, height, width), (
        f"data grid {dc}x{dh}x{dw} != expected {channels}x{height}x{width} "
        f"({'latent' if vae_run else 'pixel'}); rebuild the dataset")

    trainset = TrajectoryDataset(datadir / 'train.h5', window=window, flatten=True)
    validset = TrajectoryDataset(datadir / 'valid.h5', window=window, flatten=True)

    # Latent normalization: standardize the VAE latents to zero-mean/unit-std per
    # channel using the TRAIN statistics, so the diffusion sees a well-scaled
    # target (raw VAE latents are not unit-scaled). The stats are recorded in the
    # run config; `vae.load_decoder` reads them back and un-standardizes before
    # decoding, so every inference/DAPS script stays consistent. Pixel runs (data
    # already standardized buoyancy) skip this.
    if vae_run:
        d = trainset.data                             # (n, L, C, H, W) float32
        mean = d.mean(axis=(0, 1, 3, 4))              # (C,) per latent channel
        std = d.std(axis=(0, 1, 3, 4))                # (C,)
        std = np.where(std < 1e-8, 1.0, std)          # guard (near-)constant channels
        CONFIG['latent_mean'] = mean.tolist()
        CONFIG['latent_std'] = std.tolist()
        m = mean.reshape(1, 1, -1, 1, 1)
        s = std.reshape(1, 1, -1, 1, 1)
        trainset.data = ((trainset.data - m) / s).astype(np.float32)
        validset.data = ((validset.data - m) / s).astype(np.float32)
        print(f'latent per-channel mean {mean}  std {std}', flush=True)

    run = wandb.init(project='sda-rayleigh-benard', config=CONFIG)
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(CONFIG, runpath)

    score = make_score(**CONFIG)
    sde = VPSDE(score.kernel, shape=(window * channels, height, width)).cuda()

    # Training
    generator = loop(
        sde,
        trainset,
        validset,
        device='cuda',
        **CONFIG,
    )

    for loss_train, loss_valid, lr in generator:
        run.log({
            'loss_train': loss_train,
            'loss_valid': loss_valid,
            'lr': lr,
        })

    # Save
    torch.save(
        score.state_dict(),
        runpath / f'state.pth',
    )

    # Evaluation: sample a couple of trajectories and log a buoyancy panel. For a
    # latent run the sample lives in the standardized latent space, so un-normalize
    # (x * std + mean) then decode back to pixels.
    x = sde.sample((2,), steps=256)                  # (2, window*channels, H, W)
    x = x.unflatten(1, (window, channels))          # (2, window, channels, H, W)
    if vae is not None:
        m = torch.tensor(CONFIG['latent_mean'], device=x.device).reshape(-1, 1, 1)
        s = torch.tensor(CONFIG['latent_std'], device=x.device).reshape(-1, 1, 1)
        with torch.no_grad():
            x = vae.decode_frames(x * s + m)       # (2, window, 1, H_pix, W_pix)
    b = x[:, :, 0].cpu()                            # buoyancy

    run.log({'samples': wandb.Image(draw(b))})
    run.finish()


if __name__ == '__main__':
    schedule(
        train,
        name='Training',
        backend='async',
        export='ALL',
        env=['export WANDB_SILENT=true'],
    )
