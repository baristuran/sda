#!/usr/bin/env python
r"""Encode the consolidated pixel dataset into VAE latents for latent diffusion.

Reads ``data/{train,valid,test}.h5`` (key ``x``, standardized pixel trajectories
``(n, L, 1, H, W)``) and writes ``data_latent/{train,valid,test}.h5`` (key ``x``,
latents ``(n, L, C_z, H_l, W_l)``) using a trained VAE. The latent is the encoder
**mean** ``mu`` (deterministic) -- the standard choice for building an LDM latent
dataset. The source VAE run is recorded as an h5 attribute for provenance.

Point ``train.py`` at the resulting latents with ``vae_run=<name>`` to train the
latent diffusion (its ``config.json`` then records the latent dims and the VAE).

    python encode_latents.py --vae-run vae_smoke
    python encode_latents.py --vae-run runs_vae/my_vae --batch 8 --gpu 0
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch

from pathlib import Path

from utils import PATH
from vae import load_vae, _resolve_vae_run


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--vae-run', required=True, help='VAE run name (under runs_vae) or path')
    p.add_argument('--data', default=str(PATH / 'data'), help='pixel dataset dir')
    p.add_argument('--out', default=str(PATH / 'data_latent'), help='latent output dir')
    p.add_argument('--batch', type=int, default=8, help='trajectories encoded per chunk')
    p.add_argument('--gpu', default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    vae_path = _resolve_vae_run(args.vae_run)
    vae = load_vae(vae_path, args.device)
    Cz, Hl, Wl = vae.latent_channels, vae.latent_height, vae.latent_width
    print(f'VAE {vae_path.name}: latent ({Cz}, {Hl}, {Wl})', flush=True)

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for split in ('train', 'valid', 'test'):
        src = data / f'{split}.h5'
        if not src.exists():
            print(f'  skip {split}: {src} missing', flush=True)
            continue
        with h5py.File(src, 'r') as fin, h5py.File(out / f'{split}.h5', 'w') as fout:
            d = fin['x']                                   # (n, L, 1, H, W) standardized
            n, L = d.shape[:2]
            dset = fout.create_dataset('x', shape=(n, L, Cz, Hl, Wl), dtype='float32')
            fout.attrs['vae_run'] = str(vae_path)
            for i in range(0, n, args.batch):
                x = torch.from_numpy(d[i:i + args.batch]).float().to(args.device)   # (b,L,1,H,W)
                with torch.no_grad():
                    z = vae.encode_frames(x)                # (b, L, C_z, H_l, W_l), deterministic mu
                dset[i:i + z.shape[0]] = z.cpu().numpy().astype('float32')
            print(f'  {split}: ({n}, {L}, 1, {d.shape[-2]}, {d.shape[-1]}) '
                  f'-> ({n}, {L}, {Cz}, {Hl}, {Wl})  wrote {out/f"{split}.h5"}', flush=True)

    print(f'done -> {out}  (train with `vae_run={args.vae_run}`)', flush=True)


if __name__ == '__main__':
    main()
