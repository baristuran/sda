#!/usr/bin/env python
r"""Quick sanity check for a trained VAE (``vae.ConvVAE``).

Pulls a single standardized frame from the training set, runs it through the
VAE, and writes the original and its reconstruction as high-resolution PNGs.
Also prints the reconstruction (per-pixel MSE) and KL losses for that frame.

    python test_vae.py --vae-run vae_smoke                 # first frame
    python test_vae.py --vae-run vae_smoke --index 12345   # a specific frame
    python test_vae.py --vae-run runs_vae/vae_smoke --gpu 1 --zoom 4
"""

from __future__ import annotations

import argparse
import os
import sys

# Pick the GPU before importing torch (so device 0 maps to it).
if '--gpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[sys.argv.index('--gpu') + 1]
else:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', os.environ.get('RBC_GPU', '0'))

import h5py
import numpy as np
import torch

from pathlib import Path

from utils import PATH, draw
from vae import load_vae, _resolve_vae_run


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--vae-run', required=True,
                   help='VAE run name (under runs_vae) or a path to the run dir')
    p.add_argument('--split', default='train', choices=['train', 'valid', 'test'],
                   help='which data/<split>.h5 to sample the frame from')
    p.add_argument('--index', type=int, default=0,
                   help='flat frame index (traj*L + frame) into the split')
    p.add_argument('--beta', type=float, default=1e-4, help='KL weight for the reported total loss')
    p.add_argument('--zoom', type=int, default=4, help='integer upscale for the output PNGs')
    p.add_argument('--out-dir', default=None,
                   help='where to write the PNGs (default: the VAE run dir)')
    p.add_argument('--gpu', default=None, help='CUDA device index (sets CUDA_VISIBLE_DEVICES)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = args.device
    runpath = _resolve_vae_run(args.vae_run)
    print(runpath)
    if not runpath.exists():
        raise SystemExit(f'no such VAE run: {runpath}')
    vae = load_vae(runpath, device)                      # frozen + eval
    print(f'loaded VAE {runpath.name}  '
          f'{vae.latent_channels}x{vae.latent_height}x{vae.latent_width} latent', flush=True)

    # -- grab one frame (traj, frame) from data/<split>.h5 (key x: (n,L,1,H,W)) --
    datafile = PATH / 'data' / f'{args.split}.h5'
    if not datafile.exists():
        raise SystemExit(f'missing {datafile}; run prepare.py first')
    with h5py.File(datafile, 'r') as f:
        n, L = f['x'].shape[:2]
        idx = args.index % (n * L)
        t, l = divmod(idx, L)
        frame = np.ascontiguousarray(f['x'][t, l])       # (1, H, W)
    x = torch.from_numpy(frame).float().unsqueeze(0).to(device)   # (1, 1, H, W)
    print(f'frame {idx} -> trajectory {t}, step {l}  (of {n} traj x {L} steps)', flush=True)

    # -- reconstruct (deterministic: use the posterior mean) -----------------
    with torch.no_grad():
        mu, logvar = vae.encode(x)
        recon = vae.decode(mu)
        recon_loss = (recon - x).square().mean()
        kl = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1.0).flatten(1).sum(dim=1).mean()
        total = recon_loss + args.beta * kl

    print(f'reconstruction loss (MSE): {recon_loss.item():.6f}', flush=True)
    print(f'KL loss:                   {kl.item():.6f}', flush=True)
    print(f'total (rec + {args.beta:g}*kl):   {total.item():.6f}', flush=True)

    # -- save high-res PNGs (draw expects buoyancy frames; zoom upscales) -----
    out_dir = Path(args.out_dir) if args.out_dir else runpath
    out_dir.mkdir(parents=True, exist_ok=True)
    orig_img = out_dir / f'{args.split}_{idx}_original.png'
    recon_img = out_dir / f'{args.split}_{idx}_reconstruction.png'
    draw(x[0, 0].cpu(), vmin=-2.0, vmax=2.0, zoom=args.zoom).save(orig_img)
    draw(recon[0, 0].cpu(), vmin=-2.0, vmax=2.0, zoom=args.zoom).save(recon_img)
    print(f'saved -> {orig_img}\n         {recon_img}', flush=True)


if __name__ == '__main__':
    main()
