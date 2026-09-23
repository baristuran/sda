#!/usr/bin/env python
r"""Generate an ensemble of DPS (Diffusion Posterior Sampling) reconstructions.

Standalone version of the DPS ensemble that was previously buried in
``eval.py``. Given a trained diffusion run, it builds one sparse spatial
observation of a ground-truth test trajectory and draws ``--n-samples``
independent posterior samples with DPS (Chung et al., 2023: ``DPSGaussianScore``
guiding the reverse diffusion by the Gaussian likelihood of ``A(x)``).

DPS backpropagates the likelihood through the (decoder and) score at every step,
so it is **memory-heavy**; samples are therefore drawn in batches of ``--batch``
and accumulated, and the whole ensemble is written to a **single** ``.npz``
(never one file per batch). The output format -- ``posterior (N,L,1,H,W)``,
``truth (L,1,H,W)``, ``sub`` -- matches ``daps.py`` / ``daps_enkf.py``, so
``metrics.py`` reads it unchanged.

Latent vs pixel is transparent: ``load_decoder`` returns the identity for pixel
runs and the VAE decoder for latent runs, and the DPS guidance composes it into
the observation operator (autograd flows through the decoder), so old pixel-space
models work without change.

    python sample_dps.py --n-samples 64 --batch 8 --sub 8
    python sample_dps.py --run None_abcdef12 --n-samples 100 --batch 4 --steps 256
"""

from __future__ import annotations

import argparse
import os
import sys

# Pick the GPU before importing torch (so cuda:0 maps to it).
if '--gpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[sys.argv.index('--gpu') + 1]
else:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', os.environ.get('RBC_GPU', '0'))

import h5py
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path

from sda.utils import load_config

from utils import *
from vae import load_decoder


class IgnoreContext(nn.Module):
    r"""Adapt a guided score taking ``(x, t)`` to the ``(x, t, c)`` interface
    expected by :meth:`VPSDE.sample` (e.g. ``DPSGaussianScore``)."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x, t, c=None):
        return self.module(x, t)


def find_run(name=None) -> Path:
    if name:
        return PATH / 'runs' / name
    runs = sorted((PATH / 'runs').glob('*/state.pth'), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f'no trained runs under {PATH / "runs"}')
    return runs[-1].parent


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', default=None, help='run name under runs/ (default: most recent)')
    p.add_argument('--n-samples', type=int, default=64, help='total posterior samples to draw')
    p.add_argument('--batch', type=int, default=8,
                   help='samples per GPU batch (DPS is memory-heavy; lower if OOM)')
    p.add_argument('--length', type=int, default=None, help='trajectory length (default: model window)')
    p.add_argument('--sub', type=int, default=8, help='sparse spatial subsampling factor')
    p.add_argument('--sigma-obs', type=float, default=0.1, help='observation-noise std')
    p.add_argument('--steps', type=int, default=256, help='reverse-diffusion steps')
    p.add_argument('--corrections', type=int, default=1, help='Langevin corrections / step')
    p.add_argument('--tau', type=float, default=0.5)
    p.add_argument('--zeta', type=float, default=1.0, help='DPS likelihood guidance weight')
    p.add_argument('--gpu', default=None, help='CUDA device index (sets CUDA_VISIBLE_DEVICES)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None, help='output .npz (default: results/dps_sub<sub>.npz)')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    run = find_run(args.run)
    config = load_config(run)
    coarsen = config.get('coarsen', 1)
    H, W = HEIGHT // coarsen, WIDTH // coarsen
    L = args.length or config['window']

    # Latent vs pixel: sampling runs in the diffusion (latent) space; `decode`
    # maps a latent trajectory to standardized pixels (identity for pixel runs).
    score = load_score(run / 'state.pth').to(device).eval()
    decode, C, Hm, Wm = load_decoder(config, device)
    print(f'run {run.name} (coarsen={coarsen}, grid={H}x{W}, L={L}, '
          f'state=({C},{Hm},{Wm}))', flush=True)

    # Ground-truth test trajectory + one sparse spatial observation (pixel space),
    # shared across every sample so the ensemble targets the same posterior.
    testfile = PATH / 'data/test.h5'
    if not testfile.exists():
        raise SystemExit(f"missing {testfile}; run `python prepare.py --coarsen {coarsen}` first")
    with h5py.File(testfile, 'r') as f:
        x_star = torch.from_numpy(f['x'][0, :L]).to(device).float()      # (L,1,H,W) standardised
    sub = args.sub
    A = lambda x: x[..., ::sub, ::sub]
    y = torch.normal(A(x_star), args.sigma_obs)

    # DPS guided score (fixed given `y`): the reverse diffusion is steered by the
    # Gaussian likelihood of A(decode(z)); autograd carries the likelihood
    # gradient through the decoder into the diffusion state.
    dps_score = IgnoreContext(DPSGaussianScore(
        y, A=lambda z: A(decode(z)), sde=VPSDE(score, shape=()), zeta=args.zeta))
    dps_sde = VPSDE(dps_score, shape=(L, C, Hm, Wm)).to(device)

    # Batched sampling (memory): draw `--batch` at a time, accumulate on CPU.
    N = args.n_samples
    chunks, done = [], 0
    while done < N:
        b = min(args.batch, N - done)
        x = dps_sde.sample((b,), steps=args.steps, corrections=args.corrections, tau=args.tau)
        with torch.no_grad():
            x = decode(x).cpu()                                          # (b, L, 1, H, W) std pixel
        chunks.append(x)
        done += b
        print(f'  sampled {done}/{N}', flush=True)
    x_all = torch.cat(chunks, dim=0)                                     # (N, L, 1, H, W)

    mean = x_all.mean(0).to(device)
    rmse = (mean - x_star).square().mean().sqrt().item()
    misfit = (A(mean) - y).square().mean().sqrt().item()
    print(f'DPS ensemble (N={N}): ens-mean rmse={rmse:.4f}  obs-misfit={misfit:.4f} '
          f'(sigma_obs={args.sigma_obs})', flush=True)

    # Single ensemble npz (same format as daps.py / daps_enkf.py -> metrics.py).
    out = Path(args.out) if args.out else PATH / 'results' / f'dps_sub{sub}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        posterior=destandardize(x_all.numpy()),                         # (N, L, 1, H, W)
        truth=destandardize(x_star.cpu().numpy()),                      # (L, 1, H, W)
        sub=sub,
    )
    print(f'saved DPS ensemble ({N} samples) -> {out}', flush=True)


if __name__ == '__main__':
    main()
