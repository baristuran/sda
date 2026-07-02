#!/usr/bin/env python
r"""Generate a large set of unconditional (prior) sample trajectories in
batches, writing them incrementally to a preallocated HDF5 file.

Generating ~100 trajectories in a single batch runs the GPU out of memory, so
this pre-creates the output file at its final size and fills it a batch at a
time. Output is physical buoyancy = nondimensional temperature ``theta``.

Output: ``results/prior_fields.h5`` with dataset ``theta`` of shape
(n, length, 1, H, W), consumed by ``stats.py``.

Usage:
    python sample_prior.py --n 100 --batch 8 --length 50
    RBC_RUN=<run-name> python sample_prior.py --n 100 --batch 4 --steps 256
"""

import argparse
import os

import h5py
import numpy as np
import torch

from pathlib import Path

from sda.score import VPSDE
from sda.utils import load_config

from utils import *


def find_run() -> Path:
    r = os.environ.get('RBC_RUN')
    if r:
        return PATH / 'runs' / r
    runs = sorted((PATH / 'runs').glob('*/state.pth'), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f'no trained runs under {PATH / "runs"}')
    return runs[-1].parent


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n', type=int, default=100, help='total trajectories to generate')
    p.add_argument('--batch', type=int, default=8, help='trajectories per GPU batch')
    p.add_argument('--length', type=int, default=50, help='trajectory length (frames)')
    p.add_argument('--steps', type=int, default=256, help='reverse-diffusion steps')
    p.add_argument('--corrections', type=int, default=1, help='Langevin corrections / step')
    p.add_argument('--tau', type=float, default=0.5)
    p.add_argument('--out', default=None, help='output .h5 (default results/prior_fields.h5)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    run = find_run()
    config = load_config(run)
    coarsen = config.get('coarsen', 1)
    H, W = HEIGHT // coarsen, WIDTH // coarsen

    score = load_score(run / 'state.pth').to(args.device).eval()
    sde = VPSDE(score, shape=(args.length, 1, H, W)).to(args.device)

    results = PATH / 'results'
    results.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else results / 'prior_fields.h5'

    print(f'generating {args.n} trajectories from {run.name} '
          f'(coarsen={coarsen}, grid={H}x{W}) in batches of {args.batch}', flush=True)

    # Pre-create the file at its final size (contiguous dataset), then fill slices.
    with h5py.File(out, 'w') as f:
        dset = f.create_dataset('theta', shape=(args.n, args.length, 1, H, W), dtype='float32')
        dset.attrs['run'] = run.name
        dset.attrs['coarsen'] = coarsen

        done = 0
        while done < args.n:
            b = min(args.batch, args.n - done)
            x = sde.sample((b,), steps=args.steps, corrections=args.corrections,
                           tau=args.tau).cpu().numpy()
            dset[done:done + b] = destandardize(x).astype('float32')     # physical theta
            done += b
            print(f'  {done}/{args.n}', flush=True)

    print(f'saved -> {out}', flush=True)


if __name__ == '__main__':
    main()
