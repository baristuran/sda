#!/usr/bin/env python
r"""Consolidate the locally-simulated Dedalus trajectories into contiguous
train/valid/test HDF5 files for fast training.

The raw Dedalus output is slow to read: each ``traj_XXXX_sN.h5`` is an HDF5
*virtual dataset* pointing at per-process files, so every access chases that
indirection (~16 ms/item vs ~0.6 ms for a contiguous read). This script reads
each trajectory once through :class:`DedalusRBCDataset` (discarding the initial
transient, coarsening and standardising), then writes one contiguous array per
split so that training reads from memory instead.

    python prepare.py                 # coarsen 2 (matches train.py CONFIG)
    python prepare.py --coarsen 1     # full 128x512 resolution

Output: ``PATH/data/{train,valid,test}.h5``, each with dataset ``x`` of shape
``(n_traj, L, 1, H, W)``, standardised, ready for
:class:`sda.utils.TrajectoryDataset`.
"""

import argparse

import h5py
import numpy as np

from pathlib import Path

from utils import *


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--coarsen', type=int, default=2,
                   help='spatial coarsening factor (must match train.py CONFIG)')
    p.add_argument('--t-start', type=int, default=DEDALUS_T_START,
                   help='frames of initial transient to discard')
    p.add_argument('--interp', action='store_true',
                   help='interpolate the Chebyshev z-grid onto a uniform grid '
                        '(recommended: the U-Net assumes uniform pixel spacing)')
    p.add_argument('--root', default=DEDALUS_ROOT, help='Dedalus snapshots root')
    p.add_argument('--out', default=str(PATH / 'data'), help='output directory')
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for split in ('train', 'valid', 'test'):
        ds = DedalusRBCDataset(
            split, window=None, flatten=False,
            t_start=args.t_start, coarsen=args.coarsen, root=args.root,
            interpolate=args.interp)
        n = len(ds)
        L, C, H, W = ds[0][0].shape
        grid = 'uniform' if args.interp else 'Chebyshev'
        print(f'{split}: {n} trajectories -> (L,C,H,W)=({L},{C},{H},{W}) [{grid} z]', flush=True)

        path = out / f'{split}.h5'
        with h5py.File(path, 'w') as f:
            dset = f.create_dataset('x', shape=(n, L, C, H, W), dtype=np.float32)
            for i in range(n):
                dset[i] = ds[i][0].numpy()
        print(f'  wrote {path}', flush=True)


if __name__ == '__main__':
    main()
