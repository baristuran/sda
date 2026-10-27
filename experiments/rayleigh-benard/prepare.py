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
    python prepare.py --t-coarsen 2   # also downsample 2x in time (block-averaged)

Output: ``PATH/data/{train,valid,test}.h5``, each with dataset ``x`` of shape
``(n_traj, L, 1, H, W)``, standardised, ready for
:class:`sda.utils.TrajectoryDataset`. ``L`` is the number of *decimated*-time
frames (raw frames // ``t_coarsen``) when ``--t-coarsen`` > 1.

Standardisation is ``(theta - mean) / std`` with the mean and std of the TRAIN
split, stored in ``<out>/normalization.txt``. If that file already exists it is
reused and nothing is recomputed (delete it to recompute); every script that
converts back to physical units (``utils.destandardize``) reads it.
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
    p.add_argument('--t-coarsen', type=int, default=1,
                   help='temporal coarsening factor: block-average every N raw frames '
                        'into one (must match train.py CONFIG)')
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

    # Standardisation statistics: reuse <out>/normalization.txt if it exists,
    # otherwise compute them over the train split (processed first, below).
    norm = None
    if (out / NORMALIZATION_FILE).exists():
        norm = load_normalization(out)
        print(f'using {out / NORMALIZATION_FILE}: mean={norm[0]:.6g} std={norm[1]:.6g}', flush=True)

    for split in ('train', 'valid', 'test'):
        ds = DedalusRBCDataset(                                   # raw buoyancy
            split, window=None, flatten=False,
            t_start=args.t_start, coarsen=args.coarsen, t_coarsen=args.t_coarsen,
            root=args.root, interpolate=args.interp)
        n = len(ds)
        L, C, H, W = ds[0][0].shape
        grid = 'uniform' if args.interp else 'Chebyshev'
        print(f'{split}: {n} trajectories -> (L,C,H,W)=({L},{C},{H},{W}) [{grid} z]', flush=True)

        path = out / f'{split}.h5'
        with h5py.File(path, 'w') as f:
            dset = f.create_dataset('x', shape=(n, L, C, H, W), dtype=np.float32)
            if norm is not None:
                mean, std = norm
                for i in range(n):
                    dset[i] = (ds[i][0].numpy() - mean) / std
            else:
                # Write raw while accumulating the statistics (one read of the
                # slow Dedalus files), then standardise the file in place.
                count, s, s2 = 0, 0.0, 0.0
                for i in range(n):
                    x = ds[i][0].numpy()
                    dset[i] = x
                    x = x.astype(np.float64)
                    count, s, s2 = count + x.size, s + x.sum(), s2 + np.square(x).sum()
                mean = s / count
                std = float(np.sqrt(s2 / count - mean ** 2))
                norm = (float(mean), std)
                print(f'  computed train statistics: mean={norm[0]:.6g} std={norm[1]:.6g} '
                      f'-> {save_normalization(out, *norm)}', flush=True)
                for i in range(n):
                    dset[i] = (dset[i] - norm[0]) / norm[1]
        print(f'  wrote {path}', flush=True)


if __name__ == '__main__':
    main()
