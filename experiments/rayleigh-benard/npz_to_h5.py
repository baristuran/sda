#!/usr/bin/env python
r"""Convert a results ``*.npz`` file into an HDF5 file that ``stats.py`` can read.

``stats.py`` streams its "prior" source from an HDF5 file with a single dataset
named ``theta`` of shape ``(n, L, C, H, W)`` in *physical* buoyancy/theta units
(it is read with ``standardized=False``) -- the format ``sample_prior.py``
writes. The DAPS samplers instead save ``*.npz`` files (``daps_enkf.py`` /
``daps.py``: keys ``posterior`` / ``truth`` / ``sub``; ``eval.py``'s
``prior_fields.npz``: ``prior`` / ``truth``), whose arrays are already physical.

This converts one of those npz arrays -- the one named by ``--key`` -- into a
stats-compatible HDF5 file (dataset ``theta``). A leading ensemble axis is added
if the array is a single ``(L, C, H, W)`` trajectory (e.g. ``truth``).

Usage:
    python npz_to_h5.py results/daps_enkf.npz --key posterior
    python npz_to_h5.py results/daps_enkf.npz --key truth --out results/truth.h5
    # then feed it to stats.py as the prior source:
    python stats.py --prior-file results/daps_enkf.h5
"""

import argparse

import h5py
import numpy as np

from pathlib import Path

from utils import destandardize


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('npz', type=Path, help='path to a results *.npz file')
    p.add_argument('--key', required=True,
                   help="npz array to convert: 'posterior', 'prior', or 'truth'")
    p.add_argument('--out', type=Path, default=None,
                   help='output .h5 path (default: <npz stem>.h5 next to the npz)')
    p.add_argument('--standardized-input', action='store_true',
                   help='the npz array is in standardised units; destandardise it to '
                        'physical theta on write (the DAPS/eval npz arrays are already '
                        'physical, so this is off by default)')
    p.add_argument('--batch', type=int, default=8,
                   help='trajectories written per chunk (bounds memory for large ensembles)')
    args = p.parse_args()

    data = np.load(args.npz)
    if args.key not in data:
        raise SystemExit(f"'{args.key}' not in {args.npz} (available: {list(data.keys())})")

    x = data[args.key]
    if x.ndim == 4:                       # (L, C, H, W) single trajectory -> add ensemble axis
        x = x[None]
    if x.ndim != 5:
        raise SystemExit(
            f"'{args.key}' has shape {x.shape}; expected (n, L, C, H, W) or (L, C, H, W)")

    n, L, C, H, W = x.shape
    out = args.out or args.npz.with_suffix('.h5')
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out, 'w') as f:
        dset = f.create_dataset('theta', shape=(n, L, C, H, W), dtype='float32')
        for i in range(0, n, args.batch):
            chunk = np.asarray(x[i:i + args.batch], dtype=np.float32)
            if args.standardized_input:
                chunk = destandardize(chunk).astype('float32')     # -> physical theta
            dset[i:i + chunk.shape[0]] = chunk

    units = 'destandardised -> physical' if args.standardized_input else 'physical (as stored)'
    print(f"wrote {out}  dataset 'theta' shape ({n}, {L}, {C}, {H}, {W})  [{units}]")
    print(f"feed to stats.py:  python stats.py --prior-file {out}")


if __name__ == '__main__':
    main()
