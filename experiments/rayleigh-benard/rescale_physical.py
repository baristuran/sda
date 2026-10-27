#!/usr/bin/env python
r"""Re-express saved physical fields with the corrected destandardisation.

Until 2026-09-27 ``utils.destandardize`` used the :class:`RBCDataset` constants
(mean 0.3672, std 0.1631), but ``data/*.h5`` are standardised with the Dedalus
ones (mean 0.4993, std 0.2053, now read from ``data/normalization.txt``). Every
field it produced is therefore an exact affine image of the right one, and is
corrected without resampling:

    x_std = (x_old - 0.3672) / 0.1631
    x_new = x_std * 0.2053 + 0.4993

Handles the samplers' / ``eval.py``'s ``*.npz`` (keys ``posterior``, ``prior``,
``truth``; everything else is copied unchanged) and ``sample_prior.py`` /
``npz_to_h5.py``'s ``*.h5`` (dataset ``theta``). Corrected files are marked
(npz key / h5 attribute ``units = 'dedalus'``), and marked files are skipped,
so running the script twice cannot apply the correction twice.

    python rescale_physical.py results/*.npz                 # writes *_fixed.npz
    python rescale_physical.py results/*.npz results/*.h5 --inplace
"""

import argparse
import shutil

import h5py
import numpy as np

from pathlib import Path

from utils import load_normalization

OLD_MEAN, OLD_STD = 0.3672, 0.1631         # the constants the old destandardize used

FIELD_KEYS = ('posterior', 'prior', 'truth')
MARK = 'dedalus'


def fix(x: np.ndarray) -> np.ndarray:
    mean, std = load_normalization()       # what data/*.h5 are standardised with
    x_std = (np.asarray(x, dtype=np.float64) - OLD_MEAN) / OLD_STD
    return (x_std * std + mean).astype(np.asarray(x).dtype)


def fix_npz(src: Path, dst: Path) -> str:
    data = dict(np.load(src))
    if str(data.get('units', '')) == MARK:
        return 'already corrected, skipped'
    keys = [k for k in FIELD_KEYS if k in data]
    if not keys:
        return f'no field keys {FIELD_KEYS}, skipped'
    for k in keys:
        data[k] = fix(data[k])
    data['units'] = np.array(MARK)
    tmp = dst.with_name(dst.stem + '.tmp.npz')
    np.savez(tmp, **data)
    tmp.replace(dst)
    return 'rescaled ' + ', '.join(keys)


def fix_h5(src: Path, dst: Path, batch: int = 8) -> str:
    with h5py.File(src, 'r') as f:
        if 'theta' not in f:
            return "no 'theta' dataset, skipped"
        if f['theta'].attrs.get('units', '') == MARK:
            return 'already corrected, skipped'
    if dst != src:
        shutil.copyfile(src, dst)
    with h5py.File(dst, 'r+') as f:
        d = f['theta']
        for i in range(0, d.shape[0], batch):
            d[i:i + batch] = fix(d[i:i + batch])
        d.attrs['units'] = MARK
    return "rescaled 'theta'"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('files', type=Path, nargs='+', help='*.npz / *.h5 files to correct')
    p.add_argument('--inplace', action='store_true',
                   help='overwrite the inputs (default: write <stem>_fixed.<ext> alongside)')
    args = p.parse_args()

    for src in args.files:
        dst = src if args.inplace else src.with_name(f'{src.stem}_fixed{src.suffix}')
        if src.suffix == '.npz':
            msg = fix_npz(src, dst)
        elif src.suffix == '.h5':
            msg = fix_h5(src, dst)
        else:
            msg = 'not .npz / .h5, skipped'
        print(f'{src}: {msg}' + ('' if 'skipped' in msg or dst == src else f' -> {dst}'))


if __name__ == '__main__':
    main()
