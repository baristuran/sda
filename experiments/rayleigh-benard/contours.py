#!/usr/bin/env python
r"""Standalone matplotlib contour plots (with a theta colorbar) from a saved npz.

``draw``/``save_gif`` (in ``utils.py``) tile raw pixels with PIL; this instead
renders true matplotlib filled contours of the (physical) buoyancy/theta field,
one PNG per snapshot, each with its own colorbar labelled theta (nondimensional
temperature).

Reads one of the results ``*.npz`` files already written by ``eval.py``
(``prior_fields.npz``: ``prior``/``truth``) or ``daps.py`` / ``daps_enkf.py``
(``daps_envar.npz`` / ``daps_enkf.npz``: ``posterior``/``truth``). Snapshots are
subsampled every ``--stride`` frames (default 4, i.e. ~13 contours for a
50-snapshot trajectory) and saved under a separate directory next to the npz.

``--show-obs`` marks the sensor *locations* ``[..., ::sub, ::sub]`` on each
contour. Only the positions are drawn -- no observed values -- so the markers
show where the field was measured without obscuring what it looks like there.

Usage:
    python contours.py results/prior_fields.npz --key truth
    python contours.py results/prior_fields.npz --key prior --sample 0
    python contours.py results/daps_envar.npz --key posterior --sample 0 --stride 4
    python contours.py results/daps_envar.npz --key posterior --show-obs
    python contours.py results/prior_fields.npz --key prior --show-obs --sub 8
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

# Domain from rayleigh_benard.py (non-dimensional: box height + free-fall time).
LX, LZ = 4.0, 1.0


def load_trajectory(npz_path: Path, key: str, sample: int) -> np.ndarray:
    r"""Extract a single ``(L, H, W)`` physical-theta trajectory from an npz."""
    data = np.load(npz_path)
    if key not in data:
        raise KeyError(f"'{key}' not in {npz_path} (available: {list(data.keys())})")

    x = data[key]
    if x.ndim == 5:      # (n, L, 1, H, W) ensemble -> pick one member
        x = x[sample]
    if x.ndim == 4:      # (L, 1, H, W) -> drop the channel axis
        x = x[:, 0]
    if x.ndim != 3:
        raise ValueError(f"expected a (L, H, W) trajectory, got shape {x.shape}")
    return x


def load_obs_sub(npz_path: Path, sub_override: int = None) -> int:
    r"""Observed-pixel stride: the ``sub`` field the samplers record, or an
    explicit override (needed for npz files written without one)."""
    if sub_override is not None:
        return sub_override

    data = np.load(npz_path)
    if 'sub' not in data:
        raise SystemExit(
            f"{npz_path} has no 'sub' field; pass --sub N to say which pixels were observed")
    return int(data['sub'])


def save_contours(
    traj: np.ndarray,
    outdir: Path,
    stride: int,
    cmap: str = 'RdBu_r',
    levels: int = 21,
    obs_sub: int = None,
    obs_size: float = 10.0,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    vmin, vmax = float(traj.min()), float(traj.max())
    lvls = np.linspace(vmin, vmax, levels)

    L, H, W = traj.shape
    x = np.linspace(0.0, LX, W)
    z = np.linspace(0.0, LZ, H)

    if obs_sub is not None:
        # Sensor locations are the [::sub, ::sub] pixels, so they sit exactly on
        # the corresponding entries of the physical x/z grids.
        xo, zo = np.meshgrid(x[::obs_sub], z[::obs_sub])

    idx = range(0, L, stride)
    for i in idx:
        fig, ax = plt.subplots(figsize=(7, 7 * LZ / LX + 0.7))
        cf = ax.contourf(x, z, traj[i], levels=lvls, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect('equal')
        # ax.set_title(f'snapshot {i}')

        if obs_sub is not None:
            # Positions only -- no observed values. Black face with a white edge
            # so the markers read against both ends of the diverging colormap
            # (a single flat colour would vanish into one end or the other).
            ax.scatter(xo.ravel(), zo.ravel(), s=obs_size, c='k', marker='o',
                       edgecolors='w', linewidths=0.4, zorder=3)

        # `fraction=`/`shrink=` size the colorbar relative to the axes' nominal
        # bbox, not its actual (equal-aspect-shrunk) rendered box, so on this
        # wide, short domain the bar comes out far taller than the plot. Anchor
        # a same-height colorbar axes to the real box instead.
        cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.2)
        fig.colorbar(cf, cax=cax, label=r'$\theta$', format='%.2f',
                     ticks=MaxNLocator(nbins=6))
        fig.tight_layout()
        fig.savefig(outdir / f'frame_{i}.png', dpi=400)
        plt.close(fig)

    print(f'saved {len(idx)} contours (stride={stride}) -> {outdir}')


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('npz', type=Path, help='path to a results *.npz file')
    p.add_argument('--key', default='truth',
                   help="array to plot: 'truth', 'prior', or 'posterior'")
    p.add_argument('--sample', type=int, default=0,
                   help='ensemble member index, for arrays with a leading ensemble axis')
    p.add_argument('--stride', type=int, default=4, help='plot every Nth snapshot')
    p.add_argument('--show-obs', action='store_true',
                   help='mark the observation (sensor) locations; positions only, no values')
    p.add_argument('--sub', type=int, default=None,
                   help="observed-pixel stride for --show-obs (default: the npz's 'sub' field)")
    p.add_argument('--obs-size', type=float, default=10.0,
                   help='marker size for --show-obs')
    p.add_argument('--outdir', type=Path, default=None,
                   help='output directory (default: <npz dir>/contours_<npz stem>_<key>)')
    args = p.parse_args()

    traj = load_trajectory(args.npz, args.key, args.sample)
    obs_sub = load_obs_sub(args.npz, args.sub) if args.show_obs else None

    outdir = args.outdir or (args.npz.parent / f'contours_{args.npz.stem}_{args.key}')
    save_contours(traj, outdir, args.stride, levels=50,
                  obs_sub=obs_sub, obs_size=args.obs_size)


if __name__ == '__main__':
    main()
