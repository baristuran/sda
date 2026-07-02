#!/usr/bin/env python
r"""Generate unconditional (prior) Lorenz-63 trajectories from a trained score,
plot the 3D trajectories, and evaluate the Lorenz-ODE residual.

The residual follows the notebook (``lorenz63-diffusion-daps-envar.ipynb``):
central-difference time derivative vs the analytic right-hand side, reported as
the RHS-normalised magnitude

    |r| / |f|,   r = dx/dt - f(x),   f = Lorenz RHS,

averaged over interior points. It is computed for both the generated samples and
the ground-truth test trajectories.

Usage:
    python prior_samples.py                       # latest run, 100 samples
    python prior_samples.py --run None_fs7ebbzd --n 100 --length 1024
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


def find_run(name: str = None) -> Path:
    if name:
        return PATH / 'runs' / name
    runs = sorted((PATH / 'runs').glob('*/state.pth'), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f'no trained runs under {PATH / "runs"}')
    return runs[-1].parent


def relative_residual(traj: torch.Tensor, chain, eps: float = 1e-8) -> torch.Tensor:
    r"""|r|/|f| per interior point for physical trajectories ``traj`` (N, L, 3).

    dx/dt is a central difference; f is the exact Lorenz RHS ``chain.f``.
    """
    dt = chain.dt
    f = chain.f(traj[:, 1:-1])                       # (N, L-2, 3) analytic RHS
    dxdt = (traj[:, 2:] - traj[:, :-2]) / (2 * dt)   # (N, L-2, 3) central difference
    r = dxdt - f
    return (r.norm(dim=-1) / (f.norm(dim=-1) + eps)).reshape(-1)


def plot_trajectories_3d(gen: np.ndarray, truth: np.ndarray, path: Path, n_show: int = 60):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 5))
    for col, (name, x) in enumerate([('generated (prior)', gen), ('ground truth', truth)]):
        ax = fig.add_subplot(1, 2, col + 1, projection='3d')
        m = min(n_show, len(x))
        colors = plt.cm.viridis(np.linspace(0.15, 0.9, m))
        for i in range(m):
            ax.plot(x[i, :, 0], x[i, :, 1], x[i, :, 2], lw=0.4, alpha=0.5, color=colors[i])
        ax.set_xlabel('a'); ax.set_ylabel('b'); ax.set_zlabel('c')
        ax.set_title(f'{name}  ({m} of {len(x)} trajectories)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_residual_hist(mag_gen: np.ndarray, mag_truth: np.ndarray, path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    hi = np.percentile(np.concatenate([mag_gen, mag_truth]), 99)
    bins = np.linspace(0.0, hi, 60)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(mag_truth, bins=bins, density=True, alpha=0.55, color='tab:blue',
            label=f'ground truth (median={np.median(mag_truth):.3f})')
    ax.hist(mag_gen, bins=bins, density=True, alpha=0.55, color='tab:red',
            label=f'generated (median={np.median(mag_gen):.3f})')
    ax.set_xlabel(r'relative residual  $\|r\| / \|f(x)\|$')
    ax.set_ylabel('density')
    ax.set_title('Lorenz-ODE residual, normalised by RHS magnitude')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', default=None, help='run name under runs/ (default: latest)')
    p.add_argument('--n', type=int, default=100, help='number of prior samples')
    p.add_argument('--length', type=int, default=1024, help='trajectory length')
    p.add_argument('--steps', type=int, default=256, help='reverse-diffusion steps')
    p.add_argument('--corrections', type=int, default=1, help='Langevin corrections / step')
    p.add_argument('--tau', type=float, default=0.5)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    chain = make_chain()
    results = PATH / 'results'
    results.mkdir(parents=True, exist_ok=True)

    # -- load score (local if the config carries a 'window', else global) --
    run = find_run(args.run)
    config = load_config(run)
    local = 'window' in config
    score = load_score(run / 'state.pth', local=local).to(args.device).eval()
    print(f'run {run.name}  ({"local" if local else "global"} score)', flush=True)

    # -- generate prior samples (unconditional) ---------------------------
    sde = VPSDE(score, shape=(args.length, 3)).to(args.device)
    x = sde.sample((args.n,), steps=args.steps, corrections=args.corrections, tau=args.tau)
    gen = chain.postprocess(x.cpu())                 # (n, L, 3) physical
    print(f'generated {gen.shape[0]} trajectories of length {gen.shape[1]}', flush=True)

    # -- ground truth (physical) ------------------------------------------
    with h5py.File(PATH / 'data/test.h5', mode='r') as f:
        truth = chain.postprocess(torch.from_numpy(f['x'][:]))   # (m, L, 3) physical

    # -- 3D trajectory plot -----------------------------------------------
    plot_trajectories_3d(gen.numpy(), truth.numpy(), results / 'prior_trajectories_3d.png')

    # -- Lorenz-ODE residual (RHS-normalised) -----------------------------
    mag_gen = relative_residual(gen, chain).numpy()
    mag_truth = relative_residual(truth, chain).numpy()
    plot_residual_hist(mag_gen, mag_truth, results / 'residual_hist.png')

    print('\nrelative residual |r|/|f|   (mean / median / RMS):')
    for name, m in [('generated ', mag_gen), ('groundtruth', mag_truth)]:
        print(f'  {name}: {m.mean():.4f} / {np.median(m):.4f} / {np.sqrt((m ** 2).mean()):.4f}')
    np.savez(results / 'residual.npz', mag_gen=mag_gen, mag_truth=mag_truth)
    print(f'\nsaved -> {results/"prior_trajectories_3d.png"}, '
          f'{results/"residual_hist.png"}, {results/"residual.npz"}', flush=True)


if __name__ == '__main__':
    main()
