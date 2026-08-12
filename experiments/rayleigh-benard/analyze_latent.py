#!/usr/bin/env python
r"""PCA / t-SNE analysis of the VAE latent space (latent-diffusion Rayleigh-Benard).

Loads the encoded latents from ``data_latent/<split>.h5`` (built by
``encode_latents.py``), flattens each per-frame latent ``(C_z, H_l, W_l)`` into a
vector, standardizes the features, and projects them to 2-D with **PCA** and
**t-SNE**. Points are coloured by each frame's time step within its trajectory, so
the temporal structure of the latent manifold is visible. Two high-resolution
(400 dpi by default) PNGs are written.

t-SNE is O(n^2), so a random subset of frames is used (``--n``). For quality and
speed the t-SNE runs on the top-50 PCA components (the standard preprocessing).

    python analyze_latent.py                              # train split, 6000 frames
    python analyze_latent.py --split test --n 4000 --perplexity 40
    python analyze_latent.py --color-by traj --dpi 600
"""

from __future__ import annotations

import argparse

from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use('Agg')                       # headless: write files, no display
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils import PATH


def _scatter(ax, xy, color, cmap, cbar_label, title):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=color, cmap=cmap, s=5, alpha=0.6,
                    linewidths=0, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    ax.set_aspect('equal', adjustable='datalim')
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', default='train', choices=['train', 'valid', 'test'],
                   help='which data_latent/<split>.h5 to analyze')
    p.add_argument('--data-latent', default=None,
                   help='latent data dir (default: <PATH>/data_latent)')
    p.add_argument('--n', type=int, default=6000,
                   help='number of frames to subsample (t-SNE is O(n^2))')
    p.add_argument('--color-by', default='time', choices=['time', 'traj'],
                   help='colour points by time step within a trajectory, or by trajectory id')
    p.add_argument('--perplexity', type=float, default=30.0, help='t-SNE perplexity')
    p.add_argument('--pca-components', type=int, default=50,
                   help='PCA dimensions fed to t-SNE (0 = feed the full latent)')
    p.add_argument('--seed', type=int, default=0, help='RNG seed for the subsample and t-SNE')
    p.add_argument('--dpi', type=int, default=400, help='output PNG resolution')
    p.add_argument('--out-dir', default=None,
                   help='where to write the PNGs (default: <PATH>/latent_analysis)')
    args = p.parse_args()

    datadir = Path(args.data_latent) if args.data_latent else PATH / 'data_latent'
    datafile = datadir / f'{args.split}.h5'
    if not datafile.exists():
        raise SystemExit(f'missing {datafile}; run `python encode_latents.py --vae-run <run>` first')

    with h5py.File(datafile, 'r') as f:
        z = f['x'][:]                        # (T, L, C, H, W)
        vae_run = dict(f.attrs).get('vae_run', '?')
    T, L, C, H, W = z.shape
    print(f'{args.split}: {T} trajectories x {L} steps, latent {C}x{H}x{W}  '
          f'(vae_run {vae_run})', flush=True)

    # Flatten to one feature vector per frame; keep time-step and trajectory labels.
    X = z.reshape(T * L, C * H * W)                          # (T*L, D)
    time_label = np.tile(np.arange(L), T)                    # step within trajectory
    traj_label = np.repeat(np.arange(T), L)                  # trajectory id

    # Subsample frames (t-SNE cost); reproducible with --seed.
    rng = np.random.default_rng(args.seed)
    n = min(args.n, X.shape[0])
    sel = rng.choice(X.shape[0], size=n, replace=False)
    X, time_label, traj_label = X[sel], time_label[sel], traj_label[sel]
    color = time_label if args.color_by == 'time' else traj_label
    cbar_label = 'time step' if args.color_by == 'time' else 'trajectory'
    cmap = 'viridis' if args.color_by == 'time' else 'tab20'
    print(f'analyzing {n} frames, dim {X.shape[1]}', flush=True)

    # Standardize features, then project.
    Xs = StandardScaler().fit_transform(X)

    pca2 = PCA(n_components=2, random_state=args.seed).fit(Xs)
    xy_pca = pca2.transform(Xs)
    evr = pca2.explained_variance_ratio_
    print(f'PCA explained variance: PC1 {evr[0]:.3f}, PC2 {evr[1]:.3f}, '
          f'sum {evr.sum():.3f}', flush=True)

    # t-SNE on the top PCA components (standard, faster and less noisy).
    if args.pca_components and args.pca_components < Xs.shape[1]:
        Xt = PCA(n_components=args.pca_components, random_state=args.seed).fit_transform(Xs)
        print(f't-SNE input: top-{args.pca_components} PCA components', flush=True)
    else:
        Xt = Xs
    print(f'running t-SNE (perplexity {args.perplexity:g}) ...', flush=True)
    xy_tsne = TSNE(n_components=2, perplexity=args.perplexity, init='pca',
                   learning_rate='auto', random_state=args.seed).fit_transform(Xt)

    # -- plot + save two 400-dpi PNGs --------------------------------------
    out_dir = Path(args.out_dir) if args.out_dir else PATH / 'latent_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter(ax, xy_pca, color, cmap, cbar_label,
             f'Latent PCA ({args.split}, n={n})\n'
             f'PC1 {evr[0]:.1%} / PC2 {evr[1]:.1%} variance')
    fig.tight_layout()
    pca_png = out_dir / f'latent_pca_{args.split}.png'
    fig.savefig(pca_png, dpi=args.dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter(ax, xy_tsne, color, cmap, cbar_label,
             f'Latent t-SNE ({args.split}, n={n}, perplexity={args.perplexity:g})')
    fig.tight_layout()
    tsne_png = out_dir / f'latent_tsne_{args.split}.png'
    fig.savefig(tsne_png, dpi=args.dpi)
    plt.close(fig)

    print(f'saved -> {pca_png}\n         {tsne_png}  ({args.dpi} dpi)', flush=True)


if __name__ == '__main__':
    main()
