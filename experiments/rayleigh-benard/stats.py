#!/usr/bin/env python
r"""Statistical evaluation of generated Rayleigh-Bénard fields vs ground truth.

Compares the model's unconditional (prior) samples against real test
trajectories on physical diagnostics of the (nondimensional) temperature field
``theta`` (= buoyancy; hot bottom theta=1, cold top theta=0):

1. mean temperature profile along the wall-normal (z) axis;
2. RMS-fluctuation temperature profile along z;
3. horizontal (streamwise) power spectrum E(k_x);
4. thermal dissipation-rate profile  eps_theta(z) = kappa <|grad theta|^2>_{x,t};
5. the temperature PDF.

Prior samples are read from ``results/prior_fields.h5`` (written by
``sample_prior.py``) and ground truth from the consolidated ``data/test.h5``.
Both are streamed in batches, so nothing is fully loaded into memory. Each
diagnostic is saved as its own PNG (``results/stat_theta_*.png``,
``stat_thermal_dissipation.png``) plus ``results/stats.npz``.

Assumes a uniform grid (run ``prepare.py --interp``); on the Chebyshev grid the
z-axis / z-derivatives are only approximate, though the prior-vs-truth
comparison stays valid since both use the same grid.

Usage:
    python sample_prior.py --n 100 --batch 8    # first: generate prior fields
    python stats.py                             # then: compare vs test set
    python stats.py --prior-file results/prior_fields.h5 --batch 16 --rayleigh 1e7
"""

import argparse

import h5py
import numpy as np

from pathlib import Path

from utils import *


# Domain from rayleigh_benard.py (non-dimensional: box height + free-fall time).
LX, LZ = 4.0, 1.0


def h5_batches(path: Path, key: str, batch: int, standardized: bool):
    r"""Yield frame batches ``(m, H, W)`` of physical theta from an HDF5 dataset
    of shape (n, L, C, H, W), reading ``batch`` trajectories at a time."""
    with h5py.File(path, 'r') as f:
        d = f[key]
        n = d.shape[0]
        for i in range(0, n, batch):
            x = np.asarray(d[i:i + batch])
            if standardized:
                x = destandardize(x)
            yield to_frames(x)


def streaming_stats(batches, kappa: float, n_bins: int = 100) -> dict:
    r"""Diagnostics of physical theta accumulated over batches, without loading
    all frames at once. ``batches`` is a *factory* (a zero-arg callable returning
    a fresh iterator of ``(m, H, W)`` frame batches), so it can be traversed
    twice: pass 1 fixes the mean profile and PDF range, pass 2 accumulates the
    fluctuation-based diagnostics (matching the in-memory computation exactly).
    """
    # -- pass 1: mean profile, PDF range ----------------------------------
    sum_z = None
    count = 0
    gmin, gmax = np.inf, -np.inf
    for b in batches():
        b = b.astype(np.float64)
        m, H, W = b.shape
        if sum_z is None:
            sum_z = np.zeros(H)
        sum_z += b.sum(axis=(0, 2))
        count += m * W
        gmin, gmax = min(gmin, b.min()), max(gmax, b.max())
    if sum_z is None:
        raise ValueError('no data in stream')
    mean_prof = sum_z / count
    dz, dx = LZ / (H - 1), LX / W
    edges = np.linspace(gmin, gmax, n_bins + 1)

    # -- pass 2: rms, spectrum, turbulent dissipation, PDF ----------------
    sumsq_z = np.zeros(H)
    grad2_sum_z = np.zeros(H)
    power_sum = None
    power_count = 0
    hist = np.zeros(n_bins)
    for b in batches():
        b = b.astype(np.float64)
        m = b.shape[0]
        fluc = b - mean_prof[None, :, None]                 # fluctuation about mean profile
        sumsq_z += (fluc ** 2).sum(axis=(0, 2))

        bx = b - b.mean(axis=2, keepdims=True)              # remove k=0 (horiz. mean)
        P = (np.abs(np.fft.rfft(bx, axis=2)) ** 2).sum(axis=(0, 1))
        if power_sum is None:
            power_sum = np.zeros_like(P)
        power_sum += P
        power_count += m * H

        dbdx = (np.roll(fluc, -1, axis=2) - np.roll(fluc, 1, axis=2)) / (2 * dx)  # periodic x
        dbdz = np.gradient(fluc, dz, axis=1)                                       # wall-bounded z
        grad2_sum_z += (dbdx ** 2 + dbdz ** 2).sum(axis=(0, 2))

        hist += np.histogram(b.ravel(), bins=edges)[0]

    z = np.linspace(0.0, LZ, H)
    rms_prof = np.sqrt(np.maximum(sumsq_z / count, 0.0))
    k = 2.0 * np.pi * np.fft.rfftfreq(W, d=dx)
    power = power_sum / power_count / (W ** 2)
    eps_prof = kappa * grad2_sum_z / count
    eps_mean = float(eps_prof.mean())                       # = volume avg (uniform z)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pdf = hist / (hist.sum() * (edges[1] - edges[0]))        # density

    return dict(z=z, mean_prof=mean_prof, rms_prof=rms_prof, k=k, power=power,
                eps_prof=eps_prof, eps_mean=eps_mean, pdf_x=centers, pdf=pdf)


def to_frames(x: np.ndarray) -> np.ndarray:
    r"""(..., L, C, H, W) or (..., L, H, W) -> (M, H, W), single buoyancy channel."""
    x = np.asarray(x)
    if x.ndim == 5:            # (n, L, C, H, W)
        x = x[:, :, 0]
    elif x.ndim == 4 and x.shape[1] == 1:  # (L, C, H, W)
        x = x[:, 0]
    return x.reshape(-1, x.shape[-2], x.shape[-1])


def plot(sp: dict, sg: dict, outdir: Path) -> list:
    r"""Save each diagnostic as its own PNG (theta = nondimensional temperature).
    Ground truth in blue, prior in red. Returns the list of written paths."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cg, cp = 'tab:blue', 'tab:red'      # ground truth / prior
    paths = []

    def panel(name, draw_fn, xlabel, ylabel, title, **subplots_kw):
        fig, a = plt.subplots(figsize=(5, 4))
        draw_fn(a)
        a.set_xlabel(xlabel); a.set_ylabel(ylabel); a.set_title(title)
        a.legend(fontsize=9)
        fig.tight_layout()
        path = outdir / name
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)

    # mean theta profile (z on the vertical axis)
    def _mean(a):
        a.plot(sg['mean_prof'], sg['z'], cg, label='truth')
        a.plot(sp['mean_prof'], sp['z'], cp, label='prior')
    panel('stat_theta_mean_profile.png', _mean,
          r'$\langle\theta\rangle$', 'z', r'mean $\theta$ profile')

    # rms theta profile
    def _rms(a):
        a.plot(sg['rms_prof'], sg['z'], cg, label='truth')
        a.plot(sp['rms_prof'], sp['z'], cp, label='prior')
    panel('stat_theta_rms_profile.png', _rms,
          r'$\theta_{\mathrm{rms}}$', 'z', r'rms $\theta$ profile')

    # horizontal spectrum
    def _spec(a):
        a.loglog(sg['k'][1:], sg['power'][1:], cg, label='truth')
        a.loglog(sp['k'][1:], sp['power'][1:], cp, label='prior')
    panel('stat_theta_spectrum.png', _spec,
          r'horizontal wavenumber $k_x$', r'$E(k_x)$', r'horizontal $\theta$ spectrum')

    # thermal dissipation profile
    def _eps(a):
        a.plot(sg['eps_prof'], sg['z'], cg, label=f"truth (⟨ε⟩={sg['eps_mean']:.2e})")
        a.plot(sp['eps_prof'], sp['z'], cp, label=f"prior (⟨ε⟩={sp['eps_mean']:.2e})")
    panel('stat_thermal_dissipation.png', _eps,
          r'$\varepsilon_\theta = \kappa\langle|\nabla\theta|^2\rangle$', 'z',
          'thermal dissipation profile')

    # theta PDF
    def _pdf(a):
        a.semilogy(sg['pdf_x'], sg['pdf'], cg, label='truth')
        a.semilogy(sp['pdf_x'], sp['pdf'], cp, label='prior')
    panel('stat_theta_pdf.png', _pdf, r'$\theta$', 'pdf', r'$\theta$ PDF')

    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prior-file', default=None,
                   help="prior fields .h5 (default results/prior_fields.h5 from sample_prior.py)")
    p.add_argument('--batch', type=int, default=8,
                   help='trajectories read per batch (streaming, keeps memory bounded)')
    p.add_argument('--bins', type=int, default=100, help='PDF histogram bins')
    p.add_argument('--rayleigh', type=float, default=1e7)
    p.add_argument('--prandtl', type=float, default=1.0)
    args = p.parse_args()

    kappa = (args.rayleigh * args.prandtl) ** -0.5
    results = PATH / 'results'
    results.mkdir(parents=True, exist_ok=True)

    priorfile = Path(args.prior_file) if args.prior_file else results / 'prior_fields.h5'
    testfile = PATH / 'data/test.h5'
    for f in (priorfile, testfile):
        if not f.exists():
            raise SystemExit(f"missing {f} (run sample_prior.py / prepare.py --interp first)")

    # Stream both sources in batches; nothing is fully loaded into memory.
    sg = streaming_stats(lambda: h5_batches(testfile, 'x', args.batch, standardized=True),
                         kappa, args.bins)
    sp = streaming_stats(lambda: h5_batches(priorfile, 'theta', args.batch, standardized=False),
                         kappa, args.bins)

    paths = plot(sp, sg, results)
    np.savez(results / 'stats.npz',
             **{f'prior_{k}': v for k, v in sp.items()},
             **{f'truth_{k}': v for k, v in sg.items()})
    print(f'  prior  ⟨ε_θ⟩ = {sp["eps_mean"]:.3e}')
    print(f'  truth  ⟨ε_θ⟩ = {sg["eps_mean"]:.3e}')
    print('saved:')
    for p in paths:
        print(f'  {p}')
    print(f'  {results/"stats.npz"}', flush=True)


if __name__ == '__main__':
    main()
