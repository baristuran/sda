#!/usr/bin/env python
r"""Statistical evaluation of generated Rayleigh-Bénard fields vs ground truth.

Compares the model's unconditional (prior) samples against real test
trajectories on physical diagnostics of the (nondimensional) temperature field
``theta`` (= buoyancy; hot bottom theta=1, cold top theta=0):

1. mean temperature profile along the wall-normal (z) axis;
2. RMS-fluctuation temperature profile along z;
3. skewness and flatness profiles of the fluctuation theta';
4. horizontal (streamwise) power spectrum E(k_x): full depth, and separately
   in a near-wall band and a mid-plane band;
5. streamwise two-point correlation R(r_x), near-wall and mid-plane;
6. thermal dissipation-rate profile  eps_theta(z) = kappa <|grad theta|^2>_{x,t};
7. the wall Nusselt number Nu = -d<theta>/dz|_wall / (Delta_theta / LZ);
8. the temperature PDF.

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


def streaming_stats(batches, kappa: float, n_bins: int = 100,
                    wall_frac: float = 0.1, center_frac: float = 0.1) -> dict:
    r"""Diagnostics of physical theta accumulated over batches, without loading
    all frames at once. ``batches`` is a *factory* (a zero-arg callable returning
    a fresh iterator of ``(m, H, W)`` frame batches), so it can be traversed
    twice: pass 1 fixes the mean profile and PDF range, pass 2 accumulates the
    fluctuation-based diagnostics (matching the in-memory computation exactly).

    In addition to the mean/rms/dissipation profiles, the horizontal spectrum,
    and the PDF, this reports (i) skewness and flatness profiles of the
    fluctuation ``theta' = theta - <theta>(z)``; (ii) the horizontal spectrum and
    two-point (streamwise) correlation resolved separately in a **near-wall** band
    (``z`` within ``wall_frac`` of *either* wall, by up/down symmetry) and a
    **mid-plane** band (``|z - LZ/2| <= center_frac``); and (iii) the wall
    Nusselt number ``Nu = -d<theta>/dz|_wall / (Delta_theta / LZ)`` from the mean
    profile at each wall.
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

    z = np.linspace(0.0, LZ, H)
    # z-index bands (exact walls carry zero fluctuation -- Dirichlet BC -- so the
    # near-wall band excludes them; both walls are pooled by up/down symmetry).
    wall_mask = (((z > 0) & (z <= wall_frac * LZ)) |
                 ((z < LZ) & (z >= LZ - wall_frac * LZ)))
    center_mask = np.abs(z - 0.5 * LZ) <= center_frac * LZ
    if not wall_mask.any() or not center_mask.any():
        raise ValueError('empty wall/center band; adjust wall_frac / center_frac')

    # -- pass 2: rms/skew/flat, per-z spectrum, dissipation, PDF -----------
    sumsq_z = np.zeros(H)
    sum3_z = np.zeros(H)
    sum4_z = np.zeros(H)
    grad2_sum_z = np.zeros(H)
    spec_z_sum = None                                       # (H, n_freq): sum_frames |rfft_x theta'|^2
    n_frames = 0
    hist = np.zeros(n_bins)
    # two-point correlation: physical-space shift accumulators, one entry per
    # streamwise lag s = 0 .. W//2 (max shift = half the domain).
    max_shift = W // 2
    corr_wall_acc = np.zeros(max_shift + 1)
    corr_center_acc = np.zeros(max_shift + 1)
    for b in batches():
        b = b.astype(np.float64)
        m = b.shape[0]
        n_frames += m
        fluc = b - mean_prof[None, :, None]                 # fluctuation about mean profile
        sumsq_z += (fluc ** 2).sum(axis=(0, 2))
        sum3_z += (fluc ** 3).sum(axis=(0, 2))
        sum4_z += (fluc ** 4).sum(axis=(0, 2))

        # two-point (streamwise) correlation in PHYSICAL space: shift theta' by s
        # pixels along x (periodic -> circular roll) and accumulate the product,
        # separately for the near-wall and mid-plane z-bands. No FFT.
        fw = fluc[:, wall_mask, :]                          # (m, n_wall, W)
        fc = fluc[:, center_mask, :]                        # (m, n_center, W)
        for s in range(max_shift + 1):
            corr_wall_acc[s] += (fw * np.roll(fw, -s, axis=2)).sum()
            corr_center_acc[s] += (fc * np.roll(fc, -s, axis=2)).sum()

        bx = b - b.mean(axis=2, keepdims=True)              # remove k=0 (horiz. mean)
        F2 = np.abs(np.fft.rfft(bx, axis=2)) ** 2           # (m, H, n_freq)
        S = F2.sum(axis=0)                                  # (H, n_freq)
        if spec_z_sum is None:
            spec_z_sum = np.zeros_like(S)
        spec_z_sum += S

        dbdx = (np.roll(fluc, -1, axis=2) - np.roll(fluc, 1, axis=2)) / (2 * dx)  # periodic x
        dbdz = np.gradient(fluc, dz, axis=1)                                       # wall-bounded z
        grad2_sum_z += (dbdx ** 2 + dbdz ** 2).sum(axis=(0, 2))

        hist += np.histogram(b.ravel(), bins=edges)[0]

    # -- profiles ---------------------------------------------------------
    var_z = sumsq_z / count
    rms_prof = np.sqrt(np.maximum(var_z, 0.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        skew_prof = np.where(var_z > 1e-12, (sum3_z / count) / var_z ** 1.5, np.nan)
        flat_prof = np.where(var_z > 1e-12, (sum4_z / count) / var_z ** 2, np.nan)

    # -- spectra: full depth, near wall, mid-plane ------------------------
    k = 2.0 * np.pi * np.fft.rfftfreq(W, d=dx)
    W2 = float(W ** 2)
    power = spec_z_sum.sum(axis=0) / (n_frames * H) / W2                  # full-depth avg
    power_wall = spec_z_sum[wall_mask].mean(axis=0) / n_frames / W2       # near-wall band avg
    power_center = spec_z_sum[center_mask].mean(axis=0) / n_frames / W2   # mid-plane band avg

    # -- two-point (streamwise) correlation R(r_x): physical space --------
    # accumulated above by shifting theta' along x up to W//2 pixels (= LX/2);
    # the s=0 lag is <theta'^2>, so normalising by it gives R(0)=1. theta' is the
    # fluctuation about the mean profile (not the per-frame horizontal mean used
    # for the FFT spectrum), so this is not the exact irfft of the plotted E(k_x).
    r = np.arange(max_shift + 1) * dx
    corr_wall = corr_wall_acc / (corr_wall_acc[0] if corr_wall_acc[0] != 0.0 else 1.0)
    corr_center = corr_center_acc / (corr_center_acc[0] if corr_center_acc[0] != 0.0 else 1.0)

    # -- thermal dissipation ----------------------------------------------
    eps_prof = kappa * grad2_sum_z / count
    eps_mean = float(eps_prof.mean())                       # = volume avg (uniform z)

    # -- wall Nusselt number from the mean-profile gradient ---------------
    # Nu = -d<theta>/dz|_wall / (Delta_theta / LZ); = 1 for pure conduction.
    # 2nd-order one-sided differences at each wall; both walls reported + mean.
    dTheta = mean_prof[0] - mean_prof[-1]
    ref = dTheta / LZ
    g_bot = (-3 * mean_prof[0] + 4 * mean_prof[1] - mean_prof[2]) / (2 * dz)
    g_top = (3 * mean_prof[-1] - 4 * mean_prof[-2] + mean_prof[-3]) / (2 * dz)
    nu_bot = float(-g_bot / ref)
    nu_top = float(-g_top / ref)
    nu_wall = 0.5 * (nu_bot + nu_top)

    # -- PDF --------------------------------------------------------------
    centers = 0.5 * (edges[:-1] + edges[1:])
    pdf = hist / (hist.sum() * (edges[1] - edges[0]))        # density

    return dict(z=z, mean_prof=mean_prof, rms_prof=rms_prof,
                skew_prof=skew_prof, flat_prof=flat_prof,
                k=k, power=power, power_wall=power_wall, power_center=power_center,
                r=r, corr_wall=corr_wall, corr_center=corr_center,
                eps_prof=eps_prof, eps_mean=eps_mean,
                nu_bot=nu_bot, nu_top=nu_top, nu_wall=nu_wall,
                pdf_x=centers, pdf=pdf,
                wall_frac=wall_frac, center_frac=center_frac)


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
        fig.savefig(path, dpi=400)
        plt.close(fig)
        paths.append(path)

    # mean theta profile (z on the vertical axis); wall Nusselt in the legend
    def _mean(a):
        a.plot(sg['mean_prof'], sg['z'], cg, label=f"truth (Nu={sg['nu_wall']:.2f})")
        a.plot(sp['mean_prof'], sp['z'], cp, label=f"prior (Nu={sp['nu_wall']:.2f})")
    panel('stat_theta_mean_profile.png', _mean,
          r'$\langle\theta\rangle$', 'z', r'mean $\theta$ profile')

    # rms theta profile: bottom half of the domain only (z in (0, LZ/2]), on a
    # log z-axis to resolve the near-wall boundary layer (the z=0 wall point is
    # non-positive so it drops out of a log scale anyway).
    def _rms(a):
        for s, c, lab in [(sg, cg, 'truth'), (sp, cp, 'prior')]:
            z = s['z']
            m =  (z <= LZ / 2)
            a.plot(s['rms_prof'], z, c, label=lab)
        #a.set_yscale('log')
        # a.set_ylim(top=LZ / 2)
    panel('stat_theta_rms_profile.png', _rms,
          r'$\theta_{\mathrm{rms}}$', 'z', r'rms $\theta$ profile (lower half)')

    # horizontal spectrum: full depth, then near-wall and mid-plane separately
    def _spec_key(key):
        def f(a):
            a.loglog(sg['k'][1:], sg[key][1:], cg, label='truth')
            a.loglog(sp['k'][1:], sp[key][1:], cp, label='prior')
        return f
    panel('stat_theta_spectrum.png', _spec_key('power'),
          r'horizontal wavenumber $k_x$', r'$E(k_x)$', r'horizontal $\theta$ spectrum')
    panel('stat_theta_spectrum_wall.png', _spec_key('power_wall'),
          r'horizontal wavenumber $k_x$', r'$E(k_x)$',
          rf"near-wall $\theta$ spectrum ($z$ within {sg['wall_frac']:.2f} of a wall)")
    panel('stat_theta_spectrum_center.png', _spec_key('power_center'),
          r'horizontal wavenumber $k_x$', r'$E(k_x)$',
          rf"mid-plane $\theta$ spectrum ($|z-{LZ/2:.1f}|\leq{sg['center_frac']:.2f}$)")

    # skewness and flatness profiles of theta' (Gaussian references: 0 and 3)
    def _skew(a):
        a.plot(sg['skew_prof'], sg['z'], cg, label='truth')
        a.plot(sp['skew_prof'], sp['z'], cp, label='prior')
        a.axvline(0.0, color='k', lw=0.6, ls=':', label='Gaussian')
    panel('stat_theta_skewness.png', _skew,
          r"skewness $\langle\theta'^3\rangle/\langle\theta'^2\rangle^{3/2}$", 'z',
          r'$\theta$ skewness profile')

    def _flat(a):
        a.plot(sg['flat_prof'], sg['z'], cg, label='truth')
        a.plot(sp['flat_prof'], sp['z'], cp, label='prior')
        a.axvline(3.0, color='k', lw=0.6, ls=':', label='Gaussian')
    panel('stat_theta_flatness.png', _flat,
          r"flatness $\langle\theta'^4\rangle/\langle\theta'^2\rangle^{2}$", 'z',
          r'$\theta$ flatness profile')

    # streamwise two-point correlation R(r_x): near-wall and mid-plane
    def _corr_key(key):
        def f(a):
            a.plot(sg['r'], sg[key], cg, label='truth')
            a.plot(sp['r'], sp[key], cp, label='prior')
            a.axhline(0.0, color='k', lw=0.6, ls=':')
        return f
    panel('stat_two_point_corr_wall.png', _corr_key('corr_wall'),
          r'separation $r_x$', r'$R(r_x)$', r'near-wall two-point correlation')
    panel('stat_two_point_corr_center.png', _corr_key('corr_center'),
          r'separation $r_x$', r'$R(r_x)$', r'mid-plane two-point correlation')

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
    p.add_argument('--wall-frac', type=float, default=0.1,
                   help='near-wall band half-width (fraction of LZ from each wall) '
                        'for the wall spectrum / two-point correlation')
    p.add_argument('--center-frac', type=float, default=0.1,
                   help='mid-plane band half-width (fraction of LZ about z=LZ/2)')
    p.add_argument('--out-path', type=str, default="resuts_deneme/stats/posterior")
    args = p.parse_args()

    kappa = (args.rayleigh * args.prandtl) ** -0.5
    results = PATH / args.out_path
    results.mkdir(parents=True, exist_ok=True)

    priorfile = Path(args.prior_file) if args.prior_file else results / 'prior_fields.h5'
    testfile = PATH / 'data_coarsened_in_time/test.h5'
    for f in (priorfile, testfile):
        if not f.exists():
            raise SystemExit(f"missing {f} (run sample_prior.py / prepare.py --interp first)")

    # Stream both sources in batches; nothing is fully loaded into memory.
    kw = dict(wall_frac=args.wall_frac, center_frac=args.center_frac)
    sg = streaming_stats(lambda: h5_batches(testfile, 'x', args.batch, standardized=True),
                         kappa, args.bins, **kw)
    sp = streaming_stats(lambda: h5_batches(priorfile, 'theta', args.batch, standardized=False),
                         kappa, args.bins, **kw)

    paths = plot(sp, sg, results)
    np.savez(results / 'stats.npz',
             **{f'prior_{k}': v for k, v in sp.items()},
             **{f'truth_{k}': v for k, v in sg.items()})
    print(f'  prior  ⟨ε_θ⟩ = {sp["eps_mean"]:.3e}   Nu_wall = {sp["nu_wall"]:.3f}'
          f'  (bottom {sp["nu_bot"]:.3f}, top {sp["nu_top"]:.3f})')
    print(f'  truth  ⟨ε_θ⟩ = {sg["eps_mean"]:.3e}   Nu_wall = {sg["nu_wall"]:.3f}'
          f'  (bottom {sg["nu_bot"]:.3f}, top {sg["nu_top"]:.3f})')
    print('saved:')
    for p in paths:
        print(f'  {p}')
    print(f'  {results/"stats.npz"}', flush=True)


if __name__ == '__main__':
    main()
