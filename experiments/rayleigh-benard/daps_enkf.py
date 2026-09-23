#!/usr/bin/env python
r"""EnKF-based Decoupled Annealed Posterior Sampling (DAPS) for Rayleigh-Bénard.

Port of ``sample_daps_enkf`` / ``_enkf_analysis`` from
``lorenz63-diffusion-daps-envar-sda-unet-length-400-update.ipynb`` (section
"6.4 -- EnKF-DAPS") to the SDA continuous VP-SDE, following DAPS (Zhang et al.,
2024, "Improving Diffusion Inverse Problem Solving with Decoupled Noise
Annealing", daps.pdf).

DAPS decouples the diffusion prior from the likelihood: at each annealing level
``sigma_t`` it (1) denoises ``x_t`` to a forecast ensemble ``x0_hat`` with the
probability-flow ODE, (2) corrects it towards the observations with a **single
closed-form ensemble-Kalman analysis** (replacing the inner Langevin loop of
plain DAPS -- no step size, no autograd through the observation operator), then
(3) re-noises to the next (lower) level.

``_enkf_analysis`` is the **canonical stochastic (perturbed-observation) EnKF**
of Burgers et al. (1998) / Evensen (2003):

    Yf = H(Xf),  Xp/Yp = ensemble anomalies (/sqrt(Ne-1)),
    Pxy = Xp^T Yp,  Pyy = Yp^T Yp,
    y_pert_i = y + chol(alpha R) xi_i,
    K = Pxy (Pyy + alpha R)^-1,
    Xa_i = Xf_i + K (y_pert_i - H(Xf_i)).

The one departure from a textbook cycling-filter EnKF -- and the load-bearing
one -- is the prior scale. DAPS re-assimilates the *same* observation at every
anneal level, and the ``N`` chains are already posterior samples, so using
their raw cross-chain covariance as the prior counts the data once per level
and collapses the ensemble. Tweedie gives the correct prior scale at level
``t``, ``p(x0 | xt) ~ N(x0_hat, r_t^2 I)``, so the ensemble anomalies are
rescaled to have marginal std exactly ``r_t`` **per state coordinate**, keeping
only their *correlation shape* (``r_t=None`` recovers the plain reference
covariance):

    Xp := r_t * Xp / ||Xp||_2 (per column),   so that Xp^T Xp = r_t^2 * Corr.

This replaces the earlier ``beta``-blended hybrid covariance
``C_t = (1-beta) r_t^2 I + beta P^b`` entirely -- there is no isotropic
component left to blend; the whole prior comes from the (rescaled) ensemble.

The gain is never inverted: ``S = Pyy + alpha*R`` is a Gram matrix plus a
strictly positive diagonal, hence SPD by construction, so it is factored with
Cholesky and solved for all ``Ne`` chains at once (raises on loss of
definiteness instead of silently pivoting around it, unlike a generic
``inv``/LU).

**The one adaptation for the RBC field resolution.** The notebook forms
``Pxy = Xp^T @ Yp`` explicitly, an ``(n, dy)`` matrix -- trivial at Lorenz's
``n=1200``, but at RBC's field resolution ``n = L*C*H*W ~ 1e5-1e6`` this would
be tens of gigabytes. The final increment ``W @ Pxy.T`` is reassociated as
``(W @ Yp.T) @ Xp``, which is mathematically identical (matrix-multiply
associativity) but only ever forms ``(Ne, dy)``/``(Ne, Ne)`` intermediates --
the same inverse-free-in-ensemble-space idea used by the EnVar sampler
(``daps.py``), just applied to this closed-form analysis instead of a Langevin
loop. ``n_esmda`` (default 1) mirrors the notebook's ESMDA knob for multiple
sub-assimilations per level with inflated ``R``; ``n_esmda=1`` -- a single
analysis per level -- is what the notebook recommends (the anneal already *is*
the multi-update loop, so extra sub-steps re-count the data).

**Safe-EnKF-DAPS** (``--safe``, :func:`sample_safe_daps_enkf`). A variant ported
from ``lorenz63-three-operator-comparison.ipynb``. The prior rescaling ``r_t``
above keeps the increment on the diffusion noise manifold, but it *assumes a
linear observation operator* (inflating ``Xp`` inflates ``Pyy`` in proportion
only then). The safe variant instead runs a textbook stochastic EnKF on the
**raw** sample covariance (``r_t=None``) and then projects the level-total
increment ``dX = Xa - Xf`` onto a relative trust region,

    ||dX||_RMS <= c * sigma_eff(t),   alpha_TR = min(1, c*sigma_eff(t)/||dX||_RMS),

a closed-form scalar rescale (``c`` default 0.25) that preserves the increment's
direction and contains no model of ``H`` at all. This trades the linear-operator
assumption for a mild, operator-agnostic step-size cap -- more robust when the
observation operator (here ``A o decode``, nonlinear through the VAE decoder) is
not linear.

Run as a script to assimilate a sparse spatial observation of a test trajectory:

    python daps_enkf.py --n-samples 40 --sub 8            # EnKF-DAPS (r_t rescale)
    python daps_enkf.py --n-samples 40 --sub 8 --safe     # Safe-EnKF-DAPS (trust region)
    python daps_enkf.py --safe --c 0.5                    # looser trust region
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse

import h5py
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm

from sda.score import VPSDE
from sda.utils import load_config

from utils import *
from vae import load_decoder

# Reuse the annealing schedule / PF-ODE denoiser from the EnVar sampler so the
# two DAPS variants share exactly the same diffusion-prior machinery.
from daps import VPGrid, _build_anneal, _pf_ode_x0


# ------------------------------------------------------------- EnKF analysis
def _enkf_analysis(
    Xf: torch.Tensor,
    y: torch.Tensor,
    R_diag: torch.Tensor,
    Hop,
    alpha: float = 1.0,
    r_t: float = None,
    inflation: float = 1.0,
) -> torch.Tensor:
    r"""Stochastic (perturbed-observation) EnKF analysis -- Burgers et al.
    (1998), Evensen (2003) -- structurally the textbook scheme:

        Yf     = H(Xf)
        Xp, Yp = ensemble anomalies, scaled by 1/sqrt(Ne-1)
        Pxy    = Xp^T Yp                      Pyy = Yp^T Yp
        Ypert  = y + chol(alpha R) xi         perturbed observations
        K      = Pxy (Pyy + alpha R)^-1
        Xa     = Xf + K (Ypert - H(Xf))

    ``r_t`` rescales the prior covariance and is load-bearing here: DAPS
    re-assimilates the same ``y`` at every anneal level and ``Xf`` is already a
    posterior sample, so its raw scatter would count the data multiple times
    over. Tweedie's ``p(x0 | xt) ~ N(x0_hat, r_t^2 I)`` supplies the correct
    prior *scale*; only the *correlation shape* is taken from the ensemble:

        Xp := r_t * anomalies / their own norm     =>    Xp^T Xp = r_t^2 Corr.

    Pass ``r_t=None`` to recover the plain (unrescaled) sample covariance --
    e.g. for a one-shot / cycling-filter use, or to reproduce the reference
    implementation's own ``enkf_analysis()``. Note ``inflation`` is inert
    whenever ``r_t`` is given: it is applied to ``Xp`` before the per-column
    renormalisation, which divides it back out.

    ``Xf``: ``(Ne, n)`` prior ensemble. ``y``: ``(dy,)``. ``R_diag``: ``(dy,)``
    observation variances. ``Hop``: callable ``(Ne, n) -> (Ne, dy)`` (linear
    here -- spatial pixel selection). Returns the analysis ensemble, ``(Ne, n)``.

    THE GAIN IS NEVER INVERTED: ``S = Pyy + alpha*R`` is SPD by construction
    (a Gram matrix plus a strictly positive diagonal), so it is Cholesky-
    factored and solved for all ``Ne`` chains at once.

    Departure from the notebook, forced by the RBC field resolution: rather
    than forming ``Pxy = Xp^T @ Yp`` (an ``(n, dy)`` matrix -- tens of GB at
    ``n ~ 1e5-1e6``), the final increment ``W @ Pxy.T`` is computed as
    ``(W @ Yp.T) @ Xp`` (associativity), which only ever touches
    ``(Ne, dy)``/``(Ne, Ne)`` intermediates. Mathematically identical.
    """
    Ne = Xf.shape[0]
    sN = float(np.sqrt(max(Ne - 1, 1)))

    mean = Xf.mean(0, keepdim=True)
    Xp = (Xf - mean) * float(np.sqrt(inflation)) / sN                  # (Ne, n) anomalies
    if r_t is not None:
        Xp = r_t * Xp / torch.sqrt((Xp * Xp).sum(0, keepdim=True).clamp(min=1e-12))

    # Yf/Yp (hence Pyy) describe the Tweedie-rescaled prior; the innovation
    # below uses the ACTUAL (unrescaled) ensemble members.
    Yf = Hop(mean + sN * Xp)                                          # (Ne, dy)
    Yp = (Yf - Yf.mean(0, keepdim=True)) / sN                         # (Ne, dy)
    Pyy = Yp.t() @ Yp                                                 # (dy, dy)
    aR = R_diag * alpha

    xi = torch.randn(Ne, y.numel(), device=Xf.device)
    Ypert = y.view(1, -1) + xi * torch.sqrt(aR)                       # chol(diag) = sqrt
    innov = Ypert - Hop(Xf)                                           # (Ne, dy), per chain

    S = Pyy + torch.diag(aR)                                          # (dy, dy), SPD
    L = torch.linalg.cholesky(S.double())                             # never inv(S)
    W = torch.cholesky_solve(innov.t().double(), L).t().to(Xf.dtype)  # (Ne, dy) = innov S^-1

    # Xa = Xf + W @ Pxy.T, reassociated to avoid forming Pxy = Xp^T @ Yp (n, dy):
    return Xf + (W @ Yp.t()) @ Xp                                     # (Ne, n)


# ---------------------------------------------------------------- the sampler
@torch.no_grad()
def sample_daps_enkf(
    sde: VPSDE,
    obs_fn,                          # callable (Ne, prod(shape)) flat state -> (Ne, dy) predicted obs
    y_flat: torch.Tensor,            # (dy,) observations (standardised pixel space)
    R_diag: torch.Tensor,            # (dy,) observation-noise variances (standardised pixel space)
    shape,                           # diffusion event shape, e.g. (L, C, H, W) (latent or pixel)
    n_samples: int,
    device,
    n_anneal: int = 400,
    n_ode: int = 6,
    n_esmda: int = 1,
    prior_scale: float = 1.0,
    inflation: float = 1.02,
    sigma_max: float = 100.0,
    sigma_min: float = 0.01,
    rho: float = 7.0,
    T: int = 1000,
    progress: bool = True,
) -> torch.Tensor:
    r"""Draw ``n_samples`` posterior trajectories with EnKF-DAPS: one PF-ODE
    denoise plus one closed-form :func:`_enkf_analysis` per anneal level, in
    place of the inner Langevin chain of plain/EnVar DAPS. Returns a tensor of
    shape ``(n_samples, *shape)`` on ``device`` (standardised space).

    ``n_esmda`` > 1 gives Evensen's ESMDA (``alpha_i = n_esmda``,
    ``sum 1/alpha_i = 1``): multiple sub-assimilations per level with inflated
    ``R``. The default of 1 (a single analysis per level) is what the notebook
    recommends here -- the anneal itself already is the multi-update loop, so
    extra sub-steps just re-count the data and shrink the ensemble.
    """
    sde.eval()
    grid = VPGrid(sde, device, T)
    abar, sigma_eff = grid.abar, grid.sigma_eff
    anneal_idx = _build_anneal(sigma_eff, n_anneal, rho, sigma_max, sigma_min)

    N = n_samples
    y_flat = y_flat.to(device).float()                 # (dy,)
    R_diag = R_diag.to(device).float()                 # (dy,)

    # `obs_fn` maps the flattened diffusion state to predicted observations. For a
    # pixel run it is a plain pixel selection; for a latent run it decodes the
    # latent to a pixel field first (obs live in pixel space). The EnKF is
    # derivative-free, so no autograd through the decoder is needed.
    Hop = obs_fn

    x_t = torch.randn(N, *shape, device=device)
    steps = range(len(anneal_idx) - 1)
    if progress:
        steps = tqdm(steps, desc='EnKF-DAPS', ncols=88)

    for step in steps:
        t_idx, t_next = anneal_idx[step], anneal_idx[step + 1]
        x0_hat = _pf_ode_x0(grid, x_t, t_idx, n_ode, rho)          # (N, *shape) forecast ens.
        xf = x0_hat.flatten(1)                                     # (N, n)

        r_t = float((prior_scale * sigma_eff[t_idx]).clamp(min=float(sigma_eff[0])))
        xa = xf
        for _ in range(max(n_esmda, 1)):
            xa = _enkf_analysis(
                xa, y_flat, R_diag, Hop,
                alpha=float(max(n_esmda, 1)), r_t=r_t, inflation=inflation,
            )
        if not torch.isfinite(xa).all():
            xa = torch.where(torch.isfinite(xa), xa, xf)

        x0_a = xa.reshape(N, *shape)

        # re-noise to the next (lower) annealing level (DAPS / Prop. 1)
        if t_next == 0:
            x_t = x0_a
        else:
            x_t = torch.sqrt(abar[t_next]) * x0_a \
                + torch.sqrt(1.0 - abar[t_next]) * torch.randn_like(x0_a)

    return x_t.detach()


# ------------------------------------------------------- the Safe variant
@torch.no_grad()
def sample_safe_daps_enkf(
    sde: VPSDE,
    obs_fn,                          # callable (Ne, prod(shape)) flat state -> (Ne, dy) predicted obs
    y_flat: torch.Tensor,            # (dy,) observations (standardised pixel space)
    R_diag: torch.Tensor,            # (dy,) observation-noise variances (standardised pixel space)
    shape,                           # diffusion event shape, e.g. (L, C, H, W) (latent or pixel)
    n_samples: int,
    device,
    n_anneal: int = 400,
    n_ode: int = 6,
    n_esmda: int = 1,
    c: float = 0.25,
    sigma_max: float = 100.0,
    sigma_min: float = 0.01,
    rho: float = 7.0,
    T: int = 1000,
    progress: bool = True,
) -> torch.Tensor:
    r"""Safe-EnKF-DAPS: the same denoise -> analysis -> re-noise anneal as
    :func:`sample_daps_enkf`, but with the Tweedie prior rescaling ``r_t``
    replaced by a **relative trust region** on the level-total increment.

    Ported from ``sample_safe_daps`` of
    ``lorenz63-three-operator-comparison.ipynb`` (Safe-EnKF-DAPS).

    The published EnKF-DAPS rescales the prior ensemble anomalies to std
    ``r_t = sigma_eff(t)`` before the analysis. That keeps the increment on the
    diffusion noise manifold, but it *assumes the observation operator is linear*:
    inflating ``Xp`` inflates ``Pyy`` in proportion only when ``H`` is linear. The
    safe variant instead:

      1. runs a **textbook stochastic EnKF on the raw sample covariance**
         (:func:`_enkf_analysis` with ``r_t=None`` -- no rescaling, no model of
         ``H`` in the prior scale), then
      2. projects the whole level's increment ``dX = Xa - Xf`` onto the ball

             ||dX||_RMS <= c * sigma_eff(t),
             alpha_TR   = min(1, c * sigma_eff(t) / ||dX||_RMS),

         a closed-form scalar rescale ``Xa := Xf + alpha_TR * dX`` that preserves
         the increment's *direction* and contains no model of ``H`` at all.

    ``||dX||_RMS`` is the root-mean-square increment over all ensemble members and
    state coordinates. ``c`` (default 0.25, the value the Lorenz study selected)
    sets the trust-region radius as a fraction of the level noise scale;
    ``c=None`` disables the constraint (recovering the unconstrained no-``r``
    sampler). ``n_esmda`` > 1 replaces the single analysis with an ESMDA ladder
    (``alpha_i = n_esmda``, inflated ``R``) *inside* each level -- the
    Safe-ESMDA-DAPS variant -- with the trust region still applied to the level
    total. Returns ``(n_samples, *shape)`` on ``device`` (standardised space).
    """
    sde.eval()
    grid = VPGrid(sde, device, T)
    abar, sigma_eff = grid.abar, grid.sigma_eff
    anneal_idx = _build_anneal(sigma_eff, n_anneal, rho, sigma_max, sigma_min)

    N = n_samples
    y_flat = y_flat.to(device).float()                 # (dy,)
    R_diag = R_diag.to(device).float()                 # (dy,)
    Hop = obs_fn

    x_t = torch.randn(N, *shape, device=device)
    steps = range(len(anneal_idx) - 1)
    if progress:
        steps = tqdm(steps, desc='Safe-EnKF-DAPS', ncols=88)

    for step in steps:
        t_idx, t_next = anneal_idx[step], anneal_idx[step + 1]
        x0_hat = _pf_ode_x0(grid, x_t, t_idx, n_ode, rho)          # (N, *shape) forecast ens.
        xf = x0_hat.flatten(1)                                     # (N, n)

        # (1) pure EnKF on the raw sample covariance -- no Tweedie prior rescale.
        xr = xf
        for _ in range(max(n_esmda, 1)):
            xr = _enkf_analysis(
                xr, y_flat, R_diag, Hop,
                alpha=float(max(n_esmda, 1)), r_t=None, inflation=1.0,
            )
        delta = xr - xf                                           # level-total increment

        # (2) trust region: scalar rescale so the RMS increment <= c*sigma_eff(t).
        sig = float(sigma_eff[t_idx])
        d_rms = float(delta.pow(2).mean().sqrt())
        a = 1.0 if c is None else min(1.0, c * sig / (d_rms + 1e-12))
        # a == 1 returns xr exactly (not xf + 1*delta) so "c off" reproduces the
        # unconstrained sampler bit for bit over the whole anneal.
        xa = xr if a == 1.0 else xf + a * delta
        if not torch.isfinite(xa).all():
            xa = torch.where(torch.isfinite(xa), xa, xf)

        x0_a = xa.reshape(N, *shape)

        # re-noise to the next (lower) annealing level (DAPS / Prop. 1)
        if t_next == 0:
            x_t = x0_a
        else:
            x_t = torch.sqrt(abar[t_next]) * x0_a \
                + torch.sqrt(1.0 - abar[t_next]) * torch.randn_like(x0_a)

    return x_t.detach()


# ---------------------------------------------------------------- assimilation demo
def find_run(name=None) -> Path:
    if name:
        return PATH / 'runs' / name
    runs = sorted((PATH / 'runs').glob('*/state.pth'), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f'no trained runs under {PATH / "runs"}')
    return runs[-1].parent


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', default=None)
    p.add_argument('--n-samples', type=int, default=50)
    p.add_argument('--length', type=int, default=None, help='trajectory length (default: model window)')
    p.add_argument('--sub', type=int, default=8, help='sparse spatial subsampling factor')
    p.add_argument('--sigma-obs', type=float, default=0.1)
    p.add_argument('--n-anneal', type=int, default=400)
    p.add_argument('--n-ode', type=int, default=6)
    p.add_argument('--n-esmda', type=int, default=1,
                   help='sub-assimilations per anneal level (ESMDA); 1 = single EnKF analysis')
    p.add_argument('--inflation', type=float, default=1.02, help='multiplicative covariance inflation')
    p.add_argument('--prior-scale', type=float, default=1.0)
    p.add_argument('--safe', action='store_true',
                   help='use Safe-EnKF-DAPS: pure EnKF (no r_t rescale) + trust-region increment')
    p.add_argument('--c', type=float, default=0.25,
                   help='Safe variant trust-region radius: RMS increment <= c*sigma_eff(t)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out-path', default='results_deneme')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    run = find_run(args.run)
    config = load_config(run)
    coarsen = config.get('coarsen', 1)
    H, W = HEIGHT // coarsen, WIDTH // coarsen
    L = args.length or config['window']

    # Latent vs pixel: sampling / EnKF analysis run in the diffusion (latent)
    # space of shape `latent_shape`; `decode` maps a latent to standardized pixels
    # (identity for pixel runs). Observations live in pixel space, so `obs_fn`
    # decodes before selecting the observed pixels (the EnKF is derivative-free).
    score = load_score(run / 'state.pth').to(args.device).eval()
    decode, C, Hm, Wm = load_decoder(config, args.device)
    latent_shape = (L, C, Hm, Wm)
    sde = VPSDE(score, shape=latent_shape).to(args.device)
    print(f'run {run.name} (coarsen={coarsen}, grid={H}x{W}, L={L}, '
          f'state=({C},{Hm},{Wm}))', flush=True)

    # ground-truth test trajectory + sparse spatial observation (pixel space)
    # testfile = PATH / 'data/test.h5' # For validation
    testfile = PATH / 'data/valid.h5' # For testing

    M = 5
    N = 15 # Trajectory Index
    with h5py.File(testfile, 'r') as f:
        x_star = torch.from_numpy(f['x'][N, M*L : (M + 1)*L]).to(args.device).float()   # (L,1,H,W) standardised
    print(f"Trajectory Index:N={N}, Time Slice Index M={M}")
    sub = args.sub
    A = lambda x: x[..., ::sub, ::sub]
    y = torch.normal(A(x_star), args.sigma_obs)                            # (L,1,H/sub,W/sub)
    y_flat = y.reshape(-1)
    R_diag = torch.full_like(y_flat, args.sigma_obs ** 2)

    def obs_fn(z_flat):                              # (Ne, prod(latent_shape)) -> (Ne, dy)
        z = z_flat.reshape(z_flat.shape[0], *latent_shape)
        return A(decode(z)).reshape(z_flat.shape[0], -1)

    if args.safe:
        print(f'sampler: Safe-EnKF-DAPS (pure EnKF, trust region c={args.c})', flush=True)
        z = sample_safe_daps_enkf(
            sde, obs_fn, y_flat, R_diag, shape=latent_shape, n_samples=args.n_samples,
            device=args.device, n_anneal=args.n_anneal, n_ode=args.n_ode,
            n_esmda=args.n_esmda, c=args.c,
        )
    else:
        print('sampler: EnKF-DAPS (Tweedie r_t prior rescale)', flush=True)
        z = sample_daps_enkf(
            sde, obs_fn, y_flat, R_diag, shape=latent_shape, n_samples=args.n_samples,
            device=args.device, n_anneal=args.n_anneal, n_ode=args.n_ode, n_esmda=args.n_esmda,
            inflation=args.inflation, prior_scale=args.prior_scale,
        )
    with torch.no_grad():
        x = decode(z)                                                     # (n, L, 1, H, W) std pixel

    rmse = (x - x_star).square().mean().sqrt().item()
    misfit = (A(x) - y).square().mean().sqrt().item()
    print(f'posterior: rmse(vs truth)={rmse:.4f}  obs-misfit={misfit:.4f} (sigma_obs={args.sigma_obs})')

    results = PATH / args.out_path
    results.mkdir(parents=True, exist_ok=True)
    tag = 'daps_enkf_safe' if args.safe else 'daps_enkf'
    xb = destandardize(x.cpu().numpy())
    tb = destandardize(x_star.cpu().numpy())
    np.savez(results / f'{tag}.npz', posterior=xb, truth=tb, sub=sub)
    gt = x_star.detach().cpu()
    vmin = float(gt.mean() - torch.quantile((gt - gt.mean()).abs(), 0.99))
    vmax = float(gt.mean() + torch.quantile((gt - gt.mean()).abs(), 0.99))
    save_gif(x[0, :, 0].cpu(), results / f'{tag}_sample.gif', vmin=vmin, vmax=vmax)
    save_gif(x_star[:, 0].cpu(), results / 'daps_truth.gif', vmin=vmin, vmax=vmax)
    print(f'saved -> {results/f"{tag}.npz"}, {tag}_sample.gif', flush=True)


if __name__ == '__main__':
    main()
