#!/usr/bin/env python
r"""EnVar-based Decoupled Annealed Posterior Sampling (DAPS) for Rayleigh-Bénard.

Port of ``sample_daps_envar`` from ``lorenz63-diffusion-daps-envar.ipynb`` to the
SDA continuous VP-SDE, following DAPS (Zhang et al., 2024, "Improving Diffusion
Inverse Problem Solving with Decoupled Noise Annealing", daps.pdf).

DAPS decouples the diffusion prior from the likelihood: at each annealing level
``sigma_t`` it (1) denoises ``x_t`` to an estimate ``x0_hat`` with the
probability-flow ODE, (2) runs Langevin dynamics in ``x0`` targeting
``p(x0 | x_t, y) ∝ N(x0; x0_hat, r_t^2 I) p(y | x0)``, then (3) re-noises to the
next (lower) level. The score is therefore only called for denoising, never
inside the (cheap) likelihood MCMC.

The **EnVar** extension replaces the isotropic prior variance ``r_t^2 I`` with a
hybrid ensemble prior ``C_t = r_t^2 [(1-beta) I + beta R_t]`` (``R_t`` = the
sample correlation of the current ensemble), applied *inverse-free* by running
the Langevin in whitened control variables ``(U, V)``:

    x0 = x0_hat + sqrt(1-beta) r_t U + sqrt(beta) (V @ Z),   Z^T Z = r_t^2 R_t

whose prior is a standard normal. ``beta=0`` recovers isotropic DAPS.

After annealing, a final *manifold projection* (re-noise to ``proj_sigma`` and
denoise with the PF-ODE, iterated) snaps the samples back onto the diffusion
prior, removing off-manifold kinks left by the likelihood updates.

Run as a script to assimilate a sparse observation of a test trajectory:

    python daps.py --n-samples 16 --sub 8 --beta 0.8
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm

from sda.score import VPSDE
from sda.utils import TrajectoryDataset, load_config

from utils import *


# ------------------------------------------------------------------ scheduler
class VPGrid:
    r"""Fine VP-SDE time grid exposing the DDPM-style ``abar`` and EDM ``sigma``,
    so the DAPS algorithm (written in the DDPM/EDM coordinate) ports directly.

    Index 0 is the data end (``abar~1``, ``sigma~0``); index T-1 is pure noise.
    """

    def __init__(self, sde: VPSDE, device, T: int = 1000):
        self.sde = sde
        t = torch.linspace(0.0, 1.0, T, device=device)     # t=0 data ... t=1 noise
        self.t = t
        mu = sde.mu(t)                                      # alpha(t)
        self.abar = (mu ** 2).clamp(min=1e-8)
        self.sigma_eff = torch.sqrt((1.0 - self.abar) / self.abar)

    @torch.no_grad()
    def eps(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return self.sde.eps(x, self.t[idx])                 # scalar t (all particles same level)


# ------------------------------------------------------------ anneal helpers
def _sigma_grid(sigma_max, sigma_min, n_levels, rho=7.0):
    if n_levels <= 1:
        return [float(sigma_max)]
    return [
        float((sigma_max ** (1 / rho)
               + (i / (n_levels - 1)) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho)
        for i in range(n_levels)
    ]


def _sigma_to_idx(sigma_eff, sigma):
    return int(torch.argmin((sigma_eff - float(sigma)).abs()).item())


def _dedupe(items):
    out = []
    for it in items:
        if not out or out[-1] != it:
            out.append(it)
    return out


def _build_anneal(sigma_eff, n_anneal, rho, sigma_max, sigma_min):
    hi = min(sigma_max, float(sigma_eff[-1]))
    lo = max(sigma_min, float(sigma_eff[0]))
    idx = _dedupe([_sigma_to_idx(sigma_eff, s) for s in _sigma_grid(hi, lo, n_anneal, rho)])
    if idx[-1] != 0:
        idx.append(0)
    return idx


@torch.no_grad()
def _pf_ode_x0(grid: VPGrid, x_t, t_idx, n_ode, rho=7.0):
    r"""Deterministic DDIM / probability-flow-ODE estimate of x0 from x_t."""
    abar, sigma_eff = grid.abar, grid.sigma_eff
    if t_idx <= 0 or n_ode <= 1:
        sub = [t_idx, 0]
    else:
        sub = _dedupe([_sigma_to_idx(sigma_eff, s)
                       for s in _sigma_grid(float(sigma_eff[t_idx]), float(sigma_eff[0]),
                                            n_ode + 1, rho)])
        sub[0], sub[-1] = t_idx, 0
        sub = _dedupe(sub)
    x, x0 = x_t, x_t
    for tc, tn in zip(sub[:-1], sub[1:]):
        eps = grid.eps(x, tc)
        x0 = (x - torch.sqrt(1.0 - abar[tc]) * eps) / torch.sqrt(abar[tc]).clamp(min=1e-6)
        if tn > 0:
            x = torch.sqrt(abar[tn]) * x0 + torch.sqrt(1.0 - abar[tn]) * eps
    return x0


# ---------------------------------------------------------------- the sampler
def sample_daps_envar(
    sde: VPSDE,
    obs_sq_fn,                       # x (N,*shape) -> (N, k) weighted squared errors
    shape,                           # event shape, e.g. (L, C, H, W)
    n_samples: int,
    device,
    n_anneal: int = 200,
    n_ode: int = 6,
    n_langevin: int = 200,
    n_outer: int = 1,
    prior_scale: float = 1.0,
    beta: float = 0.8,
    inflation: float = 1.02,
    eta_0: float = 2e-4,
    eta_min_ratio: float = 1e-3,
    max_grad_norm: float = 1e3,
    proj_sigma: float = 0.8,
    proj_n_ode: int = 20,
    proj_n_iter: int = 2,
    sigma_max: float = 100.0,
    sigma_min: float = 0.01,
    rho: float = 7.0,
    T: int = 1000,
    progress: bool = True,
) -> torch.Tensor:
    r"""Draw ``n_samples`` posterior trajectories with EnVar-DAPS. Returns a
    tensor of shape ``(n_samples, *shape)`` on ``device`` (standardised space)."""
    sde.eval()
    grid = VPGrid(sde, device, T)
    abar, sigma_eff = grid.abar, grid.sigma_eff
    anneal_idx = _build_anneal(sigma_eff, n_anneal, rho, sigma_max, sigma_min)

    N = n_samples
    use_ens = beta > 0.0
    sN = float(np.sqrt(max(N - 1, 1)))
    a_stat = float(np.sqrt(1.0 - beta)) if use_ens else 1.0
    a_ens = float(np.sqrt(beta))

    x_t = torch.randn(N, *shape, device=device)
    steps = range(len(anneal_idx) - 1)
    if progress:
        steps = tqdm(steps, desc='EnVar-DAPS', ncols=88)

    for step in steps:
        t_idx, t_next = anneal_idx[step], anneal_idx[step + 1]
        x0_hat = _pf_ode_x0(grid, x_t, t_idx, n_ode, rho)          # (N, *shape)

        r_t = (prior_scale * sigma_eff[t_idx]).clamp(min=float(sigma_eff[0]))
        r2 = float(r_t ** 2)
        eta_ratio = step / max(len(anneal_idx) - 2, 1)
        eta = eta_0 * (1.0 + eta_ratio * (eta_min_ratio - 1.0)) / r2   # control-space step
        sqrt_2eta = float(np.sqrt(2.0 * eta))

        # EnRML outer loop: relinearise the ensemble covariance each sweep.
        x0_a = x0_hat.detach().clone()
        for _outer in range(max(n_outer, 1) if use_ens else 1):
            if use_ens:
                with torch.no_grad():
                    A = (x0_a - x0_a.mean(0, keepdim=True)).flatten(1) * float(np.sqrt(inflation)) / sN
                    sd = torch.sqrt((A * A).sum(0).clamp(min=1e-12))       # per-coord ens std (n,)
                    Z = r_t * (A / sd)                                     # (N, n): Z^T Z = r_t^2 R_t

            U = torch.zeros(N, *shape, device=device)
            V = torch.zeros(N, N, device=device) if use_ens else None
            for _ in range(n_langevin):
                x0 = x0_hat + a_stat * r_t * U
                if use_ens:
                    x0 = x0 + a_ens * (V @ Z).reshape(N, *shape)
                x0 = x0.detach().requires_grad_(True)
                log_lik = -0.5 * obs_sq_fn(x0).sum(dim=-1)                 # (N,)
                gx = torch.autograd.grad(log_lik.sum(), x0)[0]            # (N, *shape)
                # Per-particle gradient-norm clip: the quadratic likelihood makes
                # ||gx|| grow with x0, which the ensemble control amplifies in high
                # dimension; clipping breaks that runaway (a standard Langevin guard).
                with torch.no_grad():
                    gnorm = gx.flatten(1).norm(dim=1).clamp(min=1e-12)
                    scale = (max_grad_norm / gnorm).clamp(max=1.0)
                    gx = gx * scale.view(-1, *([1] * (gx.ndim - 1)))
                with torch.no_grad():
                    U = U + eta * (a_stat * r_t * gx - U) + sqrt_2eta * torch.randn_like(U)
                    if use_ens:
                        V = V + eta * (a_ens * (gx.flatten(1) @ Z.t()) - V) \
                            + sqrt_2eta * torch.randn_like(V)
                    if not torch.isfinite(U).all() or (use_ens and not torch.isfinite(V).all()):
                        U = torch.zeros_like(U)
                        if use_ens:
                            V = torch.zeros_like(V)
                        break
            with torch.no_grad():
                x0_a = x0_hat + a_stat * r_t * U
                if use_ens:
                    x0_a = x0_a + a_ens * (V @ Z).reshape(N, *shape)
        x0 = x0_a.detach()

        # Re-noise to the next (lower) level.
        with torch.no_grad():
            if t_next == 0:
                x_t = x0
            else:
                x_t = torch.sqrt(abar[t_next]) * x0 + torch.sqrt(1.0 - abar[t_next]) * torch.randn_like(x0)

    # Final projection onto the diffusion prior manifold: re-noise the posterior
    # samples to `proj_sigma` and denoise them with the PF-ODE, iterated, to remove
    # the off-manifold kinks the Langevin/likelihood updates leave behind (renders
    # the trajectories physically consistent). proj_sigma <= 0 disables it.
    if proj_sigma and proj_sigma > 0.0 and proj_n_iter > 0:
        j = _sigma_to_idx(sigma_eff, proj_sigma)
        with torch.no_grad():
            for _ in range(proj_n_iter):
                x_t = torch.sqrt(abar[j]) * x_t + torch.sqrt(1.0 - abar[j]) * torch.randn_like(x_t)
                x_t = _pf_ode_x0(grid, x_t, j, proj_n_ode, rho)

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
    p.add_argument('--beta', type=float, default=0.8, help='ensemble prior weight (0 = isotropic DAPS)')
    p.add_argument('--n-anneal', type=int, default=400)
    p.add_argument('--n-ode', type=int, default=6)
    p.add_argument('--n-langevin', type=int, default=400)
    p.add_argument('--n-outer', type=int, default=1, help='EnRML covariance relinearisation sweeps')
    p.add_argument('--prior-scale', type=float, default=1.0)
    p.add_argument('--eta-0', type=float, default=2e-4, help='Langevin step size')
    p.add_argument('--max-grad-norm', type=float, default=1e3, help='per-particle gradient clip')
    p.add_argument('--proj-sigma', type=float, default=0.8,
                   help='final manifold-projection noise level (<=0 disables)')
    p.add_argument('--proj-n-ode', type=int, default=20, help='PF-ODE steps for the projection')
    p.add_argument('--proj-n-iter', type=int, default=2, help='projection iterations')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    run = find_run(args.run)
    config = load_config(run)
    coarsen = config.get('coarsen', 1)
    H, W = HEIGHT // coarsen, WIDTH // coarsen
    L = args.length or config['window']

    score = load_score(run / 'state.pth').to(args.device).eval()
    sde = VPSDE(score, shape=(L, 1, H, W)).to(args.device)
    print(f'run {run.name} (coarsen={coarsen}, grid={H}x{W}, L={L})', flush=True)

    # ground-truth test trajectory + sparse spatial observation
    testfile = PATH / 'data/test.h5'
    with h5py.File(testfile, 'r') as f:
        x_star = torch.from_numpy(f['x'][0, :L]).to(args.device)      # (L,1,H,W) standardised
    sub = args.sub
    A = lambda x: x[..., ::sub, ::sub]
    y = torch.normal(A(x_star), args.sigma_obs)
    obs_sq_fn = lambda x: ((A(x) - y) ** 2 / args.sigma_obs ** 2).flatten(1)

    x = sample_daps_envar(
        sde, obs_sq_fn, shape=(L, 1, H, W), n_samples=args.n_samples, device=args.device,
        n_anneal=args.n_anneal, n_ode=args.n_ode, n_langevin=args.n_langevin, beta=args.beta,
        n_outer=args.n_outer, prior_scale=args.prior_scale, eta_0=args.eta_0,
        max_grad_norm=args.max_grad_norm, proj_sigma=args.proj_sigma,
        proj_n_ode=args.proj_n_ode, proj_n_iter=args.proj_n_iter,
    )

    rmse = (x - x_star).square().mean().sqrt().item()
    misfit = (A(x) - y).square().mean().sqrt().item()
    print(f'posterior: rmse(vs truth)={rmse:.4f}  obs-misfit={misfit:.4f} (sigma_obs={args.sigma_obs})')

    results = PATH / 'results'
    results.mkdir(parents=True, exist_ok=True)
    xb = destandardize(x.cpu().numpy())
    tb = destandardize(x_star.cpu().numpy())
    np.savez(results / 'daps_envar.npz', posterior=xb, truth=tb, sub=sub)
    gt = x_star.detach().cpu()
    vmin = float(gt.mean() - torch.quantile((gt - gt.mean()).abs(), 0.99))
    vmax = float(gt.mean() + torch.quantile((gt - gt.mean()).abs(), 0.99))
    save_gif(x[0, :, 0].cpu(), results / 'daps_envar_sample.gif', vmin=vmin, vmax=vmax)
    save_gif(x_star[:, 0].cpu(), results / 'daps_truth.gif', vmin=vmin, vmax=vmax)
    print(f'saved -> {results/"daps_envar.npz"}, daps_envar_sample.gif', flush=True)


if __name__ == '__main__':
    main()
