#!/usr/bin/env python
r"""EKS-DAPS family: DAPS with an interacting-Langevin inner step, for Rayleigh-Bénard.

Port of ``sample_eks_daps`` and its four inner steps from
``lorenz63-eks-score-daps.ipynb`` (sections 3.1-3.3) to the SDA continuous
VP-SDE, sharing the anneal schedule and PF-ODE denoiser of ``daps.py`` /
``daps_enkf.py``.

Every DAPS level (1) denoises ``x_t`` to a forecast ensemble ``x0_hat`` with the
PF-ODE, (2) corrects it towards the data, then (3) re-noises to the next level.
EnKF-DAPS implements (2) as a Kalman analysis, which is a contraction: nothing
in the level restores the spread it removes, and over hundreds of levels the
ensemble collapses. Here (2) is instead ``n_eks`` steps of the **ensemble
Kalman sampler** (Garbuno-Inigo, Hoffmann, Li & Stuart 2020),

    du_i = -C grad Phi(u_i) dt + sqrt(2 C) dW_i,

whose stationary law *is* the level target

    pi_t(u)  ∝  p(u | x_t) N(y; A(decode(u)), R),

with the derivative-free substitution ``C grad G^T ~ D(U)`` (no autograd through
the decoder or the observation operator) and the generalised square root
``C^{1/2} = U'/sqrt(J)``, so every quantity is a ``(J, J)`` Gram matrix or a
``(J, n)`` anomaly and nothing of size ``n x n`` or ``n x dy`` is ever formed.

The four methods differ ONLY in the conditional prior ``p(u | x_t)``:

    EKS-DAPS            N(x0_hat_i, sigma_eff(t)^2 I)  (DAPS eq. 5), running C(U);
                        the prior is applied implicitly by a J x J Woodbury solve.
    Ens-EKS-DAPS        N(xbar_0, gamma C_0) with C_0 = C(X0_hat) FROZEN for the
                        level. u never leaves the span of the anomalies, so the
                        prior drift is exactly (u - xbar_0)/gamma and no inverse
                        of the rank-(J-1) covariance is needed. gamma = 1 makes the
                        inner target the EnKF analysis Gaussian; larger widens it.
    Score-EKS-DAPS      grad log p(x0) + (xbar_t - x0)/sigma_t^2  (DAPS eq. 6): the
                        diffusion model's own score, re-queried at every inner step.
    ScoreLin-EKS-DAPS   the same, with the score replaced by a linear (secant) fit
                        through the J pairs (x0_hat_i, s_i): one network call per
                        level instead of one per inner step.

The inner step size is adaptive (Kovachki & Stuart), ``dt = min(dt_max, h0 /
sum of drift rates)``; ``dt_max`` is load-bearing (at large dt the Euler-Maruyama
step itself inflates the equilibrium variance). A relative trust region
``||dX||_RMS <= c * sigma_eff(t)`` on the level-total increment is shared with
Safe-EnKF-DAPS. The defaults (n_eks=16, h0=0.5, dt_max=0.02, gamma=16, c=0.25)
are the ones selected on Lorenz-63 and have not been re-tuned for RBC.

    python eks_daps.py --method EKS-DAPS --n-samples 80 --length 25
    python eks_daps.py --method Ens-EKS-DAPS --gamma 4
    python eks_daps.py --method ScoreLin-EKS-DAPS --n-eks 32 --c 0     # no trust region
    python eks_daps.py --selfcheck                                     # linear-Gaussian check
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import argparse
import math

import h5py
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm

from sda.score import VPSDE
from sda.utils import load_config

from utils import *
from vae import load_decoder

from daps import VPGrid, _build_anneal, _pf_ode_x0, _sigma_to_idx


N_EKS, H0_EKS, DT_MAX = 16, 0.5, 0.02
GAMMA_ENS = 16.0

PRIOR_OF = {
    'EKS-DAPS': 'eq5',
    'Ens-EKS-DAPS': 'ens',
    'Score-EKS-DAPS': 'eq6',
    'ScoreLin-EKS-DAPS': 'eq6lin',
}


# -------------------------------------------------------------- inner steps
# Each takes the flat ensemble u (J, n) and returns (u_new, dt used). obs_fn maps
# (J, n) -> (J, dy) predicted observations; Rinv is (dy,) = 1 / R_diag.
def _eks_inner_step(u, mu_pr, tau_pr, obs_fn, y_flat, Rinv, h0, dt_max, gen):
    r"""DAPS eq. 5 prior ``N(x0_hat_i, r_t^2 I)``, running mobility ``C(U)``.

    Garbuno-Inigo et al. (2020) eq. (4.1): explicit in the likelihood, implicit
    in the Gaussian prior (Woodbury on the J x J Gram matrix). ``tau_pr = 1/r_t^2``.
    """
    J = u.shape[0]
    Xp = u - u.mean(0, keepdim=True)                       # (J, n) anomalies
    Y = obs_fn(u)
    Yp = Y - Y.mean(0, keepdim=True)
    E = (Y - y_flat.view(1, -1)) * Rinv.view(1, -1)        # R^-1 (G(u) - y)
    Dmat = (E @ Yp.t()) / J                                # (J, J)
    Gm = Xp @ Xp.t()                                       # (J, J)
    rate = float(Dmat.norm()) + float(Gm.norm()) / J * tau_pr
    if not math.isfinite(rate) or rate <= 0.0:
        return u, 0.0                                      # ensemble has degenerated
    dt = min(h0 / rate, dt_max)
    us = u - dt * (Dmat @ Xp)                              # explicit likelihood
    lam = dt * tau_pr                                      # implicit prior, by Woodbury
    if lam > 1e-10:
        rhs = us + (lam / J) * ((mu_pr @ Xp.t()) @ Xp)
        S = Gm.double() + (J / lam) * torch.eye(J, device=u.device, dtype=torch.float64)
        us = rhs - (rhs @ Xp.t()) @ torch.linalg.solve(S, Xp.double()).to(Xp.dtype)
    xi = torch.randn(J, J, device=u.device, generator=gen)  # C^{1/2} = Xp^T / sqrt(J)
    return us + math.sqrt(2.0 * dt / J) * (xi @ Xp), dt


def _ens_inner_step(u, xbar0, Xp0, Yp0, obs_fn, y_flat, Rinv, gamma, h0, dt_max, gen):
    r"""Ensemble Gaussian prior ``N(xbar_0, gamma C_0)``, mobility frozen at ``C_0``.

    u stays in the affine span of Xp0, so ``C_0 (gamma C_0)^-1 (u - xbar0)`` is
    exactly ``(u - xbar0)/gamma``; the implicit prior solve is a division, and a
    constant mobility needs no finite-J correction term.
    """
    J = u.shape[0]
    Y = obs_fn(u)
    E = (Y - y_flat.view(1, -1)) * Rinv.view(1, -1)
    Dmat = (E @ Yp0.t()) / J                               # frozen cross-covariance
    rate = float(Dmat.norm()) + 1.0 / gamma
    if not math.isfinite(rate) or rate <= 0.0:
        return u, 0.0
    dt = min(h0 / rate, dt_max)
    xi = torch.randn(J, J, device=u.device, generator=gen)
    rhs = (u - dt * (Dmat @ Xp0) + (dt / gamma) * xbar0
           + math.sqrt(2.0 * dt / J) * (xi @ Xp0))
    return rhs / (1.0 + dt / gamma), dt                    # implicit prior


def _score_inner_step(u, s_pr, tau_trans, xbar_t, obs_fn, y_flat, Rinv, h0, dt_max, gen):
    r"""DAPS eq. 6 prior: model score ``s_pr`` evaluated at u plus the analytic
    Gaussian transition ``tau_trans (xbar_t - u)``. Fully explicit."""
    J = u.shape[0]
    Xp = u - u.mean(0, keepdim=True)
    Y = obs_fn(u)
    Yp = Y - Y.mean(0, keepdim=True)
    E = (Y - y_flat.view(1, -1)) * Rinv.view(1, -1)
    Dmat = (E @ Yp.t()) / J
    Gm = Xp @ Xp.t()
    Sp = s_pr - s_pr.mean(0, keepdim=True)
    rate = (float(Dmat.norm()) + float((Sp @ Xp.t()).norm()) / J
            + float(Gm.norm()) / J * tau_trans)
    if not math.isfinite(rate) or rate <= 0.0:
        return u, 0.0
    dt = min(h0 / rate, dt_max)
    drift = (((s_pr + tau_trans * (xbar_t - u)) @ Xp.t()) @ Xp) / J - Dmat @ Xp
    xi = torch.randn(J, J, device=u.device, generator=gen)
    return u + dt * drift + math.sqrt(2.0 * dt / J) * (xi @ Xp), dt


def _score_linear_step(u, xbar0, Xp0, Yp0, Sbar, Sp0, tau_trans, xbar_t,
                       obs_fn, y_flat, Rinv, h0, dt_max, gen):
    r"""DAPS eq. 6 prior with the score replaced by a linear fit to the ensemble.

    ``C(U) grad log p(u) ~ C sbar + (1/J) Xp0^T Sp0 (u - xbar0)``: the secant slope
    of the score across the cloud, exact for Gaussian p, never built as n x n.
    """
    J = u.shape[0]
    Y = obs_fn(u)
    E = (Y - y_flat.view(1, -1)) * Rinv.view(1, -1)
    Dmat = (E @ Yp0.t()) / J
    Gm = Xp0 @ Xp0.t()
    M2 = Sp0 @ Xp0.t()
    rate = (float(Dmat.norm()) + float(M2.norm()) / J
            + float(Gm.norm()) / J * tau_trans)
    if not math.isfinite(rate) or rate <= 0.0:
        return u, 0.0
    dt = min(h0 / rate, dt_max)
    prior = (((Sbar @ Xp0.t()) @ Xp0) / J                          # C sbar
             + (((u - xbar0) @ Sp0.t()) @ Xp0) / J                 # C Lambda (u - xbar0)
             + tau_trans * (((xbar_t - u) @ Xp0.t()) @ Xp0) / J)   # transition
    xi = torch.randn(J, J, device=u.device, generator=gen)
    return u + dt * (prior - Dmat @ Xp0) + math.sqrt(2.0 * dt / J) * (xi @ Xp0), dt


# ---------------------------------------------------------------- the sampler
@torch.no_grad()
def sample_eks_daps(
    sde: VPSDE,
    obs_fn,                          # callable (Ne, prod(shape)) flat state -> (Ne, dy) predicted obs
    y_flat: torch.Tensor,            # (dy,) observations (standardised pixel space)
    R_diag: torch.Tensor,            # (dy,) observation-noise variances
    shape,                           # diffusion event shape, e.g. (L, C, H, W) (latent or pixel)
    n_samples: int,
    device,
    prior: str = 'eq5',
    n_anneal: int = 400,
    n_ode: int = 6,
    n_eks: int = N_EKS,
    gamma: float = GAMMA_ENS,
    h0: float = H0_EKS,
    dt_max: float = DT_MAX,
    c: float = 0.25,
    prior_sigma_min: float = 0.01,
    sigma_max: float = 100.0,
    sigma_min: float = 0.01,
    rho: float = 7.0,
    T: int = 1000,
    seed: int = 0,
    progress: bool = True,
    trace: bool = False,
):
    r"""DAPS whose inner correction is ``n_eks`` ensemble-Kalman-sampler steps.

    ``prior`` selects the conditional prior (see module docstring):
    ``'eq5'`` EKS-DAPS, ``'ens'`` Ens-EKS-DAPS, ``'eq6'`` Score-EKS-DAPS,
    ``'eq6lin'`` ScoreLin-EKS-DAPS. ``c=None`` disables the trust region.
    Returns ``(n_samples, *shape)`` on ``device`` (standardised space), plus a
    per-level trace dict when ``trace=True``.
    """
    if prior not in PRIOR_OF.values():
        raise ValueError(f'unknown prior {prior!r}')
    sde.eval()
    grid = VPGrid(sde, device, T)
    abar, sigma_eff = grid.abar, grid.sigma_eff
    anneal_idx = _build_anneal(sigma_eff, n_anneal, rho, sigma_max, sigma_min)
    t_min = _sigma_to_idx(sigma_eff, prior_sigma_min)

    N = n_samples
    y_flat = y_flat.to(device).float()
    Rinv = 1.0 / R_diag.to(device).float()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    def score(u_flat, t_p):
        # grad log p_{sigma*}(u) in x0 coordinates: x_t / sqrt(abar) = x0 + sigma_eff eps.
        x = torch.sqrt(abar[t_p]) * u_flat.reshape(N, *shape)
        return -grid.eps(x, t_p).flatten(1) / float(sigma_eff[t_p])

    rec = {k: [] for k in ('sigma', 'q_raw', 'alpha', 'dt', 'spread_in', 'spread_out')}
    x_t = torch.randn(N, *shape, device=device, generator=gen)
    steps = range(len(anneal_idx) - 1)
    if progress:
        steps = tqdm(steps, desc={v: k for k, v in PRIOR_OF.items()}[prior], ncols=88)

    for step in steps:
        t_idx, t_next = anneal_idx[step], anneal_idx[step + 1]
        xf = _pf_ode_x0(grid, x_t, t_idx, n_ode, rho).flatten(1)   # (N, n) forecast ens.

        sig = float(sigma_eff[t_idx])
        tau = 1.0 / (sig * sig)        # eq. 5 prior precision / eq. 6 transition rate
        xbar0 = xf.mean(0, keepdim=True)
        Xp0 = xf - xbar0
        u, dt = xf, 0.0

        if prior == 'eq5':
            for _ in range(n_eks):
                u, dt = _eks_inner_step(u, xf, tau, obs_fn, y_flat, Rinv, h0, dt_max, gen)
        elif prior == 'ens':
            Yp0 = obs_fn(xf)
            Yp0 = Yp0 - Yp0.mean(0, keepdim=True)
            for _ in range(n_eks):
                u, dt = _ens_inner_step(u, xbar0, Xp0, Yp0, obs_fn, y_flat, Rinv,
                                        gamma, h0, dt_max, gen)
        else:
            # sigma* is floored at prior_sigma_min so the network is never queried
            # below the noise level it was trained to resolve.
            xbar_t = x_t.flatten(1) / torch.sqrt(abar[t_idx]).clamp(min=1e-6)
            t_p = max(t_idx, t_min)
            if prior == 'eq6':
                for _ in range(n_eks):
                    u, dt = _score_inner_step(u, score(u, t_p), tau, xbar_t, obs_fn,
                                              y_flat, Rinv, h0, dt_max, gen)
            else:
                Yp0 = obs_fn(xf)
                Yp0 = Yp0 - Yp0.mean(0, keepdim=True)
                S0 = score(xf, t_p)                                # one call per level
                Sbar = S0.mean(0, keepdim=True)
                Sp0 = S0 - Sbar
                for _ in range(n_eks):
                    u, dt = _score_linear_step(u, xbar0, Xp0, Yp0, Sbar, Sp0, tau,
                                               xbar_t, obs_fn, y_flat, Rinv,
                                               h0, dt_max, gen)

        # trust region on the level-total increment: ||dX||_RMS <= c * sigma_eff(t)
        delta = u - xf
        d_rms = float(delta.pow(2).mean().sqrt())
        a = 1.0 if c is None else min(1.0, c * sig / (d_rms + 1e-12))
        xa = u if a == 1.0 else xf + a * delta
        if not torch.isfinite(xa).all():
            xa = torch.where(torch.isfinite(xa), xa, xf)

        if trace:
            for k, v in (('sigma', sig), ('q_raw', d_rms / sig), ('alpha', a), ('dt', dt),
                         ('spread_in', float(xf.std(0).mean())),
                         ('spread_out', float(xa.std(0).mean()))):
                rec[k].append(v)

        x0_a = xa.reshape(N, *shape)
        if t_next == 0:
            x_t = x0_a
        else:
            x_t = torch.sqrt(abar[t_next]) * x0_a + torch.sqrt(1.0 - abar[t_next]) * torch.randn(
                x0_a.shape, device=device, generator=gen)

    out = x_t.detach()
    return (out, {k: np.array(v) for k, v in rec.items()}) if trace else out


# ---------------------------------------------------------------- self-check
def _inner_selfcheck(D=3, K=2, n_steps=20_000, burn=5_000, h0=H0_EKS, dt_max=DT_MAX):
    r"""Run the inner steps on a linear-Gaussian problem with a closed-form
    posterior. The eq.-5 step must find the mean and under-disperse at J = 8
    (Nüsken & Reich's (J-2)/J); the frozen-mobility steps must find both mean and
    variance at J = 8."""
    g = torch.Generator()
    g.manual_seed(1)
    A = torch.randn(K, D, generator=g)
    Rd = torch.rand(K, generator=g) + 0.4
    y = torch.randn(K, generator=g)
    obs = lambda z: z @ A.t()
    Ri = torch.diag(1.0 / Rd)

    def _chain(step, Pa, ma, u0, J):
        u, am, ac, n = u0.clone(), torch.zeros(D), torch.zeros(D, D), 0
        for s in range(n_steps):
            u, _ = step(u)
            if s >= burn:
                m = u.mean(0)
                am += m
                n += 1
                ac += (u - m).t() @ (u - m) / (J - 1)
        return (float((am / n - ma).norm() / Pa.diagonal().sum().sqrt()),
                float((ac / n).diagonal().sum() / Pa.diagonal().sum()))

    got = {}
    for J in (8, 100):
        u0 = torch.randn(J, D, generator=g) * 1.3 + torch.randn(1, D, generator=g)
        xbar0, Xp0 = u0.mean(0, keepdim=True), u0 - u0.mean(0, keepdim=True)
        C0 = Xp0.t() @ Xp0 / J
        Yp0 = obs(u0) - obs(u0).mean(0, keepdim=True)

        r = 1.3
        Pa = torch.linalg.inv(torch.eye(D) / r ** 2 + A.t() @ Ri @ A)
        ma = Pa @ (xbar0.view(-1) / r ** 2 + A.t() @ (y / Rd))
        got[('eq. 5 isotropic', J)] = _chain(
            lambda u: _eks_inner_step(u, xbar0, 1.0 / r ** 2, obs, y, 1.0 / Rd, h0, dt_max, g),
            Pa, ma, u0, J)

        for gm in (1.0, GAMMA_ENS):
            P0i = torch.linalg.inv(gm * C0)
            Pa = torch.linalg.inv(P0i + A.t() @ Ri @ A)
            ma = Pa @ (P0i @ xbar0.view(-1) + A.t() @ (y / Rd))
            got[(f'ensemble, gamma={gm:g}', J)] = _chain(
                lambda u: _ens_inner_step(u, xbar0, Xp0, Yp0, obs, y, 1.0 / Rd,
                                          gm, h0, dt_max, g), Pa, ma, u0, J)

        B = torch.randn(D, D, generator=g)            # analytic Gaussian standing in
        Pp = B @ B.t() / D + torch.eye(D)             # for the diffusion model's score
        mp = torch.randn(1, D, generator=g)
        Ppi = torch.linalg.inv(Pp)
        Pa = torch.linalg.inv(Ppi + A.t() @ Ri @ A)
        ma = Pa @ (Ppi @ mp.view(-1) + A.t() @ (y / Rd))
        S0 = -(u0 - mp) @ Ppi
        Sbar, Sp0 = S0.mean(0, keepdim=True), S0 - S0.mean(0, keepdim=True)
        got[('eq. 6 linearised', J)] = _chain(
            lambda u: _score_linear_step(u, xbar0, Xp0, Yp0, Sbar, Sp0, 0.0, u,
                                         obs, y, 1.0 / Rd, h0, dt_max, g), Pa, ma, u0, J)

    print(f"{'inner prior':<22}{'J':>5}{'mean err':>11}{'var ratio':>12}")
    for (nm, J), (em, vr) in got.items():
        print(f'{nm:<22}{J:>5}{em:>11.4f}{vr:>12.3f}')
    print('mean err in posterior sd; var ratio = trace(measured) / trace(exact).')
    assert got[('eq. 5 isotropic', 8)][1] < 0.80, 'eq. 5 step should under-disperse at J=8'
    assert 0.85 < got[('eq. 5 isotropic', 100)][1] < 1.15, 'the deficit must be O(1/J)'
    for key, (em, vr) in got.items():
        assert em < (0.25 if key[1] == 8 else 0.06), f'{key} misses the mean ({em:.3f} sd)'
        if not key[0].startswith('eq. 5'):
            assert 0.88 < vr < 1.12, f'{key} misses the variance ({vr:.3f})'
    print('inner self-check passed.')


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
    p.add_argument('--method', default='EKS-DAPS', choices=list(PRIOR_OF),
                   help='inner conditional prior (see above)')
    p.add_argument('--run', default=None)
    p.add_argument('--n-samples', type=int, default=50)
    p.add_argument('--length', type=int, default=None, help='trajectory length (default: model window)')
    p.add_argument('--sub', type=int, default=8, help='sparse spatial subsampling factor')
    p.add_argument('--sigma-obs', type=float, default=0.1)
    p.add_argument('--traj-index', type=int, default=15, help='trajectory index in the test file')
    p.add_argument('--time-slice', type=int, default=5, help='window index M: frames M*L .. (M+1)*L')
    p.add_argument('--n-anneal', type=int, default=400)
    p.add_argument('--n-ode', type=int, default=6)
    p.add_argument('--n-eks', type=int, default=N_EKS, help='inner EKS steps per anneal level')
    p.add_argument('--gamma', type=float, default=GAMMA_ENS,
                   help='Ens-EKS-DAPS prior inflation (1 = EnKF analysis Gaussian)')
    p.add_argument('--h0', type=float, default=H0_EKS, help='adaptive inner step: dt = h0 / rate')
    p.add_argument('--dt-max', type=float, default=DT_MAX, help='maximum inner step')
    p.add_argument('--c', type=float, default=0.25,
                   help='trust-region radius: RMS increment <= c*sigma_eff(t); <= 0 disables')
    p.add_argument('--prior-sigma-min', type=float, default=0.01,
                   help='Score/ScoreLin: floor on the noise level the score is queried at')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out-path', default='results_deneme')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--selfcheck', action='store_true',
                   help='run the linear-Gaussian check of the inner steps and exit')
    args = p.parse_args()

    if args.selfcheck:
        _inner_selfcheck()
        return

    torch.manual_seed(args.seed)
    run = find_run(args.run)
    config = load_config(run)
    coarsen = config.get('coarsen', 1)
    H, W = HEIGHT // coarsen, WIDTH // coarsen
    L = args.length or config['window']
    if L < config['window']:
        raise ValueError(f'--length {L} is shorter than the score window of run {run.name} '
                         f'({config["window"]}); the network needs L >= window')

    # Sampling runs in the diffusion (latent) space; observations live in pixel
    # space, so `obs_fn` decodes before selecting the observed pixels. EKS is
    # derivative-free, so no autograd through the decoder is needed.
    score = load_score(run / 'state.pth').to(args.device).eval()
    decode, C, Hm, Wm = load_decoder(config, args.device)
    latent_shape = (L, C, Hm, Wm)
    sde = VPSDE(score, shape=latent_shape).to(args.device)
    print(f'run {run.name} (coarsen={coarsen}, grid={H}x{W}, L={L}, '
          f'state=({C},{Hm},{Wm}))', flush=True)

    # ground-truth test trajectory + sparse spatial observation (pixel space)
    testfile = PATH / 'data/valid.h5'
    M, N = args.time_slice, args.traj_index
    with h5py.File(testfile, 'r') as f:
        n_frames = f['x'].shape[1]
        if (M + 1) * L > n_frames:
            raise ValueError(f'--time-slice {M} with L={L} needs frames up to {(M + 1) * L}, '
                             f'but {testfile.name} has {n_frames}; lower --time-slice or --length')
        x_star = torch.from_numpy(f['x'][N, M*L : (M + 1)*L]).to(args.device).float()   # (L,1,H,W)
    print(f'Trajectory Index:N={N}, Time Slice Index M={M}')
    sub = args.sub
    A = lambda x: x[..., ::sub, ::sub]
    y = torch.normal(A(x_star), args.sigma_obs)
    y_flat = y.reshape(-1)
    R_diag = torch.full_like(y_flat, args.sigma_obs ** 2)

    def obs_fn(z_flat):                              # (Ne, prod(latent_shape)) -> (Ne, dy)
        z = z_flat.reshape(z_flat.shape[0], *latent_shape)
        return A(decode(z)).reshape(z_flat.shape[0], -1)

    prior = PRIOR_OF[args.method]
    c = args.c if args.c > 0 else None
    print(f'sampler: {args.method} (n_eks={args.n_eks}, h0={args.h0}, dt_max={args.dt_max}, '
          f'c={c}' + (f', gamma={args.gamma:g}' if prior == 'ens' else '') + ')', flush=True)
    z, rec = sample_eks_daps(
        sde, obs_fn, y_flat, R_diag, shape=latent_shape, n_samples=args.n_samples,
        device=args.device, prior=prior, n_anneal=args.n_anneal, n_ode=args.n_ode,
        n_eks=args.n_eks, gamma=args.gamma, h0=args.h0, dt_max=args.dt_max, c=c,
        prior_sigma_min=args.prior_sigma_min, seed=args.seed, trace=True,
    )
    with torch.no_grad():
        x = decode(z)                                                     # (n, L, 1, H, W) std pixel

    rmse = (x - x_star).square().mean().sqrt().item()
    misfit = (A(x) - y).square().mean().sqrt().item()
    spread = x.std(0).mean().item()
    n_clip = int((rec['alpha'] < 1.0).sum())
    print(f'posterior: rmse(vs truth)={rmse:.4f}  obs-misfit={misfit:.4f} '
          f'(sigma_obs={args.sigma_obs})  spread={spread:.4f}')
    print(f'trust region clipped {n_clip}/{len(rec["alpha"])} levels '
          f'(max q_raw {rec["q_raw"].max():.3f});  inner dt range '
          f'[{rec["dt"].min():.2e}, {rec["dt"].max():.2e}]')

    results = PATH / args.out_path
    results.mkdir(parents=True, exist_ok=True)
    tag = args.method.lower().replace('-', '_')
    xb = destandardize(x.cpu().numpy(), testfile.parent)
    tb = destandardize(x_star.cpu().numpy(), testfile.parent)
    np.savez(results / f'{tag}.npz', posterior=xb, truth=tb, sub=sub,
             **{f'trace_{k}': v for k, v in rec.items()})
    gt = x_star.detach().cpu()
    vmin = float(gt.mean() - torch.quantile((gt - gt.mean()).abs(), 0.99))
    vmax = float(gt.mean() + torch.quantile((gt - gt.mean()).abs(), 0.99))
    save_gif(x[0, :, 0].cpu(), results / f'{tag}_sample.gif', vmin=vmin, vmax=vmax)
    save_gif(x_star[:, 0].cpu(), results / 'daps_truth.gif', vmin=vmin, vmax=vmax)
    print(f'saved -> {results/f"{tag}.npz"}, {tag}_sample.gif', flush=True)


if __name__ == '__main__':
    main()
