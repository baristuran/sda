#!/usr/bin/env python
r"""Standard ensemble-forecast verification metrics for Rayleigh-Bénard
posterior samples.

Reads one of the ``*.npz`` files written by ``daps.py`` / ``daps_enkf.py`` /
``eval.py``'s DPS-ensemble scenario (keys ``posterior`` (N, L, 1, H, W),
``truth`` (L, 1, H, W), ``sub``), or ``eval.py``'s unconditional prior
(``prior``/``truth``, no ``sub``), and reports, per lead time and averaged
over the trajectory:

1. **Observation misfit** -- RMSE of the ensemble mean against the (noise-free)
   truth, restricted to the observed pixel subset ``x[..., ::sub, ::sub]``. The
   noisy observation draw ``y`` used at sampling time is not persisted in the
   npz, so this is the reconstruction error at the observed locations rather
   than the misfit to the noisy ``y`` itself (which ``daps.py`` already prints
   at sampling time); pass ``--sub`` to override/supply it (e.g. for
   ``prior_fields.npz``, which has no spatial subsampling of its own).

2. **CRPS** (continuous ranked probability score) -- the proper score for a
   full ensemble forecast against a scalar truth, evaluated pointwise over the
   whole field and averaged. Uses the O(N log N) order-statistics estimator
   (Hersbach, 2000):

       CRPS = E|X - x_true| - 1/2 E|X - X'|
            = (1/N) sum_i |x_i - x_true| - (1/N^2) sum_k (2k - N - 1) x_(k)

   with x_(1) <= ... <= x_(N) the sorted ensemble (X, X' iid draws from it).

3. **Spread/skill ratio** -- sqrt(mean ensemble variance) / RMSE(ensemble mean,
   truth), evaluated over the whole field. A well-calibrated ensemble has
   ratio ~= 1: spread >> skill means the ensemble is overdispersive (posterior
   too uncertain); spread << skill means it is underdispersive (overconfident).

Usage:
    python metrics.py results/daps_envar.npz
    python metrics.py results/daps_envar.npz results/daps_enkf.npz results/dps_sub8.npz
    python metrics.py results/daps_envar.npz results/daps_enkf.npz --csv-dir results/metrics
    python metrics.py results/prior_fields.npz --key prior --sub 8

Passing more than one file also prints a side-by-side comparison table.
"""

import argparse

import numpy as np

from pathlib import Path
from typing import List
from utils import *


def load_ensemble(npz_path: Path, key: str, sub: int = None):
    r"""Load ``(posterior, truth, sub)`` as physical-units ``(N, L, H, W)`` /
    ``(L, H, W)`` / int from one of the results npz files."""
    data = np.load(npz_path)
    if key not in data:
        # raise KeyError(f"'{key}' not in {npz_path} (available: {list(data.keys())})")
        key = "prior"
    if 'truth' not in data:
        raise KeyError(f"'truth' not in {npz_path} (available: {list(data.keys())})")
    
    posterior = np.asarray(data[key])[:, :, 0]     # (N, L, 1, H, W) -> (N, L, H, W)
    truth = np.asarray(data['truth'])[:, 0]        # (L, 1, H, W) -> (L, H, W)
    # posterior = (posterior - BUOYANCY_MEAN) /BUOYANCY_STD
    # truth = (truth - BUOYANCY_MEAN) /BUOYANCY_STD

    if sub is None:
        sub = int(data['sub']) if 'sub' in data else 1

    return posterior, truth, sub

def trajectory_misfit_rmse(posterior: np.ndarray, truth: np.ndarray) -> np.ndarray:
    r"""RMSE of the ensemble mean vs. truth,
    per lead time. Returns an ``(L,)`` array."""
    mean = posterior.mean(axis=0)
    a, b = mean, truth
    return np.sqrt(((a - b) ** 2).mean(axis=(-1, -2)))


def obs_misfit_rmse(posterior: np.ndarray, truth: np.ndarray, sub: int) -> np.ndarray:
    r"""RMSE of the ensemble mean vs. truth at the observed (``::sub``) pixels,
    per lead time. Returns an ``(L,)`` array."""
    mean = posterior.mean(axis=0)
    a, b = mean[..., ::sub, ::sub], truth[..., ::sub, ::sub]
    return np.sqrt(((a - b) ** 2).mean(axis=(-1, -2)))



def crps(posterior: np.ndarray, truth: np.ndarray) -> np.ndarray:
    r"""CRPS of the full-field ensemble forecast vs. truth, per lead time (the
    ``O(N log N)`` order-statistics estimator, averaged pointwise over the
    field). Returns an ``(L,)`` array."""
    N = posterior.shape[0]
    term1 = np.abs(posterior - truth[None]).mean(axis=0)               # (L, H, W)

    sorted_ens = np.sort(posterior, axis=0)                            # (L, H, W) per rank
    k = np.arange(1, N + 1).reshape(N, *([1] * (posterior.ndim - 1)))
    term2 = ((2 * k - N - 1) * sorted_ens).sum(axis=0) / (N ** 2)      # (L, H, W)

    return (term1 - term2).mean(axis=(-1, -2))


def spread_skill(posterior: np.ndarray, truth: np.ndarray):
    r"""Spread, skill (RMSE) and their ratio over the whole field, per lead
    time. Returns three ``(L,)`` arrays ``(spread, skill, ratio)``.

    Matches ``_spread_skill`` in lorenz63-diffusion-daps-envar.ipynb (Part
    4e.3): population variance (``ddof=0``, numpy's default -- *not* the
    unbiased ``ddof=1`` estimator) and an additive epsilon on the denominator.
    """
    mean = posterior.mean(axis=0)
    var = posterior.var(axis=0)                                        # ddof=0, as in the notebook

    spread = np.sqrt(var.mean(axis=(-1, -2)))
    skill = np.sqrt(((mean - truth) ** 2).mean(axis=(-1, -2)))
    ratio = spread / (skill + 1e-8)

    return spread, skill, ratio


def evaluate(npz_path: Path, key: str, sub_override: int = None) -> dict:
    posterior, truth, sub = load_ensemble(npz_path, key, sub_override)
    L = truth.shape[0]

    traj_rmse = trajectory_misfit_rmse(posterior, truth)
    misfit = obs_misfit_rmse(posterior, truth, sub)
    c = crps(posterior, truth)
    spread, skill, ratio = spread_skill(posterior, truth)

    return dict(
        L=L, sub=sub, n_samples=posterior.shape[0],
        traj_rmse=traj_rmse, obs_misfit=misfit, crps=c, spread=spread, skill=skill, ratio=ratio,
    )


def print_summary(name: str, m: dict) -> None:
    print(f'{name}  (N={m["n_samples"]}, L={m["L"]}, sub={m["sub"]})')
    print(f'  trajectory RMSE   : {m["traj_rmse"].mean():.8f}  (final lead: {m["traj_rmse"][-1]:.4f})')

    print(f'  obs misfit (RMSE)   : {m["obs_misfit"].mean():.8f}  (final lead: {m["obs_misfit"][-1]:.4f})')
    print(f'  CRPS                : {m["crps"].mean():.4f}  (final lead: {m["crps"][-1]:.4f})')
    print(f'  spread/skill ratio  : {m["ratio"].mean():.4f}  '
          f'(spread={m["spread"].mean():.4f}, skill={m["skill"].mean():.4f})')


def print_comparison(names: List[str], metrics: List[dict]) -> None:
    r"""Side-by-side comparison table (trajectory-averaged) across methods."""
    header = f'{"method":<24}{"N":>5}{"obs misfit":>12}{"CRPS":>10}{"spread/skill":>14}'
    print()
    print(header)
    print('-' * len(header))
    for name, m in zip(names, metrics):
        print(f'{name:<24}{m["n_samples"]:>5}{m["obs_misfit"].mean():>12.4f}'
              f'{m["crps"].mean():>10.4f}{m["ratio"].mean():>14.4f}')


def save_csv(path: Path, m: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write('lead,obs_misfit,crps,spread,skill,ratio\n')
        for i in range(m['L']):
            f.write(f'{i},{m["obs_misfit"][i]:.6f},{m["crps"][i]:.6f},'
                     f'{m["spread"][i]:.6f},{m["skill"][i]:.6f},{m["ratio"][i]:.6f}\n')
    print(f'  saved per-lead-time table -> {path}')


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('npz', type=Path, nargs='+', help='one or more results *.npz files')
    p.add_argument('--key', default='posterior',
                   help="ensemble array to evaluate: 'posterior' (daps.py/daps_enkf.py) "
                        "or 'prior' (eval.py's prior_fields.npz)")
    p.add_argument('--sub', type=int, default=None,
                   help="observed-pixel stride (default: read from the npz's 'sub' field, "
                        "or 1 -- fully observed -- if absent)")
    p.add_argument('--csv-dir', type=Path, default=None,
                   help='if given, also write a per-lead-time CSV per input file here')
    args = p.parse_args()

    results = []
    for path in args.npz:
        m = evaluate(path, args.key, args.sub)
        print_summary(path.name, m)
        if args.csv_dir is not None:
            save_csv(args.csv_dir / f'{path.stem}_metrics.csv', m)
        results.append(m)

    if len(args.npz) > 1:
        print_comparison([path.stem for path in args.npz], results)


if __name__ == '__main__':
    main()
