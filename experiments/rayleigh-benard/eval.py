#!/usr/bin/env python
r"""Evaluation for the Rayleigh-Bénard case.

Mirrors the structure of ``lorenz/eval.py`` but, as the state is high-dimensional
(like Kolmogorov), the evaluation is qualitative: zero-shot data assimilation
from sparse/noisy observations of a ground-truth test trajectory, plus
unconditional prior samples. For each scenario we save the ground truth, the
observation and the SDA reconstruction (and a DPS reconstruction for comparison),
and we log reconstruction RMSE / observation misfit.

Usage (single machine):

    # evaluate the most recent run under ./runs
    python eval.py

    # or a specific run, with fewer diffusion steps
    RBC_RUN=eager-firefly-1_ab12cd34 EVAL_STEPS=128 python eval.py
"""

import os

import numpy as np
import torch

from dawgz import job, schedule
from typing import *

from sda.mcs import *
from sda.score import *
from sda.utils import *

from utils import *
from vae import load_decoder


# Run to evaluate (most recent under PATH/runs if unset) and eval budget.
RUN = os.environ.get('RBC_RUN', None)
LENGTH = int(os.environ.get('EVAL_LENGTH', 25))     # assimilation horizon (frames)
STEPS = int(os.environ.get('EVAL_STEPS', 256))      # reverse-diffusion steps
CORRECTIONS = int(os.environ.get('EVAL_CORR', 1))   # Langevin corrections / step
TAU = float(os.environ.get('EVAL_TAU', 0.5))
N_DPS = int(os.environ.get('EVAL_N_DPS', 8))       # DPS ensemble size (for CRPS / spread-skill)


def find_run() -> Path:
    if RUN is not None:
        return PATH / 'runs' / RUN
    runs = sorted((PATH / 'runs').glob('*/state.pth'), key=os.path.getmtime)
    if not runs:
        raise FileNotFoundError(f'no trained runs under {PATH / "runs"} (train first)')
    return runs[-1].parent


def rmse(x: Tensor, x_star: Tensor) -> float:
    r"""Reconstruction RMSE in units of the buoyancy std (data is standardised)."""
    return (x - x_star).square().mean().sqrt().item()


class IgnoreContext(nn.Module):
    r"""Adapts a guided score that takes ``(x, t)`` to the ``(x, t, c)`` interface
    expected by :meth:`VPSDE.sample` (e.g. ``DPSGaussianScore``)."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        return self.module(x, t)


@job(cpus=4, gpus=1, ram='32GB', time='02:00:00')
def evaluate(i: int = 0):
    runpath = find_run()
    config = load_config(runpath)
    coarsen = config.get('coarsen', 1)
    H, W = HEIGHT // coarsen, WIDTH // coarsen

    score = load_score(runpath / 'state.pth').cuda()
    # Latent vs pixel: `decode` maps the diffusion state to a standardized pixel
    # trajectory (identity for pixel runs); `(C, Hm, Wm)` is the event shape.
    # Ground truth / observations stay in pixel space; guidance composes decode
    # into the observation operator so the likelihood is evaluated on pixels.
    decode, C, Hm, Wm = load_decoder(config, 'cuda')

    results = PATH / 'results'
    results.mkdir(parents=True, exist_ok=True)
    csv = results / 'metrics.csv'
    with open(csv, mode='w') as f:
        f.write('scenario,method,rmse,misfit\n')

    print(f'evaluating {runpath.name} (coarsen={coarsen}, grid={H}x{W})', flush=True)

    # -- 1. Ground-truth trajectory (also fixes the shared colour scale) ---
    # Same consolidated data the model was trained on (run prepare.py first).
    testfile = PATH / 'data/test.h5'
    if not testfile.exists():
        raise SystemExit(f"missing {testfile}; run `python prepare.py --coarsen {coarsen}` first")
    testset = TrajectoryDataset(testfile, window=None, flatten=False)
    x_star = testset[0][0][:LENGTH].cuda()          # (L, 1, H, W)

    # One set of colour limits, anchored to the ground truth, applied to every
    # panel (truth, prior samples, reconstructions) so comparisons are on the
    # same scale.
    gt = x_star.detach().cpu()
    center = gt.mean()
    amp = torch.quantile((gt - center).abs(), 0.99)
    vlim = dict(vmin=float(center - amp), vmax=float(center + amp))

    draw(x_star.cpu()[::4, 0], **vlim).save(results / 'truth.png')
    save_gif(x_star.cpu()[:, 0], results / 'truth.gif', **vlim)

    # -- 2. Unconditional (prior) samples ----------------------------------
    sde = VPSDE(score, shape=(LENGTH, C, Hm, Wm)).cuda()
    x = sde.sample((1,), steps=STEPS, corrections=CORRECTIONS, tau=TAU)
    with torch.no_grad():
        x = decode(x).cpu()                              # (1, L, 1, H, W) std pixel

    draw(x[:, ::4, 0], **vlim).save(results / 'prior_samples.png')
    save_gif(x[0, :, 0], results / 'prior_sample.gif', **vlim)

    # # Save the raw generated fields (and the ground truth) as physical buoyancy
    # # (= nondimensional temperature), for downstream statistical analysis.
    np.savez(
        results / 'prior_fields.npz',
        prior=destandardize(x.numpy()),                  # (n, L, 1, H, W)
        truth=destandardize(x_star.cpu().numpy()),       # (L, 1, H, W)
    )
    print('saved ground truth + prior samples (+ prior_fields.npz)', flush=True)
  
    def assimilate(A, y_star, std, tag, mask=None):
        r"""Posterior sampling (SDA + DPS) and bookkeeping for one observation.

        Sampling runs in the diffusion (latent) space; the observation operator
        ``A`` (pixel space) is composed with ``decode`` so the Gaussian likelihood
        is evaluated on decoded pixels (autograd flows through the decoder). For
        pixel runs ``decode`` is the identity and this is unchanged."""
        A_lat = lambda z: A(decode(z))
        for method, build in [
            ('sda', lambda: GaussianScore(y_star, A=A_lat, std=std, sde=VPSDE(score, shape=()))),
            ('dps', lambda: IgnoreContext(
                DPSGaussianScore(y_star, A=A_lat, sde=VPSDE(score, shape=()), zeta=1.0))),
        ]:
            sde = VPSDE(build(), shape=(LENGTH, C, Hm, Wm)).cuda()
            x = sde.sample(steps=STEPS, corrections=CORRECTIONS, tau=TAU)
            x = decode(x).cpu()                          # latent -> std pixel (L,1,H,W)
            e = rmse(x.cuda(), x_star)
            misfit = (A(x.cuda()) - y_star).square().mean().sqrt().item()
            with open(csv, mode='a') as f:
                f.write(f'{tag},{method},{e:.4f},{misfit:.4f}\n')
            print(f'  [{tag}] {method}: rmse={e:.4f} misfit={misfit:.4f}', flush=True)

            draw(x[::4, 0], **vlim).save(results / f'{tag}_{method}.png')
            if method == 'sda':
                save_gif(x[:, 0], results / f'{tag}_{method}.gif', **vlim)

        # observation panel (greyed-out unobserved cells)
        # draw(x_star.cpu()[::4, 0], mask=None if mask is None else mask[::4], **vlim).save(
        #     results / f'{tag}_obs.png')

    # -- 2a. Sparse spatial sensors (subsample H x W) ----------------------
    for sub in (4, 8):
        for sub in (4,8):
            def A(x, sub=sub):
                return x[..., ::sub, ::sub]

            y_star = torch.normal(A(x_star), 0.1)
            mask = torch.zeros(LENGTH, H, W, dtype=torch.bool)
            mask[..., ::sub, ::sub] = True
            assimilate(A, y_star, std=0.1, tag=f'spatial_sub{sub}', mask=mask)

        # -- 2b. Temporal gaps (observe every s-th frame fully) ----------------
        for step in (4,):
            def A(x, step=step):
                return x[..., ::step, :, :, :]

            y_star = torch.normal(A(x_star), 0.1)
            mask = torch.zeros(LENGTH, H, W, dtype=torch.bool)
            mask[::step] = True
            assimilate(A, y_star, std=0.1, tag=f'temporal_step{step}', mask=mask)

    # -- 2c. DPS ensemble for the sub-8 spatial scenario --------------------
    # `assimilate` above draws a single DPS sample per scenario; CRPS and the
    # spread/skill ratio (see metrics.py) need an actual ensemble, so draw
    # N_DPS independent posterior samples under the same sub=8 observation
    # setup as daps_envar.npz / daps_enkf.npz, in one batched `sde.sample`
    # call, and save it in the same (posterior, truth, sub) npz format so
    # metrics.py can evaluate it unchanged.
    sub = 8

    def A_dps(x):
        return x[..., ::sub, ::sub]

    for i in range(1):
        y_star = torch.normal(A_dps(x_star), 0.1)
        dps_score = IgnoreContext(
            DPSGaussianScore(y_star, A=lambda z: A_dps(decode(z)),
                             sde=VPSDE(score, shape=()), zeta=1.0))
        dps_sde = VPSDE(dps_score, shape=(LENGTH, C, Hm, Wm)).cuda()
        x_dps = dps_sde.sample((N_DPS,), steps=STEPS, corrections=CORRECTIONS, tau=TAU)
        x_dps = decode(x_dps).cpu()                      # (N_DPS, L, 1, H, W) std pixel

        mean = x_dps.mean(0).cuda()
        e = rmse(mean, x_star)
        misfit = (A_dps(mean) - y_star).square().mean().sqrt().item()
        with open(csv, mode='a') as f:
            f.write(f'spatial_sub{sub}_ensemble,dps,{e:.4f},{misfit:.4f}\n')
        print(f'  [spatial_sub{sub}] dps ensemble (N={N_DPS}): rmse={e:.4f} misfit={misfit:.4f}', flush=True)

        np.savez(
            results / f'dps_sub{sub}_{i:04d}.npz',
            posterior=destandardize(x_dps.numpy()),          # (N, L, 1, H, W)
            truth=destandardize(x_star.cpu().numpy()),       # (L, 1, H, W)
            sub=sub,
        )
        print(f'saved DPS ensemble -> {results}/dps_sub{sub}_{i:04d}.npz', flush=True)

        print(f'done -> {results}', flush=True)

    # -- 2d. Combine the 20 per-batch DPS files into a single ensemble -------
    # Each batch above was saved separately; concatenate them along the
    # ensemble axis so metrics.py (CRPS, spread/skill) sees one N=20*N_DPS
    # ensemble instead of 20 small ones.
    batches, truth_ref, sub_ref = [], None, None
    for i in range(10):
        file_name = results / f'dps_sub{sub}_{i:04d}.npz'
        with np.load(file_name) as d:
            batches.append(d['posterior'])        # (N_DPS, L, 1, H, W)
            if truth_ref is None:
                truth_ref = d['truth']
                sub_ref = int(d['sub'])

    combined = np.concatenate(batches, axis=0)     # (20 * N_DPS, L, 1, H, W)
    np.savez(
        results / f'dps_sub{sub}.npz',
        posterior=combined,
        truth=truth_ref,
        sub=sub_ref,
    )
    print(f'combined {len(batches)} batches ({combined.shape[0]} samples total) -> '
          f'{results}/dps_sub{sub}.npz', flush=True)

if __name__ == '__main__':
    (PATH / 'results').mkdir(parents=True, exist_ok=True)

    schedule(
        evaluate,
        name='Evaluation',
        backend='async',
        export='ALL',
    )
