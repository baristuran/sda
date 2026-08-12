import sys, time, collections
sys.path.insert(0, '/home/baris-turan/sda/experiments/rayleigh-benard')

import numpy as np
import torch
import h5py

import daps
import daps_enkf
from daps_enkf import _enkf_analysis, find_run
from daps import VPGrid, _build_anneal, _pf_ode_x0
from sda.score import VPSDE
from sda.utils import load_config
from utils import PATH, HEIGHT, WIDTH, load_score

T_ACC = collections.defaultdict(float)
N_CALLS = collections.defaultdict(int)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        sync(); self.t0 = time.perf_counter(); return self
    def __exit__(self, *a):
        sync(); T_ACC[self.name] += time.perf_counter() - self.t0; N_CALLS[self.name] += 1


# ---- instrument the score network (the thing PF-ODE calls repeatedly) -------
_orig_eps = VPGrid.eps
def counted_eps(self, x, idx):
    with timer('  score net (sde.eps)'):
        return _orig_eps(self, x, idx)
VPGrid.eps = counted_eps


def profiled_sample(sde, obs_rows, y_flat, R_diag, shape, n_samples, device,
                    n_anneal=400, n_ode=6, n_esmda=1, prior_scale=1.0,
                    inflation=1.02, sigma_max=100.0, sigma_min=0.01, rho=7.0,
                    T=1000):
    sde.eval()
    with timer('setup (VPGrid + anneal)'):
        grid = VPGrid(sde, device, T)
        abar, sigma_eff = grid.abar, grid.sigma_eff
        anneal_idx = _build_anneal(sigma_eff, n_anneal, rho, sigma_max, sigma_min)

    N = n_samples
    obs_rows = obs_rows.to(device).long()
    y_flat = y_flat.to(device).float()
    R_diag = R_diag.to(device).float()

    def Hop(x0_flat):
        return x0_flat[:, obs_rows]

    x_t = torch.randn(N, *shape, device=device)

    n_levels = len(anneal_idx) - 1
    for step in range(n_levels):
        t_idx, t_next = anneal_idx[step], anneal_idx[step + 1]
        with timer('PF-ODE denoise (total)'):
            x0_hat = _pf_ode_x0(grid, x_t, t_idx, n_ode, rho)
            xf = x0_hat.flatten(1)

        r_t = float((prior_scale * sigma_eff[t_idx]).clamp(min=float(sigma_eff[0])))
        with timer('EnKF analysis'):
            xa = xf
            for _ in range(max(n_esmda, 1)):
                xa = _enkf_analysis(xa, y_flat, R_diag, Hop,
                                    alpha=float(max(n_esmda, 1)), r_t=r_t,
                                    inflation=inflation)
            if not torch.isfinite(xa).all():
                xa = torch.where(torch.isfinite(xa), xa, xf)
            x0_a = xa.reshape(N, *shape)

        with timer('re-noise'):
            if t_next == 0:
                x_t = x0_a
            else:
                x_t = torch.sqrt(abar[t_next]) * x0_a \
                    + torch.sqrt(1.0 - abar[t_next]) * torch.randn_like(x0_a)

    return x_t.detach(), n_levels


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    N_ANNEAL = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    N_ODE = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    run = find_run("None_4o07btvu")
    config = load_config(run)
    coarsen = config.get('coarsen', 1)
    H, W = HEIGHT // coarsen, WIDTH // coarsen
    L = 12
    shape = (L, 1, H, W)

    score = load_score(run / 'state.pth').to(device).eval()
    sde = VPSDE(score, shape=shape).to(device)

    with h5py.File(PATH / 'data/test.h5', 'r') as f:
        x_star = torch.from_numpy(f['x'][0, :L]).to(device).float()

    sub = 8
    A = lambda x: x[..., ::sub, ::sub]
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    mask[..., ::sub, ::sub] = True
    obs_rows = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    y = torch.normal(A(x_star), 0.1)
    y_flat = y.reshape(-1)
    R_diag = torch.full_like(y_flat, 0.01)

    n = int(np.prod(shape))
    print(f'config: N={N_SAMPLES} ensemble, n_anneal={N_ANNEAL}, n_ode={N_ODE}')
    print(f'state:  shape={shape}  n={n:,}  dy={obs_rows.numel():,}  grid={H}x{W}')
    print()

    # warm-up (cuDNN autotune / allocator) so it doesn't pollute the numbers
    _ = _pf_ode_x0(VPGrid(sde, device, 1000), torch.randn(N_SAMPLES, *shape, device=device), 500, 2)
    sync()
    T_ACC.clear(); N_CALLS.clear()

    t0 = time.perf_counter(); sync()
    x, n_levels = profiled_sample(sde, obs_rows, y_flat, R_diag, shape,
                                  N_SAMPLES, device, n_anneal=N_ANNEAL, n_ode=N_ODE)
    sync(); total = time.perf_counter() - t0

    print(f'{"phase":<28}{"total s":>10}{"% ":>8}{"calls":>8}{"ms/call":>10}')
    print('-' * 64)
    order = ['setup (VPGrid + anneal)', 'PF-ODE denoise (total)',
             '  score net (sde.eps)', 'EnKF analysis', 're-noise']
    for k in order:
        if k not in T_ACC:
            continue
        v = T_ACC[k]
        print(f'{k:<28}{v:>10.3f}{100*v/total:>7.1f}%{N_CALLS[k]:>8}{1000*v/max(N_CALLS[k],1):>10.2f}')
    print('-' * 64)
    print(f'{"TOTAL":<28}{total:>10.3f}{100.0:>7.1f}%')
    print()
    print(f'anneal levels actually run: {n_levels}  (requested n_anneal={N_ANNEAL})')
    print(f'score-net calls per level:  {N_CALLS["  score net (sde.eps)"]/n_levels:.2f}')


if __name__ == '__main__':
    main()