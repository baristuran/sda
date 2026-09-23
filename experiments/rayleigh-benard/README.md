# Rayleigh–Bénard convection — score-based data assimilation

Diffusion-model priors for 2D Rayleigh–Bénard (RB) convection, and posterior
sampling (data assimilation) from sparse/noisy observations. Built on the
[SDA](https://github.com/francois-rozet/sda) framework (Rozet & Louppe, 2023);
this directory is one experiment inside that repository.

The modelled field is the standardized **buoyancy** (a single channel), oriented
with the vertical (gravity) direction first: `(H, W)` with native `128 × 512`,
usually spatially coarsened to `64 × 256` (`coarsen=2`).

---

## Overview

### Data

The trajectories currently under `data/` are **locally simulated with Dedalus**
(`~/dedalus/rayleigh_benard.py`, `generate_dataset.py`) at **Ra = 1e7, Pr = 1**
in a box of aspect ratio `Lx × Lz = 4 × 1`, written by `prepare.py`. Physical
diagnostics therefore default to `--rayleigh 1e7 --prandtl 1.0`.

`utils.py` also still contains the loader for the Ra = 1e9 buoyancy trajectories
of the [Well](https://github.com/PolymathicAI/the_well) dataset (`RBCDataset`,
`RAYLEIGH = 1e9`), which is what the original fork was trained on. The two
sources have different standardisation constants — `BUOYANCY_MEAN/STD` (Well)
and `DEDALUS_BUOYANCY_MEAN/STD` (Dedalus) — so check which one `destandardize`
is using before comparing absolute values across datasets.

### Diffusion prior

The prior over short trajectories is a **variance-preserving (VP) SDE** with a
cosine noise schedule. Its noise predictor (score network) follows SDA's
**local, Markov-blanket** construction: a single shared kernel is applied to
overlapping temporal windows of `window` consecutive frames (`MCScoreNet`), so a
network trained on short windows can score trajectories of **arbitrary length** at
inference time. A fixed **normalized-height context channel** is fed to the kernel
because RB convection is not vertically translation-invariant (hot lower plate,
cold upper plate).

Two score-kernel architectures are available (selected by the `arch` config key,
default `2d`; fully backward compatible — old runs have no `arch` key and rebuild
as `2d`):

- **`2d`** (`LocalScoreUNet`): the temporal window is folded into the channel
  axis and processed by a 2D U-Net.
- **`3d`** (`LocalScoreUNet3D`): time is a genuine dimension processed by a 3D
  U-Net (downsampling the two spatial dims only). Same external interface, so
  training and all inference scripts are unchanged.

### Pixel vs. latent diffusion

- **Pixel diffusion** (default): the SDE runs directly on standardized buoyancy
  frames.
- **Latent diffusion**: a per-frame convolutional VAE (`vae.ConvVAE`) compresses
  each `(1, 64, 256)` frame to a `(C_z, H_l, W_l)` latent (e.g. `(4, 8, 32)`); the
  diffusion then runs in that latent space and decodes back at inference. The
  latents are standardized per channel using training-set statistics (recorded in
  the run config). This is enabled by setting `vae_run` in the training config;
  everything is backward compatible — runs without `vae_run` use the identity
  "decoder" via `vae.load_decoder`, so all inference scripts work with old
  pixel-space models unchanged.

### Posterior sampling (data assimilation)

Observations always live in **pixel space**; for latent runs the observation
operator is composed with the VAE decoder. Several posterior samplers are
implemented:

- **DAPS-EnVar** (`daps.py`) — Decoupled Annealing Posterior Sampling (Zhang et
  al., 2024) with a variational/Langevin inner correction.
- **EnKF-DAPS** and **Safe-EnKF-DAPS** (`daps_enkf.py`) — DAPS with a closed-form
  ensemble-Kalman analysis; the `--safe` variant drops the Tweedie prior rescale
  and instead constrains each level's increment with a trust region.
- **DPS** (`sample_dps.py`) and **SDA / DPS** guided sampling (`eval.py`).

### Running the scripts

Use the project's conda environment (it provides `torch`, `h5py`, `sklearn`,
etc.) and run from this directory so the local `utils` / `vae` modules import:

```bash
cd sda/experiments/rayleigh-benard
/home/baris-turan/miniconda3/envs/sda/bin/python <script>.py ...
```

A typical end-to-end flow:

```
prepare.py                 →  data/*.h5
# pixel diffusion:
train.py                   →  runs/<name>/state.pth
# OR latent diffusion:
train_vae.py → encode_latents.py → train.py (with vae_run set) → runs/<name>/
# then assimilate / evaluate:
daps.py | daps_enkf.py | sample_dps.py | eval.py | sample_prior.py → results*/*.npz
metrics.py                                        → ensemble scores (CRPS, RMSE, ...)
compute_stats.sh (→ npz_to_h5.py → stats.py)      → physical statistics & figures
contours.py                                       → contour panels
```

---

## Scripts

### Data preparation

**`prepare.py`** — Consolidate the locally-simulated Dedalus snapshots into
contiguous `train/valid/test` HDF5 files (loaded fully into memory for fast
training). Choose spatial (`--coarsen`) and temporal (`--t-coarsen`)
downsampling; `--interp` maps the Chebyshev z-grid onto a uniform one
(recommended — the U-Net and `stats.py` both assume uniform pixel spacing).
```bash
python prepare.py --coarsen 2 --t-coarsen 1 --interp
```

**`check_dataset.py`** — Quick sanity check on `data/test.h5`: prints the array
shape / per-channel maxima and dumps the first 50 frames of one trajectory as
PNGs into `data/`. Run directly (no arguments): `python check_dataset.py`.

**`data/check_dataset_transient.py`** — Diagnose how long the initial transient
lasts: tracks the Nusselt number and θ-RMS over time for a split and writes
`data/stationarity_*.png`. Use it to choose `DEDALUS_T_START` (frames discarded
by `prepare.py`) and to pick the stationary frame window used for statistics.

### VAE / latent diffusion

**`vae.py`** — The per-frame `ConvVAE` model plus builders/loaders
(`make_vae`, `load_vae`, `load_decoder`). Imported by other scripts; not run
directly.

**`train_vae.py`** — Train the VAE on pixel frames (recon MSE + `beta`·KL).
Supports AMP, frame subsampling, checkpoint resume, and multi-GPU via `torchrun`
(single-process otherwise).
```bash
python train_vae.py --epochs 100 --batch-size 128 --beta 1e-5 --gpu 0
# multi-GPU (per-GPU batch; global batch = batch * nproc):
torchrun --nproc_per_node=3 train_vae.py --gpus 0,1,2 --batch-size 128 --name vae_big
python train_vae.py --resume runs_vae/vae_big        # continue a run
```

**`test_vae.py`** — Reconstruct a single frame with a trained VAE, print the
reconstruction / KL loss, and save high-resolution original vs. reconstruction
PNGs.
```bash
python test_vae.py --vae-run vae_big --split test --index 12345 --gpu 0
```

**`encode_latents.py`** — Encode `data/{train,valid,test}.h5` into
`data_latent/` using a trained VAE (deterministic posterior mean). Run this
before training a latent diffusion model.
```bash
python encode_latents.py --vae-run vae_big --gpu 0
```

**`analyze_latent.py`** — PCA and t-SNE of the VAE latent space, colored by time
step (or trajectory); writes two high-resolution (400 dpi) PNGs.
```bash
python analyze_latent.py --split train --n 6000 --perplexity 30
```

### Training the diffusion model

**`train.py`** — Train the score network (the diffusion prior). Edit the
`CONFIG` dict at the top to set architecture (`window`, `hidden_channels`,
`arch`), the data mode (`coarsen`; set `vae_run` for latent diffusion), and
optimization. Uses Weights & Biases for logging and `dawgz` to schedule the job;
the target GPU is set near the top via `CUDA_VISIBLE_DEVICES`. Latent runs
automatically standardize the latents per channel and record the stats in
`config.json`.
```bash
python train.py
```

**`utils.py`** — Experiment helpers shared by everything: dataset paths,
`TrajectoryDataset`, `make_score` / `load_score`, the score-net kernels
(`LocalScoreUNet`, `LocalScoreUNet3D`), standardization, and plotting (`draw`,
`save_gif`). Imported, not run directly.

**`num_params.py`** — Print the parameter count of the score network for a given
run's `config.json` (edit the `path` variable at the top).
```bash
python num_params.py
```

---

## Posterior sampling (data assimilation)

`daps.py` and `daps_enkf.py` are the two main assimilation drivers. Both do the
same thing end to end:

1. load a trained diffusion run (`--run`, default = most recent under `runs/`)
   and its `config.json` (which also decides pixel vs. latent, and the grid);
2. read one ground-truth trajectory of `L = --length` frames (default: the
   model's training `window`) from the dataset;
3. build a **sparse spatial observation** `y = A(x*) + N(0, sigma_obs²)`, where
   `A` keeps every `--sub`-th pixel in each direction;
4. draw `--n-samples` posterior trajectories jointly (the ensemble is one batch —
   the samplers are *ensemble* methods, so `--n-samples` is a statistical knob,
   not just a throughput one);
5. print `rmse(vs truth)` and `obs-misfit`, and save an `.npz`
   (`posterior`, `truth`, `sub`) plus GIFs of the first sample and the truth.

The `.npz` arrays are written in **physical units** (`destandardize` is applied),
which is the format `metrics.py`, `contours.py` and `compute_stats.sh` expect.

### `daps.py` — DAPS-EnVar

Annealed posterior sampling where each level's likelihood correction is a
**Langevin / EnRML** inner loop in whitened control variables, with a hybrid
ensemble prior `C_t = r_t²[(1-beta) I + beta R_t]` (`--beta 0` recovers isotropic
DAPS). A final manifold projection re-noises to `--proj-sigma` and denoises with
the PF-ODE to remove off-manifold kinks.

```bash
# defaults: 50 samples, L = model window, sub = 8, sigma_obs = 0.1
python daps.py

# the setting recorded as best in hyperparams.txt (window-7 run)
python daps.py --n-samples 80 --length 25 --sub 8 \
               --n-anneal 400 --n-ode 10 --n-langevin 400 \
               --n-outer 1 --eta-0 2e-5 --proj-n-ode 1

python daps.py --run None_4o07btvu --beta 0.0 --device cuda:2   # isotropic DAPS
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--run` | most recent | run directory under `runs/` |
| `--n-samples` | 50 | ensemble size |
| `--length` | model `window` | trajectory length in frames |
| `--sub` / `--sigma-obs` | 8 / 0.1 | observation stride and noise std |
| `--beta` | 0.8 | ensemble-prior weight (0 = isotropic DAPS) |
| `--n-anneal` / `--n-ode` | 400 / 6 | anneal levels, PF-ODE steps per level |
| `--n-langevin` / `--eta-0` | 400 / 2e-4 | Langevin steps per level and step size |
| `--n-outer` | 1 | EnRML covariance relinearisation sweeps |
| `--prior-scale` | 1.0 | scaling of the Tweedie prior std `r_t` |
| `--max-grad-norm` | 1e3 | per-particle likelihood-gradient clip |
| `--proj-sigma` / `--proj-n-ode` / `--proj-n-iter` | 0.8 / 1 / 1 | final manifold projection (`--proj-sigma <= 0` disables it) |
| `--device` / `--seed` | cuda / 0 | — |

**Fixed in the file, not on the CLI:** the ground truth is always the **first**
trajectory of `data/test.h5` (`f['x'][0, :L]`), and the output always goes to
`results/daps_envar.npz` (+ `daps_envar_sample.gif`, `daps_truth.gif`). Change
the trajectory or the destination by editing the bottom of `main()`, or copy the
`.npz` out after each run — a second run overwrites the first.

### `daps_enkf.py` — EnKF-DAPS / Safe-EnKF-DAPS

Replaces the inner Langevin loop with a **single closed-form stochastic
(perturbed-observation) EnKF analysis** per anneal level: no step size, no
autograd through the observation operator, and the gain is never inverted
(Cholesky on the ensemble-space Gram matrix). The prior anomalies are rescaled
to the Tweedie std `r_t` per coordinate, so only their *correlation shape* is
used — this is what stops the ensemble collapsing when the same observation is
re-assimilated at every level.

`--safe` switches to **Safe-EnKF-DAPS**: no `r_t` rescale (pure EnKF on the raw
sample covariance) and instead a **trust region** on each level's increment,
`RMS(increment) <= c * sigma_eff(t)`, with radius `--c`.

```bash
python daps_enkf.py --n-samples 40 --sub 8                        # EnKF-DAPS
python daps_enkf.py --n-samples 40 --sub 8 --safe                 # Safe-EnKF-DAPS
python daps_enkf.py --safe --c 0.5                                # looser trust region

# the settings recorded as best in hyperparams.txt (window-7 run)
python daps_enkf.py --length 25 --n-samples 80                    # rmse ~0.236
python daps_enkf.py --length 25 --n-samples 80 --safe             # rmse ~0.229

# pick the GPU and keep each run's output separate
CUDA_VISIBLE_DEVICES=0 python daps_enkf.py --safe --length 25 --n-samples 80 \
    --out-path results_enkf_test_stationary_part/daps_enkf_safe_1
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--run` | most recent | run directory under `runs/` |
| `--n-samples` | 50 | ensemble size (also the EnKF rank — small ensembles under-rank the covariance) |
| `--length` | model `window` | trajectory length in frames |
| `--sub` / `--sigma-obs` | 8 / 0.1 | observation stride and noise std |
| `--n-anneal` / `--n-ode` | 400 / 6 | anneal levels, PF-ODE steps per level |
| `--n-esmda` | 1 | sub-assimilations per level (ESMDA); 1 is recommended — the anneal already is the multi-update loop |
| `--inflation` | 1.02 | multiplicative covariance inflation (non-safe variant) |
| `--prior-scale` | 1.0 | scaling of `r_t` (non-safe variant) |
| `--safe` | off | Safe-EnKF-DAPS (pure EnKF + trust region) |
| `--c` | 0.25 | trust-region radius, `--safe` only |
| `--out-path` | `results_deneme` | output directory, relative to this one |
| `--device` / `--seed` | cuda / 0 | — |

Outputs land in `<--out-path>/` as `daps_enkf.npz` or `daps_enkf_safe.npz`
(depending on `--safe`), plus `<tag>_sample.gif` and `daps_truth.gif`. Pass a
fresh `--out-path` per run, since the filename itself is fixed.

**Fixed in the file, not on the CLI** (edit near the top of `main()` if you need
to change them):

- the GPU is pinned with `os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")` at
  the top of the module — override it from the shell, as in the example above;
- the ground truth is read from **`data/valid.h5`** (the `data/test.h5` line is
  commented out just above it);
- the target slice is chosen by two constants, `N = 15` (trajectory index) and
  `M = 5` (time-slice index): frames `M*L : (M+1)*L` of trajectory `N`. Both are
  echoed to stdout at the start of the run.

To build a large ensemble over many slices, loop over `N`/`M` with one
`--out-path` per run and concatenate the resulting `.npz` files afterwards —
`results_enkf_test_stationary_part/combine_trajectories.py` is the throwaway
script used for exactly that (it concatenates `daps_enkf_safe_<i>/daps_enkf_safe.npz`
into one combined `.npz`, stacking 5-D arrays along the time axis).

### Other samplers

**`sample_dps.py`** — Draw an ensemble of **DPS** (Diffusion Posterior Sampling)
reconstructions for one sparse spatial observation (the standalone version of the
DPS block formerly inside `eval.py`). DPS is memory-heavy, so samples are drawn in
batches of `--batch` and accumulated; the whole ensemble is written to a **single**
`.npz` (`posterior`, `truth`, `sub`) in the same format as `daps*.py`, so
`metrics.py` reads it unchanged.
```bash
python sample_dps.py --n-samples 64 --batch 8 --sub 8            # 64 samples, 8 per GPU batch
python sample_dps.py --run <run-name> --n-samples 100 --batch 4 --steps 256
```

**`eval.py`** — Broader qualitative evaluation of a run: unconditional prior
samples plus SDA / DPS guided assimilation under several observation scenarios
(sparse spatial, temporal gaps), writing figures, GIFs, a metrics CSV, and DPS
ensembles. Configured through environment variables; scheduled with `dawgz`.
```bash
RBC_RUN=<run-name> EVAL_STEPS=256 python eval.py
```

**`sample_prior.py`** — Generate many **unconditional** prior trajectories in
batches, written incrementally to a preallocated HDF5 file (dataset `theta`, in
physical units) — this is the input `stats.py` expects when you want the
statistics of the *prior* rather than of a posterior ensemble. The GPU is pinned
at the top of the file (`CUDA_VISIBLE_DEVICES = "2"`).
```bash
RBC_RUN=<run-name> python sample_prior.py --n 100 --batch 4 --length 50 --steps 256 \
    --out results/prior_fields.h5
```

**`profile_daps.py`** — Micro-benchmark / profile the DAPS sampler
(`python profile_daps.py [n_samples] [n_anneal] [n_ode]`).

---

## Post-processing, metrics & figures

**`metrics.py`** — Ensemble-forecast verification scores (CRPS, energy /
variogram scores, spread–skill, coverage, RMSE, ...) computed against the ground
truth for one or more results `*.npz` files. `--sub` defaults to the value stored
in the `.npz`.
```bash
python metrics.py results_enkf_test/daps_enkf_safe.npz --csv-dir results_enkf_test/scores
python metrics.py results*/daps_enkf*.npz --key posterior
```

### Physical statistics: `compute_stats.sh`

`compute_stats.sh` is the one-command wrapper for the physical-statistics
pipeline. It chains the two scripts that would otherwise be run by hand:

1. **`npz_to_h5.py`** — extract one array (`--key`) from a results `.npz` and
   write it as the streamable HDF5 (`theta`, physical units) that `stats.py` reads;
2. **`stats.py`** — compute the diagnostics of those generated fields and of the
   ground-truth test set, and write the comparison figures.

```bash
./compute_stats.sh <generated.npz> <output_dir> [options]
```

| Argument / option | Default | Meaning |
|-------------------|---------|---------|
| `<generated.npz>` | — | results file, e.g. `results_enkf_test/daps_enkf_safe.npz` |
| `<output_dir>` | — | destination for the `.h5` and every figure (created if missing) |
| `--key KEY` | `posterior` | which array to analyse: `posterior`, `prior` or `truth` |
| `--rayleigh RA` | `1e7` | Rayleigh number — sets `kappa` for the dissipation/Nusselt diagnostics |
| `--prandtl PR` | `1.0` | Prandtl number |
| `--batch N` | `8` | trajectories per streaming batch (bounds memory) |
| `--standardized` | off | the `.npz` holds standardized values; de-standardize on write |
| `--python PATH` | the `sda` conda env | interpreter to use |
| `-h`, `--help` | — | print the header block |

Examples:

```bash
# statistics of a Safe-EnKF-DAPS posterior ensemble
./compute_stats.sh results_enkf_test/daps_enkf_safe.npz results_enkf_test/stats_posterior

# same trajectories, but analysing the ground truth stored in the same npz,
# at a different Rayleigh number and with bigger read batches
./compute_stats.sh results_enkf_test/daps_enkf_safe.npz results_enkf_test/stats_truth \
    --key truth --rayleigh 1e7 --batch 16

# a combined multi-slice ensemble
./compute_stats.sh results_enkf_test_stationary_part/daps_enkf_safe_combined_stationary.npz \
    results_enkf_test_stationary_part --key posterior
```

Notes:

- The script `cd`s into this directory itself (the Python scripts import the
  local `utils`), and resolves the `.npz` and output paths to absolute paths
  first, so it can be called from anywhere.
- The intermediate HDF5 is written as `<output_dir>/<npz stem>.h5` and kept, so a
  re-run of `stats.py` alone with different diagnostics flags is cheap:
  `python stats.py --prior-file <that>.h5 --out-path <output_dir> --rayleigh 1e7`.
- **`--standardized` is off by default and that is usually right**: the arrays
  written by `daps.py` / `daps_enkf.py` / `sample_dps.py` are already in physical
  units. Pass it only for a `.npz` you know is standardized, otherwise the field
  is scaled twice.
- The ground truth is not configurable here: `stats.py` always compares against
  `data/test.h5`, using the **last 20 frames** of each test trajectory (the
  stationary part) and de-standardizing them.
- `set -euo pipefail` is on, so a failure in step 1 stops the run before
  `stats.py` is invoked.

**`stats.py`** — The diagnostics themselves; run it directly if you already have
the `.h5` (or the prior fields from `sample_prior.py`). It streams both sources
in batches and writes, into `--out-path`: `stat_theta_mean_profile.png`,
`stat_theta_rms_profile.png`, `stat_theta_skewness.png`, `stat_theta_flatness.png`,
`stat_theta_spectrum{,_wall,_center}.png`, `stat_two_point_corr_{wall,center}.png`,
`stat_thermal_dissipation.png`, `stat_theta_pdf.png`,
`stat_theta_fluc_pdf_z.png`, and `stats.npz` with all the underlying arrays. The
wall-normal probe locations are `--spec-z-wall` / `--spec-z-center` / `--fluc-z`.
Assumes a uniform z-grid — generate the data with `prepare.py --interp`.
```bash
python stats.py --prior-file results/prior_fields.h5 --out-path results/stats_prior \
    --rayleigh 1e7 --prandtl 1.0 --batch 8
```

**`npz_to_h5.py`** — The conversion step on its own (`--key`, `--out`,
`--batch`, `--standardized-input`).
```bash
python npz_to_h5.py results_enkf_test/daps_enkf.npz --key posterior --out results_enkf_test/posterior.h5
```

**`contours.py`** — Standalone matplotlib contour panels (with a θ colorbar) from
a saved `*.npz`, optionally overlaying the observation mask.
```bash
python contours.py results_enkf_test/daps_enkf.npz --key truth --stride 4 --show-obs --sub 8
```

---

## Notes

- Scripts that touch a GPU either accept `--gpu` / `--device`, read
  `CUDA_VISIBLE_DEVICES` / `RBC_GPU`, or (in `train.py`, `daps_enkf.py`,
  `sample_prior.py`) pin the device near the top of the file — check the file, or
  just prefix the command with `CUDA_VISIBLE_DEVICES=<n>`, which overrides the
  `setdefault` pins.
- Several scripts have run-specific constants at the top of `main()` rather than
  CLI flags (the assimilated trajectory in `daps*.py`, the run path in
  `num_params.py`, the split in `check_dataset.py`). They are noted per script
  above.
- Backward compatibility is a hard requirement throughout: any model trained
  before the latent-diffusion or 3D-U-Net additions still loads and runs with the
  current code, because the latent/arch switches default to the original
  pixel-space, 2D behaviour when their config keys are absent.
