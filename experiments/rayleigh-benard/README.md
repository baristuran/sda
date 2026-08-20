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
- **SDA / DPS** guided sampling (`eval.py`).

### Directory / data layout

| Path | Contents |
|------|----------|
| `data/{train,valid,test}.h5` | Consolidated pixel dataset, key `x`, shape `(n, L, 1, H, W)` |
| `data_latent/{train,valid,test}.h5` | VAE latents, shape `(n, L, C_z, H_l, W_l)` |
| `runs/<name>/` | Trained diffusion models (`state.pth`, `config.json`) |
| `runs_vae/<name>/` | Trained VAEs (`state.pth`, `config.json`, checkpoints) |
| `results*/` | Posterior samples (`*.npz`), GIFs, metrics |

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
daps.py | daps_enkf.py | eval.py | sample_prior.py → results*/*.npz
metrics.py | stats.py | contours.py               → scores & figures
```

---

## Scripts

### Data preparation

**`prepare.py`** — Consolidate the locally-simulated Dedalus snapshots into
contiguous `train/valid/test` HDF5 files (loaded fully into memory for fast
training). Choose spatial (`--coarsen`) and temporal (`--t-coarsen`)
downsampling.
```bash
python prepare.py --coarsen 2 --t-coarsen 1
```

**`check_dataset.py`** — Quick sanity check on `data/train.h5`: prints the array
shape / per-channel maxima and dumps a few sample frames as PNGs. Run directly
(no arguments): `python check_dataset.py`.

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

### Posterior sampling (data assimilation)

All samplers take a trained run (`--run`, default = most recent under `runs/`),
build a sparse spatial observation of a test trajectory (`--sub`, `--sigma-obs`),
draw `--n-samples` posterior trajectories, and save an `*.npz` (posterior +
truth) plus GIFs.

**`daps.py`** — DAPS-EnVar: annealed posterior sampling with a Langevin/EnRML
inner correction (`--beta` blends the ensemble prior, `--n-langevin`,
`--n-anneal`, `--n-ode`, ...).
```bash
python daps.py --n-samples 40 --sub 8
```

**`daps_enkf.py`** — EnKF-DAPS (closed-form ensemble-Kalman analysis per anneal
level). Add `--safe` for **Safe-EnKF-DAPS** (pure EnKF on the raw sample
covariance + a trust region on the level increment, radius `--c`).
```bash
python daps_enkf.py --n-samples 40 --sub 8            # EnKF-DAPS
python daps_enkf.py --n-samples 40 --sub 8 --safe     # Safe-EnKF-DAPS
```

**`eval.py`** — Broader qualitative evaluation of a run: unconditional prior
samples plus SDA / DPS guided assimilation under several observation scenarios
(sparse spatial, temporal gaps), writing figures, GIFs, a metrics CSV, and DPS
ensembles. Configured through environment variables; scheduled with `dawgz`.
```bash
RBC_RUN=<run-name> EVAL_STEPS=256 python eval.py
```

**`sample_prior.py`** — Generate many **unconditional** prior trajectories in
batches, written incrementally to a preallocated HDF5 file (for distributional
statistics).
```bash
RBC_RUN=<run-name> python sample_prior.py --n 100 --batch 4 --length 50 --steps 256
```

**`profile_daps.py`** — Micro-benchmark / profile the DAPS sampler
(`python profile_daps.py [n_samples] [n_anneal] [n_ode]`).

### Post-processing, metrics & figures

**`metrics.py`** — Ensemble-forecast verification scores (CRPS, energy /
variogram scores, spread–skill, coverage, RMSE, ...) computed against the ground
truth for one or more results `*.npz` files.
```bash
python metrics.py results/daps_enkf.npz --sub 8 --csv-dir results/scores
```

**`stats.py`** — Physical-statistics evaluation of generated fields vs. ground
truth (spectra, PDFs, wall/center diagnostics for a given Rayleigh/Prandtl
number).
```bash
python stats.py --prior-file results/prior_fields.h5 --rayleigh 1e9 --prandtl 1.0
```

**`npz_to_h5.py`** — Convert a results `*.npz` (e.g. `posterior` or `truth`) into
an HDF5 file that `stats.py` can read (optionally de-standardizing).
```bash
python npz_to_h5.py results/daps_enkf.npz --key posterior --out results/posterior.h5
```

**`contours.py`** — Standalone matplotlib contour panels (with a θ colorbar) from
a saved `*.npz`, optionally overlaying the observation mask.
```bash
python contours.py results/daps_enkf.npz --key truth --stride 4 --show-obs --sub 8
```

---

## Notes

- Scripts that touch a GPU either accept `--gpu` / `--device`, read
  `CUDA_VISIBLE_DEVICES` / `RBC_GPU`, or (in `train.py` / `daps*.py`) set the
  device near the top of the file — check the file if you need a specific card.
- Backward compatibility is a hard requirement throughout: any model trained
  before the latent-diffusion or 3D-U-Net additions still loads and runs with the
  current code, because the latent/arch switches default to the original
  pixel-space, 2D behaviour when their config keys are absent.
