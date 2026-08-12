# Rayleigh–Bénard convection — progress report

Score-based generative modelling of 2D Rayleigh–Bénard convection within the
SDA framework (Rozet & Louppe, 2023). This note summarises the **unconditional**
model: the training setup and the quality of the fields it generates. Figures are
from [`results_window_5/`](results_window_5).

## 1. Data

Generated from scratch with a **Dedalus (d3)** 2D Rayleigh–Bénard solver
([`dedalus/rayleigh_benard.py`](../../../dedalus/rayleigh_benard.py)); **1048**
independent trajectories, each from a distinct random seed.

- **Physics:** **Ra = 10⁷, Pr = 1**, non-dimensionalised by box height and
  free-fall time (κ = (Ra·Pr)⁻¹ᐟ², ν = (Ra/Pr)⁻¹ᐟ²). Domain **Lx×Lz = 4×1**,
  horizontally periodic, no-slip isothermal walls: hot bottom **b(z=0)=1**, cold
  top **b(z=1)=0**. RK222 time-stepping with CFL-adaptive Δt; each run integrates
  50 free-fall times from a conduction profile + small (10⁻³) wall-damped noise.
- **Field:** buoyancy = nondimensional temperature **θ** (single channel).
- **Grid:** Fourier in x (uniform, Nx=512), Chebyshev in z (Nz=128), then
  **interpolated onto a uniform z-grid** (`chebyshev_to_uniform_matrix`). Oriented
  as (H=z, W=x) so the vertical (gravity) axis is first: native **128×512**,
  coarsened 2× → **64×256**.
- **Preprocessing:** snapshots saved every 0.25 free-fall times (~200 per run);
  the initial transient (first **80** frames) is discarded, leaving **120**
  stationary-regime frames per trajectory. θ is standardised with a fixed regime
  mean/std (**0.4993 / 0.2053**).
- **Splits (consolidated contiguous HDF5 via `prepare.py`, loaded in memory):**
  train **838** trajectories × 120 snapshots, valid **105**, test **105**
  — each `(120, 1, 64, 256)`.

## 2. Unconditional training setup

A VP-SDE score model over short **temporal windows** of the field; long
trajectories are generated at inference by SDA's Markov-blanket composition of
overlapping windows.

| | |
|---|---|
| Objective | VP-SDE denoising score matching (`VPSDE`, SDA) |
| Backbone | local score U-Net (`make_score`), 2D conv |
| Window | **5** frames, modelled jointly as `(window, 64, 256)` |
| Embedding dim | 64 |
| Hidden channels | (64, 128, 256), 3 residual blocks per level |
| Kernel / activation | 3×3 / SiLU |
| Optimiser | AdamW, lr 2e-4, weight decay 1e-3, exponential schedule |
| Batch size / epochs | 128 / 4096 |
| Hardware | 3× GPU (DDP) |

Run: `runs/None_qxh9vkvh` (window = 5, coarsen = 2). A window-7 variant
(`None_4o07btvu`) was also trained.

## 3. Unconditional generation results

### 3.1 Generated fields (contour visualisation)

Prior samples unrolled to 50 snapshots via SDA window composition. The model
reproduces the qualitative structure of Ra = 10⁷ convection: thin thermal
boundary layers at the top/bottom walls, detaching **plumes**, and turbulent
mixing in the well-mixed bulk.

![Unconditional prior samples of the buoyancy field θ](results/contours_prior_fields_prior/frame_0000.png)
![Unconditional prior samples of the buoyancy field θ](results/contours_prior_fields_prior/frame_0004.png)
![Unconditional prior samples of the buoyancy field θ](results/contours_prior_fields_prior/frame_0008.png)
![Unconditional prior samples of the buoyancy field θ](results/contours_prior_fields_prior/frame_0012.png)


### 3.2 Statistical comparison (generated "prior" vs. ground-truth test set)

Statistics are accumulated over generated trajectories and the held-out test
trajectories (`stats.py`).

**Wall-normal RMS θ profile** — the model captures the characteristic
double-hump shape (fluctuations peak inside the thermal boundary layers, flat in
the bulk). It slightly **under-predicts the boundary-layer peak**
(0.102 vs. 0.123) while matching the bulk level (~0.047).

![RMS θ profile: generated vs. truth](results_window_5/prior_stats/stat_theta_rms_profile.png)

**Horizontal θ spectrum** — energy across scales matches the ground truth over
the full inertial range; the only discrepancy is a mild **excess of energy at the
smallest scales** (highest wavenumbers), i.e. the generator is slightly less
dissipative than the data at grid scale.

![Horizontal θ spectrum: generated vs. truth](results_window_5/prior_stats/stat_theta_spectrum.png)

**Summary numbers**

| quantity | truth | prior (generated) |
|---|---|---|
| RMS θ, boundary-layer peak | 0.123 | 0.102 |
| RMS θ, bulk (median) | 0.049 | 0.047 |
| mean thermal dissipation ⟨ε_θ⟩ | 8.3e-4 | 7.0e-4 |

### Takeaways

- The unconditional model produces **visually and statistically realistic**
  Ra = 10⁷ buoyancy fields: correct plume/boundary-layer morphology, matching
  bulk fluctuation level, and a horizontal spectrum that tracks the data across
  scales.
- Two mild biases remain: a **slightly weak boundary-layer RMS peak** and a
  **small excess of small-scale energy** (⟨ε_θ⟩ ≈ 16 % low). Both are consistent
  with the model marginally under-resolving the sharpest near-wall gradients.

## 4 Conditional Generation Results
The observation operator is linear selection.  
Every 8th grid point is observed with an isotropic Gaussian observaiton noise. 

|Truth  | Generated 
|:---:  | :---:     |
|![Unconditional prior samples of the buoyancy field θ](results/contours_daps_envar_truth/frame_0000.png)     |![Unconditional prior samples of the buoyancy field θ](results/contours_daps_envar_posterior/frame_0000.png)          |
| ![Unconditional prior samples of the buoyancy field θ](results/contours_daps_envar_truth/frame_0002.png) | ![Unconditional prior samples of the buoyancy field θ](results/contours_daps_envar_posterior/frame_0002.png)|
| ![Unconditional prior samples of the buoyancy field θ](results/contours_daps_envar_truth/frame_0004.png) | ![Unconditional prior samples of the buoyancy field θ](results/contours_daps_envar_posterior/frame_0004.png)|
| ![Unconditional prior samples of the buoyancy field θ](results/contours_daps_envar_truth/frame_0006.png) | ![Unconditional prior samples of the buoyancy field θ](results/contours_daps_envar_posterior/frame_0006.png)|