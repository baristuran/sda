import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils import destandardize

# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
FILE = "valid.h5"
DATASET_KEY = None     # e.g. "temperature"; leave None to auto-detect
DT = 0.25              # snapshot spacing in free-fall times, if applicable


# ------------------------------------------------------------
# Load dataset
# Expected shape: (Ntraj, Nt, 1, Nz, Nx)
# ------------------------------------------------------------
def find_5d_dataset(group):
    """Find the first 5D dataset in an HDF5 file."""
    for key, item in group.items():
        if isinstance(item, h5py.Dataset) and item.ndim == 5:
            return item
        elif isinstance(item, h5py.Group):
            result = find_5d_dataset(item)
            if result is not None:
                return result
    return None


with h5py.File(FILE, "r") as f:
    if DATASET_KEY is None:
        dataset = find_5d_dataset(f)
        if dataset is None:
            raise RuntimeError("Could not find a 5D dataset in the HDF5 file.")
        dataset = dataset
        print(f"Using dataset: {dataset.name}")
    else:
        dataset = f[DATASET_KEY]

    theta = dataset[:]   # (Ntraj, Nt, 1, Nz, Nx)

DEDALUS_BUOYANCY_MEAN = 0.4993  # stationary-regime standardisation
DEDALUS_BUOYANCY_STD = 0.2053
theta = theta[:, :, 0] * DEDALUS_BUOYANCY_STD + DEDALUS_BUOYANCY_MEAN   # -> (Ntraj, Nt, Nz, Nx)

Ntraj, Nt, Nz, Nx = theta.shape
print(f"Loaded temperature field with shape {theta.shape}")


# ------------------------------------------------------------
# Grid
#
# Assumes the vertical domain is z/H in [0, 1] and that the
# first and last grid points lie on the walls.
# ------------------------------------------------------------
z = np.linspace(0.0, 1.0, Nz)
dz = z[1] - z[0]


# ============================================================
# 1. Nusselt number as a function of time
# ============================================================
#
# For nondimensional temperature theta and H = Delta theta = 1:
#
#       Nu_wall = - d theta / dz |_wall
#
# Use second-order one-sided finite differences at both walls.
#

# Horizontal mean first:
# shape -> (Ntraj, Nt, Nz)
theta_xmean = theta.mean(axis=-1)

# Bottom-wall derivative
dtheta_dz_bottom = (
    -3.0 * theta_xmean[:, :, 0]
    + 4.0 * theta_xmean[:, :, 1]
    - theta_xmean[:, :, 2]
) / (2.0 * dz)

# Top-wall derivative
dtheta_dz_top = (
    3.0 * theta_xmean[:, :, -1]
    - 4.0 * theta_xmean[:, :, -2]
    + theta_xmean[:, :, -3]
) / (2.0 * dz)

Nu_bottom_each = -dtheta_dz_bottom    # (Ntraj, Nt)
Nu_top_each = -dtheta_dz_top          # (Ntraj, Nt)

# Ensemble means at each physical time
Nu_bottom = Nu_bottom_each.mean(axis=0)
Nu_top = Nu_top_each.mean(axis=0)

# Also useful: average of top and bottom estimates
Nu_mean = 0.5 * (Nu_bottom + Nu_top)


# ============================================================
# 2. Temperature RMS as a function of time
# ============================================================
#
# At each time t, define the mean vertical profile using all
# trajectories and horizontal positions:
#
#   theta_bar(z,t) = <theta(x,z,t)>_{x, trajectories}
#
# Then
#
#   theta'(x,z,t) = theta - theta_bar
#
# and calculate a global RMS over x, z and trajectories.
#

# shape: (Nt, Nz)
theta_mean_profile = theta.mean(axis=(0, 3))

# Broadcast back to (Ntraj, Nt, Nz, Nx)
theta_prime = theta - theta_mean_profile[None, :, :, None]

# Global fluctuation RMS at each time
theta_rms = np.sqrt(np.mean(theta_prime**2, axis=(0, 2, 3)))


# ============================================================
# Plotting
# ============================================================

time = np.arange(Nt) * DT


plt.figure(figsize=(7, 4.5))
plt.plot(time, Nu_bottom, label="Bottom wall")
plt.plot(time, Nu_top, label="Top wall")
plt.plot(time, Nu_mean, label="Wall average", linewidth=2)

plt.xlabel(r"$t/t_f$")
plt.ylabel(r"$Nu$")
plt.title("Nusselt number vs. time")
plt.legend()
plt.tight_layout()
plt.savefig("stationarity_nusselt.png", dpi=300)
plt.show()


plt.figure(figsize=(7, 4.5))
plt.plot(time, theta_rms)

plt.xlabel(r"$t/t_f$")
plt.ylabel(r"$\theta_{\mathrm{rms}}$")
plt.title("Temperature RMS vs. time")
plt.tight_layout()
plt.savefig("stationarity_theta_rms.png", dpi=300)
plt.show()