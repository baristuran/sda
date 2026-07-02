r"""Rayleigh-Bénard experiment helpers.

Trains an SDA score network on 2D Rayleigh-Bénard convection trajectories from
the Well dataset, restricted to Rayleigh number ``1e9``. The buoyancy field is
modelled (single channel). The installed Well HDF5 files are read directly, so
no data-generation step is required.
"""

import glob
import os

import h5py
import numpy as np
import torch

from numpy.typing import ArrayLike
from pathlib import Path
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from typing import *

from sda.mcs import *
from sda.score import *
from sda.utils import *


if 'SCRATCH' in os.environ:
    SCRATCH = os.environ['SCRATCH']
    PATH = Path(SCRATCH) / 'sda/rayleigh-benard'
else:
    PATH = Path('.')

PATH.mkdir(parents=True, exist_ok=True)


# -- Dataset selection / preprocessing -------------------------------------
DATASET_ROOT = os.environ.get('RBC_DATASET_ROOT', '/hdd/rbc-dataset')
RAYLEIGH = 1e9          # use only the Ra = 1e9 trajectories
PRANDTL = 1.0          # None -> all Prandtl numbers at Ra=1e9; or fix to a float
T_START = 120           # discard the initial transient (steps 0 .. T_START-1)

# Snapshots are stored as (x=512, y=128); we orient them as (H=y, W=x) so the
# vertical (gravity) direction is the first spatial axis.
HEIGHT, WIDTH = 128, 512

# Standardisation of the buoyancy field over the Ra=1e9 stationary regime.
BUOYANCY_MEAN = 0.3672
BUOYANCY_STD = 0.1631


def _matches(path: str, rayleigh: float, prandtl: Optional[float]) -> bool:
    with h5py.File(path, 'r') as f:
        ra = float(f['scalars/Rayleigh'][()])
        pr = float(f['scalars/Prandtl'][()])
    if abs(ra - rayleigh) > 1e-3 * rayleigh:
        return False
    if prandtl is not None and abs(pr - prandtl) > 1e-3 * prandtl:
        return False
    return True


class RBCDataset(Dataset):
    r"""Rayleigh-Bénard buoyancy trajectories (Ra=1e9) from the Well dataset.

    Mirrors :class:`sda.utils.TrajectoryDataset`: each item is a standardised
    trajectory ``(L, C, H, W)`` (a random ``window``-length slice when ``window``
    is given), flattened to ``(window * C, H, W)`` when ``flatten`` is set. The
    initial transient is discarded and the Well HDF5 files are read lazily, so no
    on-disk copy of the data is created.
    """

    def __init__(
        self,
        split: str = 'train',
        window: int = None,
        flatten: bool = False,
        rayleigh: float = RAYLEIGH,
        prandtl: Optional[float] = PRANDTL,
        t_start: int = T_START,
        coarsen: int = 1,
        mean: float = BUOYANCY_MEAN,
        std: float = BUOYANCY_STD,
        dataset_root: str = DATASET_ROOT,
    ):
        super().__init__()

        files = sorted(glob.glob(os.path.join(dataset_root, split, '*.hdf5')))
        self.files = [p for p in files if _matches(p, rayleigh, prandtl)]
        if not self.files:
            raise FileNotFoundError(
                f"no Ra={rayleigh:g} (Pr={prandtl}) files under "
                f"{os.path.join(dataset_root, split)}")

        self.window = window
        self.flatten = flatten
        self.t_start = t_start
        self.coarsen = coarsen
        self.mean = mean
        self.std = std
        self._handles: Dict[int, h5py.File] = {}

        # Index one entry per (file, trajectory); all files share the time length.
        self.index: List[Tuple[int, int]] = []
        with h5py.File(self.files[0], 'r') as f:
            n_time = f['t0_fields/buoyancy'].shape[1]
        self.length = n_time - t_start
        if window is not None and window > self.length:
            raise ValueError(f"window={window} exceeds available length {self.length}")

        for fi, path in enumerate(self.files):
            with h5py.File(path, 'r') as f:
                n_traj = f['t0_fields/buoyancy'].shape[0]
            self.index.extend((fi, tr) for tr in range(n_traj))

    def _handle(self, fi: int) -> h5py.File:
        h = self._handles.get(fi)
        if h is None:
            h = h5py.File(self.files[fi], 'r')
            self._handles[fi] = h
        return h

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Tuple[Tensor, Dict]:
        fi, tr = self.index[i]
        b = self._handle(fi)['t0_fields/buoyancy']     # (n_traj, time, x, y)

        if self.window is None:
            ts, L = self.t_start, self.length
        else:
            off = int(torch.randint(0, self.length - self.window + 1, size=()))
            ts, L = self.t_start + off, self.window

        crop = np.asarray(b[tr, ts : ts + L], dtype=np.float32)   # (L, x, y)
        x = torch.from_numpy(crop).movedim(1, 2)                  # (L, y, x) = (L, H, W)
        x = (x - self.mean) / self.std
        x = x.unsqueeze(1)                                        # (L, C=1, H, W)

        if self.coarsen > 1:
            x = torch.nn.functional.avg_pool2d(x, self.coarsen)

        if self.flatten:
            return x.flatten(0, 1), {}
        return x, {}


def destandardize(x: ArrayLike) -> ArrayLike:
    r"""Map a standardised buoyancy field back to physical units."""
    return np.asarray(x) * BUOYANCY_STD + BUOYANCY_MEAN


# -- Locally-simulated Dedalus dataset -------------------------------------
DEDALUS_ROOT = os.environ.get('DEDALUS_ROOT', '/home/baris-turan/dedalus/snapshots')
DEDALUS_T_START = 80            # discard the initial transient (frames 0 .. T_START-1)
DEDALUS_BUOYANCY_MEAN = 0.4993  # stationary-regime standardisation
DEDALUS_BUOYANCY_STD = 0.2053
DEDALUS_SPLIT = {'train': (0.0, 0.8), 'valid': (0.8, 0.9), 'test': (0.9, 1.0)}
DEDALUS_LZ = 1.0               # box height (z in [0, Lz]); must match the solver


def _set_index(path: str) -> int:
    r"""'.../traj_0003_s2.h5' -> 2 (Dedalus snapshot-set ordering)."""
    return int(path.rsplit('_s', 1)[-1].split('.')[0])


def chebyshev_to_uniform_matrix(z_src: ArrayLike, Lz: float = DEDALUS_LZ):
    r"""Build the matrix that maps values on the (non-uniform) Chebyshev z-grid
    to a uniform z-grid of the same size.

    The Dedalus solver runs on a Chebyshev grid in z (clustered at the walls); a
    Chebyshev field is exactly the polynomial interpolating its values at those
    nodes, so barycentric Lagrange interpolation through the stored nodes
    reproduces Dedalus' own spectral interpolation to machine precision -- with
    no Dedalus dependency. Apply along the last (z) axis as ``data @ M.T``.
    """
    z_src = np.asarray(z_src, dtype=np.float64)
    n = len(z_src)
    z_dst = np.linspace(0.0, Lz, n)

    # Capacity-scaled barycentric weights (Berrut & Trefethen) to avoid overflow.
    C = (z_src.max() - z_src.min()) / 4.0
    w = np.ones(n)
    for j in range(n):
        d = (z_src[j] - z_src) / C
        d[j] = 1.0
        w[j] = 1.0 / np.prod(d)

    M = np.zeros((n, n))
    for i, x in enumerate(z_dst):
        diff = x - z_src
        hit = np.isclose(diff, 0.0, atol=1e-12)
        if hit.any():
            M[i, int(np.argmax(hit))] = 1.0
        else:
            t = w / diff
            M[i] = t / t.sum()

    return M.astype(np.float32), z_dst


class DedalusRBCDataset(Dataset):
    r"""Rayleigh-Bénard buoyancy trajectories simulated locally with Dedalus.

    The data lives under ``root`` as one directory per trajectory
    (``traj_XXXX/``), each split into ordered Dedalus snapshot sets
    (``traj_XXXX_sN.h5``, 50 frames each) that concatenate along time into a
    full ~200-step trajectory. We model the ``tasks/buoyancy`` field, oriented as
    ``(H=z, W=x)`` to match :class:`RBCDataset`, discard the initial transient
    and standardise. Trajectories are split into train/valid/test by index.

    Same interface as :class:`RBCDataset` / :class:`sda.utils.TrajectoryDataset`:
    each item is a standardised ``(L, C, H, W)`` trajectory (a random
    ``window``-length slice when ``window`` is given), flattened to
    ``(window * C, H, W)`` when ``flatten`` is set. Windows straddling two
    snapshot files are read transparently across the boundary. Files are read
    lazily and trajectories that are still being written (too short) are skipped,
    so the dataset can be used while it keeps growing.
    """

    def __init__(
        self,
        split: str = 'train',
        window: int = None,
        flatten: bool = False,
        t_start: int = DEDALUS_T_START,
        coarsen: int = 1,
        mean: float = DEDALUS_BUOYANCY_MEAN,
        std: float = DEDALUS_BUOYANCY_STD,
        root: str = DEDALUS_ROOT,
        field: str = 'buoyancy',
        interpolate: bool = False,
        Lz: float = DEDALUS_LZ,
    ):
        super().__init__()
        self.window = window
        self.flatten = flatten
        self.t_start = t_start
        self.coarsen = coarsen
        self.mean = mean
        self.std = std
        self.field = field
        self.interpolate = interpolate
        self._handles: Dict[str, h5py.File] = {}

        dirs = sorted(glob.glob(os.path.join(root, 'traj_*')))
        if not dirs:
            raise FileNotFoundError(f'no traj_* directories under {root}')

        # Per trajectory: ordered (file, n_frames) layout and total length.
        need = t_start + (window if window is not None else 1)
        layouts = []
        for d in dirs:
            files = sorted(glob.glob(os.path.join(d, '*_s*.h5')), key=_set_index)
            if not files:
                continue
            try:
                counts = [h5py.File(f, 'r')[f'tasks/{field}'].shape[0] for f in files]
            except (OSError, KeyError):
                continue  # skip trajectories that are mid-write / unreadable
            total = sum(counts)
            if total < need:
                continue  # skip too-short / still-spinning-up trajectories
            layouts.append((files, counts, total))

        if not layouts:
            raise RuntimeError(f'no usable trajectories under {root} (need >= {need} frames)')

        # Split by trajectory index.
        lo, hi = DEDALUS_SPLIT[split]
        i, j = int(lo * len(layouts)), int(hi * len(layouts))
        self.layouts = layouts[i:j]
        if not self.layouts:
            raise RuntimeError(
                f'empty {split} split ({len(layouts)} usable trajectories total)')

        # Common available length across the split (so full trajectories batch).
        self.length = min(total for _, _, total in self.layouts) - t_start
        if window is not None and window > self.length:
            raise ValueError(f"window={window} exceeds available length {self.length}")

        # Chebyshev -> uniform z interpolation operator (applied along the z axis).
        self._Mt = None
        if interpolate:
            with h5py.File(self.layouts[0][0][0], 'r') as f:
                zname = next(k for k in f['scales'] if k.startswith('z_hash'))
                z_src = f['scales'][zname][:]
            M, _ = chebyshev_to_uniform_matrix(z_src, Lz)
            self._Mt = np.ascontiguousarray(M.T)        # (z_src, z_uniform)

    def _handle(self, path: str) -> h5py.File:
        h = self._handles.get(path)
        if h is None:
            h = h5py.File(path, 'r')
            self._handles[path] = h
        return h

    def _read(self, files, counts, start: int, length: int) -> np.ndarray:
        r"""Read the global frame range ``[start, start + length)`` across the
        ordered snapshot files of a trajectory."""
        out, pos = [], 0
        for path, n in zip(files, counts):
            a, b = max(start, pos), min(start + length, pos + n)
            if a < b:
                out.append(self._handle(path)[f'tasks/{self.field}'][a - pos : b - pos])
            pos += n
        return np.concatenate(out, axis=0)

    def __len__(self) -> int:
        return len(self.layouts)

    def __getitem__(self, i: int) -> Tuple[Tensor, Dict]:
        files, counts, total = self.layouts[i]

        if self.window is None:
            ts, L = self.t_start, self.length
        else:
            off = int(torch.randint(0, self.length - self.window + 1, size=()))
            ts, L = self.t_start + off, self.window

        crop = np.asarray(self._read(files, counts, ts, L), dtype=np.float32)  # (L, x, z)
        if self._Mt is not None:
            crop = crop @ self._Mt                                # interpolate z -> uniform
        x = torch.from_numpy(crop).movedim(1, 2)                  # (L, z, x) = (L, H, W)
        x = (x - self.mean) / self.std
        x = x.unsqueeze(1)                                        # (L, C=1, H, W)

        if self.coarsen > 1:
            x = torch.nn.functional.avg_pool2d(x, self.coarsen)

        if self.flatten:
            return x.flatten(0, 1), {}
        return x, {}


# -- Score network ---------------------------------------------------------
class LocalScoreUNet(ScoreUNet):
    r"""Score U-Net with a fixed vertical-coordinate (height) context channel.

    Rayleigh-Bénard convection is not translation-invariant in the vertical: the
    bottom plate is hot and the top is cold. Feeding the normalised height as a
    context channel (analogous to the Kolmogorov forcing channel) lets the local
    score break that symmetry.
    """

    def __init__(self, channels: int, height: int = HEIGHT, width: int = WIDTH, **kwargs):
        super().__init__(channels, 1, **kwargs)

        h = torch.linspace(-1, 1, height).reshape(1, height, 1)
        self.register_buffer('height', h.expand(1, height, width).clone())

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        return super().forward(x, t, self.height)


def make_score(
    window: int = 5,
    coarsen: int = 1,
    embedding: int = 64,
    hidden_channels: Sequence[int] = (64, 128, 256),
    hidden_blocks: Sequence[int] = (3, 3, 3),
    kernel_size: int = 3,
    activation: str = 'SiLU',
    **absorb,
) -> nn.Module:
    height, width = HEIGHT // coarsen, WIDTH // coarsen

    score = MCScoreNet(1, order=window // 2)
    score.kernel = LocalScoreUNet(
        channels=window,            # window * (1 buoyancy channel)
        height=height,
        width=width,
        embedding=embedding,
        hidden_channels=hidden_channels,
        hidden_blocks=hidden_blocks,
        kernel_size=kernel_size,
        activation=ACTIVATIONS[activation],
        spatial=2,
        # No circular padding: x is periodic but y has walls, so we use the
        # default zero padding rather than wrapping the vertical boundaries.
    )

    return score


def load_score(file: Path, device: str = 'cpu', **kwargs) -> nn.Module:
    state = torch.load(file, map_location=device)
    config = load_config(file.parent)
    config.update(kwargs)

    score = make_score(**config)
    score.load_state_dict(state)

    return score


# -- Visualisation ---------------------------------------------------------
def buoyancy2rgb(b: ArrayLike, vmin: float = -2.0, vmax: float = 2.0) -> ArrayLike:
    r"""Map a (standardised) buoyancy field to RGB with a diverging colormap."""
    import matplotlib.cm as cm

    b = np.asarray(b)
    b = (b - vmin) / (vmax - vmin)
    b = np.clip(b, 0.0, 1.0)
    b = cm.get_cmap('RdBu_r')(b)
    return (256 * b[..., :3]).astype(np.uint8)


def draw(
    w: ArrayLike,
    mask: ArrayLike = None,
    pad: int = 4,
    zoom: int = 1,
    **kwargs,
) -> Image.Image:
    r"""Tile a grid of buoyancy frames into a single image (cf. kolmogorov.draw).

    An optional boolean ``mask`` greys out the unobserved cells, which is handy
    for displaying sparse observations.
    """
    w = buoyancy2rgb(w, **kwargs)
    w = w[(None,) * (5 - w.ndim)]

    M, N, H, W, _ = w.shape

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        mask = mask[(None,) * (4 - mask.ndim)]

    img = Image.new(
        'RGB',
        size=(N * (W + pad) + pad, M * (H + pad) + pad),
        color=(255, 255, 255),
    )

    for i in range(M):
        for j in range(N):
            offset = (j * (W + pad) + pad, i * (H + pad) + pad)
            img.paste(Image.fromarray(w[i][j]), offset)

            if mask is not None:
                img.paste(
                    Image.new('L', size=(W, H), color=240),
                    offset,
                    Image.fromarray(~mask[i][j]),
                )

    if zoom > 1:
        return img.resize((img.width * zoom, img.height * zoom), resample=0)
    return img


def save_gif(
    w: ArrayLike,
    file: Path,
    dt: float = 0.2,
    **kwargs,
) -> None:
    r"""Save a buoyancy trajectory ``(L, H, W)`` as an animated GIF."""
    w = buoyancy2rgb(w, **kwargs)

    imgs = [Image.fromarray(img) for img in w]
    imgs[0].save(
        file,
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 * dt),
        loop=0,
    )
