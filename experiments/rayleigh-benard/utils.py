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
        t_coarsen: int = 1,
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
        self.t_coarsen = t_coarsen
        self.mean = mean
        self.std = std
        self.field = field
        self.interpolate = interpolate
        self._handles: Dict[str, h5py.File] = {}

        dirs = sorted(glob.glob(os.path.join(root, 'traj_*')))
        if not dirs:
            raise FileNotFoundError(f'no traj_* directories under {root}')

        # Per trajectory: ordered (file, n_frames) layout and total length.
        # `window` is expressed in decimated-time units; reading it needs
        # `window * t_coarsen` raw frames.
        need = t_start + (window if window is not None else 1) * t_coarsen
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

        # Common available length across the split (so full trajectories batch),
        # in decimated-time units (raw frames // t_coarsen).
        self.length = (min(total for _, _, total in self.layouts) - t_start) // t_coarsen
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

        # `off`, `L` are in decimated-time units; `ts_raw` (start) and
        # `L * t_coarsen` (length) convert that to the raw frame range to read.
        if self.window is None:
            off, L = 0, self.length
        else:
            off = int(torch.randint(0, self.length - self.window + 1, size=()))
            L = self.window
        ts_raw = self.t_start + off * self.t_coarsen

        crop = np.asarray(
            self._read(files, counts, ts_raw, L * self.t_coarsen),
            dtype=np.float32,
        )  # (L * t_coarsen, x, z)
        if self.t_coarsen > 1:
            # Block-average raw frames down to the decimated time resolution
            # (matches the box-filter spatial coarsening below).
            crop = crop.reshape(L, self.t_coarsen, *crop.shape[1:]).mean(axis=1)
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


class LocalScoreUNet3D(nn.Module):
    r"""3D-U-Net score kernel: time is a genuine (depth) dimension processed with
    3D convolutions, instead of the default :class:`LocalScoreUNet`, which folds
    the temporal window into the channel dimension and uses 2D convolutions.

    The **external interface is identical** to the 2D kernel -- it maps an input
    ``(..., T*C, H, W)`` (the window-major folded window produced by
    ``MCScoreNet.unfold``) to an output of the same shape -- so ``MCScoreNet``'s
    window unfold/fold and every train/inference script work unchanged; the only
    difference is internal. Internally it un-folds the ``T*C`` channels back to a
    ``(N, C, T, H, W)`` volume, runs a 3D U-Net that **downsamples only the two
    spatial dimensions** (``stride=(1, 2, 2)``, so the short time window is never
    reduced), and re-folds. A fixed normalised-height context channel is broadcast
    over ``(T, H, W)`` exactly as in the 2D kernel.

    ``features`` is the per-frame channel count ``C`` (1 for pixel, ``C_z`` for
    latent); ``window`` is the temporal window ``T = 2*order + 1``.
    """

    def __init__(
        self,
        features: int,
        window: int,
        height: int = HEIGHT,
        width: int = WIDTH,
        embedding: int = 64,
        hidden_channels: Sequence[int] = (64, 128, 256),
        hidden_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: int = 3,
        activation: Callable[[], nn.Module] = nn.SiLU,
        **kwargs,
    ):
        super().__init__()

        self.features = features
        self.window = window
        self.embedding = TimeEmbedding(embedding)
        self.network = UNet(
            in_channels=features + 1,            # + normalised-height context channel
            out_channels=features,
            mod_features=embedding,
            hidden_channels=hidden_channels,
            hidden_blocks=hidden_blocks,
            kernel_size=kernel_size,
            stride=(1, 2, 2),                    # downsample H, W only; keep the time window
            activation=activation,
            spatial=3,
            # Zero padding: x is periodic but y has walls (as in the 2D kernel).
        )

        h = torch.linspace(-1, 1, height).reshape(1, 1, 1, height, 1)
        self.register_buffer('height', h.expand(1, 1, window, height, width).clone())

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        H, W = x.shape[-2:]
        lead = x.shape[:-3]                                    # (B,) train / (B, L') compose
        # (..., T*C, H, W) -> (N, C, T, H, W); T*C is window-major (t0c0..t0c_{C-1}, t1c0..).
        v = x.reshape(-1, self.window, self.features, H, W).transpose(1, 2)
        n = v.shape[0]
        hc = self.height.expand(n, -1, -1, -1, -1)            # (N, 1, T, H, W)
        v = torch.cat((v, hc), dim=1)                         # (N, C+1, T, H, W)
        y = self.embedding(t.reshape(-1))                     # scalar t (compose) or (B,) (train)
        v = self.network(v, y)                                # (N, C, T, H, W)
        return v.transpose(1, 2).reshape(*lead, self.window * self.features, H, W)


def make_score(
    window: int = 5,
    coarsen: int = 1,
    embedding: int = 64,
    hidden_channels: Sequence[int] = (64, 128, 256),
    hidden_blocks: Sequence[int] = (3, 3, 3),
    kernel_size: int = 3,
    activation: str = 'SiLU',
    latent_channels: int = 1,
    latent_height: int = None,
    latent_width: int = None,
    arch: str = '2d',
    **absorb,
) -> nn.Module:
    r"""Build the trajectory score network.

    Pixel diffusion (default): ``latent_channels=1`` and no ``latent_*`` grid, so
    the state is single-channel buoyancy on the coarsen-``coarsen`` grid --
    identical to before. Latent diffusion: pass ``latent_channels=C_z`` and the
    latent grid ``latent_height/latent_width`` (recorded in the run's config from
    the VAE), giving a ``C_z``-channel state on the smaller latent grid. The
    height context channel is still a single normalised-height map at whatever
    grid is used.

    ``arch`` selects the score kernel: ``'2d'`` (default) folds the temporal
    window into channels and uses 2D convolutions (:class:`LocalScoreUNet`) --
    unchanged from before, so old runs (whose config has no ``arch`` key) rebuild
    exactly as they did. ``'3d'`` keeps time as a real dimension and uses 3D
    convolutions (:class:`LocalScoreUNet3D`); both kernels share the same
    ``(..., window*C, H, W)`` interface, so ``MCScoreNet``, training and inference
    are identical regardless of the choice.
    """
    if latent_height is not None and latent_width is not None:
        height, width = latent_height, latent_width
    else:
        height, width = HEIGHT // coarsen, WIDTH // coarsen

    score = MCScoreNet(latent_channels, order=window // 2)
    if arch == '3d':
        score.kernel = LocalScoreUNet3D(
            features=latent_channels,
            window=window,
            height=height,
            width=width,
            embedding=embedding,
            hidden_channels=hidden_channels,
            hidden_blocks=hidden_blocks,
            kernel_size=kernel_size,
            activation=ACTIVATIONS[activation],
        )
    elif arch == '2d':
        score.kernel = LocalScoreUNet(
            channels=window * latent_channels,   # window * C (folded temporal window)
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
    else:
        raise ValueError(f"unknown arch {arch!r} (expected '2d' or '3d')")

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


# Below this height a colorbar rendered at the image's native height becomes a
# short, fat box rather than a slim bar (its width is fixed by the tick/label
# text, which does not shrink with the figure). So the colorbar height is
# floored here and the (shorter) image is padded to match, instead of squashing
# the bar down to the image's height.
_CBAR_MIN_HEIGHT = 160


def _colorbar_image(
    vmin: float,
    vmax: float,
    height_px: int,
    label: str = r'$\theta$',
    cmap: str = 'RdBu_r',
    dpi: int = 400,
) -> Image.Image:
    r"""Render a standalone vertical colorbar for the diverging buoyancy/theta
    colormap as a PIL image at least ``_CBAR_MIN_HEIGHT`` px tall, for
    compositing next to ``draw``/``save_gif`` output."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # height_px = max(height_px, _CBAR_MIN_HEIGHT)
    height_px = 75
    fig = plt.figure(figsize=(0.3, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0.25, 0.06, 0.28, 0.88])
    cb = matplotlib.colorbar.ColorbarBase(
        ax, cmap=plt.get_cmap(cmap), norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
    )
    cb.ax.tick_params(labelsize=9)
    cb.set_label(label, fontsize=11)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)

    return Image.fromarray(buf)


def _append_colorbar(img: Image.Image, cbar: Image.Image) -> Image.Image:
    r"""Paste ``img`` and ``cbar`` side by side, vertically centering whichever
    is shorter (the colorbar has a minimum height, so small frames are padded
    rather than the colorbar being squashed to their height)."""
    h = max(img.height, cbar.height)
    combined = Image.new('RGB', (img.width + cbar.width, h), color=(255, 255, 255))
    combined.paste(img, (0, (h - img.height) // 2))
    combined.paste(cbar, (img.width, (h - cbar.height) // 2))
    return combined


def draw(
    w: ArrayLike,
    mask: ArrayLike = None,
    pad: int = 4,
    zoom: int = 1,
    cbar_label: str = r'$\theta$',
    **kwargs,
) -> Image.Image:
    r"""Tile a grid of buoyancy frames into a single image (cf. kolmogorov.draw).

    An optional boolean ``mask`` greys out the unobserved cells, which is handy
    for displaying sparse observations. A colorbar for theta (nondimensional
    temperature) is appended on the right unless ``colorbar=False``.
    """
    vmin = kwargs.get('vmin', -2.0)
    vmax = kwargs.get('vmax', 2.0)

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
    colorbar: bool = False,
    cbar_label: str = r'$\theta$',
    **kwargs,
) -> None:
    r"""Save a buoyancy trajectory ``(L, H, W)`` as an animated GIF. A (static)
    colorbar for theta (nondimensional temperature) is appended on the right of
    every frame unless ``colorbar=False``."""
    vmin = kwargs.get('vmin', -2.0)
    vmax = kwargs.get('vmax', 2.0)

    w = buoyancy2rgb(w, **kwargs)
    imgs = [Image.fromarray(img) for img in w]

    if colorbar and imgs:
        cbar = _colorbar_image(vmin, vmax, imgs[0].height, label=cbar_label)
        imgs = [_append_colorbar(im, cbar) for im in imgs]

    imgs[0].save(
        file,
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 * dt),
        loop=0,
    )
