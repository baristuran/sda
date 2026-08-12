#!/usr/bin/env python
r"""Per-frame convolutional VAE for latent-diffusion Rayleigh-Bénard.

The pixel diffusion in this experiment operates on standardized buoyancy frames
``(1, H, W)`` with ``H, W = HEIGHT//coarsen, WIDTH//coarsen`` (64x256 at
coarsen=2). Latent diffusion instead runs on a compressed latent
``(C_z, H_l, W_l)`` produced by this VAE, applied **independently per frame** so
the entire temporal-window score architecture (``MCScoreNet`` +
``LocalScoreUNet``) carries over unchanged, only with more channels and a smaller
grid.

The encoder/decoder are built from the same primitives as the score U-Net
(``sda/nn.py``): a ``LayerNorm -> Conv -> act -> Conv`` residual block, strided
convolutions to downsample, ``Upsample(nearest)+Conv`` to upsample, zero padding
(the vertical has walls, the horizontal is periodic but we keep zero padding to
match ``make_score``). As in ``LocalScoreUNet`` a fixed normalized-height context
channel is concatenated to the encoder input (and, at latent resolution, to the
decoder input) so the VAE can respect the non-homogeneous vertical.

Latent grid: ``len(hidden_channels) - 1`` stride-2 stages, i.e. downsample factor
``2**(len(hidden_channels)-1)``. With ``hidden_channels`` of length 4 this is 8x
(64x256 -> 8x32).

Helpers:
- ``make_vae(**config)`` / ``load_vae(run, device)`` mirror ``make_score`` /
  ``load_score`` in ``utils.py``.
- ``load_decoder(diffusion_config, device)`` returns ``(decode_fn, C, H_m, W_m)``:
  if the diffusion run references a VAE (``vae_run`` key) it loads it and returns
  a latent->standardized-pixel decoder plus the latent shape; otherwise it
  returns the identity and the pixel shape. This is the single backward-compat
  switch used by every inference / DAPS script.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pathlib import Path
from torch import Tensor
from typing import Sequence, Tuple

from zuko.nn import LayerNorm

from sda.nn import ResidualBlock
from sda.utils import ACTIVATIONS, load_config

from utils import HEIGHT, WIDTH, PATH


def _conv(in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=kernel_size // 2)


def _resblock(channels: int, kernel_size: int, activation) -> ResidualBlock:
    r"""``x + Conv(act(Conv(LayerNorm(x))))`` -- the unmodulated analogue of the
    score U-Net's residue stack (``sda/nn.py`` block lambda)."""
    return ResidualBlock(
        LayerNorm(-3),                                  # over (C, H, W), spatial=2
        _conv(channels, channels, kernel_size),
        activation(),
        _conv(channels, channels, kernel_size),
    )


class ConvVAE(nn.Module):
    r"""Per-frame 2D convolutional VAE: ``(1, H, W) <-> (C_z, H_l, W_l)``."""

    def __init__(
        self,
        latent_channels: int = 4,
        hidden_channels: Sequence[int] = (64, 128, 256, 256),
        hidden_blocks: Sequence[int] = (2, 2, 2, 2),
        kernel_size: int = 3,
        activation: str = 'SiLU',
        height_context: bool = True,
        height: int = HEIGHT // 2,
        width: int = WIDTH // 2,
    ):
        super().__init__()

        hc = tuple(hidden_channels)
        hb = tuple(hidden_blocks)
        assert len(hc) == len(hb), 'hidden_channels and hidden_blocks must align'
        n_levels = len(hc)
        n_down = n_levels - 1
        act = ACTIVATIONS[activation]

        self.latent_channels = latent_channels
        self.height_context = height_context
        self.pixel_height, self.pixel_width = height, width
        self.latent_height = height // (2 ** n_down)
        self.latent_width = width // (2 ** n_down)

        in_c = 1 + (1 if height_context else 0)
        dec_in_c = latent_channels + (1 if height_context else 0)

        # -- encoder: full-res -> latent-res, channels grow with depth ---------
        enc = [_conv(in_c, hc[0], kernel_size)]
        for i in range(n_levels):
            enc += [_resblock(hc[i], kernel_size, act) for _ in range(hb[i])]
            if i < n_down:
                enc.append(_conv(hc[i], hc[i + 1], kernel_size, stride=2))
        self.encoder = nn.Sequential(*enc)
        self.to_moments = _conv(hc[-1], 2 * latent_channels, kernel_size)   # (mu, logvar)

        # -- decoder: latent-res -> full-res, mirror of the encoder ------------
        self.from_latent = _conv(dec_in_c, hc[-1], kernel_size)
        dec = []
        for i in reversed(range(n_levels)):
            dec += [_resblock(hc[i], kernel_size, act) for _ in range(hb[i])]
            if i > 0:
                dec.append(nn.Upsample(scale_factor=2, mode='nearest'))
                dec.append(_conv(hc[i], hc[i - 1], kernel_size))
        self.decoder = nn.Sequential(*dec)
        self.to_pixels = _conv(hc[0], 1, kernel_size)

        # -- fixed normalized-height context channels (encoder & decoder) ------
        if height_context:
            he = torch.linspace(-1, 1, height).reshape(1, 1, height, 1)
            self.register_buffer('height_enc', he.expand(1, 1, height, width).clone())
            hl = torch.linspace(-1, 1, self.latent_height).reshape(1, 1, self.latent_height, 1)
            self.register_buffer(
                'height_dec', hl.expand(1, 1, self.latent_height, self.latent_width).clone())

    # -- core ----------------------------------------------------------------
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        r"""``x`` (B, 1, H, W) -> ``(mu, logvar)`` each (B, C_z, H_l, W_l)."""
        if self.height_context:
            x = torch.cat((x, self.height_enc.expand(x.shape[0], -1, -1, -1)), dim=1)
        h = self.encoder(x)
        mu, logvar = self.to_moments(h).chunk(2, dim=1)
        return mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        r"""``z`` (B, C_z, H_l, W_l) -> ``x`` (B, 1, H, W) (standardized pixel)."""
        if self.height_context:
            z = torch.cat((z, self.height_dec.expand(z.shape[0], -1, -1, -1)), dim=1)
        h = self.from_latent(z)
        h = self.decoder(h)
        return self.to_pixels(h)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x: Tensor, beta: float = 1e-4) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Reconstruction (per-pixel MSE) + ``beta`` * KL (per-sample, summed over
        latent dims then averaged over the batch). ``beta`` is small so the KL
        penalty does not swamp reconstruction. Returns ``(total, recon, kl)``."""
        recon, mu, logvar = self.forward(x)
        recon_loss = (recon - x).square().mean()
        kl = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1.0).flatten(1).sum(dim=1).mean()
        return recon_loss + beta * kl, recon_loss, kl

    # -- batched frame helpers (accept arbitrary leading dims) ---------------
    def encode_frames(self, x: Tensor, sample: bool = False) -> Tensor:
        r"""``x`` (..., 1, H, W) -> latent (..., C_z, H_l, W_l). Deterministic
        (mean) by default; set ``sample=True`` to draw from the posterior."""
        lead = x.shape[:-3]
        mu, logvar = self.encode(x.reshape(-1, *x.shape[-3:]))
        # z = self.reparameterize(mu, logvar) if sample else mu
        # return z.reshape(*lead, *z.shape[-3:])
        return mu.reshape(*lead, *mu.shape[-3:])

    def decode_frames(self, z: Tensor) -> Tensor:
        r"""``z`` (..., C_z, H_l, W_l) -> standardized pixel (..., 1, H, W).
        Differentiable w.r.t. ``z`` (used inside DAPS/DPS guidance)."""
        lead = z.shape[:-3]
        x = self.decode(z.reshape(-1, *z.shape[-3:]))
        return x.reshape(*lead, *x.shape[-3:])


# ------------------------------------------------------------------ builders
def make_vae(
    latent_channels: int = 4,
    hidden_channels: Sequence[int] = (64, 128, 256, 256),
    hidden_blocks: Sequence[int] = (2, 2, 2, 2),
    kernel_size: int = 3,
    activation: str = 'SiLU',
    height_context: bool = True,
    coarsen: int = 2,
    **absorb,
) -> ConvVAE:
    r"""Build a :class:`ConvVAE` for the coarsen-``coarsen`` pixel grid. Extra
    config keys (epochs, beta, ...) are swallowed, mirroring ``make_score``."""
    height, width = HEIGHT // coarsen, WIDTH // coarsen
    return ConvVAE(
        latent_channels=latent_channels,
        hidden_channels=hidden_channels,
        hidden_blocks=hidden_blocks,
        kernel_size=kernel_size,
        activation=activation,
        height_context=height_context,
        height=height,
        width=width,
    )


def load_vae(run: Path, device: str = 'cpu', freeze: bool = True, **kwargs) -> ConvVAE:
    r"""Rebuild a VAE from ``run/config.json`` and load ``run/state.pth`` (mirrors
    ``utils.load_score``). Frozen and in eval mode by default -- for inference the
    VAE is fixed; gradients through the decoder are taken w.r.t. its *input*."""
    run = Path(run)
    config = load_config(run)
    config.update(kwargs)
    vae = make_vae(**config)
    vae.load_state_dict(torch.load(run / 'ckpt_epoch0045.pth', map_location=device))
    vae = vae.to(device).eval()
    if freeze:
        for p in vae.parameters():
            p.requires_grad_(False)
    return vae


def _resolve_vae_run(vae_run) -> Path:
    r"""Accept a full path or a bare run name under ``PATH/runs_vae``."""
    p = Path(vae_run)
    if p.exists():
        return p
    return PATH / 'runs_vae' / str(vae_run)


def load_decoder(diffusion_config: dict, device: str = 'cpu'):
    r"""Return ``(decode_fn, C, H_m, W_m)`` for a diffusion run.

    - If ``diffusion_config`` has a truthy ``vae_run`` -> load that VAE and return
      a ``decode_fn`` mapping the diffusion state (...,C_z,H_l,W_l) to standardized
      pixels (...,1,H,W), plus the latent event shape ``(C_z, H_l, W_l)``. When the
      config also carries per-channel latent stats (``latent_mean``/``latent_std``,
      written by ``train.py``), the diffusion operates on *standardized* latents, so
      ``decode_fn`` first un-standardizes ``z * std + mean`` before the VAE decoder
      (autograd flows through it for guidance). Absent stats -> raw-latent decode.
    - Otherwise -> identity decode and the pixel shape ``(1, H, W)``.

    This is the single latent-vs-pixel switch; older pixel runs (no ``vae_run``)
    get the identity path and behave exactly as before.
    """
    vae_run = diffusion_config.get('vae_run', None)
    if vae_run:
        vae = load_vae(_resolve_vae_run(vae_run), device)
        mean, std = diffusion_config.get('latent_mean'), diffusion_config.get('latent_std')
        if mean is not None and std is not None:
            m = torch.tensor(mean, dtype=torch.float32, device=device).reshape(-1, 1, 1)
            s = torch.tensor(std, dtype=torch.float32, device=device).reshape(-1, 1, 1)
            decode = lambda z: vae.decode_frames(z * s + m)     # un-standardize, then decode
        else:
            decode = vae.decode_frames
        return decode, vae.latent_channels, vae.latent_height, vae.latent_width
    coarsen = diffusion_config.get('coarsen', 1)
    identity = lambda x: x
    return identity, 1, HEIGHT // coarsen, WIDTH // coarsen
