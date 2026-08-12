#!/usr/bin/env python
r"""Train the per-frame VAE (``vae.ConvVAE``) for latent-diffusion Rayleigh-Bénard.

Trains on the standardized pixel frames in ``data/{train,valid}.h5`` (the same
consolidated files the pixel diffusion uses, key ``x``, shape
``(n, L, 1, H, W)``), treating every frame ``(1, H, W)`` as an independent
sample. Loss = per-pixel MSE reconstruction + ``beta`` * KL, with ``beta`` small
(default 1e-4) so the KL penalty does not swamp reconstruction.

Saves to ``runs_vae/<name>/{config.json, state.pth}`` (the same layout as the
diffusion ``runs/``), so ``vae.load_vae`` / ``vae.load_decoder`` can rebuild it,
and ``encode_latents.py`` / ``train.py`` (with ``vae_run=<name>``) can consume it.

    python train_vae.py --epochs 100 --batch-size 64 --beta 1e-4
    python train_vae.py --amp --subsample 4          # ~faster: fp16 + 1/4 frames
    python train_vae.py --resume runs_vae/vae_big    # continue that run

Multi-GPU uses DistributedDataParallel via ``torchrun`` (one process per card;
``batch-size`` is then *per-GPU*, so the global batch is ``batch-size * nproc``):

    torchrun --nproc_per_node=3 train_vae.py --gpus 0,1,2 --batch-size 128 --name vae_big

Only rank 0 logs, saves checkpoints and writes the reconstruction snapshot; the
script disables GPU P2P (``NCCL_P2P_DISABLE``) automatically since it deadlocks on
this host. ``--resume`` works under ``torchrun`` too (every rank loads the same
checkpoint).

Levers (see the training-cost note in the plan): ``--amp`` (fp16 autocast,
~1.5-2x), ``--subsample N`` (use every N-th frame per trajectory; adjacent frames
are highly correlated), ``--gpu`` to pick one card, ``--resume`` to restart.
"""

from __future__ import annotations

import argparse
import os
import sys

# Pick the GPU(s) before importing torch (so cuda:0 maps to the first one).
# `--gpu 1` picks one card; `--gpus 0,1,2` exposes several for multi-GPU DDP
# (launch with `torchrun --nproc_per_node=3 ... --gpus 0,1,2`). Under torchrun
# (LOCAL_RANK set) with no explicit flag, leave every GPU visible so each rank
# can bind its own card.
if '--gpus' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[sys.argv.index('--gpus') + 1]
elif '--gpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[sys.argv.index('--gpu') + 1]
elif 'LOCAL_RANK' not in os.environ:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', os.environ.get('RBC_GPU', '0'))

# This host's GPU peer-to-peer access deadlocks NCCL/DataParallel collectives, so
# disable P2P by default (overridable). Harmless for single-GPU runs.
os.environ.setdefault('NCCL_P2P_DISABLE', '1')

import math
import time

import h5py
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from sda.utils import save_config

from utils import PATH, draw
from vae import ConvVAE, make_vae


CONFIG = {
    # Architecture (must round-trip through vae.make_vae)
    'latent_channels': 4,
    # 'hidden_channels': (64, 128, 256, 256),   # 4 levels -> 3 downsamples -> 8x
    # 'hidden_blocks': (2, 2, 2, 2),
    'hidden_channels': (64, 128, 128),   # 3 levels -> 2 downsamples -> 4x
    'hidden_blocks': (4, 4, 4),
    'kernel_size': 3,
    'activation': 'SiLU',
    'height_context': True,
    'coarsen': 2,                              # must match the data grid (64x256)
    # Training
    'beta': 1e-5,
    'epochs': 100,
    'batch_size': 128,
    'learning_rate': 2e-4,
    'weight_decay': 1e-3,
    'num_workers': 4,
    'subsample': 4,                            # keep every N-th frame per trajectory
    'amp': False,
}


class FrameDataset(Dataset):
    r"""Every frame of ``data/<split>.h5`` (key ``x``, shape ``(n,L,1,H,W)``) as an
    independent standardized sample ``(1,H,W)``. ``subsample`` keeps every N-th
    frame along time (adjacent frames are highly correlated), loaded into memory."""

    def __init__(self, file: Path, subsample: int = 1):
        with h5py.File(file, 'r') as f:
            data = f['x'][:, ::subsample]          # strided read, (n, L', 1, H, W)
        self.data = torch.from_numpy(np.ascontiguousarray(data)).float()
        self.n, self.L = self.data.shape[:2]

    def __len__(self) -> int:
        return self.n * self.L

    def __getitem__(self, i: int) -> torch.Tensor:
        t, l = divmod(i, self.L)
        return self.data[t, l]                     # (1, H, W)


def _lr_lambda(epochs: int):
    # 'exponential' schedule, matching sda.utils.loop.
    return lambda t: math.exp(-7.0 * (t / epochs) ** 2)


class VAELoss(nn.Module):
    r"""Compute the VAE losses inside ``forward`` so the module can be wrapped in
    ``DistributedDataParallel`` (whose gradient all-reduce hooks fire on the
    backward of *its* forward). Returns ``(total, recon, kl)`` for the rank's
    batch; ``total`` is the scalar that gets ``.backward()``."""

    def __init__(self, vae: ConvVAE, beta: float):
        super().__init__()
        self.vae = vae
        self.beta = beta

    def forward(self, x: torch.Tensor):
        return self.vae.loss(x, self.beta)


def _ddp_info():
    r"""``(ddp, world_size, rank, local_rank)`` from the env torchrun sets."""
    world = int(os.environ.get('WORLD_SIZE', 1))
    if world > 1:
        return True, world, int(os.environ.get('RANK', 0)), int(os.environ.get('LOCAL_RANK', 0))
    return False, 1, 0, 0


def _run_epoch(loss_fn, module, loader, device, opt=None, scaler=None, amp=False, desc=None):
    train = opt is not None
    module.train(train)
    tot = rec = kl = 0.0
    nb = 0
    bar = tqdm(loader, desc=desc, leave=False, ncols=100) if desc is not None else loader
    for x in bar:
        x = x.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast('cuda', enabled=amp):
            loss, r, k = loss_fn(x)
        if train:
            opt.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
        tot += float(loss); rec += float(r); kl += float(k); nb += 1
        if desc is not None:
            bar.set_postfix(rec=f'{rec / nb:.4f}', kl=f'{kl / nb:.1f}')
    return tot / nb, rec / nb, kl / nb


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, val in CONFIG.items():
        if isinstance(val, bool):
            p.add_argument(f'--{key.replace("_", "-")}', action='store_true', default=val)
        elif isinstance(val, (tuple, list)):
            p.add_argument(f'--{key.replace("_", "-")}', type=int, nargs='+', default=list(val))
        else:
            p.add_argument(f'--{key.replace("_", "-")}', type=type(val), default=val)
    p.add_argument('--name', default=None, help='run name (default: vae_<timestamp>)')
    p.add_argument('--gpu', default=None, help='single CUDA device index (sets CUDA_VISIBLE_DEVICES)')
    p.add_argument('--gpus', default=None,
                   help='comma-separated CUDA devices to expose for torchrun DDP, e.g. 0,1,2')
    p.add_argument('--resume', default=None,
                   help='resume from a checkpoint: a run dir (uses its last.pth) or a .pth file')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--valid-batches', type=int, default=40, help='cap validation batches/epoch')
    p.add_argument('--ckpt-every', type=int, default=5,
                   help='also save ckpt_epoch<N>.pth every N epochs (0 disables)')
    args = p.parse_args()

    config = {k: (tuple(v) if isinstance(v, list) else v)
              for k, v in vars(args).items()
              if k in CONFIG}

    # Distributed (launched via torchrun): one process per GPU. Rank 0 is the only
    # one that logs, validates, saves and draws; every rank trains its data shard
    # and DDP all-reduces the gradients on each backward.
    ddp, world, rank, local_rank = _ddp_info()
    is_main = rank == 0
    if ddp:
        import torch.distributed as dist
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
        device = f'cuda:{local_rank}'
    else:
        device = args.device

    if args.name:
        name = args.name
    elif args.resume:                       # continue in the resumed run's directory
        rp = Path(args.resume)
        name = (rp if rp.is_dir() else rp.parent).name
    else:
        name = f'vae_{int(time.time())}'
    runpath = PATH / 'runs_vae' / name
    if is_main:
        runpath.mkdir(parents=True, exist_ok=True)
        if not (runpath / 'config.json').exists():   # keep the original config on resume
            save_config(config, runpath)
        print(f'run {name} -> {runpath}' + (f'  [DDP world {world}]' if ddp else ''),
              flush=True)
        print(f'config: {config}', flush=True)

    datadir = PATH / 'data'
    if not (datadir / 'train.h5').exists():
        raise SystemExit(f"missing {datadir/'train.h5'}; run prepare.py first")

    trainset = FrameDataset(datadir / 'train.h5', subsample=config['subsample'])
    validset = FrameDataset(datadir / 'valid.h5', subsample=config['subsample'])
    if is_main:
        print(f'train frames: {len(trainset):,}   valid frames: {len(validset):,}', flush=True)

    dl_kw = dict(batch_size=config['batch_size'], num_workers=config['num_workers'],
                 pin_memory=str(device).startswith('cuda'),
                 persistent_workers=config['num_workers'] > 0)
    # Under DDP each rank trains a disjoint shard (DistributedSampler); validation
    # runs only on rank 0 over the full set, so it needs no sampler.
    train_sampler = (DistributedSampler(trainset, num_replicas=world, rank=rank,
                                        shuffle=True, drop_last=True) if ddp else None)
    train_loader = DataLoader(trainset, shuffle=(train_sampler is None),
                              sampler=train_sampler, drop_last=True, **dl_kw)
    valid_loader = DataLoader(validset, shuffle=False, drop_last=False, **dl_kw)

    vae = make_vae(**config).to(device)
    nparam = sum(pp.numel() for pp in vae.parameters())
    if is_main:
        print(f'VAE {vae.latent_channels}x{vae.latent_height}x{vae.latent_width} latent, '
              f'{nparam/1e6:.2f}M params', flush=True)

    opt = torch.optim.AdamW(vae.parameters(), lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(config['epochs']))
    scaler = torch.cuda.amp.GradScaler() if (config['amp'] and device == 'cuda') else None

    # fixed validation frames for the reconstruction snapshot
    vis = torch.stack([validset[i] for i in
                       np.linspace(0, len(validset) - 1, 6).astype(int)]).to(device)

    # optional resume: a full checkpoint (model+optimizer+scheduler+epoch) or,
    # for backward compat, a bare state_dict (weights only, restart at epoch 0).
    start_epoch = 0
    best = math.inf
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / 'last.pth'
        if not ckpt_path.exists():
            raise SystemExit(f'--resume: no checkpoint at {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            vae.load_state_dict(ckpt['model'])
            opt.load_state_dict(ckpt['optimizer'])
            sched.load_state_dict(ckpt['scheduler'])
            if scaler is not None and ckpt.get('scaler') is not None:
                scaler.load_state_dict(ckpt['scaler'])
            start_epoch = int(ckpt.get('epoch', 0))
            best = float(ckpt.get('best', math.inf))
            if is_main:
                print(f'resumed {ckpt_path} at epoch {start_epoch} '
                      f'(best valid rec {best:.4f})', flush=True)
        else:                               # bare state_dict -> weights only
            vae.load_state_dict(ckpt)
            if is_main:
                print(f'resumed weights only from {ckpt_path} '
                      f'(no optimizer/epoch state; starting at epoch 0)', flush=True)

    # Wrap for DDP: gradients are all-reduced across ranks on each backward. `vae`
    # stays the unwrapped model used for validation, saving and the snapshot.
    if ddp:
        module = nn.parallel.DistributedDataParallel(
            VAELoss(vae, config['beta']), device_ids=[local_rank])
        loss_fn = lambda x: module(x)
        if is_main:
            print(f'DDP across {world} GPUs (per-GPU batch {config["batch_size"]}, '
                  f'global {config["batch_size"] * world})', flush=True)
    else:
        module = vae
        loss_fn = lambda x: vae.loss(x, config['beta'])

    for epoch in range(start_epoch, config['epochs']):
        t0 = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)   # reshuffle the shards each epoch
        tr_tot, tr_rec, tr_kl = _run_epoch(
            loss_fn, module, train_loader, device, opt=opt, scaler=scaler,
            amp=config['amp'],
            desc=(f'epoch {epoch + 1}/{config["epochs"]}' if is_main else None))
        if ddp:                              # average the per-shard metrics for logging
            stats = torch.tensor([tr_tot, tr_rec, tr_kl], device=device)
            dist.all_reduce(stats)
            tr_tot, tr_rec, tr_kl = (stats / world).tolist()

        # validation + all checkpointing on rank 0 only (uses the unwrapped `vae`)
        if is_main:
            vae.eval()
            va_rec = va_kl = 0.0
            nb = 0
            with torch.no_grad():
                for x in valid_loader:
                    x = x.to(device)
                    _, r, k = vae.loss(x, config['beta'])
                    va_rec += float(r); va_kl += float(k); nb += 1
                    if nb >= args.valid_batches:
                        break
            va_rec /= max(nb, 1); va_kl /= max(nb, 1)
        sched.step()

        if is_main:
            print(f'epoch {epoch:4d}  train[tot {tr_tot:.4f} rec {tr_rec:.4f} kl {tr_kl:.1f}]  '
                  f'valid[rec {va_rec:.4f} kl {va_kl:.1f}]  lr {sched.get_last_lr()[0]:.2e}  '
                  f'{time.time()-t0:.1f}s', flush=True)

            if va_rec < best:
                best = va_rec
                torch.save(vae.state_dict(), runpath / 'state.pth')

            # full checkpoint (overwritten each epoch) -- the canonical resume point
            torch.save({
                'epoch': epoch + 1,
                'model': vae.state_dict(),
                'optimizer': opt.state_dict(),
                'scheduler': sched.state_dict(),
                'scaler': scaler.state_dict() if scaler is not None else None,
                'best': best,
                'config': config,
            }, runpath / 'last.pth')

            # periodic intermediate checkpoint (bare weights, for inference snapshots)
            if args.ckpt_every and (epoch + 1) % args.ckpt_every == 0:
                torch.save(vae.state_dict(), runpath / f'ckpt_epoch{epoch + 1:04d}.pth')

        if ddp:
            dist.barrier()                   # keep ranks in lockstep across epochs

    # final save + reconstruction snapshot (originals top row, recons bottom)
    if is_main:
        torch.save(vae.state_dict(), runpath / 'state.pth')
        vae.eval()
        with torch.no_grad():
            rec = vae.decode(vae.encode(vis)[0])
        panel = torch.stack([vis[:, 0].cpu(), rec[:, 0].cpu()])      # (2, 6, H, W)
        draw(panel, vmin=-2.0, vmax=2.0).save(runpath / 'reconstruction.png')
        print(f'saved -> {runpath}/state.pth, reconstruction.png  '
              f'(best valid rec {best:.4f})', flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
