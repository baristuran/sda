#!/usr/bin/env python

import wandb

from dawgz import job, schedule
from typing import *

from sda.mcs import *
from sda.score import *
from sda.utils import *

from utils import *


CONFIG = {
    # Architecture
    'window': 5,
    'embedding': 64,
    'hidden_channels': (64, 128, 256),
    'hidden_blocks': (3, 3, 3),
    'kernel_size': 3,
    'activation': 'SiLU',
    # Data
    'coarsen': 2,            # 512x128 -> 256x64 to keep memory/compute tractable
    # Training
    'epochs': 4096,
    'batch_size': 128,
    'optimizer': 'AdamW',
    'learning_rate': 2e-4,
    'weight_decay': 1e-3,
    'scheduler': 'exponential',
    'num_workers': 4,
}


@job(array=1, cpus=4, gpus=3, ram='32GB', time='24:00:00')
def train(i: int):
    run = wandb.init(project='sda-rayleigh-benard', config=CONFIG)
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(CONFIG, runpath)

    # Network
    window = CONFIG['window']
    coarsen = CONFIG['coarsen']
    height, width = HEIGHT // coarsen, WIDTH // coarsen

    score = make_score(**CONFIG)
    sde = VPSDE(score.kernel, shape=(window, height, width)).cuda()

    # Data: consolidated, contiguous HDF5 (run `python prepare.py --coarsen N`
    # first). TrajectoryDataset loads it into memory, so reads are fast -- unlike
    # the raw Dedalus virtual datasets.
    datadir = PATH / 'data'
    if not (datadir / 'train.h5').exists():
        raise SystemExit(
            f"missing {datadir/'train.h5'}; run `python prepare.py --coarsen {coarsen}` first")
    with h5py.File(datadir / 'train.h5', 'r') as f:
        _, _, _, dh, dw = f['x'].shape
    assert (dh, dw) == (height, width), (
        f"data grid {dh}x{dw} != coarsen-{coarsen} grid {height}x{width}; "
        f"re-run `python prepare.py --coarsen {coarsen}`")

    trainset = TrajectoryDataset(datadir / 'train.h5', window=window, flatten=True)
    validset = TrajectoryDataset(datadir / 'valid.h5', window=window, flatten=True)

    # Training
    generator = loop(
        sde,
        trainset,
        validset,
        device='cuda',
        **CONFIG,
    )

    for loss_train, loss_valid, lr in generator:
        run.log({
            'loss_train': loss_train,
            'loss_valid': loss_valid,
            'lr': lr,
        })

    # Save
    torch.save(
        score.state_dict(),
        runpath / f'state.pth',
    )

    # Evaluation
    x = sde.sample((2,), steps=64).cpu()
    x = x.unflatten(1, (-1, 1))        # (2, window, C=1, H, W)
    b = x[:, :, 0]                     # buoyancy

    run.log({'samples': wandb.Image(draw(b))})
    run.finish()


if __name__ == '__main__':
    schedule(
        train,
        name='Training',
        backend='async',
        export='ALL',
        env=['export WANDB_SILENT=true'],
    )
