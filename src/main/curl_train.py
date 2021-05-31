import os
import sys
import wandb

from pathlib import Path
from config import setSeed, getConfig
from main.curl import CURL

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from IPython import embed

setSeed(0)
assert len(sys.argv) == 2, "Indicate a configuration file like 'config_0.0'"
conf = getConfig(sys.argv[1])

if not os.path.exists('./results'):
    os.mkdir('./results')

wandb_logger = WandbLogger(
    project='mineRL',
    name=conf['experiment'],
    tags=['curl']
)

wandb_logger.log_hyperparams(conf)

curl = CURL(conf)

trainer = pl.Trainer(
    gpus=1,
    max_epochs=conf['epochs'],
    progress_bar_refresh_rate=20,
    weights_summary='full',
    logger=wandb_logger,
    default_root_dir=f"./results/{conf['experiment']}"
)

trainer.fit(curl)
