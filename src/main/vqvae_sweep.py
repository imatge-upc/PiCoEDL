
import wandb
alg = 'vqvae'

def update_custom(conf, d):
    for k,v in d.items():
        if '.' in k:
            key = k.split('.')
            conf[key[0]][key[1]] = v
        else:
            conf[k] = v
    return conf

def main():
    import sys
    import wandb

    from config import setSeed, getConfig
    from main.vqvae import VQVAE
    from pytorch_lightning.loggers import WandbLogger
    import pytorch_lightning as pl

    from IPython import embed

    run = wandb.init()
    conf = getConfig(sys.argv[1])

    conf = update_custom(conf, run.config)
    wandb_logger = WandbLogger(
        project='mineRL',
        name=conf['experiment'],
        tags=[alg, 'sweep']
    )

    wandb_logger.log_hyperparams(conf)

    vqvae = VQVAE(conf)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=conf['epochs'],
        progress_bar_refresh_rate=20,
        weights_summary='full',
        logger=wandb_logger,
        default_root_dir=f"./results/{conf['experiment']}"
    )

    trainer.fit(vqvae)

sweep_config = {
    "name": f"{alg}_CW0_pixels_coords.sweep",
    "method": 'grid',
    "metric": {
        "name": "perplexity/train",
        "goal": "maximize"
    },
    "parameters": {
        "coord_cost": {
              "values": [0.5,0.3,0.2,0.1,0.05]
        }
  }
}


import os
del os.environ["SLURM_NTASKS"]
del os.environ["SLURM_JOB_NAME"]

sweep_id = wandb.sweep(sweep_config, project="mineRL")
wandb.agent(sweep_id, function=main, count=30, project="mineRL", entity="juanjo3ns")
