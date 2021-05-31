import os
import sys
import wandb
import torch

from os.path import join
from pathlib import Path
from config import setSeed, getConfig

from main.vqvae import VQVAE

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from IPython import embed

setSeed(0)
assert len(sys.argv) == 2, "Indicate a configuration file like 'config_0.0'"
conf = getConfig(sys.argv[1])


if os.getenv('USER') == 'juanjo':
    path_weights = Path('./results')
elif os.getenv('USER') == 'juan.jose.nieto':
    path_weights = Path('/home/usuaris/imatge/juan.jose.nieto/mineRL/src/results')
else:
    raise Exception("Sorry user not identified!")

vqvae = VQVAE(conf).cuda()
checkpoint = torch.load(join(path_weights, conf['test']['path_weights']))

# uncomment for compatibility with older models
# for c in checkpoint['state_dict'].copy():
#     checkpoint['state_dict']['model.' + c] = checkpoint['state_dict'].pop(c)

vqvae.load_state_dict(checkpoint['state_dict'])
vqvae._construct_map()
