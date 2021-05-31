import os
import sys
import cv2
import time
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pylab as plt
import torch.nn.functional as F

from models.PixelSNAIL import PixelSNAIL
from models.VQVAE2 import VQVAE2

from config import setSeed, getConfig
from customLoader import LatentDataset

from tqdm import tqdm
from os.path import join
from pathlib import Path
from pprint import pprint


from torchvision.utils import save_image
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter


from scheduler import CycleScheduler

from IPython import embed


@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0])):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row


if __name__ == '__main__':

    setSeed(0)
    assert len(sys.argv) == 2, "Indicate a configuration file like 'config_0.0'"
    conf = getConfig(sys.argv[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    path_weights = Path('/mnt/gpid07/users/juan.jose.nieto/weights/')

    model_top = PixelSNAIL(
        [conf['pixelsnail']['top_dim'], conf['pixelsnail']['top_dim']],
        conf['pixelsnail']['n_class'],
        conf['pixelsnail']['channel'],
        conf['pixelsnail']['kernel_size'],
        conf['pixelsnail']['n_block'],
        conf['pixelsnail']['n_res_block'],
        conf['pixelsnail']['n_res_channel'],
        dropout=conf['pixelsnail']['dropout'],
        n_out_res_block=conf['pixelsnail']['n_out_res_block']
    )
    model_bottom = PixelSNAIL(
        [conf['pixelsnail']['bottom_dim'], conf['pixelsnail']['bottom_dim']],
        conf['pixelsnail']['n_class'],
        conf['pixelsnail']['channel'],
        conf['pixelsnail']['kernel_size'],
        conf['pixelsnail']['n_block'],
        conf['pixelsnail']['n_res_block'],
        conf['pixelsnail']['n_res_channel'],
        attention=False,
        dropout=conf['pixelsnail']['dropout'],
        n_cond_res_block=conf['pixelsnail']['n_cond_res_block'],
        cond_res_channel=conf['pixelsnail']['n_res_channel']
    )
    
    vqvae2 = VQVAE2().to(device)

    print('Loading weights...')

    vqvae2_w = torch.load(path_weights / 'vqvae2_0' / '59197.pt')['state_dict']
    vqvae2.load_state_dict(vqvae2_w)

    pixelsnail_bottom_w = torch.load(path_weights / 'pixelsnail_1' / 'bottom_005.pt')['state_dict']
    model_bottom.load_state_dict(pixelsnail_bottom_w)

    pixelsnail_top_w = torch.load(path_weights / 'pixelsnail_1' / 'top_015.pt')['state_dict']
    model_top.load_state_dict(pixelsnail_top_w)

    model_top = model_top.to(device)
    model_bottom = model_bottom.to(device)

    print('All models loaded!')

    batch = 8
    temp = 1

    for i in range(8):
        print(f'Sampling... {i}')
        top_sample = sample_model(model_top, device, batch, [8,8], temp)
        bottom_sample = sample_model(
            model_bottom, device, batch, [16,16], temp, condition=top_sample
        )

        decoded_sample = vqvae2.decode_code(top_sample, bottom_sample)
        decoded_sample = decoded_sample.clamp(-1, 1)

        save_image(decoded_sample, f'../images/img_sampled/sample_toprandom_{i}.png', normalize=True, range=(-1, 1))
