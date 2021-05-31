import os
import sys
import cv2
import torch

import numpy as np
import matplotlib.pylab as plt
import torch.optim as optim
import torch.nn.functional as F

from models.VQVAE2 import VQVAE2
from config import setSeed, getConfig
from customLoader import MinecraftData

from pprint import pprint
from os.path import join
from pathlib import Path

from scipy.signal import savgol_filter
from torchvision.utils import make_grid
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from IPython import embed

setSeed(0)
assert len(sys.argv) == 2, "Indicate a configuration file like 'config_0.0'"
conf = getConfig(sys.argv[1])


transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                        ])


mrl_val = MinecraftData(conf['environment'], 'val', conf['split'], False, transform=transform)

validation_loader = DataLoader(mrl_val, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VQVAE2().to(device)

model.eval()

pprint(conf)

valid_originals = next(iter(validation_loader))
valid_originals = valid_originals.to(device)

path_imgs = Path('../images')
weights = sorted(os.listdir('../weights/vqvae2_0'), key=lambda x: int(x.split('.')[0]))

num_imgs = len(weights)

for count, i in enumerate(weights):
    print(f"Loading model {i}...")
    weights = torch.load(f"../weights/vqvae2_0/{i}")['state_dict']
    model.load_state_dict(weights)
    quant_t, quant_b,_, id_t, id_b, _,_ = model.encode(valid_originals)
    valid_reconstructions = model.decode(quant_t, quant_b)

    id_t = id_t.cpu().numpy()
    id_b = id_b.cpu().numpy()
    valid_reconstructions = valid_reconstructions.permute(0,2,3,1)
    valid_reconstructions = valid_reconstructions.cpu().detach().numpy() + 0.5
    fig, ax = plt.subplots(6,8, figsize=(16,10))
    for q, imgs in enumerate([valid_reconstructions, id_b, id_t]):
        for j,img in enumerate(imgs):
            ax[int(j/8) + q*2, int(j%8)].imshow(img)
            ax[int(j/8) + q*2, int(j%8)].axis('off')
            ax[int(j/8) + q*2, int(j%8)].axis("tight")
    name = str(i.split('.')[0]) + '.png'
    plt.suptitle(f'{count}/{num_imgs}')
    plt.savefig(path_imgs / name)
    plt.close()
