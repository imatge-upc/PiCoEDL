import os
import sys
import cv2
import torch

import numpy as np
import matplotlib.pylab as plt
import torch.optim as optim
import torch.nn.functional as F

from models.VQVAE import VQVAE
from models.GatedPixelCNN import GatedPixelCNN
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


mrl_train = MinecraftData(conf['environment'], 'train', conf['split'], False, transform=transform)
mrl_val = MinecraftData(conf['environment'], 'val', conf['split'], False, transform=transform)

training_loader = DataLoader(mrl_train, batch_size=1, shuffle=True)
validation_loader = DataLoader(mrl_val, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VQVAE(**conf['vqvae']).to(device)
weights = torch.load(f"../weights/{conf['experiment']}/24999.pt")['state_dict']
model.load_state_dict(weights)


pprint(conf)


if not os.path.exists(join('../data', 'latent_blocks')):
    os.mkdir(join('../data', 'latent_blocks'))

def saveLatents(file, data):
    with open(f"../data/latent_blocks/{file}.npy", 'wb') as f:
        np.save(f, data)



for i, loader in enumerate([training_loader, validation_loader]):
    batch = iter(loader)
    encodings = []
    print("Storing latent vectors...")
    for b in batch:
        data = b.to(device)
        encoding_indices = model.save_encoding_indices(data)
        encoding_indices = encoding_indices.squeeze().cpu().detach().numpy().tolist()
        encodings.append(encoding_indices)

    saveLatents('train' if i==0 else 'val', np.array(encodings))
