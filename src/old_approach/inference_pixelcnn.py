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
from customLoader import MinecraftData, LatentBlockDataset

from pprint import pprint
from os.path import join
from pathlib import Path

from torchvision.utils import make_grid
from torchvision.transforms import transforms

from torch.utils.data import DataLoader

from IPython import embed

setSeed(0)
assert len(sys.argv) == 2, "Indicate a configuration file like 'config_0.0'"
conf = getConfig(sys.argv[1])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vqvae = VQVAE(**conf['vqvae']).to(device)
pixelcnn = GatedPixelCNN(conf['vqvae']['num_embeddings'], conf['pixelcnn']['img_dim']**2).to(device)

vqvae.eval()
pixelcnn.eval()

weights_vqvae = torch.load(f"../weights/{conf['experiment']}/24999.pt")['state_dict']
vqvae.load_state_dict(weights_vqvae)

weights_pixelcnn = torch.load(f"../weights/pixel/pixel_024.pt")['state_dict']
pixelcnn.load_state_dict(weights_pixelcnn)

pprint(conf)

batch = 8
a = []
for i in range(2):
    print("Sampling from GatedPixelCNN...")
    label = torch.arange(batch).contiguous().view(-1)
    label = label.long().to(device)
    indices = pixelcnn.generate(label, batch_size=batch)

    print("Generating images")
    quantized = vqvae._vq_vae.indices2quantized(indices, batch)
    generated = vqvae._decoder(quantized)
    a.append(generated)

generated = torch.concat((a[0], a[1]), 0)
grid = make_grid(generated.cpu().data, normalize=True)

print("Saving images")
plt.imsave("../images/pixel_2.png", grid.permute(1,2,0).numpy())
