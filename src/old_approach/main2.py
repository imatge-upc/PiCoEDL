import os
import sys
import cv2
import torch

import numpy as np
import matplotlib.pylab as plt
import torch.optim as optim
import torch.nn.functional as F

from models.VQVAE2 import VQVAE2
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

training_loader = DataLoader(mrl_train, batch_size=conf['batch_size'], shuffle=True)
validation_loader = DataLoader(mrl_val, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VQVAE2().to(device)

# weights = torch.load(f'../weights/vqvae2_0/37798.pt')['state_dict']
# model.load_state_dict(weights)

pprint(conf)

optimizer = optim.Adam(model.parameters(), lr=conf['learning_rate'], amsgrad=False)

model.train()

train_res_recon_error = []
train_res_perp_t = []
train_res_perp_b = []

writer = SummaryWriter(log_dir=f"../tensorboard/{conf['experiment']}/")

if not os.path.exists(join('../weights', conf['experiment'])):
    os.mkdir(join('../weights', conf['experiment']))

def saveModel(model, optim, iter):
	path = Path(f"../weights/{conf['experiment']}/{iter}.pt")
	torch.save({
        'state_dict': model.state_dict(),
		'optimizer': optim},
		path)


valid_originals = next(iter(validation_loader))
valid_originals = valid_originals.to(device)


latent_loss_weight = 0.25

for i in range(conf['num_training_updates']):
    batch = next(iter(training_loader))
    data = batch.to(device)

    optimizer.zero_grad()
    vq_loss, data_recon, perp_t, perp_b = model(data)
    # recon_error = F.mse_loss(data_recon, data) / mrl.data_variance
    recon_error = F.mse_loss(data_recon, data)
    loss = recon_error + vq_loss * latent_loss_weight
    loss.backward()

    optimizer.step()

    train_res_recon_error.append(recon_error.item())
    train_res_perp_t.append(perp_t.item())
    train_res_perp_b.append(perp_b.item())

    writer.add_scalar('Reconstruction Error', recon_error.item(), i)
    writer.add_scalar('Perplexity/Top', perp_t.item(), i)
    writer.add_scalar('Perplexity/Bottom', perp_b.item(), i)

    if (i+1) % 200 == 0:
        print('%d iterations' % (i+1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-200:]))
        print('perplexity top: %.3f' % np.mean(train_res_perp_t[-200:]))
        print('perplexity bottom: %.3f' % np.mean(train_res_perp_b[-200:]))
        print()
    #     model.eval()
    #     _, valid_reconstructions, _, _ = model(valid_originals)
    #     grid = make_grid(valid_reconstructions.cpu().data)+0.5
    #     writer.add_image('images', grid, i)
    #     saveModel(model, optimizer, i)
    #     model.train()
