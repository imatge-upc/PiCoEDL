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

from models.GatedPixelCNN import GatedPixelCNN
from config import setSeed, getConfig
from customLoader import LatentBlockDataset

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



mrl_train = LatentBlockDataset('../data/latent_blocks/train.npy', True, conf['pixelcnn']['img_dim'], transform=transforms.ToTensor())
mrl_test = LatentBlockDataset('../data/latent_blocks/val.npy', False, conf['pixelcnn']['img_dim'], transform=transforms.ToTensor())

train_loader = DataLoader(mrl_train, batch_size=conf['pixelcnn']['batch_size'], shuffle=True)
test_loader = DataLoader(mrl_test, batch_size=conf['pixelcnn']['batch_size'], shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GatedPixelCNN(conf['vqvae']['num_embeddings'], conf['pixelcnn']['img_dim']**2).to(device)

pprint(conf)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=conf['pixelcnn']['lr'])

model.train()


writer = SummaryWriter(log_dir=f"../tensorboard/{conf['experiment']}/")

if not os.path.exists(join('../weights', conf['experiment'])):
    os.mkdir(join('../weights', conf['experiment']))

def saveModel(model, optim, iter):
	path = Path(f"../weights/{conf['experiment']}/pixel_{iter}.pt")
	torch.save({
        'state_dict': model.state_dict(),
		'optimizer': optim},
		path)

def generate_samples(epoch):
    label = torch.arange(10).expand(10, 10).contiguous().view(-1)
    label = label.long().to(device)

    x_tilde = model.generate(
        label,
        shape=(conf['pixelcnn']['img_dim'],conf['pixelcnn']['img_dim']),
        batch_size=100
    )

    print(x_tilde[0])

def train():
    train_loss = []
    for batch_idx, (x, label) in enumerate(train_loader):
        start_time = time.time()
        x = x.squeeze().to(device)
        label = label.to(device)
        # Train PixelCNN with images
        logits = model(x, label)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        loss = criterion(
            logits.view(-1, conf['vqvae']['num_embeddings']),
            x.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        if (batch_idx + 1) % conf['pixelcnn']['log_interval'] == 0:
            print('\tIter: [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                batch_idx * len(x), len(train_loader.dataset),
                conf['pixelcnn']['log_interval'] * batch_idx / len(train_loader),
                np.asarray(train_loss)[-conf['pixelcnn']['log_interval']:].mean(0),
                time.time() - start_time
            ))
    return np.asarray(train_loss).mean(0)


def test():
    start_time = time.time()
    val_loss = []
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            x = x.squeeze().to(device)
            label = label.to(device)

            logits = model(x, label)

            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss = criterion(
                logits.view(-1, conf['vqvae']['num_embeddings']),
                x.view(-1)
            )

            val_loss.append(loss.item())

    print('Validation Completed!\tLoss: {} Time: {}'.format(
        np.asarray(val_loss).mean(0),
        time.time() - start_time
    ))
    return np.asarray(val_loss).mean(0)


writer = SummaryWriter(log_dir=f"../tensorboard/{conf['experiment']}/")

BEST_LOSS = 999
LAST_SAVED = -1
for epoch in range(1, conf['pixelcnn']['epochs']):
    print("\nEpoch {}:".format(epoch))
    train_loss = train()
    test_loss = test()

    writer.add_scalar('GatedPixelCNN/Train Loss', train_loss, epoch)
    writer.add_scalar('GatedPixelCNN/Test Loss', test_loss, epoch)

    if conf['pixelcnn']['save'] or test_loss <= BEST_LOSS:
        BEST_LOSS = test_loss
        LAST_SAVED = epoch

        print("Saving model!")
        saveModel(model, optimizer, epoch)
    else:
        print("Not saving model! Last saved: {}".format(LAST_SAVED))
    if conf['pixelcnn']['gen_samples']:
        generate_samples(epoch)
