import os
import csv
import time
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from os.path import join
from pathlib import Path
from pprint import pprint
from config import setSeed, getConfig
from collections import Counter, defaultdict
from main.utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from customLoader import *
from torchvision.transforms import transforms

from models.CustomVQVAE import VQVAE_PL

from pytorch_lightning.loggers import WandbLogger

from mod.q_functions import parse_arch
from sklearn.cluster import KMeans

class VQVAE(VQVAE_PL):
    def __init__(self, conf):
        super(VQVAE, self).__init__(conf['data_type'], **conf['vqvae'])

        self.experiment = conf['experiment']
        self.batch_size = conf['batch_size']
        self.lr = conf['lr']
        self.split = conf['split']
        self.num_clusters = conf['vqvae']['num_embeddings']

        self.delay = conf['delay']
        self.trajectories = conf['trajectories']
        self.trajectories_train, self.trajectories_val = get_train_val_split(self.trajectories, self.split)

        self.conf = {
            'k_std': conf['k_std'], 
            'k_mean': conf['k_mean'],
            'data_type': conf['data_type']
            }

        self.transform = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                ])

        self.test = conf['test']
        self.type = self.test['type']
        self.shuffle = self.test['shuffle']
        self.limit = self.test['limit']


    def on_train_start(self):
        embeddings = []

        print("Computing embeddings...")
        for batch in self.trainer.train_dataloader:
            z_1 = self.model.compute_embedding(batch, self.device)
            embeddings.append(z_1.detach().cpu().numpy())

        e = np.concatenate(np.array(embeddings))

        print("Computing kmeans...")
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(e)

        kmeans_tensor = torch.from_numpy(kmeans.cluster_centers_).to(self.device)
        self.model._vq_vae._embedding.weight = nn.Parameter(kmeans_tensor)
        self.model._vq_vae._ema_w = nn.Parameter(kmeans_tensor)
        
    def training_step(self, batch, batch_idx):

        loss = self.model(batch, batch_idx, self.logger, "train")

        return loss

    def validation_step(self, batch, batch_idx):


        loss = self.model(batch, batch_idx, self.logger, "val")

        return loss

    def on_epoch_end(self):
        self.model.log_reconstructions(self.trainer.train_dataloader, self.logger)


    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.lr, weight_decay=1e-5)

    def train_dataloader(self):
        train_dataset = CustomMinecraftData(self.trajectories_train, transform=self.transform, delay=self.delay, **self.conf)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = CustomMinecraftData(self.trajectories_val, transform=self.transform, delay=self.delay, **self.conf)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        return val_dataloader

    def _construct_map(self):
        construct_map(self)
