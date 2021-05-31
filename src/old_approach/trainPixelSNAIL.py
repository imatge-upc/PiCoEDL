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
from config import setSeed, getConfig
from customLoader import LatentDataset

from tqdm import tqdm
from os.path import join
from pathlib import Path
from pprint import pprint

from scipy.signal import savgol_filter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter


from scheduler import CycleScheduler

from IPython import embed

def train(hier, epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()
    train_loss = []

    for i, (top, bottom, label) in enumerate(loader):
        model.zero_grad()

        top = top.to(device)
        top = top.squeeze()
        bottom = bottom.squeeze()

        if hier == 'top':
            target = top
            out, _ = model(top)

        elif hier == 'bottom':
            bottom = bottom.to(device)
            target = bottom
            out, _ = model(bottom, condition=top)


        loss = criterion(out, target)
        loss.backward()

        train_loss.append(loss.item())


        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                f'acc: {accuracy:.5f}; lr: {lr:.5f}'
            )
        )
    return np.asarray(train_loss).mean(0)

def test(hier, epoch, loader, model, device):
    model.eval()
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()
    test_loss = []

    for i, (top, bottom, label) in enumerate(loader):

        top = top.to(device)
        top = top.squeeze()
        bottom = bottom.squeeze()

        if hier == 'top':
            target = top
            out, _ = model(top)

        elif hier == 'bottom':
            bottom = bottom.to(device)
            target = bottom
            out, _ = model(bottom, condition=top)


        loss = criterion(out, target)

        test_loss.append(loss.item())

    model.train()
    return np.asarray(test_loss).mean(0)


def saveModel(path_weights, hier, model, optim, iter):
    file_name = hier + '_' + str(iter).zfill(3) + ".pt"
    path = path_weights / conf['experiment'] / file_name
	
    torch.save({
        'state_dict': model.state_dict(),
		'optimizer': optim},
		path)


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()


if __name__ == '__main__':

    setSeed(0)
    assert len(sys.argv) == 2, "Indicate a configuration file like 'config_0.0'"
    conf = getConfig(sys.argv[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    mrl_train = LatentDataset('train', transform=transforms.ToTensor())
    mrl_test = LatentDataset('val', transform=transforms.ToTensor())

    train_loader = DataLoader(mrl_train, batch_size=conf['pixelsnail']['batch_size'], shuffle=True)
    test_loader = DataLoader(mrl_test, batch_size=conf['pixelsnail']['batch_size'], shuffle=True)

    path_weights = Path('/mnt/gpid07/users/juan.jose.nieto/weights/')

    if not os.path.exists(path_weights / conf['experiment']):
        os.mkdir(path_weights / conf['experiment'])

    hier = conf['pixelsnail']['hier']
    if hier == 'top':
        model = PixelSNAIL(
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
        if conf['pixelsnail']['load']:
            weights = torch.load(path_weights / 'pixelsnail_1' / conf['pixelsnail']['name_top'])['state_dict']
            model.load_state_dict(weights)
    elif hier == 'bottom':
        model = PixelSNAIL(
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
        if conf['pixelsnail']['load']:
            weights = torch.load(path_weights / 'pixelsnail_1' / conf['pixelsnail']['name_bottom'])['state_dict']
            model.load_state_dict(weights)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=conf['pixelsnail']['lr'])

    

    scheduler = None
    if conf['pixelsnail']['sched'] == 'cycle':
        scheduler = CycleScheduler(
            optimizer, conf['pixelsnail']['lr'], n_iter=len(train_loader) * conf['pixelsnail']['epochs'], momentum=None
        )

    writer = SummaryWriter(log_dir=f"../tensorboard/{conf['experiment']}_{hier}/")

    for i in range(conf['pixelsnail']['epochs']):
        train_loss = train(hier, i, train_loader, model, optimizer, scheduler, device)
        test_loss = test(hier, i, test_loader, model, device)
        writer.add_scalar('PixelSNAIL/Train Loss', train_loss, i)
        writer.add_scalar('PixelSNAIL/Test Loss', test_loss, i)

        saveModel(path_weights, hier, model, optimizer, i)
