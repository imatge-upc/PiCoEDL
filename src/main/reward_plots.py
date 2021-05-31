import os
import sys
import cv2
import gym
import json
import time
import minerl
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from os.path import join
from pathlib import Path
from pprint import pprint
from config import setSeed, getConfig

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from main.encoder import PixelEncoder
from main.model import CURL

from IPython import embed

setSeed(2)
assert len(sys.argv) == 2, "Indicate a configuration file like 'config_0.0'"
conf = getConfig(sys.argv[1])

MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLNavigate-v0')
# MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechop-v0')
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', '/home/usuaris/imatge/juan.jose.nieto/mineRL/data/')
data = minerl.data.make(MINERL_GYM_ENV, data_dir=MINERL_DATA_ROOT, num_workers=1)

feature_dim = conf['curl']['embedding_dim']
img_size = conf['curl']['img_size']
obs_shape = (3, img_size, img_size)
batch_size = conf['batch_size']

if os.getenv('USER') == 'juanjo':
    path_weights = Path('../weights/')
elif os.getenv('USER') == 'juan.jose.nieto':
    path_weights = Path('/mnt/gpid07/users/juan.jose.nieto/weights/')
else:
    raise Exception("Sorry user not identified!")


pixel_encoder = PixelEncoder(obs_shape, feature_dim)
pixel_encoder_target = PixelEncoder(obs_shape, feature_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

curl = CURL(obs_shape, feature_dim, batch_size, pixel_encoder, pixel_encoder_target).to(device)


curl.eval()

if conf['curl']['load']:
    weights = torch.load(path_weights / conf['experiment'] / conf['curl']['epoch'])['state_dict']
    curl.load_state_dict(weights)

def save_image(img, name, type='goal_state_'):
    fig, ax = plt.subplots()
    plt.imsave(f'../images/reward_plots/{type}{name}.png',img)
    plt.close()

def save_fig(img, name):
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(f'../images/inference_attention_2/{name}.svg')
    plt.close()

def store_random(batch, seq, step):
    obs_pos = batch[seq,step,:,:,:]
    save_image(obs_pos, '_'.join([str(seq), str(step)]), type='seq_step_')



final_step = 1000
goal_states = [10, int(final_step/2), final_step-1]

for i, (current_state, action, reward, next_state, done) in enumerate(data.batch_iter(batch_size=64, num_epochs=1, seq_len=final_step)):

    batch = current_state['pov']
    # store_random(batch, 2, 600)
    # store_random(batch, 3, 850)
    # store_random(batch, 1, 780)
    # store_random(batch, 1, 785)
    # store_random(batch, 1, 790)
    # store_random(batch, 3, 350)
    seq = 60
    for goal in goal_states:
        print(f"seq {seq} goal {goal}")
        # obs_anchor = batch[:,0,:,:,:]
        obs_pos = batch[seq,goal,:,:,:]
        # save_image(obs_pos, '_'.join([str(seq), str(goal)]))
        obs_pos = torch.from_numpy(obs_pos).float().unsqueeze(dim=0).to(device)
        obs_pos = obs_pos.permute(0,3,1,2)
        z_pos = curl.encode(obs_pos, ema=True)
        print("\tsaved goal state")
        rewards = []
        a = batch[:5]
        b = batch[60]
        b = b[None, ...]
        asd = np.vstack((a,b))
        for b in asd:
            aux = []
            for img in b[0:final_step-1]:
                obs_anchor = torch.from_numpy(img).float().unsqueeze(dim=0).to(device)
                obs_anchor = obs_anchor.permute(0,3,1,2)
                z_a = curl.encode(obs_anchor)
                logits = curl.compute_logits_(z_a, z_pos)
                aux.append(logits.detach().cpu().numpy().item())
            rewards.append(aux)
        print("\tcomputed rewards")
        data = np.array(rewards).T
        df = pd.DataFrame(data)
        df = df.rolling(5).sum()
        ax = df.plot.line()
        ax.set_xlabel('step')
        ax.set_ylabel('reward')
        plt.axvline(goal, color='red')
        plt.savefig(f"../images/reward_plots/{conf['experiment']}_{seq}_{goal}.png")
        plt.close()
        print("\tsaved fig")
    break
