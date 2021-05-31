import os
import numpy as np
import torch
import torch.nn as nn
from random import randint
import pytorch_lightning as pl
from models.PixelEncoder import PixelEncoder

from IPython import embed

class CURL_PL(pl.LightningModule):
    """
    CURL
    """

    def __init__(self,
            obs_shape=(3,64,64),
            z_dim=50,
            output_type="continuous",
            load_goal_states=False,
            device=None,
            path_goal_states=None,
            **kwargs
            ):
        super(CURL_PL, self).__init__()

        self.reward_type = ""
        
        self.encoder = PixelEncoder(obs_shape, z_dim)
        self.encoder_target = PixelEncoder(obs_shape, z_dim)

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

        if load_goal_states:
            self.path_gs = path_goal_states
            self.dev = device
            self.goal_states = self.load_goal_states()
            self.num_goal_states = self.goal_states.shape[0]


    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_embedding(self, batch, device):
        x = batch[:, 0]
        x = x.to(device)
        return self.encode(x)
        
    def compute_logits(self, z_a, z_pos=None):
        if z_pos == None:
            z_pos = self.goal_states
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        return logits

    def compute_train(self, z_a, z_pos):

        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        logits = self.compute_logits(z_a, z_pos)
        return logits - torch.max(logits, 1)[0][:, None]

    def compute_argmax(self, z_a):
        logits = self.compute_logits(z_a)
        return torch.argmax(logits).cpu().item()

    def compute_first_second_argmax(self, z_a):
        logits = self.compute_logits(z_a)
        first = torch.argmax(logits).cpu().item()
        first_value = logits[0,first].cpu().item()
        logits[0,first] = 0
        second = torch.argmax(logits).cpu().item()
        second_value = logits[0,second].cpu().item()
        return first_value, second_value

    def load_goal_states(self):
        goal_states = []
        for gs in sorted(os.listdir(self.path_gs)):
            if 'npy' in gs:
                goal_states.append(np.load(os.path.join(self.path_gs, gs)))
        goal_states = np.array(goal_states)
        goal_states = torch.from_numpy(goal_states).squeeze().float().to(self.dev)
        return goal_states

    def get_goal_state(self, idx):
        return self.goal_states[idx].detach().cpu().numpy()

    def compute_reward(self, z_a, goal, coord=None):
        k = self.compute_argmax(z_a)
        return int(k==goal)
