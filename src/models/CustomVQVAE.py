import wandb
import torch

import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn

import matplotlib.pyplot as plt
import torch.nn.functional as F
import pytorch_lightning as pl

from plot import *
from scipy.signal import savgol_filter
from torchvision.utils import make_grid

from IPython import embed

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def indices2quantized(self, indices, batch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoding_indices = indices.view(-1).unsqueeze(1)  # [B*256, 1]
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=device)
        encodings.scatter_(1, encoding_indices, 1) # [B*256,512]

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight) # [256,64]
        quantized = quantized.view((batch,16,16,64)) # [B,16,16,64]

        return quantized.permute(0, 3, 1, 2).contiguous()

    def compute_distances(self, inputs):
        # inputs = inputs.permute(0, 2, 3, 1).contiguous() # [1,16,16,64]
        # input_shape = inputs.shape
        # flat_input = inputs.view(-1, self._embedding_dim) # [256,64]

        distances = (torch.sum(inputs**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self._embedding.weight.t()))
        return distances

    def forward(self, inputs):
        # Comments on the right correspond to example for one image
        # convert inputs from BCHW -> BHWC

        # Calculate distances
        distances = (torch.sum(inputs**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self._embedding.weight.t()))
        # distances shape [256,512]

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # [256,1]

        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1) # [256,512]

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight) # [256,64]

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), inputs)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encoding_indices


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//4,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//4,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_4 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_5 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        x = F.relu(x)

        x = self._conv_4(x)
        x = F.relu(x)

        x = self._conv_5(x)

        return self._residual_stack(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        # self._conv_1 = nn.Conv2d(in_channels=in_channels,
        #                          out_channels=num_hiddens,
        #                          kernel_size=3,
        #                          stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens//2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                                out_channels=num_hiddens//4,
                                                kernel_size=4,
                                                stride=2, padding=1)
        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=num_hiddens//4,
                                                out_channels=num_hiddens//4,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_4 = nn.ConvTranspose2d(in_channels=num_hiddens//4,
                                                out_channels=num_hiddens//4,
                                                kernel_size=4,
                                                stride=2, padding=1)
        self._conv_trans_5 = nn.ConvTranspose2d(in_channels=num_hiddens//4,
                                                out_channels=3,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        # x = self._conv_1(inputs)

        x = self._residual_stack(inputs)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        x = self._conv_trans_2(x)
        x = F.relu(x)

        x = self._conv_trans_3(x)
        x = F.relu(x)

        x = self._conv_trans_4(x)
        x = F.relu(x)

        return self._conv_trans_5(x)


class PixelVQVAE(pl.LightningModule):
    def __init__(self, num_hiddens=64, num_residual_layers=2, num_residual_hiddens=32,
                 num_embeddings=10, embedding_dim=256, commitment_cost=0.25, decay=0.99,
                 img_size=64, coord_cost=0.05, reward_type="sparse"):
        
        super(PixelVQVAE, self).__init__()

        self.img_size = img_size

        self.n_h = num_hiddens
        self.k = int(2 * (self.img_size / self.n_h))

        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self.img_mlp = nn.Sequential(
            nn.Linear(self.n_h * self.k * self.k, int(embedding_dim)),
            nn.ReLU()
        )

        self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                          commitment_cost, decay)

        self.img_mlp_inv = nn.Sequential(
            nn.Linear(int(embedding_dim), self.n_h * self.k * self.k),
            nn.ReLU()
        )

        self._decoder = Decoder(embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

    def forward(self, batch, batch_idx, logger, set):
        img = batch

        i1, i2 = img[:, 0], img[:, 1]

        z = self.encode(i1)

        vq_loss, quantized, perplexity, _ = self._vq_vae(z)

        img_recon = self.decode(quantized)

        img_recon_error = F.mse_loss(img_recon, i2)

        loss = img_recon_error + vq_loss

        logs = {
            f'loss/{set}': loss,
            f'perplexity/{set}': perplexity,
            f'loss_img_recon/{set}': img_recon_error,
            f'loss_vq_loss/{set}': vq_loss
        }

        self.log_metrics(logger, logs, img_recon, batch_idx, set)

        return loss

    def encode(self, img):
        z_1 = self._encoder(img)
        z_1_shape = z_1.shape
        z_1 = z_1.view(z_1_shape[0], -1)
        return self.img_mlp(z_1)

    def decode(self, z):
        z = self.img_mlp_inv(z)
        h_i = z.view(-1, self.n_h, self.k, self.k)
        return self._decoder(h_i)

    def compute_embedding(self, batch, device):
        img = batch

        i1, _ = img[:, 0], img[:, 1]

        return self.encode(i1.to(device))

    def log_metrics(self, logger, logs, img_recon, batch_idx, set='train'):
        logger.experiment.log(logs)
        if batch_idx == 0 and set == 'val':
            grid = make_grid(img_recon[:64].cpu().data)
            grid = grid.permute(1, 2, 0)
            logger.experiment.log({"Images": [wandb.Image(grid.numpy())]})

    def list_reconstructions(self):
        with torch.no_grad():
            img_list = []
            for e in self._vq_vae._embedding.weight:
                img_recon = self.decode(e)
                img_recon = img_recon.squeeze().permute(1, 2, 0)
                img_list.append(img_recon.detach().cpu().numpy())
        return img_list, None

    def log_reconstructions(self, loader, logger):

        img_list, _ = self.list_reconstructions()
        fig_img = plot_img_centroides(img_list)
        logger.experiment.log({'Centroides images': fig_img})
        plt.close()


class CoordVQVAE(pl.LightningModule):
    def __init__(self, num_hiddens=64, num_residual_layers=2, num_residual_hiddens=32,
                 num_embeddings=10, embedding_dim=256, commitment_cost=0.25, decay=0.99,
                 img_size=64, coord_cost=0.05, reward_type="sparse"):
        
        super(CoordVQVAE, self).__init__()

        self.coord_mlp = nn.Sequential(
            nn.Linear(3, int(embedding_dim/2)),
            nn.ReLU(),
            nn.Linear(int(embedding_dim/2), embedding_dim)
        )

        self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                          commitment_cost, decay)

        self.coord_mlp_inv = nn.Sequential(
            nn.Linear(embedding_dim, int(embedding_dim/2)),
            nn.ReLU(),
            nn.Linear(int(embedding_dim/2), 3)
        )

    def forward(self, batch, batch_idx, logger, set):
        coords = batch

        c1, c2 = coords[:, 0], coords[:, 1]

        z = self.encode(c1)

        vq_loss, quantized, perplexity, _ = self._vq_vae(z)

        coord_recon = self.decode(quantized)

        coord_recon_error = F.mse_loss(coord_recon, c2)


        loss = coord_recon_error + vq_loss

        logs = {
            f'loss/{set}': loss,
            f'perplexity/{set}': perplexity,
            f'loss_coord_recon/{set}': coord_recon_error,
            f'loss_vq_loss/{set}': vq_loss
        }

        logger.experiment.log(logs)

        return loss

    def encode(self, coords):
        return self.coord_mlp(coords)

    def decode(self, z):
        return self.coord_mlp_inv(z)

    def compute_embedding(self, batch, device):
        coords = batch

        c1, _ = coords[:, 0], coords[:, 1]

        return self.encode(c1.to(device))

    def list_reconstructions(self):
        with torch.no_grad():
            coord_list = []
            for e in self._vq_vae._embedding.weight:
                coord_recon = self.decode(e)
                coord_list.append(coord_recon.detach().cpu().numpy())
        return None, coord_list

    def log_reconstructions(self, loader, logger):

        _, coord_list = self.list_reconstructions()
        fig_coord = plot_coord_centroides(coord_list, loader)
        logger.experiment.log({'Centroides coordinates': fig_coord})
        plt.close()


class PixelCoordVQVAE(pl.LightningModule):
    def __init__(self, num_hiddens=64, num_residual_layers=2, num_residual_hiddens=32,
                 num_embeddings=10, embedding_dim=256, commitment_cost=0.25, decay=0.99,
                 img_size=64, coord_cost=0.05, reward_type="sparse"):

        super(PixelCoordVQVAE, self).__init__()

        self.img_size = img_size
        self.coord_cost = coord_cost
        
        self.n_h = num_hiddens
        self.k = int(2 * (self.img_size / self.n_h))

        self._encoder = Encoder(3, self.n_h,
                        num_residual_layers,
                        num_residual_hiddens)


        self.img_mlp = nn.Sequential(
            nn.Linear(self.n_h * self.k * self.k, int(embedding_dim)),
            nn.ReLU()
        )   


        self.coord_mlp = nn.Sequential(
            nn.Linear(3, int(embedding_dim/2)),
            nn.ReLU(),
            nn.Linear(int(embedding_dim/2), embedding_dim)
        )

        self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                        commitment_cost, decay)

        self.img_mlp_inv = nn.Sequential(
            nn.Linear(int(embedding_dim), self.n_h * self.k * self.k),
            nn.ReLU()
        )

        self.coord_mlp_inv = nn.Sequential(
            nn.Linear(embedding_dim, int(embedding_dim/2)),
            nn.ReLU(),
            nn.Linear(int(embedding_dim/2), 3)
        )
        self._decoder = Decoder(embedding_dim,
                                self.n_h,
                                num_residual_layers,
                                num_residual_hiddens)

    def forward(self, batch, batch_idx, logger, set):
        img, coords = batch

        i1, i2 = img[:, 0], img[:, 1]
        c1, c2 = coords[:, 0], coords[:, 1]

        z = self.encode((i1, c1))

        vq_loss, quantized, perplexity, _ = self._vq_vae(z)

        img_recon, coord_recon = self.decode(quantized)

        img_recon_error = F.mse_loss(img_recon, i2)
        coord_recon_error = F.mse_loss(coord_recon, c2)

        coord_recon_error = self.coord_cost*coord_recon_error

        loss = img_recon_error + coord_recon_error + vq_loss
        
        logs = {
            f'loss/{set}': loss,
            f'perplexity/{set}': perplexity,
            f'loss_img_recon/{set}': img_recon_error,
            f'loss_coord_recon/{set}': coord_recon_error,
            f'loss_vq_loss/{set}': vq_loss
        }

        self.log_metrics(logger, logs, img_recon, batch_idx, set)

        return loss


    def encode(self, batch):
        img, coords = batch
        z_1 = self._encoder(img)
        z_1_shape = z_1.shape
        z_1 = z_1.view(z_1_shape[0], -1)
        z_1 = self.img_mlp(z_1)
        z_2 = self.coord_mlp(coords)
        return torch.add(z_1, z_2)

    def decode(self, z):
        z = self.img_mlp_inv(z)
        h_i = z.view(-1, self.n_h, self.k, self.k)
        img = self._decoder(h_i)
        coord = self.coord_mlp_inv(z)
        return img, coord

    def compute_embedding(self, batch, device):
        img, coords = batch

        i1, _ = img[:, 0], img[:, 1]
        c1, _ = coords[:, 0], coords[:, 1]

        return self.encode((i1.to(device), c1.to(device)))

    def log_metrics(self, logger, logs, img_recon, batch_idx, set='train'):
        logger.experiment.log(logs)

        if batch_idx == 0 and set == 'val':
            grid = make_grid(img_recon[:64].cpu().data)
            grid = grid.permute(1,2,0)
            logger.experiment.log({"Images": [wandb.Image(grid.numpy())]})


    def list_reconstructions(self):
        with torch.no_grad():
            img_list = []
            coord_list = []
            for e in self._vq_vae._embedding.weight:
                img_recon, coord_recon = self.decode(e)
                img_recon = img_recon.squeeze().permute(1, 2, 0)
                img_list.append(img_recon.detach().cpu().numpy())
                coord_list.append(coord_recon.detach().cpu().numpy())
        return img_list, coord_list

    def log_reconstructions(self, loader, logger):

        img_list, coord_list = self.list_reconstructions()

        fig_coord = plot_coord_centroides(coord_list, loader)
        logger.experiment.log({'Centroides coordinates': fig_coord})
        plt.close()

        fig_img = plot_img_centroides(img_list)
        logger.experiment.log({'Centroides images': fig_img})
        plt.close()



class VQVAE_PL(pl.LightningModule):
    def __init__(self, input, **kwargs):
        super(VQVAE_PL, self).__init__()

        self.num_goal_states = kwargs["num_embeddings"]
        self.reward_type = kwargs["reward_type"]
        self.input = input
        
        if input == "pixel":
            self.model = PixelVQVAE(**kwargs)
        elif input == "coord":
            self.model = CoordVQVAE(**kwargs)
        elif input == "pixelcoord":
            self.model = PixelCoordVQVAE(**kwargs)
        else:
            self.model = None


    def encode(self, batch):
        return self.model.encode(batch)
        
    def compute_embedding(self, batch, device):
        return self.model.compute_embedding(batch, device)

    def compute_logits_(self, z_a, z_pos):
        distances = self.model._vq_vae.compute_distances(z_a)
        return -distances.squeeze()[z_pos].detach().cpu().item()

    def compute_argmax(self, z_a):
        distances = self.model._vq_vae.compute_distances(z_a)
        # it's the same as argmax of (-distances)
        return torch.argmin(distances).cpu().item()

    def compute_reward(self, z_a, goal, coord=None):
        distances = self.model._vq_vae.compute_distances(z_a).squeeze()
        k = torch.argmin(distances).cpu().item()
        if self.reward_type == "dense":
            return - (1/z_a.view(-1).shape[0]) * distances[goal].detach().cpu().item()
        elif self.reward_type == "sparse":
            return int(k==goal)
            # if k == goal:
            #     return - (1/z_a.view(-1).shape[0]) * distances[goal].detach().cpu().item()
            # else:
            #     return -0.5
        elif self.reward_type == "comb":
            if k == goal:
                return - (1/z_a.view(-1).shape[0]) * distances[goal].detach().cpu().item()
            else:
                if not self.input == "pixel":
                    with torch.no_grad():
                        z_idx = torch.tensor(goal).cuda()
                        goal_embedding = torch.index_select(self.model._vq_vae._embedding.weight.detach(), dim=0, index=z_idx)
                        _, coord_goal = self.model.decode(goal_embedding)
                        coord_goal = coord_goal.detach().cpu().numpy()
                    return - np.linalg.norm(coord-coord_goal)
                return -0.5
        else:
            raise NotImplementedError()

    def get_goal_state(self, idx):
        z_idx = torch.tensor(idx).cuda()
        embeddings = torch.index_select(
            self.model._vq_vae._embedding.weight.detach(), dim=0, index=z_idx)
        return embeddings.squeeze().detach().cpu().numpy()
