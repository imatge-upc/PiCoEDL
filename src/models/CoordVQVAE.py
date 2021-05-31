import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import savgol_filter
import pytorch_lightning as pl
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



class VQVAE_PL(pl.LightningModule):
    def __init__(self, num_hiddens=64, num_residual_layers=2, num_residual_hiddens=32,
                 num_embeddings=10, embedding_dim=256, commitment_cost=0.25, decay=0.99, goals=[]):
        super(VQVAE_PL, self).__init__()

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

        self.goals = goals
        self.num_goal_states = len(goals)

    def forward(self, _, coords):
        z = self.coord_mlp(coords)
        
        loss, quantized, perplexity, _ = self._vq_vae(z)

        coord_recon = self.coord_mlp_inv(quantized)

        return loss, 0, coord_recon, perplexity

    def decode(self, img=True, coords=True):
        img_list = []
        coord_list = []
        for e in self._vq_vae._embedding.weight:
            if img:
                h_i = e.view(1, 64, 2, 2)
                img_list.append(self._decoder(h_i).detach().cpu().numpy())
            if coords:
                coord_list.append(self.coord_mlp_inv(e).detach().cpu().numpy())
        return img_list, coord_list
        
    def get_centroids(self, idx):
        z_idx = torch.tensor(idx).cuda()
        embeddings = torch.index_select(self._vq_vae._embedding.weight.detach(), dim=0, index=z_idx)
        embeddings = embeddings.view((1,2,2,64))
        embeddings = embeddings.permute(0, 3, 1, 2).contiguous()

        return self._decoder(embeddings)

    def save_encoding_indices(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        _, _, _, encoding_indices = self._vq_vae(z)
        return encoding_indices

    def encode(self, imgs, coords):

        return self.coord_mlp(coords)

    def compute_logits_(self, z_a, z_pos):
        distances = self._vq_vae.compute_distances(z_a)
        return -distances.squeeze()[z_pos].detach().cpu().item()

    def compute_argmax(self, z_a):
        distances = self._vq_vae.compute_distances(z_a)
        # it's the same as argmax of (-distances)
        return torch.argmin(distances).cpu().item()

    def compute_second_argmax(self, z_a):
        distances = self._vq_vae.compute_distances(z_a)
        # it's the same as argmax of (-distances)
        first = torch.argmin(distances).cpu().item()
        distances[0][first] = 100
        return torch.argmin(distances).cpu().item()

    def compute_reward(self, z_a, goal):
        distances = self._vq_vae.compute_distances(z_a).squeeze()
        return - (1/z_a.view(-1).shape[0]) * distances[goal].detach().cpu().item()



    def get_goal_state(self, idx):
        z_idx = torch.tensor(idx).cuda()
        embeddings = torch.index_select(self._vq_vae._embedding.weight.detach(), dim=0, index=z_idx)
        return embeddings.squeeze().detach().cpu().numpy()


