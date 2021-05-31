import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import savgol_filter
import pytorch_lightning as pl
from IPython import embed


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

        self.emb_indexes = []

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        self.emb_indexes.extend(encoding_indices.cpu().detach().numpy()[0])

        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

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
        inputs = inputs.permute(0, 2, 3, 1).contiguous() # [1,16,16,64]
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim) # [256,64]

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        return distances

    def forward(self, inputs):
        # Comments on the right correspond to example for one image
        # convert inputs from BCHW -> BHWC

        # inputs shape [1,64,16,16]
        inputs = inputs.permute(0, 2, 3, 1).contiguous() # [1,16,16,64]
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim) # [256,64]

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        # distances shape [256,512]

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # [256,1]

        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1) # [256,512]

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight) # [256,64]
        quantized = quantized.view(input_shape) # [1,16,16,64]

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encoding_indices

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

class VQVAE_PL(pl.LightningModule):
    def __init__(self, num_hiddens=64, num_residual_layers=2, num_residual_hiddens=32,
                 num_embeddings=10, embedding_dim=256, commitment_cost=0.25, decay=0.99, goals=[]):
        super(VQVAE_PL, self).__init__()
        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self.goals = goals
        self.num_goal_states = len(goals)

    def forward(self, x):
        z = self._encoder(x)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity

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

    def encode(self, x):
        return self._encoder(x)

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
