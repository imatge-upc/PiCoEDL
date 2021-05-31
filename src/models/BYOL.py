import torchvision.models as models
import torch
from torch import nn
import pytorch_lightning as pl


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


class ResNet18(torch.nn.Module):
    def __init__(self, mlp_hidden_size, projection_size):
        super(ResNet18, self).__init__()
        resnet = models.resnet18(pretrained=False)
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(
            in_channels=resnet.fc.in_features,
            mlp_hidden_size=mlp_hidden_size,
            projection_size=projection_size)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)

class BYOL_PL(pl.LightningModule):
    def __init__(self, mlp_hidden_size=512, projection_size=128):
        super(BYOL_PL, self).__init__()

        self.online_network = ResNet18(mlp_hidden_size, projection_size)
        self.target_network = ResNet18(mlp_hidden_size, projection_size)

        self.predictor = MLPHead(
            in_channels=self.online_network.projetion.net[-1].out_features,
            mlp_hidden_size=mlp_hidden_size,
            projection_size=projection_size)

    def forward(self, batch_view_1, batch_view_2):
        p1 = self.predictor(self.online_network(batch_view_1))
        p2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            t2 = self.target_network(batch_view_1)
            t1 = self.target_network(batch_view_2)

        return p1,p2,t1,t2
