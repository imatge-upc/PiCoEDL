import sys
import torch

from pip._internal.operations import freeze
from logging import getLogger

from models.PixelEncoder import PixelEncoder
from models.CURL import CURL_PL
from models.CustomVQVAE import VQVAE_PL
from IPython import embed

logger = getLogger(__name__)


def log_versions():
    logger.info(sys.version)  # Python version
    logger.info(','.join(freeze.freeze()))  # pip freeze

def load_encoder(conf, path_weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_type = conf['type']
    img_size = conf['img_size']
    data_type = conf['data_type']

    conf = conf[enc_type]
    encoder_version = conf['encoder_version']
    load_epoch = conf['load_epoch']
    embedding_dim = conf['embedding_dim']


    if enc_type == 'curl':
        obs_shape = (3, img_size, img_size)

        model = CURL_PL(
            obs_shape=obs_shape,
            z_dim=embedding_dim,
            load_goal_states=True,
            device=device,
            path_goal_states=conf['path_goal_states'],
        )


    elif enc_type == 'vqvae':
        model = VQVAE_PL(data_type, 
                         num_hiddens=conf['num_hiddens'],
                         num_residual_layers=conf['num_residual_layers'],
                         num_residual_hiddens=conf['num_residual_hiddens'],
                         num_embeddings=conf['num_embeddings'],
                         embedding_dim=conf['embedding_dim'],
                         commitment_cost=conf['commitment_cost'],
                         reward_type=conf['reward_type'])
    else:
        raise NotImplementedException()

    model = model.to(device)
    weights = torch.load(path_weights / encoder_version / load_epoch)['state_dict']
    model.load_state_dict(weights)
    return model
