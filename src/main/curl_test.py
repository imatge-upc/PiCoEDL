import os
import sys
import torch

from os.path import join
import numpy as np
from pathlib import Path
from config import setSeed, getConfig
from main.curl import CURL

from IPython import embed

setSeed(0)
assert len(sys.argv) == 2, "Indicate a configuration file like 'config_0.0'"
conf = getConfig(sys.argv[1])


if os.getenv('USER') == 'juanjo':
    path_weights = Path('./results')
elif os.getenv('USER') == 'juan.jose.nieto':
    path_weights = Path('/home/usuaris/imatge/juan.jose.nieto/mineRL/src/results')
else:
    raise Exception("Sorry user not identified!")

conf['curl']['path_goal_states'] = conf['test']['path_goal_states']
conf['curl']['load_goal_states'] = True
conf['curl']['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(conf['test']['path_goal_states']):
    os.mkdir(conf['test']['path_goal_states'])

curl = CURL(conf).cuda()
checkpoint = torch.load(join(path_weights, conf['test']['path_weights']))
curl.load_state_dict(checkpoint['state_dict'])

# Do it in two steps since anyways we need to store goal states in numpy arrays.
# First compute and store goal states (centroides with kmeans)
# Then only compute index_maps or reward_maps

# curl.store_goal_states()

curl._construct_map()
