"""
Calls the data viewer.
"""

import argparse
import logging
import random
import coloredlogs
import time
import numpy as np
import os
import imageio
import csv
import minerl
from minerl.viewer_custom import get_parser
from minerl.viewer_custom.trajectory_display_controller import TrajectoryDisplayController

coloredlogs.install(logging.DEBUG)
logger = logging.getLogger(__name__)


from IPython import embed

def load_csv(csv_path, type='rewards'):
    with open(csv_path) as csv_file:
        whatever = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if type == 'coords':
                whatever.append(row[::2]) # skip y coordinate
            else:
                whatever.extend(row)
    return np.array(whatever, dtype=np.float32)

def main(opts):
    logger.info("Welcome to the MineRL Stream viewer! \n")

    logger.info("Building data pipeline for {}".format(opts.environment))
    data = minerl.data.make(opts.environment)

    # for _ in data.seq_iter( 1, -1, None, None, include_metadata=True):
    #     print(_[-1])
    #     pass
    if opts.stream_name is None:
        trajs = data.get_trajectory_names()
        opts.stream_name = random.choice(trajs)

    # added videostream name
    gs = opts.goal_state
    tr = opts.trajectory
    files = [x for x in os.listdir(f"../data/MineRLTreechop-v0/{opts.stream_name}") if f"0{tr}.mp4" in x]
    name = files[0]
    data_frames = list(data.load_data(opts.stream_name, include_metadata=False, video_name=name))
    meta = data_frames[0][-1]
    logger.info("Data loading complete!".format(opts.stream_name))
    logger.info("META DATA: {}".format(meta))

    # load rewards
    csv_path_rewards = os.path.join(data.data_dir, opts.stream_name, f"rewards_{gs}.{tr}.csv")

    rewards = load_csv(csv_path_rewards)

    # load goal state using opts.goal_state
    path_gs = './goal_states/vqvae_Expert_pixels_0/'
    goal_state_pov = imageio.imread(os.path.join(path_gs, f'{opts.goal_state}_img.png'))
    goal_state_pov = goal_state_pov[:,:,:3]
    # coord_centroide = np.load(os.path.join(
    #     path_gs, f'{opts.goal_state}_coord.npy'))
    coord_centroide = np.array([0,0,0])
    mean = np.array([2.6988769, 65.450745,  -3.170297])
    std = np.array([18.587471,   3.0484843, 20.42835  ])
    coord_centroide = coord_centroide*std + mean
    print(coord_centroide)

    # load coords
    csv_path_coords = os.path.join(data.data_dir, opts.stream_name, f"coords_{gs}.{tr}.csv")
    coords = load_csv(csv_path_coords, type='coords')
    coords = np.repeat(coords, 10, axis=0)
    rewards = np.repeat(rewards, 10)

    trajectory_display_controller = TrajectoryDisplayController(
        data_frames,
        rewards,
        opts.goal_state,
        goal_state_pov,
        coord_centroide,
        coords,
        header=opts.environment,
        subtext=opts.stream_name,
        vector_display='VectorObf' in opts.environment
    )
    trajectory_display_controller.run()


if __name__ == '__main__':
    main(get_parser().parse_args())
