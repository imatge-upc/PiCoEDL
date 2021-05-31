"""A module for viewing individual streams from the dataset!

To use:
```
    python3 -m minerl.viewer <environment name> <trajectory name>
```
"""
import argparse

from minerl.data import FILE_PREFIX

_DOC_TRAJ_NAME = "{}absolute_zucchini_basilisk-13_36805-50154".format(FILE_PREFIX)


def get_parser():
    parser = argparse.ArgumentParser("python3 -m minerl.viewer")
    parser.add_argument("environment", type=str,
                        help='The MineRL environment to visualize. e.g. MineRLObtainDiamondDense-v0')

    parser.add_argument("stream_name", type=str, nargs='?', default=None,
                        help="(optional) The name of the trajectory to visualize. "
                             "e.g. {}."
                             "".format(_DOC_TRAJ_NAME))
                             
    parser.add_argument("goal_state", type=int, default=0,
                        help="The goal state to visualize. e.g. 0")
    parser.add_argument("trajectory", type=int, default=0,
                        help="The trajectory to visualize. e.g. 0")

    
    return parser
