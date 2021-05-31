import os
import csv
import cv2
import torch

import numpy as np
import matplotlib.pylab as plt

from random import choice, randint, shuffle
from os.path import join
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset

from IPython import embed


class CustomMinecraftData(Dataset):
    def __init__(self, traj_list, transform=None, path='../data', delay=False, **kwargs) -> None:
        self.path = Path(path)
        self.traj_list = traj_list
        self.delay = delay
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.transform = transform
        self.k_std = kwargs['k_std']
        self.k_mean = kwargs['k_mean']
        self.data_type = kwargs['data_type']
        self.loadData()

    """
    Given an index from self.data it returns the last index
    of the trajectory that it belongs to.
    """
    def getTrajLastIdx(self, idx):
        list_idxs = self.list_idxs
        idx_acc = 0
        for i in list_idxs:
            if idx >= idx_acc and idx < (idx_acc + i):
                return idx_acc + i - 1
            idx_acc += i
        return None


    def customLoad(self):
        data, list_idxs = [], []

        for i, traj in enumerate(self.traj_list):
            print(f"\tTraj: {i}", end ='\r')
            obs = np.load(traj, allow_pickle=True)
            data.append(obs)
            list_idxs.append(obs.shape[0])

        print()
        data = np.array(data).reshape(-1, 64, 64, 3)
        self.data = data
        self.list_idxs = list_idxs

    def expertLoad(self):
        data, list_idxs = [], []

        for i, vid in enumerate(self.traj_list):
            print(f"\tVid: {i}", end ='\r')
            video = []

            vid_path = vid / 'recording.mp4'
            frames = cv2.VideoCapture(str(vid_path))
            ret = True
            fc = 0
            while(frames.isOpened() and ret):
                ret, frame = frames.read()
                # if ret and fc % 3 == 0:
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video.append(frame)
                fc += 1

            data.append(video)
            list_idxs.append(len(video))
        data = [y for x in data for y in x]
        self.data = np.array(data)
        self.list_idxs = list_idxs


    def load_trajectories(self):
        print("Loading trajectories...")

        all_trajectories = []

        coord_list = [
            join(
                str(x).split('/')[2],
                'coords_0.' +
                str(x).split('/')[3].split('_')[1].split('.')[0]+'.csv'
            )
            for x in self.traj_list
        ]
        for trajectories in coord_list:
            with open(f"./results/{trajectories}") as csv_file:
                trajectory = []
                csv_reader = csv.reader(csv_file, delimiter=',')
                for i, row in enumerate(csv_reader):
                    trajectory.append(row)
                all_trajectories.append(trajectory)
        coords = np.array(all_trajectories, dtype=np.float32).reshape(-1, 3)
        
        self.coord_mean = coords.mean(axis=0)
        self.coord_std = coords.std(axis=0)

        self.coords = (coords - self.coord_mean)/self.coord_std

    def loadData(self) -> list:
        print('Loading data...')
        if 'Custom' in str(self.traj_list[0]):
            if self.data_type == "pixel":
                self.customLoad()
            elif self.data_type == "coord":
                self.load_trajectories()
            elif self.data_type == "pixelcoord":
                self.customLoad()
                self.load_trajectories()
        else:
            self.expertLoad()
            self.coords = None

    def getCoords(self, idx, key_idx):
        return torch.from_numpy(self.coords[idx]), torch.from_numpy(self.coords[key_idx])

    def getImages(self, idx, key_idx):
        query = self.data[idx]
        key = self.data[key_idx]
        if self.transform is not None:
            key = self.transform(key)
            query = self.transform(query)
        return key, query

    def __len__(self) -> int:
        if self.data_type == "coord":
            return len(self.coords)
        else:
            return len(self.data)

    def __getitem__(self, index):
        # Get query obs
        if self.delay:
            # Make sure that we pick a frame from the same trajectory
            fin_idx = self.getTrajLastIdx(index)
            key_idx = index + int(np.random.rand()*self.k_std + self.k_mean)

            # Get key obs
            key_idx = min(key_idx, fin_idx)
        else:
            key_idx = index

        if self.data_type == "pixel":
            key, query = self.getImages(index, key_idx)
            return torch.stack((key, query))

        elif self.data_type == "coord":
            coord_query, coord_key = self.getCoords(index, key_idx)
            return torch.stack((coord_query, coord_key))

        elif self.data_type == "pixelcoord":
            key, query = self.getImages(index, key_idx)
            coord_query, coord_key = self.getCoords(index, key_idx)
            return torch.stack((query, key)), torch.stack((coord_query, coord_key))



class LatentDataset(Dataset):
    """
    Loads latent block dataset
    """

    def __init__(self, file_name, transform=None):
        print('Loading latent block data')
        p = Path('../data')
        self.fname= file_name
        file_name = self.fname + '_t.npy'
        self.data_t = np.load(p / 'latent_blocks' / file_name, allow_pickle=True)
        file_name = self.fname + '_b.npy'
        self.data_b = np.load(p / 'latent_blocks' / file_name, allow_pickle=True)
        self.transform = transform

    def __getitem__(self, index):
        top = self.data_t[index]
        bottom = self.data_b[index]
        if self.transform is not None:
            top = self.transform(top)
            bottom = self.transform(bottom)
        label = 0
        return top, bottom, label

    def __len__(self):
        return len(self.data_t)

class LatentBlockDataset(Dataset):
    """
    Loads latent block dataset
    """

    def __init__(self, file_path, train=True, shape=16, transform=None):
        print('Loading latent block data')
        self.data = np.load(file_path, allow_pickle=True)
        self.transform = transform
        self.shape = shape

    def __getitem__(self, index):
        img = self.data[index]
        img = img.reshape((self.shape, self.shape))
        if self.transform is not None:
            img = self.transform(img)
        label = 0
        return img, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                            ])
    # env_list = ['MineRLNavigate-v0', 'MineRLNavigateVectorObf-v0']
    # mrl = MultiMinecraftData(env_list, 'train', 1, False, transform=transform)
    # embed()
    # img = mrl[10]
    # plt.imshow(img)
    # plt.show()
    # c = CustomMinecraftData('CustomTrajectories4', 'train', 0.95, transform=transform)
    # embed()
