import os
import csv
import torch

import numpy as np
import pandas as pd
import seaborn as sns

from plot import *
from os.path import join
from pathlib import Path
from sklearn.cluster import KMeans
from collections import Counter
from torch.utils.data import DataLoader, Subset
from customLoader import *
from torchvision.transforms import transforms

from IPython import embed


def get_loader(trajectories, transform, conf, shuffle=False, limit=None):
    train, _ = get_train_val_split(trajectories, 1)
    train_dataset = CustomMinecraftData(train, transform=transform, delay=False, **conf)

    if not limit == None:
        train_dataset = Subset(train_dataset, limit)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=shuffle, num_workers=0)
    return train_dataloader

def compute_kmeans(embeddings, num_clusters):
    return KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)

def compute_embeddings(loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return np.array([model.compute_embedding(batch, device).detach().cpu().numpy() for batch in loader]).squeeze()

def get_images(loader):
    return torch.cat([data[:,0] for data in loader])


def load_trajectories(trajectories, limit=None):
    print("Loading trajectories...")

    all_trajectories = []
    files = sorted([x for x in os.listdir(f"./results/{trajectories}/") if 'coords' in x], key=lambda x: int(x.split('.')[1]))
    for file in files:
        with open(f"./results/{trajectories}/{file}") as csv_file:
            trajectory = []
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for i, row in enumerate(csv_reader):
                trajectory.append(row)
            all_trajectories.append(trajectory)
    trajs = np.array(all_trajectories).reshape(-1, 3)
    if not limit == None:
        return trajs[limit]
    return trajs


def construct_map(enc):
    if not enc.limit == None:
        limit = [x*10 for x in range(enc.limit)]
    else: limit = None
    loader = get_loader(
        enc.trajectories,
        enc.transform,
        enc.conf,
        shuffle=enc.shuffle,
        limit=limit)
    if 'Custom' in enc.trajectories[0]:
        trajectories = load_trajectories(enc.trajectories[0], limit)

    embeddings = compute_embeddings(loader, enc)

    if enc.type == "index":
        indexes = get_indexes(trajectories, embeddings, enc)
        index_map(enc, indexes)
    elif enc.type == "reward":
        reward_map(trajectories, embeddings, enc, loader)
    elif enc.type == "embed":
        images = get_images(loader) + 0.5
        embed_map(embeddings, images, enc.experiment)
    elif enc.type == "centroides":
        indexes = get_indexes(trajectories, embeddings, enc)
        centroides_map(enc, loader, indexes)
    else:
        raise NotImplementedError()


def get_indexes(trajectories, embeddings, enc):
    print("Get index from all data points...")
    values = pd.DataFrame(columns=['x', 'y', 'Code:'])
    for i, (e, p) in enumerate(zip(embeddings, trajectories)):
        x = float(p[2])
        y = float(p[0])
        e = torch.from_numpy(e).cuda()
        k = enc.compute_argmax(e.unsqueeze(dim=0))
        if k==3:
            values = values.append(
                {'x': x, 'y': y, 'Code:': int(k)}, ignore_index=True)

    values['Code:'] = values['Code:'].astype('int32')
    return values

def centroides_map(encoder, loader, indexes):
    experiment = encoder.experiment

    _, coord_list = encoder.model.list_reconstructions()
    world = getWorld(encoder.trajectories[0])
    palette = sns.color_palette("Paired", n_colors=encoder.num_clusters)

    experiment = encoder.test['path_weights'].split('/')[0]
    centroides_indexmap(coord_list, indexes, palette, experiment, world, loader)

def index_map(enc, indexes):
    code_list = indexes['Code:'].tolist()
    codes_count = Counter(code_list)
    palette = sns.color_palette("Paired", n_colors=len(list(set(code_list))))

    experiment = enc.test['path_weights'].split('/')[0]
    world = getWorld(enc.trajectories[0])

    plot_idx_maps(indexes, palette, experiment, world)
    skill_appearance(codes_count, palette, experiment, world)


def reward_map(trajectories, embeddings, enc, loader):
    print("Get index from all data points...")
    data_list = []
    for g in range(enc.num_clusters):
        print(f"Comparing data points with goal state {g}", end="\r")
        values = pd.DataFrame(columns=['x', 'y', 'reward'])
        for i, (e, p) in enumerate(zip(embeddings, trajectories)):
            x = float(p[2])
            y = float(p[0])
            e = torch.from_numpy(e).cuda()

            coord = None
            if not enc.conf["data_type"] == "pixel":
                coord = np.array(p, dtype=np.float32)
                mu = loader.dataset.coord_mean
                std = loader.dataset.coord_std
                coord = (coord-mu)/std
            r = enc.compute_reward(e.unsqueeze(dim=0), g, coord)

            values = values.append({'x': x, 'y': y, 'reward': r}, ignore_index=True)


        data_list.append(values)
    
    experiment = enc.test['path_weights'].split('/')[0]
    
    plot_reward_maps(data_list, experiment, getWorld(enc.trajectories[0]), enc.reward_type)

def embed_map(embeddings, images, exp):
    import tensorflow
    from torch.utils.tensorboard import SummaryWriter
    import tensorboard

    tensorflow.io.gfile = tensorboard.compat.tensorflow_stub.io.gfile
    writer = SummaryWriter(log_dir=os.path.join("./results", exp))
    writer.add_embedding(embeddings, label_img=images)
    writer.close()

def trainValSplit(traj_list, split):
    num_traj = len(traj_list)
    if split == 1:
        return traj_list, []
    else:
        # Since we can mix trajectories from different tasks, we want to shuffle them
        # e.g: otherwise we could have all treechop trajectories as validation
        shuffle(traj_list)
        return traj_list[:int(split*num_traj)], traj_list[int(split*num_traj):]

def get_train_val_split(trajectories, split):
    path = Path('../data')
    total_t = []
    if 'Custom' in trajectories[0]:
        for t in trajectories:
            items = sorted(os.listdir(path / t), key=lambda x: int(x.split('.')[0].split('_')[1]))
            items = [path / t / x for x in items]
            total_t.extend(items)
    else:
        for t in trajectories:
            items = sorted(os.listdir(path / t))
            items = [path / t / x for x in items]
            total_t.extend(items)
    return trainValSplit(total_t, split)


'''
Mapping from Trajectories to Worlds.
We can have multiple datasets of trajectories that belong to a unique world.
'''
def getWorld(t):
    if '13' in t or '14' in t:
        return 'Simple'
    elif '15' in t or '16' in t:
        return 'Toy'
    elif '17' in t or '18' in t:
        return '5Circles'
    elif '8' in t or '9' in t:
        return 0
    elif '10' in t:
        return 1
    elif '11' in t:
        return 2
    elif '12' in t:
        return 4
    elif 'Test' in t:
        return 'Test'
    else:
        raise NotImplementedError()
