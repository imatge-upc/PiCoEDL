import os
import sys
import csv
import numpy as np
import imageio
import matplotlib

from abc import ABC, abstractmethod

from matplotlib.animation import PillowWriter
from mpl_toolkits import mplot3d

import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go
import pandas as pd

from collections import defaultdict
plt.rcParams.update({'font.size': 15})



class Trajectory(ABC):
    def __init__(self, experiment, init_state, goal_states, fix_lim=False, filter_traj=None):

        self.current_goal_state = None
        self.experiment = experiment
        self.fix_lim = fix_lim
        self.filter = filter_traj

        self.initial_state = init_state
        self.goal_states = goal_states

        self.hist_gstate = defaultdict(int)

        # raw data
        self.files_ = os.listdir(f"../results/{self.experiment}/")
        self.files = os.listdir(f"../results/{self.experiment}/")
        self.data = self.load_data()
        self.rewards = self.load_rewards()

        # data organized by goal states
        self.data_list = self.dict2list(self.data)
        self.reward_list = self.dict2list(self.rewards)

        self.num_gstates = len(self.reward_list)

        # all trajectories together
        self.data_reshaped = self.list_reshape(self.data_list)
        self.reward_reshaped = self.list_reshape(self.reward_list)

        # no trajectories at all
        d = np.array(self.data_reshaped, dtype=np.float32)
        r = np.array(self.reward_reshaped, dtype=np.float32)
        self.data_points = np.reshape(d, (d.shape[0]*d.shape[1], -1))
        self.reward_points = np.reshape(r, -1)

        self.path_gif = '../../images/gifs/'

        if not os.path.exists(os.path.join(self.path_gif, self.experiment)):
            os.mkdir(os.path.join(self.path_gif, self.experiment))

    def load_rewards(self):
        data = defaultdict(list)
        files = sorted([x for x in self.files if 'rewards' in x], key=lambda x: int(x.split('.')[1]))
        for file in files:
            goal_state = int(file.split('_')[1][0])
            trajectory = int(file.split('.')[1])

            if self.filter is not None and trajectory < self.filter:
                continue
            self.hist_gstate[goal_state] += 1


            with open(f"../results/{self.experiment}/{file}") as csv_file:
                trajectory = []
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for i, row in enumerate(csv_reader):
                    trajectory.append(row)
            data[goal_state].append(trajectory)
        return data

    def load_data(self):
        data = defaultdict(list)
        files = sorted([x for x in self.files if 'coords' in x], key=lambda x: int(x.split('.')[1]))
        for file in files:
            goal_state = int(file.split('_')[1][0])
            trajectory = int(file.split('.')[1])

            if self.filter is not None and trajectory < self.filter:
                continue
            self.hist_gstate[goal_state] += 1


            with open(f"../results/{self.experiment}/{file}") as csv_file:
                trajectory = []
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for i, row in enumerate(csv_reader):
                    trajectory.append(row)
            data[goal_state].append(trajectory)
        return data

    def dict2list(self, data):
        return [v for k,v in sorted(data.items())]

    def list_reshape(self, data):
        return [j for d in data for j in d]

    def plot_list(self, traj_list, ax):
        for traj in traj_list:
            self.plot_trajectory(traj, ax)

    def plot_standard(self, ax):
        self.plot_goal_states(ax)
        self.plot_init_state(ax)
        self.plot_labels(ax)
        self.plot_lim(ax)

    def save_imgs(self):
        # Save figures as png files
        for i in range(len(self.data_reshaped)):
            all_traj = self.data_reshaped[0:i]
            fig, ax = plt.subplots(figsize=(15,15))
            for i,traj in enumerate(all_traj):
                self.plot_trajectory(traj, ax)
            self.plot_standard(ax)
            plt.savefig(os.path.join(self.path_gif, self.experiment, str(i) + '.png'))
            plt.close()

    def save_gif(self):
        # Load files and create gif
        path_exp = os.path.join(self.path_gif, self.experiment)
        imgs = []
        for file in sorted(os.listdir(path_exp), key=lambda x: int(x.split('.')[0])):
            imgs.append(imageio.imread(os.path.join(path_exp, file)))
            os.remove(os.path.join(path_exp, file))
        imageio.mimsave(os.path.join(path_exp, self.experiment + '.gif'), imgs, 'GIF', duration=0.1)


    def generate_gif(self):
        self.save_imgs()
        self.save_gif()

    @abstractmethod
    def plot_lim(self):

        raise NotImplementedError()

    @abstractmethod
    def plot_trajectory(self):

        raise NotImplementedError()

    @abstractmethod
    def plot_goal_states(self):

        raise NotImplementedError()

    @abstractmethod
    def plot_init_state(self):

        raise NotImplementedError()

    @abstractmethod
    def plot_point(self):

        raise NotImplementedError()

    @abstractmethod
    def plot_labels(self, ax):

        raise NotImplemetedError()

    @abstractmethod
    def plot_per_goal_state(self):

        raise NotImplementedError()


    @abstractmethod
    def plot_all_together(self):

        raise NotImplementedError()

    @abstractmethod
    def plot_pointcloud(self):

        raise NotImplementedError()

    @abstractmethod
    def plot_histogram(self, labels):

        raise NotImplementedError()


class Trajectory3D(Trajectory):
    def __init__(self, experiment, init_state, goal_states, fix_lim=False, filter_traj=None):
        super().__init__(experiment, init_state, goal_states, fix_lim=fix_lim, filter_traj=filter_traj)

        self.current_goal_state = None

        self.vertex = [
            [
                [-10, -10, 10, 10, -10, -10, 10, 10],
                [10, 30, 30, 10, 10, 30, 30, 10],
            ],
            [
                [10, 10, 30, 30, 10, 10, 30, 30],
                [10, 30, 30, 10, 10, 30, 30, 10],
            ],
            [
                [10, 10, 30, 30, 10, 10, 30, 30],
                [-10, 10, 10, -10, -10, 10, 10, -10],
            ],
            [
                [10, 10, 30, 30, 10, 10, 30, 30],
                [-30, -10, -10, -30, -30, -10, -10, -30],
            ],
            [
                [-10, -10, 10, 10, -10, -10, 10, 10],
                [-30, -10, -10, -30, -30, -10, -10, -30],
            ],
            [
                [-30, -30, -10, -10, -30, -30, -10, -10],
                [-30, -10, -10, -30, -30, -10, -10, -30],
            ],
            [
                [-30, -30, -10, -10, -30, -30, -10, -10],
                [-10, 10, 10, -10, -10, 10, 10, -10],
            ],
            [
                [-30, -30, -10, -10, -30, -30, -10, -10],
                [10, 30, 30, 10, 10, 30, 30, 10],
            ],
        ]


    def plot_trajectory(self, traj, ax):
        d = np.array(traj, dtype=np.float32)
        ax.plot3D(d[:,0], d[:,2], d[:,1])


    def plot_goal_states(self, ax):
        for i,(x,y,z) in enumerate(self.goal_states):
            if self.current_goal_state == i:
                ax.plot(x, y, z, marker='o', color='blue', markersize=13)
            else:
                ax.plot(x, y, z, marker='o', color='black', markersize=10)
            text = '   ' + str(x) + ', ' + str(y) + ', ' + str(z)
            ax.text(x, y, z, text)


    def plot_init_state(self, ax):
        x,y,z = self.initial_state
        ax.plot(x,y,z, marker='o', markersize=10)
        text = '   ' + str(x) + ', ' + str(y) + ', ' + str(z)
        ax.text(x, y, z, text)


    def plot_point(self, point, reward, ax):
        x,_,y = point
        z = reward
        ax.plot(x, y, z, color='blue', marker='.', markersize=1)


    def plot_labels(self, ax):
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    def plot_lim(self, ax):
        if self.fix_lim:
            ax.set_xlim(-55, 55)
            ax.set_ylim(-55, 55)
            ax.set_zlim(-10, 30)
        ax.view_init(elev=80, azim=-30)


    def plot_per_goal_state(self):
        fig = plt.figure(figsize=(20,20))
        for i, goal_state in enumerate(self.data_list):
            self.current_goal_state = i
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            for traj in goal_state:
                self.plot_trajectory(traj, ax)

            self.plot_standard(ax)
        self.current_goal_state = None

    def plot_all_together(self):
        fig, ax = plt.subplots(figsize=(15,15))
        ax = plt.axes(projection='3d')
        for traj in self.data_reshaped:
            self.plot_trajectory(traj, ax)

        self.plot_standard(ax)

    def plot_pointcloud(self):
        fig, ax = plt.subplots(figsize=(15,15))
        ax = plt.axes(projection='3d')
        colorsMap = 'jet'
        cs = self.reward_points.copy()
        cm = plt.get_cmap(colorsMap)
        cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        b = self.data_points
        ax.scatter(b[:,0], b[:,2], self.reward_points, c=scalarMap.to_rgba(cs))
        scalarMap.set_array(cs)
        fig.colorbar(scalarMap)
        self.plot_standard(ax)
        plt.show()

    def plot_pointcloud_(self):
        data_dict = {'x': self.data_points[:,0], 'y': self.data_points[:,2], 'z': self.reward_points}
        df = pd.DataFrame.from_dict(data_dict)
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='z')

        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.write_html(f"./vqvae_1/{self.experiment}.html")

    def plot_histogram(self, labels):
        pass

class Trajectory2D(Trajectory):
    def __init__(self, experiment, init_state, goal_states, fix_lim=False, filter_traj=None):
        super().__init__(experiment, init_state, goal_states, fix_lim=fix_lim, filter_traj=filter_traj)

        self.current_goal_state = None

    def plot_trajectory(self, traj, ax):
        d = np.array(traj, dtype=np.float32)
        ax.plot(d[:,0], d[:,2])

    def plot_goal_states(self, ax):

        for i,(x,y) in enumerate(self.goal_states):
            if self.current_goal_state == i:
                ax.plot(x, y, marker='o', color='blue', markersize=13)
            else:
                ax.plot(x, y, marker='o', color='black', markersize=10)
            text = '   ' + str(x) + ', ' + str(y)
            ax.text(x, y, text)

    def plot_init_state(self, ax):

        x,y = self.initial_state
        ax.plot(x,y, marker='o', markersize=10)
        text = '   ' + str(x) + ', ' + str(y)
        ax.text(x, y, text)

    def plot_point(self, point, ax):

        x,z = point
        ax.plot(x, z, color='blue', marker='.', markersize=1)

    def plot_labels(self, ax):

        ax.set_xlabel('x')
        ax.set_ylabel('z')

    def plot_lim(self, ax):
        if self.fix_lim:
            ax.set_xlim(-55, 55)
            ax.set_ylim(-55, 55)

    def plot_per_goal_state(self):

        fig = plt.figure(figsize=(20,20))
        for i, goal_state in enumerate(self.data_list):
            self.current_goal_state = i
            ax = fig.add_subplot(3, 3, i+1)
            for traj in goal_state:
                self.plot_trajectory(traj, ax)

            self.plot_standard(ax)
        self.current_goal_state = None


    def plot_all_together(self):

        fig, ax = plt.subplots(figsize=(15,15))
        for traj in self.data_reshaped:
            self.plot_trajectory(traj, ax)
        self.plot_standard(ax)

    def plot_pointcloud(self):

        fig, ax = plt.subplots(figsize=(15,15))
        for traj in self.data_reshaped:
            traj = np.array(traj, dtype=np.float32)
            for point in traj:
                self.plot_point(point, ax)
        self.plot_standard(ax)

    def plot_histogram(self, labels):
        fig, ax = plt.subplots()
        histogram = defaultdict(int)
        for point in self.data_points:
            x,y,z = point
            if -10<x<10 and -10<z<10:
                histogram[0] += 1

            elif -10<x<10 and 10<z:
                histogram[1] += 1

            elif 10<x and 10<z:
                histogram[2] += 1

            elif 10<x and -10<z<10:
                histogram[3] += 1

            elif 10<x and -10>z:
                histogram[4] += 1

            elif -10<x<10 and -10>z:
                histogram[5] += 1

            elif -10>x and -10>z:
                histogram[6] += 1

            elif -10>x and -10<z<10:
                histogram[7] += 1

            elif -10>x and 10<z:
                histogram[8] += 1

        plt.xticks(rotation=45)
        values = [ v for k,v in sorted(histogram.items(), key=lambda item: item[0])]
        plt.bar(labels, values)
