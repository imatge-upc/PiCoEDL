import os
import csv
import gym
import cv2
import copy
import wandb
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plot import plot_positive_rewards
from logging import getLogger
from random import randint, choice
from collections import deque, Counter
from torchvision import transforms
from pfrl.wrappers.atari_wrappers import ScaledFloatFrame, LazyFrames
from pfrl.wrappers import ContinuingTimeLimit, RandomizeAction, Monitor

from IPython import embed

cv2.ocl.setUseOpenCL(False)
logger = getLogger(__name__)


def wrap_env(
        env, 
        test,
        monitor, 
        outdir,
        frame_skip, 
        data_type,
        gray_scale, 
        frame_stack,
        randomize_action, 
        eval_epsilon,
        encoder, 
        device, 
        sampling,
        train_encoder, 
        downstream_task, 
        coords
        ):
    # wrap env: time limit...
    # Don't use `ContinuingTimeLimit` for testing, in order to avoid unexpected behavior on submissions.
    # (Submission utility regards "done" as an episode end, which will result in endless evaluation)
    if not test and isinstance(env, gym.wrappers.TimeLimit):
        logger.info('Detected `gym.wrappers.TimeLimit`! Unwrap it and re-wrap our own time limit.')
        env = env.env
        max_episode_steps = env.spec.max_episode_steps
        env = ContinuingTimeLimit(env, max_episode_steps=max_episode_steps)

    # wrap env: observation...
    # NOTE: wrapping order matters!
    if not train_encoder:
        env = ResetWrapper(env, encoder, test, sampling)

    if test and monitor:
        env = Monitor(
            env, os.path.join(outdir, env.spec.id, 'monitor'),
            mode='evaluation' if test else 'training', video_callable=lambda episode_id: True)
    if frame_skip is not None:
        env = FrameSkip(env, skip=frame_skip)
    if gray_scale:
        env = GrayScaleWrapper(env, dict_space_key='pov')

    if test:
        env = ObtainCoordWrapper(env, outdir)
    env = ObtainPoVWrapper(env)
    env = MoveAxisWrapper(env, source=-1, destination=0)  # convert hwc -> chw as Pytorch requires.

    # NEW
    if not train_encoder:
        env = ObtainEmbeddingWrapper(env, encoder, data_type, device, test, coords)
        env = ConcatenateWrapper(env, encoder)
        ngs = encoder.num_goal_states
        env.skill_probs = np.ones(ngs)/ngs
    else:
        env = ScaledFloatFrame(env)

    # Since we have our own encoder, we won't use scaling
    if frame_stack is not None and frame_stack > 0:
        env = FrameStack(env, frame_stack, channel_order='chw')

    env = ClusteredActionWrapper(env, frame_skip)

    # NEW
    env = TransformReward(env, encoder, train_encoder, downstream_task)

    if randomize_action:
        env = RandomizeAction(env, eval_epsilon)

    return env


class ResetWrapper(gym.Wrapper):
    """
    ResetWrapper
    """
    def __init__(self, env, encoder, test, sampling):
        super().__init__(env)
        self.env = env
        self.model = encoder
        self.test = test
        self.sampling = sampling
        self.env.idx_buffer = []
        self.env.idx_buffer_size = 10000

        self.log_interval = 500
        self.initialize_dataframes()

    def initialize_dataframes(self):
        self.env.p_rewards = [pd.DataFrame(columns=['x', 'y']) for i in range(self.model.num_goal_states)]

    def log_positive_rewards(self):
        if self.env.resets % self.log_interval == 0:
            fig = plot_positive_rewards(self.env.p_rewards)
            wandb.log({'Positive rewards per goal state': fig})
            self.initialize_dataframes()

    def update_probs(self):

        ngs = self.model.num_goal_states
        ibs = self.env.idx_buffer_size

        if len(self.env.idx_buffer) < ibs:
            # if buffer not full return maximum entropy
            return np.ones(ngs)/ngs
        else:
            probs = []
            probs = np.zeros(ngs)
            # compute probability of each skill
            d = dict(Counter(self.env.idx_buffer))
            for i, (k,v) in enumerate(sorted(d.items())):
                probs[k] = v/ibs
            # update skill probs
            return probs

    def plot_histogram(self, data):
        fig, ax = plt.subplots()
        ngs_rng = range(self.model.num_goal_states)
        plt.bar(ngs_rng, data)
        plt.xticks(ngs_rng)
        ax.set_ylim(0,1)
        wandb.log({'Skills probabilities': fig})

    def reset(self):
        ob = self.env.reset()
        probs = self.update_probs()
        self.log_positive_rewards()
        self.plot_histogram(probs)
        # Sample goal state
        num_goal_states = self.model.num_goal_states
        # num_goal_states = 8
        if self.sampling == 'weighted':
            goal_state = np.random.choice(np.arange(num_goal_states), p=probs)
        elif self.sampling == 'uniform':
            if self.test:
                goal_state = (self.env.resets - 1) % num_goal_states
                # goal_state = self.model.goals[(self.env.resets - 1) % num_goal_states]

            else:
                # goal_state = randint(0, num_goal_states-1)
                goal_state = np.random.randint(num_goal_states)
        else:
            raise NotImplementedException()


        self.env.current_step = 0
        self.env.goal_state = goal_state
        self.env.current_step = 0
        self.env.prev_reward = None
        return ob


class FrameSkip(gym.Wrapper):
    """Return every `skip`-th frame and repeat given action during skip.

    Note that this wrapper does not "maximize" over the skipped frames.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)

        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, channel_order='hwc', use_tuple=False):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.observations = deque([], maxlen=k)
        self.stack_axis = {'hwc': 2, 'chw': 0}[channel_order]
        self.use_tuple = use_tuple

        if self.use_tuple:
            pov_space = env.observation_space[0]
            inv_space = env.observation_space[1]
        else:
            pov_space = env.observation_space

        low_pov = np.repeat(pov_space.low, k, axis=self.stack_axis)
        high_pov = np.repeat(pov_space.high, k, axis=self.stack_axis)
        pov_space = gym.spaces.Box(low=low_pov, high=high_pov, dtype=pov_space.dtype)

        if self.use_tuple:
            low_inv = np.repeat(inv_space.low, k, axis=0)
            high_inv = np.repeat(inv_space.high, k, axis=0)
            inv_space = gym.spaces.Box(low=low_inv, high=high_inv, dtype=inv_space.dtype)
            self.observation_space = gym.spaces.Tuple(
                (pov_space, inv_space))
        else:
            self.observation_space = pov_space

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.observations.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.observations.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.observations) == self.k
        if self.use_tuple:
            frames = [x[0] for x in self.observations]
            inventory = [x[1] for x in self.observations]
            return (LazyFrames(list(frames), stack_axis=self.stack_axis),
                    LazyFrames(list(inventory), stack_axis=0))
        else:
            return LazyFrames(list(self.observations), stack_axis=self.stack_axis)


class ObtainPoVWrapper(gym.ObservationWrapper):
    """Obtain 'pov' value (current game display) of the original observation."""
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = self.env.observation_space.spaces['pov']

    def observation(self, observation):
        return observation['pov'], observation['coords']

class ObtainCoordWrapper(gym.ObservationWrapper):
    """Obtain 'coord' value (coordinate values) of the original observation."""
    def __init__(self, env, outdir):
        super().__init__(env)
        self.env = env
        self.outdir = outdir


    def observation(self, observation):
        if 'coords' in observation:
            csvfile = open(os.path.join(self.outdir, f"coords_{self.env.goal_state}.{self.env.resets-1}.csv"), 'a')
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(observation['coords'])
            csvfile.close()
        return observation

class ObtainEmbeddingWrapper(gym.ObservationWrapper):
    """Obtain embedding vector corresponding to current observation."""
    def __init__(self, env, encoder, data_type, device, test, coords):
        super().__init__(env)
        self.env = env
        self.model = encoder
        self.data_type = data_type
        self.device = device
        self.test = test
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))])
        self.coord_mean = coords['mean']
        self.coord_std = coords['std']

        

    def store_idx(self, idx):
        ibs = self.env.idx_buffer_size

        # if buffer full remove random element
        if len(self.env.idx_buffer) >= ibs:
            self.env.idx_buffer.pop(randint(0, ibs - 1))

        self.env.idx_buffer.append(idx)


    def observation(self, observation):
        # Uncomment only when testing on a sequence of skills
        # seq_num = int(self.env.current_step / 30)
        # if seq_num == 0:
        #     self.env.goal_state = self.model.goals[(self.env.resets - 1 + seq_num) % num_goal_states]
        #
        # self.env.current_step += 1
        ###

        goal_state = self.env.goal_state

        coord_ori = np.array(observation[1], dtype=np.float32)
        coord_np = (coord_ori-self.coord_mean)/self.coord_std

        coord = torch.from_numpy(coord_np).float()
        # We don't need .ToTensor since shape already 3,64,64 but we need to divide by 255
        # Then, we substract the mean 0.5 and divide by 1 like in the encoder training
        obs = self.transform(observation[0])

        obs = obs.unsqueeze(dim=0).to(self.device)
        coord = coord.unsqueeze(dim=0).to(self.device)

        if self.data_type == "pixel":
            z_a = self.model.encode(obs)
        elif self.data_type == "coord":
            z_a = self.model.encode(coord)
        elif self.data_type == "pixelcoord":
            z_a = self.model.encode((obs, coord))
        else: z_a = None

        g = self.model.compute_argmax(z_a)
        # Compute reward as distance similarity in the embedding space - baseline reward (max)
        # r = self.model.compute_logits_(z_a, goal_state)
        # reward = int(r > self.model.threshold)

        # Compute reward as a classification problem. If the goal state with highest similarity
        # is the current selected, give reward of 1.
        self.model.reward = self.model.compute_reward(z_a, goal_state, coord_np)

        if self.model.reward:
            self.env.p_rewards[g] = self.env.p_rewards[g].append({'x': coord_ori[2], 'y': coord_ori[0]}, ignore_index=True)

        self.store_idx(g)

        if self.test:
            csvfile = open(os.path.join(self.outdir, f"rewards_{self.env.goal_state}.{self.env.resets-1}.csv"), 'a')
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow([self.model.reward])
            csvfile.close()

        return z_a.detach().cpu().numpy()

class ConcatenateWrapper(gym.ObservationWrapper):
    """Obtain embedding vector corresponding to current observation."""
    def __init__(self, env, encoder):
        super().__init__(env)
        self.model = encoder

    def observation(self, observation):
        goal_state = self.model.get_goal_state(self.env.goal_state)
        obs_gs = np.concatenate((observation.reshape(-1), goal_state))
        return obs_gs.reshape(1,obs_gs.shape[0])


class UnifiedObservationWrapper(gym.ObservationWrapper):
    """Take 'pov', 'compassAngle', 'inventory' and concatenate with scaling.
    Each element of 'inventory' is converted to a square whose side length is region_size.
    The color of each square is correlated to the reciprocal of (the number of the corresponding item + 1).
    """
    def __init__(self, env, region_size=8):
        super().__init__(env)

        self._compass_angle_scale = 180 / 255  # NOTE: `ScaledFloatFrame` will scale the pixel values with 255.0 later
        self.region_size = region_size

        pov_space = self.env.observation_space.spaces['pov']
        low_dict = {'pov': pov_space.low}
        high_dict = {'pov': pov_space.high}

        if 'compassAngle' in self.env.observation_space.spaces:
            compass_angle_space = self.env.observation_space.spaces['compassAngle']
            low_dict['compassAngle'] = compass_angle_space.low
            high_dict['compassAngle'] = compass_angle_space.high

        if 'inventory' in self.env.observation_space.spaces:
            inventory_space = self.env.observation_space.spaces['inventory']
            low_dict['inventory'] = {}
            high_dict['inventory'] = {}
            for key in inventory_space.spaces.keys():
                low_dict['inventory'][key] = inventory_space.spaces[key].low
                high_dict['inventory'][key] = inventory_space.spaces[key].high

        low = self.observation(low_dict)
        high = self.observation(high_dict)

        self.observation_space = gym.spaces.Box(low=low, high=high)

    def observation(self, observation):
        obs = observation['pov']
        pov_dtype = obs.dtype

        if 'compassAngle' in observation:
            compass_scaled = observation['compassAngle'] / self._compass_angle_scale
            compass_channel = np.ones(shape=list(obs.shape[:-1]) + [1], dtype=pov_dtype) * compass_scaled
            obs = np.concatenate([obs, compass_channel], axis=-1)
        if 'inventory' in observation:
            assert len(obs.shape[:-1]) == 2
            region_max_height = obs.shape[0]
            region_max_width = obs.shape[1]
            rs = self.region_size
            if min(region_max_height, region_max_width) < rs:
                raise ValueError("'region_size' is too large.")
            num_element_width = region_max_width // rs
            inventory_channel = np.zeros(shape=list(obs.shape[:-1]) + [1], dtype=pov_dtype)
            for idx, key in enumerate(observation['inventory']):
                item_scaled = np.clip(255 - 255 / (observation['inventory'][key] + 1),  # Inversed
                                      0, 255)
                item_channel = np.ones(shape=[rs, rs, 1], dtype=pov_dtype) * item_scaled
                width_low = (idx % num_element_width) * rs
                height_low = (idx // num_element_width) * rs
                if height_low + rs > region_max_height:
                    raise ValueError("Too many elements on 'inventory'. Please decrease 'region_size' of each component")
                inventory_channel[height_low:(height_low + rs), width_low:(width_low + rs), :] = item_channel
            obs = np.concatenate([obs, inventory_channel], axis=-1)
        return obs


class FullObservationSpaceWrapper(gym.ObservationWrapper):
    """Returns as observation a tuple with the frames and a list of
    compassAngle and inventory items.
    compassAngle is scaled to be in the interval [-1, 1] and inventory items
    are scaled to be in the interval [0, 1]
    """
    def __init__(self, env):
        super().__init__(env)

        pov_space = self.env.observation_space.spaces['pov']

        low_dict = {'pov': pov_space.low, 'inventory': {}}
        high_dict = {'pov': pov_space.high, 'inventory': {}}

        for obs_name in self.env.observation_space.spaces['inventory'].spaces.keys():
            obs_space = self.env.observation_space.spaces['inventory'].spaces[obs_name]
            low_dict['inventory'][obs_name] = obs_space.low
            high_dict['inventory'][obs_name] = obs_space.high

        if 'compassAngle' in self.env.observation_space.spaces:
            compass_angle_space = self.env.observation_space.spaces['compassAngle']
            low_dict['compassAngle'] = compass_angle_space.low
            high_dict['compassAngle'] = compass_angle_space.high

        low = self.observation(low_dict)
        high = self.observation(high_dict)

        pov_space = gym.spaces.Box(low=low[0], high=high[0])
        inventory_space = gym.spaces.Box(low=low[1], high=high[1])
        self.observation_space = gym.spaces.Tuple((pov_space, inventory_space))

    def observation(self, observation):
        frame = observation['pov']
        inventory = []

        if 'compassAngle' in observation:
            compass_scaled = observation['compassAngle'] / 180
            inventory.append(compass_scaled)

        for obs_name in observation['inventory'].keys():
            inventory.append(observation['inventory'][obs_name] / 2304)

        inventory = np.array(inventory)
        return (frame, inventory)


class MoveAxisWrapper(gym.ObservationWrapper):
    """Move axes of observation ndarrays."""
    def __init__(self, env, source, destination, use_tuple=False):
        if use_tuple:
            assert isinstance(env.observation_space[0], gym.spaces.Box)
        else:
            assert isinstance(env.observation_space, gym.spaces.Box)
        super().__init__(env)

        self.source = source
        self.destination = destination
        self.use_tuple = use_tuple

        if self.use_tuple:
            low = self.observation(
                tuple([space.low for space in self.observation_space]))
            high = self.observation(
                tuple([space.high for space in self.observation_space]))
            dtype = self.observation_space[0].dtype
            pov_space = gym.spaces.Box(low=low[0], high=high[0], dtype=dtype)
            inventory_space = self.observation_space[1]
            self.observation_space = gym.spaces.Tuple(
                (pov_space, inventory_space))
        else:
            low = self.observation(self.observation_space.low)
            high = self.observation(self.observation_space.high)
            dtype = self.observation_space.dtype
            self.observation_space = gym.spaces.Box(
                low=low, high=high, dtype=dtype)

    def observation(self, observation):
        if self.use_tuple:
            new_observation = list(observation)
            new_observation[0] = np.moveaxis(
                observation[0], self.source, self.destination)
            return tuple(new_observation)
        else:
            return np.moveaxis(observation, self.source, self.destination)


class GrayScaleWrapper(gym.ObservationWrapper):
    def __init__(self, env, dict_space_key=None):
        super().__init__(env)

        self._key = dict_space_key

        if self._key is None:
            original_space = self.observation_space
        else:
            original_space = self.observation_space.spaces[self._key]
        height, width = original_space.shape[0], original_space.shape[1]

        # sanity checks
        ideal_image_space = gym.spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)
        if original_space != ideal_image_space:
            raise ValueError('Image space should be {}, but given {}.'.format(ideal_image_space, original_space))
        if original_space.dtype != np.uint8:
            raise ValueError('Image should `np.uint8` typed, but given {}.'.format(original_space.dtype))

        height, width = original_space.shape[0], original_space.shape[1]
        new_space = gym.spaces.Box(low=0, high=255, shape=(height, width, 1), dtype=np.uint8)
        if self._key is None:
            self.observation_space = new_space
        else:
            new_space_dict = copy.deepcopy(self.observation_space)
            new_space_dict.spaces[self._key] = new_space
            self.observation_space = new_space_dict

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = np.expand_dims(frame, -1)
        if self._key is None:
            obs = frame
        else:
            obs[self._key] = frame
        return obs


class ClusteredActionWrapper(gym.ActionWrapper):
    def __init__(self, env, frame_skip):
        super().__init__(env)
        self.env = env
        self.delta_degree = 45
        if not frame_skip == None:
            self.delta_degree /= frame_skip

        self.action_space = gym.spaces.Discrete(4)

        base = self.env.action_space.no_op()

        forward = base.copy()
        forward['forward'] = np.array(1)
        forward['sprint'] = np.array(1)

        right = forward.copy()
        right['camera'] = np.array([0,self.delta_degree], dtype=np.float32)

        left = forward.copy()
        left['camera'] = np.array([0,-self.delta_degree], dtype=np.float32)

        jump_forward = forward.copy()
        jump_forward['jump'] = np.array(1)

        if env.unwrapped.custom_config['num'] == 'Simple' or env.unwrapped.custom_config['num'] == 'ToyCool':
            self.actions = [base, forward, right, left]
        else:
            self.actions = [forward, right, left, jump_forward]

    def action(self, action):
        return self.actions[action]

    def seed(self, seed):
        super().seed(seed)

class TransformReward(gym.RewardWrapper):
    """Transform the reward via an arbitrary function.
        Args:
            env (Env): environment
    """
    def __init__(self, env, encoder, train_enc, downstream_task):
        super(TransformReward, self).__init__(env)

        self.model = encoder
        self.train_encoder = train_enc
        self.downstream_task = downstream_task
        self.env = env
        self.env.prev_reward = None

    def reward(self, reward):
        if self.train_encoder:
            return reward

        elif not self.train_encoder and not self.downstream_task:
            reward = self.model.reward
            #if not self.env.prev_reward == None:
            #    reward -= self.env.prev_reward
            #self.env.prev_reward = self.model.reward
            return reward

        elif not self.train_encoder and self.downstream_task:
            return self.model.reward + reward

        else:
            return reward
