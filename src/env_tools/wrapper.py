import gym
import numpy as np

from collections import OrderedDict
from itertools import product
from skimage import color

class PreprocessWrapperPytorch(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.transpose_tuple = (2,1,0)
        self.observation_space = env.observation_space

        old_shape = env.observation_space.spaces["image"].shape
        new_shape = [old_shape[dim] for dim in self.transpose_tuple]

        self.observation_space.spaces["image"] = gym.spaces.Box(0,1, shape=new_shape, dtype=np.float32)

    def observation(self, obs):
        return self._preprocess_obs(obs)

    def _preprocess_obs(self, obs):
        obs['image'] = np.transpose(obs['image'], self.transpose_tuple)
        return obs


class ObsSpaceWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        new_obs_space = OrderedDict()

        new_obs_space['image'] = self.observation_space.spaces["image"]
        new_obs_space['direction'] = gym.spaces.Discrete(4)
        #At the moment, mission is not necessary
        self.observation_space = gym.spaces.Dict(new_obs_space)

    def observation(self, obs):
        return self._preprocess_obs(obs)

    def _preprocess_obs(self, obs):

        ordered_obs = OrderedDict()
        ordered_obs['image'] = obs['image']
        ordered_obs['direction'] = obs['direction']
        return ordered_obs


class CarActionWrapper(gym.core.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.action_map = np.array(
            [k for k in product([-1, 0, 1], [1, 0], [0.2, 0])]
        )
        self.action_space = gym.spaces.Discrete(len(self.action_map))
        self.dont_move_action = np.where((self.action_map == 0).sum(axis=1) == 3)[0][0] # Compute where all angles == 0


    def action(self, action):
        converted_action = self.action_map[action]
        return converted_action

class FrameStackWrapper(gym.Wrapper):

    def __init__(self, env, n_frameskip=3, early_reset=True):
        self.n_frameskip = n_frameskip

        assert self.n_frameskip > 1, "Frameskip is useless"
        super(FrameStackWrapper, self).__init__(env)

        if isinstance(env.observation_space, gym.spaces.Dict):
            raise NotImplementedError("Todo later")
        else:
            base_shape = env.observation_space.shape[:2] # Deleting the channel because it's converted to grey
            extended_shape = (n_frameskip, *base_shape)
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=extended_shape)

        self.empty_image = np.zeros((1, *base_shape))
        self.empty_feedback = 0

        self.frames_before_zoom_end = 48 # Number of step for the zoom to end, fuck this shit.

        self.early_reset = early_reset
        self.min_reward_before_reset = 12

    def convert(self, frame):
        """
        Convert to RGB, normalize and add dummy channel dimension so its shape is (1,96,96)
        """
        frame = 2 * color.rgb2gray(frame) - 1.0
        frame = np.expand_dims(frame, axis=0)

        return frame

    def _unconvert(self, converted_frame):

        #converted_frame = np.squeeze(converted_frame, 0)
        original_frame = color.gray2rgb(converted_frame + 1)/2 #* 255
        return original_frame

    def stack(self, obs_list):
        """
        Concatenate obs on the first dimension
        """
        if isinstance(obs_list[0], np.ndarray):
            return np.concatenate(obs_list)
        elif isinstance(obs_list[0], dict):
            raise NotImplementedError("Need to check feedback, if stack on dim=0 or 1")
            stacked_obs = dict()
            for key in obs_list.keys():
                stacked_obs[key] = np.concatenate([obs[key] for obs in obs_list])

            return stacked_obs
        else:
            raise NotImplementedError("Problem in obs list")

    def check_early_reset(self, reward):
        pass

    def step(self, action):
        sum_reward = 0
        stacked_obs = []

        for current_frame in range(self.n_frameskip):
            obs, reward, done, _ = super().step(action)

            sum_reward += reward
            stacked_obs.append(self.convert(obs))

            self.render('rgb_array')

            if done:
                stacked_obs.extend([self.empty_image] * (self.n_frameskip - len(stacked_obs)))
                break

        if self.early_reset:
            self.check_early_reset(sum_reward)

        array_obs = self.stack(stacked_obs)
        assert self.observation_space.contains(array_obs), "Problem, observation don't match observation space."

        return array_obs, sum_reward, done, None

    def reset(self):
        """
        Beginning observation is a duplicate of the starting frame
        (to avoid having black frame at the beginning)
        """

        super().reset()
        dont_move_action = self.env.dont_move_action

        # Don't apply early reset when initializing
        early_reset = self.early_reset
        self.early_reset = False

        for _ in range(self.frames_before_zoom_end):
            obs, rew, done, info = self.step(dont_move_action)

        # Re-apply early reset
        self.early_reset = early_reset
        self.reward_neg_counter = 0

        self.render('rgb_array')
        return obs


if __name__ == "__main__" :

    #from xvfbwrapper import Xvfb
    import matplotlib.pyplot as plt
    from skimage import data
    import time

    #with Xvfb(width=100, height=100, colordepth=16) as disp:
    game = gym.make("CarRacing-v0")

    game = CarActionWrapper(game)
    game = FrameStackWrapper(game)

    init_time = time.time()
    game.reset()
    print("Time taken to init in seconds :", time.time() - init_time)

    init_time = time.time()

    done = False
    step = 0

    while not done:
        a = game.action_space.sample()
        obs, rew, done, _ = game.step(a)

        print(rew)

        assert obs.shape == (3,96,96)

        # plt.imshow(game._unconvert(obs[0, :, :]))
        # plt.savefig('test{:04d}1'.format(step))
        # plt.imshow(game._unconvert(obs[1, :, :]))
        # plt.savefig('test{:04d}2'.format(step))
        # plt.imshow(game._unconvert(obs[2, :, :]))
        # plt.savefig('test{:04d}3'.format(step))

        step += 1

    print("FPS = ", (step*3) / (time.time() - init_time))