import gym
import numpy as np

from collections import OrderedDict

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