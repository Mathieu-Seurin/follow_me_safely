import gym
import numpy as np


class PreprocessWrapper(gym.core.ObservationWrapper):
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