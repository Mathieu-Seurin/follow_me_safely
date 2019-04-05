import gym
import numpy as np

from collections import OrderedDict
from itertools import product

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


    def action(self, action):
        a = self.action_map[action]
        return self._action(a)


class FrameStackWrapperList(gym.Wrapper):

    def __init__(self, env, n_frameskip=3):
        self.n_frameskip = n_frameskip

        assert self.n_frameskip > 1, "Frameskip is useless"
        super(FrameStackWrapperList, self).__init__(env)

        if isinstance(env.observation_space, gym.spaces.Dict):
            raise NotImplementedError("Todo later")
        else:
            base_shape = env.observation_space.shape
            extended_shape = (n_frameskip, *base_shape)
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=extended_shape)

        self.empty_image = np.zeros((1, *env.observation_space.shape))
        self.empty_feedback = 0

    def stack(self, obs_list):
        """
        Concatenate obs on the first dimension
        """
        if isinstance(obs_list[0], np.ndarray):
            return np.concatenate([np.expand_dims(obs,0) for obs in obs_list])
        elif isinstance(obs_list[0], dict):
            raise NotImplementedError("Need to check feedback, if stack on dim=0 or 1")
            stacked_obs = dict()
            for key in obs_list.keys():
                stacked_obs[key] = np.concatenate([obs[key] for obs in obs_list])
            return stacked_obs

    def step(self, action):
        sum_reward = 0
        stacked_obs = []

        for current_frame in range(self.n_frameskip):
            obs, reward, done, _ = super().step(action)
            sum_reward += reward
            stacked_obs.append(obs)

            if done:
                stacked_obs.extend(self.empty_feedback * (self.n_frameskip - len(stacked_obs)))
                break

        return self.stack(stacked_obs), sum_reward, done, None

    def reset(self):
        """
        Beginning observation is a duplicate of the starting frame
        (to avoid having black frame at the beginning)
        """
        obs = super().reset()
        return self.stack([obs] * self.n_frameskip)
