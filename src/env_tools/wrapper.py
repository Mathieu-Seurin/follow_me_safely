import gym
import numpy as np

from collections import OrderedDict
from itertools import product
from skimage import color

from env_tools.car_racing import CarRacingSafe

from gym_minigrid.envs.safe_crossing import SafeCrossing

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
            [(-1, 1, 0.2),
             (-1, 1, 0),
             (-1, 0, 0.2),
             (-1, 0, 0),
             (0, 1, 0.2),
             (0, 1, 0),
             (0, 0, 0.2),
             (0, 0, 0),
             (1, 1, 0.2),
             (1, 1, 0),
             (1, 0, 0.2),
             (1, 0, 0)]
        )

        self.action_space = gym.spaces.Discrete(len(self.action_map))
        self.dont_move_action = 7
        self.brake_action = 6

    def action(self, action):
        converted_action = self.action_map[action]
        return converted_action

class CarFrameStackWrapper(gym.Wrapper):

    def __init__(self, env, n_frameskip, early_reset=True):
        self.n_frameskip = n_frameskip

        assert self.n_frameskip > 1, "Frameskip is useless"
        super(CarFrameStackWrapper, self).__init__(env)

        if isinstance(env.observation_space, gym.spaces.Dict):
            base_shape = env.observation_space.spaces['state'].shape[:2]  # Deleting the channel because it's converted to grey
            extended_shape = (n_frameskip, *base_shape)

            new_obs_space = dict()
            new_obs_space['state'] = gym.spaces.Box(low=-1, high=1, shape=extended_shape)
            new_obs_space['gave_feedback'] = gym.spaces.Discrete(2)
            self.observation_space = gym.spaces.Dict(new_obs_space)

        else:
            base_shape = env.observation_space.shape[:2] # Deleting the channel because it's converted to grey
            extended_shape = (n_frameskip, *base_shape)
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=extended_shape)

        self.empty_image = np.zeros(env.observation_space.spaces['state'].shape)
        self.empty_feedback = 0

        self.frames_before_zoom_end = 5 # Number of step for the zoom to end

        self.early_reset = early_reset
        self.min_reward_before_reset = 12

        # todo : Ugly
        self.brake_action = env.brake_action

    def convert(self, frame):
        """
        Convert to greyscale, normalize and add dummy channel dimension so its shape is (1,96,96)
        """
        frame = 2 * color.rgb2gray(frame) - 1.0
        frame = np.expand_dims(frame, axis=0)

        return frame

    def _unconvert(self, converted_frame, frame=2):

        converted_frame = converted_frame['state'].numpy()[0, frame, :, :]
        #converted_frame = np.squeeze(converted_frame, 0)
        original_frame = color.gray2rgb(converted_frame + 1)/2 #* 255
        return original_frame

    def stack(self, obs_list):
        """
        Concatenate obs on the first dimension
        """
        if isinstance(obs_list[0], np.ndarray):
            return np.concatenate([self.convert(obs) for obs in obs_list])
        elif isinstance(obs_list[0], dict):
            stacked_obs = dict()
            stacked_obs['state'] = np.concatenate([self.convert(obs['state']) for obs in obs_list])
            # todo : stacked them and add dimension
            stacked_obs['gave_feedback'] = np.max(np.array([obs['gave_feedback'] for obs in obs_list]))
            #assert stacked_obs['gave_feedback'].sum() <= 1, "Received 2 feedback in same step, bad."

            return stacked_obs
        else:
            raise NotImplementedError("Problem in obs list")

    def stack_inf(self, info_dict):

        new_info = dict()
        new_info['gave_feedback'] = max([info['gave_feedback'] for info in info_dict])
        new_info['percentage_road_visited'] = max([info['percentage_road_visited'] for info in info_dict])
        return new_info

    def step(self, action):
        sum_reward = 0
        stacked_obs = []
        stacked_info = []

        for current_frame in range(self.n_frameskip):
            obs, reward, done, info = super().step(action)

            sum_reward += reward

            stacked_obs.append(obs)
            stacked_info.append(info)

            self.render('human')

            if obs['gave_feedback'] and self.unwrapped.reset_when_out :
                action = self.brake_action

            if done:
                stacked_obs.extend([{'state' : self.empty_image, 'gave_feedback' : False}] * (self.n_frameskip - len(stacked_obs)))
                break

        array_obs = self.stack(stacked_obs)
        assert self.observation_space.contains(array_obs), "Problem, observation don't match observation space."

        return array_obs, sum_reward, done, self.stack_inf(stacked_info)

    def reset(self):
        """
        Beginning observation is a duplicate of the starting frame
        (to avoid having black frame at the beginning)
        """

        obs = super().reset()
        dont_move_action = self.env.dont_move_action

        # Don't apply early reset when initializing
        early_reset = self.early_reset
        self.early_reset = False

        for _ in range(self.frames_before_zoom_end):
            obs, rew, done, info = self.step(dont_move_action)

        # Re-apply early reset
        self.early_reset = early_reset
        self.reward_neg_counter = 0

        self.render('human')

        assert self.observation_space.contains(obs), "Problem, observation don't match observation space."
        return obs


class MinigridFrameStacker(gym.Wrapper):

    def __init__(self, env, n_frameskip):
        """
        In MiniGrid, we want frame stacking, to avoid using a recurrent policy
        But the action repeating is irrelevant

        A state is the last *n* frames, so the wrapper job is to remember to last n-1 frames and stack the new frame
        """

        self.n_frameskip = n_frameskip
        super().__init__(env)

        observation_space = dict()
        observation_space['gave_feedback'] = self.observation_space.spaces['gave_feedback']
        observation_space['state'] = gym.spaces.Box(low=0, high=10, shape=(3*n_frameskip, 7, 7))
        self.observation_space = gym.spaces.Dict(observation_space)

    def _unconvert(self, state):
        x = self.render("rgb_array")
        return x

    def stack_last_frame(self, obs):
        new_obs = np.concatenate((*self.last_frames, obs), axis=2)
        new_obs = new_obs.transpose((2, 0, 1))

        self.last_frames.append(obs)
        self.last_frames.pop(0)

        assert len(self.last_frames) == self.n_frameskip-1
        return new_obs

    def reset(self):
        obs = super().reset()

        self.last_frames = [obs['state'] for i in range(self.n_frameskip-1)]

        new_obs = dict()
        new_obs['state'] = self.stack_last_frame(obs['state'])
        new_obs['gave_feedback'] = False

        assert self.observation_space.contains(new_obs), "Observation don't match observation space"
        return new_obs

    def step(self, action):

        obs, reward, done, info = super().step(action)

        self.last_frames = [obs['state'] for i in range(self.n_frameskip - 1)]

        new_obs = dict()
        new_obs['state'] = self.stack_last_frame(obs['state'])
        new_obs['gave_feedback'] = obs['gave_feedback']

        assert self.observation_space.contains(new_obs), "Problem, observation don't match observation space."
        return new_obs, reward, done, info


if __name__ == "__main__" :


    test_car_racing = False
    test_minigrid = True

    # Car Racing env test
    if test_car_racing:

        #from xvfbwrapper import Xvfb
        import matplotlib.pyplot as plt
        from skimage import data
        import time

        #with Xvfb(width=100, height=100, colordepth=16) as disp:
        game = CarRacingSafe()

        game = CarActionWrapper(game)
        game = CarFrameStackWrapper(game)

        init_time = time.time()
        game.reset()
        print("Time taken to init in seconds :", time.time() - init_time)

        init_time = time.time()

        done = False
        step = 0

        while not done:
            a = game.action_space.sample()
            obs, rew, done, info = game.step(a)

            print(rew)

            assert obs['state'].shape == (3,96,96)

            plt.imshow(game._unconvert(obs['state'][0, :, :]))
            plt.show()
            plt.imshow(game._unconvert(obs['state'][1, :, :]))
            plt.show()
            plt.imshow(game._unconvert(obs['state'][2, :, :]))
            plt.show()

            step += 1

        print("FPS = ", (step*3) / (time.time() - init_time))


    if test_minigrid:

        # from xvfbwrapper import Xvfb
        import matplotlib.pyplot as plt
        from skimage import data
        import time

        n_frameskip = 3

        # with Xvfb(width=100, height=100, colordepth=16) as disp:
        game = SafeCrossing(reward_when_falling=-10)
        game = MinigridFrameStacker(game, n_frameskip=n_frameskip)

        game.reset()

        done = False
        step = 0
        game.render('human')

        while not done:

            #a = game.action_space.sample()
            a = int(input())
            obs, rew, done, info = game.step(a)

            game.render('human')

            assert obs['state'].shape == (3*n_frameskip, 7, 7)
            step += 1

            if obs['gave_feedback']:
                time.sleep(2)
                print(rew, done, step)

        print(rew, done, step)


