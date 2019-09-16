import gym
import numpy as np

from collections import OrderedDict
from itertools import product
from skimage import color

#from env_tools.car_racing import CarRacingSafe

from gym_minigrid.envs.safe_crossing import SafeCrossing
from gym_minigrid.minigrid import COLORS

from typing import Tuple, List, Iterable, Optional, Any
from textworld.gym import spaces as tw_spaces
from textworld.envs.wrappers.filter import EnvInfos

import copy
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from math import floor

import platform

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
        observation_space['zone'] = self.observation_space.spaces['zone']

        observation_space['state'] = gym.spaces.Box(low=0, high=len(COLORS), shape=(3*n_frameskip, 7, 7))
        self.observation_space = gym.spaces.Dict(observation_space)

    def get_feedback_actions(self, states):
        zone_per_state = np.maximum(states[:,1,4, -1], states[:,1,2, -1]) - 1

        feedback_list = []
        for zone in zone_per_state:
            feedback_list.append([])
            for action in range(len(self.env.action_to_zone.keys())):
                if self.env.action_to_zone[action] == zone:
                    feedback_list[-1].append(-1)
                else:
                    feedback_list[-1].append(1)

        return np.array(feedback_list), zone_per_state

    def _unconvert(self, state):
        x = self.render("rgb_array")
        return x

    def stack_last_frame(self, obs):

        obs = obs.transpose(2, 0, 1)
        new_obs = np.concatenate((*self.last_frames, obs), axis=0)
        self.last_frames.append(obs)
        self.last_frames.pop(0)

        assert len(self.last_frames) == self.n_frameskip-1
        return new_obs

    def reset(self):
        obs = super().reset()

        self.last_frames = [obs['state'].transpose(2, 0, 1) for i in range(self.n_frameskip-1)]

        new_obs = dict()
        new_obs['state'] = self.stack_last_frame(obs['state'])
        new_obs['gave_feedback'] = False
        new_obs['zone'] = self.env.current_zone_num

        assert self.get_feedback_actions(np.expand_dims(new_obs['state'],0))[1][0] == self.env.current_zone_num
        assert self.observation_space.contains(new_obs), "Observation don't match observation space"
        return new_obs

    def step(self, action):

        obs, reward, done, info = super().step(action)

        new_obs = dict()
        new_obs['state'] = self.stack_last_frame(obs['state'])
        new_obs['gave_feedback'] = obs['gave_feedback']
        new_obs['zone'] = obs['zone']

        assert self.get_feedback_actions(np.expand_dims(new_obs['state'],0))[1][0] == self.env.current_zone_num
        assert self.observation_space.contains(new_obs), "Problem, observation don't match observation space."
        return new_obs, reward, done, info

# class ZorkPreproc(gym.Wrapper):
#     def __init__(self, env):
#         pass

class TextWorldWrapper(gym.Wrapper):
    """
    Simple wrapper to preprocess text_world game observation
    """
    def __init__(self, env: gym.Env, encode_raw_text: bool = True,
                 encode_extra_fields: Iterable[str] = ('description', 'inventory'),
                 choose_among_useful_actions: bool = True,
                 use_admissible_commands: bool = False,
                 use_intermediate_reward: bool = True,
                 tokens_limit: Optional[int] = None,
                 n_dummy_action: int = 0):
        """
        :param env: env to be wrapped. Has to provide Word observations
        :param encode_raw_text: do we need to encode raw observation from environment, if true, adds an extra encoder
        :param encode_extra_fields: tuple of field names to be encoded, expected to be string values
        :param use_admissible_commands: if true, admissible commands used for action wrapping
        :param use_intermediate_reward: take intermediate reward into account
        :param tokens_limit: optional limit of tokens in the encoded fields
        """
        super(TextWorldWrapper, self).__init__(env)
        if not isinstance(env.observation_space, tw_spaces.Word):
            raise ValueError("Env should expose text_world compatible observation space, "
                             "this one gives %s" % env.observation_space)
        self._encode_raw_text = encode_raw_text
        self._encode_extra_field = tuple(encode_extra_fields)
        self._use_admissible_commands = use_admissible_commands
        self._choose_among_useful_actions = choose_among_useful_actions
        self._use_intermedate_reward = use_intermediate_reward
        self._num_fields = len(self._encode_extra_field) + int(self._encode_raw_text)
        self._last_admissible_commands = None
        self._last_extra_info = None
        self._tokens_limit = tokens_limit
        self._cmd_hist = []

        self._all_doable_actions = self.compute_all_doable_actions()

        if n_dummy_action:
            self._all_doable_actions = self._all_doable_actions.extend([str(i) for i in range(n_dummy_action)])

        self.action_space = gym.spaces.Discrete(len(self._all_doable_actions))
        self.env.action_map = self._all_doable_actions

        self.env.max_steps = 100

        # Hidden action, shouldn't be used by agent
        #self._all_doable_actions.append('undo')

    @property
    def num_fields(self):
        return self._num_fields


    def _unconvert(self, state):

        def fit_state(raw_sentences, split_every=8):
            keys = ['obs', 'description', 'inventory']
            state_str = ''

            for key in keys:
                state_str_tmp = raw_sentences[key].replace('\n', ' ')
                list_sentence = state_str_tmp.split()

                n_words = len(list_sentence)

                for idx in range(round(floor(n_words/split_every)*split_every), 0, -split_every):
                    list_sentence.insert(idx, '\n')

                sentence = ' '.join(list_sentence)
                state_str += sentence + '\n\n'

            return state_str


        font_path = '/usr/share/fonts/truetype/lato/Lato-Medium.ttf' if 'Linux' in platform.system() else '/Library/Fonts/Arial.ttf'

        blank_image = Image.new('RGBA', (400, 300), 'black')
        img_draw = ImageDraw.Draw(blank_image)
        fnt = ImageFont.truetype(font_path, 17)
        img_draw.text((0, 0), fit_state(state['raw']), fill='green', font=fnt)
        # plt.imshow(blank_image)
        # plt.show()
        return np.array(blank_image)

    def compute_all_doable_actions(self):

        def _complete_action_recurs(command, obj_list):

            command_list = []

            start_index_brack = command.find('{')
            if start_index_brack == -1:
                return [command]
            else:
                for obj in obj_list:
                    # obj[1] is the type of the object
                    obj_type = '{'+obj[1]+'}'
                    obj_name = obj[0]

                    if obj_type == command[start_index_brack:start_index_brack+3]:
                        partially_completed_command = command.replace(obj_type, obj_name)
                        command_list.extend(_complete_action_recurs(partially_completed_command, obj_list))

            return command_list

        #self.env.compute_intermediate_reward()
        obs, extra = self.env.reset()
        internal_game = self.env.env.textworld_env._wrapped_env.game_state._env.game
        obj_list = internal_game.objects_names_and_types

        # Cleaning obj list because 'type 1 safe' => 'safe'
        for obj_id in range(len(obj_list)):
            if 'type ' in obj_list[obj_id][0]:
                elem = obj_list[obj_id][0] # obj_list[obj_id] is a tuple (obj_name: str, obj_type: str)
                elem = elem.split()

                type_idx = elem.index('type')
                elem.pop(type_idx+1) # Delete number
                elem.pop(type_idx) # Delete type

                elem = ' '.join(elem)
                obj_list[obj_id] = (elem, obj_list[obj_id][1])

        # Delete double
        obj_list = list(set(obj_list))

        # Some type are subclasses of other type, need to add them
        additionnal_obj = []
        for obj in obj_list:
            if obj[1] in ['f', 'k']:
                additionnal_obj.append((obj[0],'o'))

        obj_list.extend(additionnal_obj)

        all_doable_actions = []
        for command in internal_game.command_templates :
            all_doable_actions.extend(_complete_action_recurs(command, obj_list))

        # Check that at least all necessary actions are in the possible actions list.

        actions_not_available = []
        necessary_command = extra.get('policy_commands', [])

        if necessary_command == []:
            print("Cannot check policy commands, run with 'intermediate_reward' set to true")
        else:
            for act in necessary_command:
                if act not in all_doable_actions:
                    actions_not_available.append(act)

            assert len(actions_not_available) == 0,\
                "Some useful actions are not available, check all_doable_actions \n{}".format(actions_not_available)

        print("Number of available actions : ", len(all_doable_actions))
        return all_doable_actions


    def _encode(self, obs: str, extra_info: dict) -> dict:

        state_result = OrderedDict()
        if self._encode_raw_text:
            tokens = self.env.observation_space.tokenize(obs)
            if self._tokens_limit is not None:
                tokens = tokens[:self._tokens_limit]
            state_result['obs'] = tokens
        for field in self._encode_extra_field:
            tokens = self.env.observation_space.tokenize(extra_info[field])
            if self._tokens_limit is not None:
                tokens = tokens[:self._tokens_limit]
            state_result[field] = tokens
        if self._use_admissible_commands:
            adm_result = []
            for cmd in extra_info['admissible_commands']:
                adm_result.append(self.env.action_space.tokenize(cmd))
            state_result['admissible_commands'] = adm_result
            self._last_admissible_commands = extra_info['admissible_commands']
        self._last_extra_info = extra_info
        return state_result

    # TextWorld environment has a workaround of gym drawback:
    # reset returns tuple with raw observation and extra dict
    def reset(self):
        res = self.env.reset()
        self._cmd_hist = []
        encoded_state = self._encode(res[0], res[1])
        state = dict()
        state['state'] = encoded_state
        state['raw'] = dict(**res[1])
        state['raw']['obs'] = res[0]
        state['gave_feedback'] = False

        self._last_state = copy.deepcopy(state)
        return state

    def step(self, action):
        if self._use_admissible_commands:
            action = self._last_admissible_commands[action]
            self._cmd_hist.append(action)
        elif self._choose_among_useful_actions:
            action = self._all_doable_actions[action]
            self._cmd_hist.append(action)

        obs, r, is_done, extra = self.env.step(action)
        if self._use_intermedate_reward:
            r += extra.get('intermediate_reward', 0)
        new_extra = dict(extra)
        for f in self._encode_extra_field + ('admissible_commands', 'intermediate_reward'):
            if f in new_extra:
                new_extra.pop(f)
        # if is_done:
        #     self.log.info("Commands: %s", self._cmd_hist)
        #     self.log.info("Reward: %s, extra: %s", r, new_extra)
        encoded_state = self._encode(obs, extra)

        state = dict()
        state['state'] = encoded_state
        state['raw'] = dict(**extra)
        state['raw']['obs'] = obs

        if self._last_state['raw']['admissible_commands'] == state['raw']['admissible_commands']:
            state['gave_feedback'] = True

            # Check that the action was really useless
            assert action in ['inventory', 'look'] or 'examine' in action \
                   or action not in self._last_state['raw']['admissible_commands'], \
                "Problem, action should have done something, action was {}".format(action)
        else:
            state['gave_feedback'] = False

        new_extra['gave_feedback'] = state['gave_feedback']

        self._last_state = copy.deepcopy(state)
        return state, r, is_done, new_extra

    @property
    def last_admissible_commands(self):
        return tuple(self._last_admissible_commands) if self._last_admissible_commands else None

    @property
    def last_extra_info(self):
        return self._last_extra_info


if __name__ == "__main__" :


    test_car_racing = False
    test_minigrid = False
    test_text_world = True

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

        obs = game.reset()

        done = False
        step = 0
        game.render('human')

        while not done:

            a = game.action_space.sample()
            #a = int(input())
            new_obs, rew, done, info = game.step(a)

            game.render('human')

            assert (new_obs['state'][3:6,:,:] == obs['state'][6:,:,:]).all()

            obs = new_obs


            assert obs['state'].shape == (3*n_frameskip, 7, 7)
            step += 1

            if obs['gave_feedback']:
                time.sleep(2)
                print(rew, done, step)

        print(rew, done, step)

    if test_text_world:

        import textworld.gym as tw_gym
        import os

        EXTRA_GAME_INFO = {
            "inventory": True,
            "description": True,
            "intermediate_reward": True,
            "admissible_commands": True,
            "policy_commands": True,
        }

        game_path = os.path.join("text_game_files","simple10.ulx")

        env_id = tw_gym.register_game(game_path, max_episode_steps=1000,
                                      name="simple1", request_infos=EnvInfos(**EXTRA_GAME_INFO))
        game = gym.make(env_id)
        game = TextWorldWrapper(env=game)

        game.reset()

        done = False
        while not done:
            act = game.action_space.sample()
            state, reward, done, info = game.step(act)

            if state['gave_feedback']:
                print("Feedback")
            else:
                print("No feedback")


