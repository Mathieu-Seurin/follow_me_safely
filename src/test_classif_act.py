from rl_agent.models import TextModel
from rl_agent.agent_utils import ReplayMemory
import os

import gym
import textworld.gym as tw_gym
from textworld.envs.wrappers.filter import EnvInfos
from env_tools.wrapper import TextWorldWrapper

import numpy as np

import pickle as pkl
import copy


EXTRA_GAME_INFO = {
    "inventory": True,
    "description": True,
    "intermediate_reward": True,
    "admissible_commands": True,
    "policy_commands": True,
}

games_directory = "text_game_files"


game_path = os.path.join(games_directory, 'simple10.ulx')
env_id = tw_gym.register_game(game_path, max_episode_steps=100000,
                              name="simple1", request_infos=EnvInfos(**EXTRA_GAME_INFO))
game = gym.make(env_id)
game = TextWorldWrapper(env=game, use_intermediate_reward=EXTRA_GAME_INFO["intermediate_reward"])


dataset_path = os.path.join(games_directory, 'simple10.ulx.dataset')

if os.path.isfile(dataset_path):
    dataset = np.load(dataset_path)

else:

    dataset = ReplayMemory(capacity=1e6)
    done = False
    current_state = game.reset()

    all_commands_to_exec = [list(range(0, game.action_space.n))]

    policy_commands = current_state['raw']['policy_commands']
    policy_commands_idx = []
    for n_command, command in enumerate(policy_commands):

        current_command_idx = game._all_doable_actions.index(command)
        policy_commands_idx.append(current_command_idx)

        command_list_tmp = copy.deepcopy(policy_commands_idx)
        command_list_tmp.e

        command_list_tmp.pop(current_command_idx)
        command_list_tmp.append(current_command_idx)

        all_commands_to_exec.extend(command_list_tmp)

    for act in all_commands_to_exec:
        next_state, reward, done, info = game.step(action=act)

        dataset.push(current_state['state'], act, next_state['state'], reward, next_state['gave_feedback'])


    print("Done is True : ", done is True)

