import argparse
import ray

from rl_agent.rllib_utils import select_agent, env_basic_creator
from ray.tune import register_env

from ray.tune.logger import pretty_print

import numpy as np

from config import load_config

from tensorboardX import SummaryWriter

import gym
import gym_minigrid


parser = argparse.ArgumentParser('Log Parser arguments!')

parser.add_argument("-env_config",    type=str)
parser.add_argument("-env_ext",       type=str)
parser.add_argument("-model_config",  type=str)
parser.add_argument("-model_ext",     type=str)
parser.add_argument("-out_dir",       type=str, default="out", help="Directory all results")
parser.add_argument("-seed",          type=int, default=0, help="Random seed used")

args = parser.parse_args()


full_config, save_path = load_config(env_config_file=args.env_config,
                          model_config_file=args.model_config,
                          env_ext_file=args.env_ext,
                          model_ext_file=args.model_ext,
                          out_dir=args.out_dir,
                          seed=args.seed)

ray.init()
writer = SummaryWriter(save_path)

env_name = full_config["env_name"]
register_env(full_config["env_name"], lambda x: env_basic_creator(full_config["env_name"]))

agent = select_agent(algo_name=full_config["algo"],
                     algo_config=full_config["algo_config"],
                     env_name=env_name)

mean_reward = 0
n_episodes = 0
success = 0

while n_episodes < full_config["stop"]["episodes_total"]:

    result = agent.train()
    print(pretty_print(result))

    timesteps_total = result['timesteps_total']
    mean_reward = result['episode_reward_mean']

    writer.add_scalar("data/reward_max", result['episode_reward_max'], timesteps_total)
    writer.add_scalar("data/reward_mean", mean_reward, timesteps_total)
    writer.add_scalar("data/reward_min", result['episode_reward_min'], timesteps_total)
    writer.add_scalar("data/episode_len_mean", result['episode_len_mean'], timesteps_total)

    n_episodes = result["episodes_total"]

    if mean_reward > full_config["stop"]["episode_reward_mean"]:
        success += 1

    if success >=3:
        break

checkpoint = agent.save()
print("checkpoint saved at", checkpoint)
