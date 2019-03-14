import random
import time
import argparse

from os import path

import gym
import gym_minigrid
import gym.wrappers

import numpy as np

from config import load_config
from rl_agent.dqn_agent import DQNAgent, FullRandomAgent

import tensorboardX
import os

parser = argparse.ArgumentParser('Log Parser arguments!')

parser.add_argument("-env_config",   type=str)
parser.add_argument("-env_ext",      type=str)
parser.add_argument("-model_config", type=str)
parser.add_argument("-model_ext",    type=str)
parser.add_argument("-exp_dir",      type=str, default="out", help="Directory all results")
parser.add_argument("-seed",         type=int, default=42, help="Random seed used")


args = parser.parse_args()

full_config, expe_path = load_config(env_config_file=args.env_config,
                          model_config_file=args.model_config,
                          env_ext_file=args.env_ext,
                          model_ext_file=args.model_ext,
                          out_dir=args.exp_dir,
                          seed=args.seed
                          )

writer = tensorboardX.SummaryWriter(expe_path)

game = gym.make(full_config["env_name"])

game = gym.wrappers.Monitor(game, directory=expe_path, force=True)

model_type = full_config["agent_type"]
if model_type == "dqn" :
    model = DQNAgent(config=full_config["dqn_params"],
                     n_action=game.action_space,
                     state_dim=game.observation_space)
elif model_type == "random" :
    model = FullRandomAgent(config=full_config, n_action=game.action_space, state_dim=game.observation_space)



episodes = full_config["stop"]["episodes_total"]
score_success = full_config["stop"]["episode_reward_mean"]

rew_list = []
discount_factor = full_config["discount_factor"]
total_iter = 0
success_count = 0


for num_episode in range(1, episodes):
    state = game.reset()
    game.render()
    done = False
    iter_this_ep = 0
    reward_total_discounted = 0
    reward_total_not_discounted = 0

    while not done:

        action = model.forward(state)
        next_state, reward, done, info = game.step(action=action)
        game.render()
        model.optimize(state=state, action=action, next_state=next_state, reward=reward)

        state = next_state

        iter_this_ep = iter_this_ep + 1
        reward_total_discounted += reward * (discount_factor ** iter_this_ep)
        reward_total_not_discounted += reward

        if done:
            model.callback(epoch=num_episode)

            total_iter += iter_this_ep

            rew_list.append(reward_total_discounted)
            writer.add_scalar("data/sum_reward_discounted", reward_total_discounted, total_iter)
            writer.add_scalar("data/sum_reward_not_discounted", reward_total_not_discounted, total_iter)
            writer.add_scalar("data/iter_per_ep", iter_this_ep, total_iter)

            # todo create logging
            if num_episode%100 == 0:
                print("Reward at the end of ep #{} discounted rew : {} undiscounted : {}".format(
                    total_iter, reward_total_discounted, reward_total_not_discounted))

            if reward_total_discounted > score_success:
                success_count += 1
            else:
                success_count = 0

    #print("Steps iter main {}, iter in model {}, eps {}".format(total_iter, model.n_step_eps, model.current_eps))

    if success_count > 5:
        break


print("Experiment over")
print("Last 5 epoch's rewards  : ", rew_list[-4:])