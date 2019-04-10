import argparse

import gym.wrappers
from env_tools.wrapper import PreprocessWrapperPytorch, FrameStackWrapper, CarActionWrapper

from config import load_config
from rl_agent.dqn_agent import DQNAgent

import numpy as np
import torch

import os

import matplotlib.pyplot as plt


from xvfbwrapper import Xvfb
display = Xvfb(width=100, height=100, colordepth=16)

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


game = gym.make(full_config["env_name"])

# Apply wrapper necessary to the env
wrapper_translate = dict([("frameskip", FrameStackWrapper), ("action", CarActionWrapper)])
if full_config.get("wrappers", False):
    for wrap_key in full_config["wrappers"]:
        game = wrapper_translate[wrap_key](game)


episodes = 5 # Number of episodes to see what the policy is doing.

discount_factor = full_config["discount_factor"]
total_iter = 0

num_episode = 0

model_type = full_config["agent_type"]
if model_type == "dqn" :
    model = DQNAgent(config=full_config["dqn_params"],
                     n_action=game.action_space,
                     state_dim=game.observation_space,
                     discount_factor=discount_factor,
                     biased_sampling=full_config["biased_sampling"])
else:
    raise NotImplementedError("{} not available for model".format(full_config["agent_type"]))


model.policy_net.load_state_dict(torch.load(os.path.join(expe_path, 'best_model.pth')))
print(expe_path)

with display as xvfb:

    while num_episode < episodes :
        state = torch.FloatTensor(game.reset()).unsqueeze(0)

        done = False
        iter_this_ep = 0
        reward_total_discounted = 0
        reward_total_not_discounted = 0

        while not done:

            action = model.select_action_greedy(state)
            next_state, reward, done, info = game.step(action=action.item())

            plt.imshow(game._unconvert(next_state[0, :, :]))
            plt.savefig(os.path.join(expe_path, 'test_ep{:03d}_step{:04d}_f1'.format(num_episode, iter_this_ep)))
            plt.close()
            plt.imshow(game._unconvert(next_state[1, :, :]))
            plt.savefig(os.path.join(expe_path, 'test_ep{:03d}_step{:04d}_f2'.format(num_episode, iter_this_ep)))
            plt.close()
            plt.imshow(game._unconvert(next_state[2, :, :]))
            plt.savefig(os.path.join(expe_path, 'test_ep{:03d}_step{:04d}_f3'.format(num_episode, iter_this_ep)))
            plt.close()

            reward = torch.FloatTensor([reward])
            next_state = torch.FloatTensor(next_state).unsqueeze(0)

            if done:
                next_state = None

            state = next_state

            total_iter += 1
            iter_this_ep = iter_this_ep + 1

            reward_total_discounted += reward * (discount_factor ** iter_this_ep)
            reward_total_not_discounted += reward

        print(
            "Reward at the end of ep #{}, n_timesteps {}, discounted rew : {} undiscounted : {}, current_eps {}".format(
                num_episode, total_iter, reward_total_discounted.item(), reward_total_not_discounted.item(), model.current_eps))


        num_episode += 1

    print("Experiment over")