import random
import time
import argparse

from os import path

import gym
import gym_minigrid
import gym.wrappers

import numpy as np

from config import load_config
#from env.env_tool import env_basic_creator


parser = argparse.ArgumentParser('Log Parser arguments!')

parser.add_argument("-env_config",   type=str)
parser.add_argument("-env_ext",      type=str)
parser.add_argument("-model_config", type=str)
parser.add_argument("-model_ext",    type=str)
parser.add_argument("-exp_dir",      type=str, default="out", help="Directory all results")
parser.add_argument("-seed",         type=int, default=0, help="Random seed used")


args = parser.parse_args()

full_config = load_config(env_config_file=args.env_config,
                          model_config_file=args.model_config,
                          env_ext_file=args.env_ext,
                          model_ext_file=args.model_ext
                          )


#game = env_basic_creator(full_config["env_config"])
#game.render('human')
game = gym.make(full_config["env_name"])

#game = gym.wrappers.Monitor(game, "test_out", resume=False, force=True)

episodes = 100
mean = 0
rew_list = []
total_frame = 0

longest = 0

a = time.time()
for i in range(1, episodes):
    state = game.reset()
    done = False
    j = 0
    reward_total = 0
    while not done:
        action = game.action_space.sample()
        observation, reward, done, info = game.step(action=action)

        #longest = max(longest, len(observation['mission']))

        j = j+1
        reward_total += reward
        if done:
            rew_list.append(reward_total)
            total_frame += j

            if i == 0 :
                mean = reward_total
            else:
                mean = mean + reward_total/episodes
            break

print("time elpase", time.time() - a)
print("framerate :", total_frame / (time.time() - a))

print(mean)
print(np.mean(rew_list))

# print("Number gen double : ", game.env.count_double, "  ", game.env.count_gen)
# print("Number of double : ", game.env.count_double/game.env.count_gen)
