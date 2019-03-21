import argparse

import gym.wrappers
from env_tools.wrapper import PreprocessWrapperPytorch

from config import load_config
from rl_agent.dqn_agent import DQNAgent, FullRandomAgent

import tensorboardX
import numpy as np

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
log_every = 100

if "minigrid" in full_config["env_name"]:
    game = PreprocessWrapperPytorch(gym.make(full_config["env_name"]))
else:
    game = gym.make(full_config["env_name"])

#game = gym.wrappers.Monitor(game, directory=expe_path, force=True)


episodes = full_config["stop"]["episodes_total"]
score_success = full_config["stop"]["episode_reward_mean"]

discount_factor = full_config["discount_factor"]
total_iter = 0
success_count = 0

reward_undiscount_list = []
reward_discount_list = []


model_type = full_config["agent_type"]
if model_type == "dqn" :
    model = DQNAgent(config=full_config["dqn_params"],
                     n_action=game.action_space,
                     state_dim=game.observation_space,
                     discount_factor=discount_factor)
elif model_type == "random" :
    model = FullRandomAgent(config=full_config, n_action=game.action_space, state_dim=game.observation_space)


for num_episode in range(1, episodes):
    state = game.reset()
    #game.render()
    done = False
    iter_this_ep = 0
    reward_total_discounted = 0
    reward_total_not_discounted = 0

    while not done:

        action = model.forward(state)
        next_state, reward, done, info = game.step(action=action)

        #game.render()
        if done:
            next_state = None

        model.push(state, action, next_state, reward)
        model.optimize()

        state = next_state

        total_iter += 1
        iter_this_ep = iter_this_ep + 1
        model.callback(timestep=total_iter)

        reward_total_discounted += reward * (discount_factor ** iter_this_ep)
        reward_total_not_discounted += reward

        if done:

            reward_undiscount_list.append(reward_total_not_discounted)
            reward_discount_list.append(reward_total_discounted)

            # todo create logging
            if num_episode%100 == 0:
                print("Reward at the end of ep #{}, n_timesteps {}, discounted rew : {} undiscounted : {}, current_eps {}".format(
                    num_episode, total_iter, reward_total_discounted, reward_total_not_discounted, model.current_eps))

                writer.add_scalar("data/sum_reward_discounted", np.mean(reward_discount_list), total_iter)
                writer.add_scalar("data/sum_reward_not_discounted", np.mean(reward_undiscount_list), total_iter)
                writer.add_scalar("data/iter_per_ep", iter_this_ep, total_iter)
                writer.add_scalar("data/epsilon", model.current_eps, total_iter)
                writer.add_scalar("data/model_update", model.num_update, total_iter)

            reward_undiscount_list = []
            reward_discount_list = []

            if reward_total_discounted > score_success:
                success_count += 1
            else:
                success_count = 0

    #print("Steps iter main {}, iter in model {}, eps {}".format(total_iter, model.n_step_eps, model.current_eps))

    if success_count > 5:
        break


print("Experiment over")