import argparse

import gym.wrappers
from env_tools.wrapper import CarFrameStackWrapper, CarActionWrapper, MinigridFrameStacker

from gym_minigrid.envs.safe_crossing import SafeCrossing

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

from rl_agent.agent_utils import render_state_and_q_values

from config import load_config
from rl_agent.dqn_agent import DQNAgent

import tensorboardX

import tensorflow as tf
import numpy as np
import torch

import os
import time

import ray

from env_tools.car_racing import CarRacingSafe

@ray.remote(num_gpus=0.24)
def train(env_config, env_ext, model_config, model_ext, exp_dir, seed, local_test, override_expe=True):

    print("Expe",env_config, env_ext, model_config, model_ext, exp_dir, seed, sep='  ')
    print("Is cuda available ?", torch.cuda.is_available())

    if local_test:
        display = open('nothing.txt', 'w')
    else :
        from xvfbwrapper import Xvfb
        display = Xvfb(width=100, height=100, colordepth=16)


    full_config, expe_path = load_config(env_config_file=env_config,
                              model_config_file=model_config,
                              env_ext_file=env_ext,
                              model_ext_file=model_ext,
                              out_dir=exp_dir,
                              seed=seed
                              )

    if override_expe == False:
        # Check that the experiment has run more than a few episodes
        # If so, DON'T rerun everything (useful for grid search)

        rerun_expe = True

        for dir in os.listdir(expe_path):

            last_ep = 0

            if "tfevents" in dir:
                tf_event_path = os.path.join(expe_path,dir)

                for i, elem in enumerate(tf.train.summary_iterator(tf_event_path)):
                    for v in elem.summary.value:
                        if 'episode' in v.tag:
                            last_ep = max(int(v.simple_value), last_ep)

                if last_ep < 10:
                    os.remove(tf_event_path)
                    print("Experiment doesn't seem to be over, rerun.")
                else:
                    rerun_expe = False

        if rerun_expe == False:
            print("Expe was over, don't rerun")
            return False


    writer = tensorboardX.SummaryWriter(expe_path)

    MAX_STATE_TO_REMEMBER = 50 # To avoid storing too much images in tensorboard
    DEFAULT_LOG_STATS = 3
    log_stats_every = full_config.get("log_stats_every", DEFAULT_LOG_STATS)

    if "racing" in full_config["env_name"].lower():
        reset_when_out = full_config["reset_when_out"]
        reward_when_out = full_config["reward_when_out"]
        max_steps = full_config["max_steps"]

        game = CarRacingSafe(reset_when_out=reset_when_out,
                             reward_when_out=reward_when_out,
                             max_steps=max_steps)

        DEFAULT_FRAME_SKIP = 3
        n_frameskip = full_config.get("frameskip", DEFAULT_FRAME_SKIP)

        game = CarActionWrapper(game)
        game = CarFrameStackWrapper(game, n_frameskip=n_frameskip)

    elif "minigrid" in full_config['env_name'].lower():

        reward_when_falling = full_config["reward_when_out"]
        size = full_config["size_env"]

        game = SafeCrossing(size=size, num_crossings=1, seed=seed, reward_when_falling=reward_when_falling)
        game = MinigridFrameStacker(game, full_config["n_frameskip"])

    else:
        game = gym.make(full_config["env_name"])

    episodes = full_config["stop"]["episodes_total"]
    score_success = full_config["stop"]["episode_reward_mean"]

    discount_factor = full_config["discount_factor"]
    total_iter = 0
    success_count = 0

    num_episode = 0
    early_stopping = False

    reward_undiscount_list = []
    reward_discount_list = []
    feedback_per_ep_list = []
    percentage_tile_seen_list = []
    iter_this_ep_list = []

    best_undiscount_reward = -float("inf")

    model_type = full_config["agent_type"]
    if model_type == "dqn" :
        model = DQNAgent(config=full_config["dqn_params"],
                         n_action=game.action_space,
                         state_dim=game.observation_space,
                         discount_factor=discount_factor,
                         writer=writer
                         )
    else:
        raise NotImplementedError("{} not available for model".format(full_config["agent_type"]))

    save_images_at = {0, 3, 20, 100, 200}

    with display as xvfb:

        while num_episode < episodes and not early_stopping :

            state = game.reset()
            state['state'] = torch.FloatTensor(state['state']).unsqueeze(0)
            state['gave_feedback'] = torch.FloatTensor([state['gave_feedback']])

            #game.render('human')
            done = False
            iter_this_ep = 0
            reward_total_discounted = 0
            reward_total_not_discounted = 0
            percentage_tile_seen = 0
            n_feedback_this_ep = 0

            rendered_images = []

            # Do we store images of state and q function associated with it ?
            if num_episode in save_images_at or num_episode == episodes - 1:
                save_images_and_q_this_ep = True
            else:
                save_images_and_q_this_ep = False

            while not done:

                action = model.select_action(state['state'])
                next_state, reward, done, info = game.step(action=action.item())

                reward = torch.FloatTensor([reward])
                next_state['state'] = torch.FloatTensor(next_state['state']).unsqueeze(0)
                next_state['gave_feedback'] = torch.FloatTensor([next_state['gave_feedback']])

                model.push(state['state'].to('cpu'), action, next_state['state'], reward, next_state['gave_feedback'])
                if not local_test:
                    model.optimize()

                # Render state, and compute q values to visualize them later
                if save_images_and_q_this_ep:
                    array_rendered = render_state_and_q_values(model=model, game=game, state=state)
                    rendered_images.append(array_rendered)

                    # Save only the last frames, to avoid overloading tensorboard
                    if len(rendered_images) > MAX_STATE_TO_REMEMBER:
                        rendered_images.pop(0)

                state = next_state

                total_iter += 1
                iter_this_ep = iter_this_ep + 1

                percentage_tile_seen = max(info.get('percentage_road_visited', 0), percentage_tile_seen)
                n_feedback_this_ep += info['gave_feedback']

                assert max(next_state['gave_feedback']) == info['gave_feedback'], "Problem, info should contain the same info as state"

                reward_total_discounted += reward * (discount_factor ** iter_this_ep)
                reward_total_not_discounted += reward

                # print("step",total_iter)
                # print(time.time() - timer)

            # DONE, GO HERE :
            # ================

            # Save images of state and q func associated
            if save_images_and_q_this_ep:
                for i, array_rendered in enumerate(rendered_images):
                    num_iter = iter_this_ep - len(rendered_images) + i + 1
                    writer.add_image('data/{}/state_and_q'.format(num_episode), global_step=num_iter,
                                     img_tensor=array_rendered, dataformats="HWC")

            # Update target network if needed
            model.callback(epoch=num_episode)

            reward_undiscount_list.append(reward_total_not_discounted.item())
            reward_discount_list.append(reward_total_discounted.item())

            feedback_per_ep_list.append(n_feedback_this_ep)
            percentage_tile_seen_list.append(percentage_tile_seen)
            iter_this_ep_list.append(iter_this_ep)

            if reward_total_discounted > score_success:
                success_count += 1
                if success_count > 5:
                    early_stopping = True
            else:
                success_count = 0

            if num_episode % log_stats_every == 0 or early_stopping:
                reward_discount_mean = np.mean(reward_discount_list)
                reward_undiscount_mean = np.mean(reward_undiscount_list)

                last_rewards_discount = np.mean(reward_discount_list[-log_stats_every:])
                last_rewards_undiscount = np.mean(reward_undiscount_list[-log_stats_every:])

                iter_this_ep_mean = np.mean(iter_this_ep_list)

                last_feedback_mean = np.mean(feedback_per_ep_list)

                writer.add_scalar("data/percentage_tile_seen", np.mean(percentage_tile_seen_list), total_iter)
                writer.add_scalar("data/number_of feedback", last_feedback_mean, total_iter)

                writer.add_scalar("data/reward_discounted", last_rewards_discount, total_iter)
                writer.add_scalar("data/reward_not_discounted", last_rewards_undiscount, total_iter)

                writer.add_scalar("data/running_mean_reward_discounted", reward_discount_mean, total_iter)
                writer.add_scalar("data/running_mean_reward_not_discounted", reward_undiscount_mean, total_iter)
                writer.add_scalar("data/iter_per_ep", iter_this_ep_mean, total_iter)
                writer.add_scalar("data/epsilon", model.current_eps, total_iter)
                writer.add_scalar("data/model_update", model.num_update_target, total_iter)
                writer.add_scalar("data/n_episode", num_episode, total_iter)
                # writer.add_scalar("data/model_update_ep", model.num_update_target, num_episode)

                print(
                    "End of ep #{}, n_timesteps {}, iter_this_ep : {}, current_eps {}".format(
                        num_episode, total_iter, iter_this_ep_mean, model.current_eps))

                print("Discounted rew : {} undiscounted : {}, n_feedback {} \n\n".format(
                    last_rewards_discount, last_rewards_undiscount, last_feedback_mean))

                if reward_discount_mean > best_undiscount_reward :
                    best_undiscount_reward = reward_discount_mean
                    torch.save(model.policy_net.state_dict(), os.path.join(expe_path,"best_model.pth"))

                torch.save(model.policy_net.state_dict(), os.path.join(expe_path, "last_model.pth"))

                # Reset feedback and percentage
                feedback_per_ep_list = []
                percentage_tile_seen_list = []
                iter_this_ep_list = []

            num_episode += 1

        print("Experiment over")


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Log Parser arguments!')

    parser.add_argument("-env_config", type=str)
    parser.add_argument("-env_ext", type=str)
    parser.add_argument("-model_config", type=str)
    parser.add_argument("-model_ext", type=str)
    parser.add_argument("-exp_dir", type=str, default="out", help="Directory all results")
    parser.add_argument("-seed", type=int, default=42, help="Random seed used")
    parser.add_argument("-local_test", type=bool, default=False, help="If env is run on my PC or a headless server")

    args = parser.parse_args()

    ext_test = {"dqn_params" : {"feedback_proportion_replayed" : 0.05}}

    ray.init(num_gpus=1, local_mode=True)
    ray.get(train.remote(args.env_config,
                         args.env_ext,
                         args.model_config,
                         args.model_ext,
                         args.exp_dir,
                         args.seed,
                         args.local_test,
                         override_expe=True))

    #ray.get(train.remote(args.env_config, args.env_ext, args.model_config, ext_test, args.exp_dir, args.seed, args.local_test))
