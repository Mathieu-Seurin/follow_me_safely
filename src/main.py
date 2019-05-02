import argparse

import gym.wrappers
from env_tools.wrapper import PreprocessWrapperPytorch, FrameStackWrapper, CarActionWrapper

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')


from config import load_config
from rl_agent.dqn_agent import DQNAgent

import tensorboardX
import numpy as np
import torch

import os
import time

import ray

from env_tools.car_racing import CarRacingSafe

@ray.remote(num_gpus=0.5)
def train(env_config, env_ext, model_config, model_ext, exp_dir, seed, local_test, save_images_freq=3):

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

    writer = tensorboardX.SummaryWriter(expe_path)
    log_every = 100

    if "safe" in full_config["env_name"].lower():
        reset_when_out = full_config["reset_when_out"]
        reward_when_out = full_config["reward_when_out"]
        max_steps = full_config["max_steps"]

        game = CarRacingSafe(reset_when_out=reset_when_out,
                             reward_when_out=reward_when_out,
                             max_steps=max_steps)

        DEFAULT_FRAME_SKIP = 3
        n_frameskip = full_config.get("frameskip", DEFAULT_FRAME_SKIP)

        game = CarActionWrapper(game)
        game = FrameStackWrapper(game, n_frameskip=n_frameskip)

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

            while not done:

                action = model.select_action(state['state'])
                next_state, reward, done, info = game.step(action=action.item())

                reward = torch.FloatTensor([reward])
                next_state['state'] = torch.FloatTensor(next_state['state']).unsqueeze(0)
                next_state['gave_feedback'] = torch.FloatTensor([next_state['gave_feedback']])

                model.push(state['state'].to('cpu'), action, next_state['state'], reward, next_state['gave_feedback'])
                if not local_test:
                    model.optimize()

                if num_episode % save_images_freq == 0:

                    q = model.get_q_values(state['state'])
                    max_action = torch.max(q, dim=1)[1].item()

                    fig = plt.figure()
                    fig.add_subplot(121)

                    plt.imshow(game._unconvert(next_state['state'].numpy()[0, 2, :, :]))

                    f = plt.gcf()
                    f.set_size_inches(18.5, 10.5)

                    fig.add_subplot(122)

                    plt.bar(list(range(game.action_space.n)), height=q[0, :].cpu(),
                            color=[(0.1, 0.2, 0.8) if i != max_action else (0.8, 0.1, 0.1) for i in
                                   range(game.action_space.n)], tick_label=[str(l) for l in game.env.action_map])

                    plt.xticks(fontsize=18, rotation=70)

                    plt.xlabel('action', fontsize=16)
                    plt.ylabel('q_value', fontsize=16)

                    plt.tight_layout()

                    fig.canvas.draw()
                    array_rendered = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    array_rendered = array_rendered.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                    #X = np.array(fig.canvas)

                    writer.add_image('data/state_and_q', img_tensor=array_rendered, dataformats="HWC")
                    #plt.show()
                    plt.close()

                state = next_state

                total_iter += 1
                iter_this_ep = iter_this_ep + 1

                percentage_tile_seen = max(info['percentage_road_visited'], percentage_tile_seen)
                n_feedback_this_ep += info['gave_feedback']

                assert max(next_state['gave_feedback']) == info['gave_feedback'], "Problem, info should contain the same info as state"

                reward_total_discounted += reward * (discount_factor ** iter_this_ep)
                reward_total_not_discounted += reward

                # print("step",total_iter)
                # print(time.time() - timer)

            # DONE GO HERE :
            model.callback(epoch=num_episode)

            reward_undiscount_list.append(reward_total_not_discounted.item())
            reward_discount_list.append(reward_total_discounted.item())

            if reward_total_discounted > score_success:
                success_count += 1
                if success_count > 5:
                    early_stopping = True
            else:
                success_count = 0

            if num_episode % 2 == 0 or early_stopping:
                reward_discount_mean = np.mean(reward_discount_list)
                reward_undiscount_mean = np.mean(reward_undiscount_list)

                print(
                    "Reward at the end of ep #{}, n_timesteps {}, discounted rew : {} undiscounted : {}, current_eps {}".format(
                        num_episode, total_iter, reward_discount_mean, reward_undiscount_mean, model.current_eps))

                writer.add_scalar("data/percentage_tile_seen", percentage_tile_seen, total_iter)
                writer.add_scalar("data/number_of feedback", n_feedback_this_ep, total_iter)
                writer.add_scalar("data/sum_reward_discounted", reward_discount_mean, total_iter)
                writer.add_scalar("data/sum_reward_not_discounted", reward_undiscount_mean, total_iter)
                writer.add_scalar("data/iter_per_ep", iter_this_ep, total_iter)
                writer.add_scalar("data/epsilon", model.current_eps, total_iter)
                writer.add_scalar("data/model_update", model.num_update, total_iter)
                writer.add_scalar("data/model_update_ep", model.num_update, num_episode)

                if reward_discount_mean > best_undiscount_reward :
                    best_undiscount_reward = reward_discount_mean
                    torch.save(model.policy_net.state_dict(), os.path.join(expe_path,"best_model.pth"))

                torch.save(model.policy_net.state_dict(), os.path.join(expe_path, "last_model.pth"))



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

    ray.init(num_gpus=1, local_mode=True)
    ray.get(train.remote(args.env_config, args.env_ext, args.model_config, args.model_ext, args.exp_dir, args.seed, args.local_test))
