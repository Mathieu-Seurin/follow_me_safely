import ray
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')

import tensorboardX

import tensorflow as tf
import numpy as np

import time

@ray.remote(num_gpus=0.24)
def train(env_config, env_ext, model_config, model_ext, exp_dir, seed, local_test, override_expe=True, save_images=False):
    import argparse

    import gym.wrappers
    from env_tools.wrapper import CarFrameStackWrapper, CarActionWrapper, MinigridFrameStacker

    from gym_minigrid.envs.safe_crossing import SafeCrossing

    from rl_agent.agent_utils import render_state_and_q_values

    from config import load_config
    from rl_agent.dqn_agent import DQNAgent

    import torch

    from env_tools.car_racing import CarRacingSafe

    print("Expe",env_config, env_ext, model_config, model_ext, exp_dir, seed, sep='  ')
    print("Is cuda available ?", torch.cuda.is_available())

    if not local_test:
        assert len(ray.get_gpu_ids()) == 1

    assert torch.cuda.device_count() == 1, "Should be only 1, is {}".format(torch.cuda.device_count())

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


    MAX_STATE_TO_REMEMBER = 50 # To avoid storing too much images in tensorboard
    DEFAULT_LOG_STATS = 500
    log_stats_every = full_config.get("log_stats_every", DEFAULT_LOG_STATS)

    max_iter_expe = full_config["stop"]["max_iter_expe"]
    score_success = full_config["stop"]["episode_reward_mean"]

    if override_expe == False:
        # Check that the experiment has run more than a few episodes
        # If so, DON'T rerun everything (useful for grid search)

        rerun_expe = True

        for dir in os.listdir(expe_path):

            last_iter = 0

            if "tfevents" in dir:
                tf_event_path = os.path.join(expe_path,dir)

                for i, elem in enumerate(tf.train.summary_iterator(tf_event_path)):
                    if elem.step:
                        last_iter = max(last_iter, elem.step)

                if last_iter < max_iter_expe - log_stats_every:
                    os.remove(tf_event_path)
                    print("Experiment doesn't seem to be over, rerun.")
                else:
                    rerun_expe = False

        if rerun_expe == False:
            print("Expe was over, don't rerun")
            return True


    writer = tensorboardX.SummaryWriter(expe_path)


    if "racing" in full_config["env_name"].lower():
        reset_when_out = full_config["reset_when_out"]
        reward_when_falling = full_config["reward_when_out"]
        max_steps = full_config["max_steps"]

        game = CarRacingSafe(reset_when_out=reset_when_out,
                             reward_when_out=reward_when_falling,
                             max_steps=max_steps)

        DEFAULT_FRAME_SKIP = 3
        n_frameskip = full_config.get("frameskip", DEFAULT_FRAME_SKIP)

        game = CarActionWrapper(game)
        game = CarFrameStackWrapper(game, n_frameskip=n_frameskip)

    elif "minigrid" in full_config['env_name'].lower():

        reward_when_falling = full_config["reward_when_out"]
        size = full_config["size_env"]
        feedback_when_wall_hit = full_config["feedback_when_wall_hit"]
        proba_reset = full_config["proba_reset"]
        use_lava = full_config["use_lava"]
        n_zone = full_config["n_zone"]
        good_zone_action_proba = full_config["good_zone_action_proba"]
        bad_zone_action_proba = full_config["bad_zone_action_proba"]
        obstacle_type = full_config["obstacle_type"]

        game = SafeCrossing(size=size,
                            reward_when_falling=reward_when_falling,
                            proba_reset = proba_reset,
                            feedback_when_wall_hit=feedback_when_wall_hit,
                            use_lava=use_lava,
                            n_zone=n_zone,
                            good_zone_action_proba=good_zone_action_proba,
                            bad_zone_action_proba=bad_zone_action_proba,
                            obstacle_type=obstacle_type,
                            seed=seed)

        game = MinigridFrameStacker(game, full_config["n_frameskip"])

    else:
        game = gym.make(full_config["env_name"])


    discount_factor = full_config["discount_factor"]
    total_iter = 0
    success_count = 0

    num_episode = 0
    early_stopping = False

    reward_wo_feedback_list = []
    reward_undiscount_list = []
    reward_discount_list = []
    feedback_per_ep_list = []
    percentage_tile_seen_list = []

    iter_this_ep_list = []
    last_reward_undiscount_list = []
    last_reward_discount_list = []

    self_destruct_list = []
    self_destruct_trial_list = []

    best_undiscount_reward = -float("inf")

    model_type = full_config["agent_type"]
    if model_type == "dqn" :
        model = DQNAgent(config=full_config["dqn_params"],
                         n_action=game.action_space,
                         state_dim=game.observation_space,
                         discount_factor=discount_factor,
                         writer=writer,
                         log_stats_every=log_stats_every
                         )
    else:
        raise NotImplementedError("{} not available for model".format(full_config["agent_type"]))

    #save_images_at = {1, 2, 3, 20, 100, 1000, 4000, 8000, 8001, 8002, 8003}
    save_images_at = {50, 51, 52, 53, 2000, 2001, 2002, 2003, 2004}

    with display as xvfb:

        while total_iter < max_iter_expe and not early_stopping :

            state = game.reset()
            state['state'] = torch.FloatTensor(state['state']).unsqueeze(0)
            state['gave_feedback'] = torch.FloatTensor([state['gave_feedback']])

            #game.render('human')
            done = False
            iter_this_ep = 0
            reward_wo_feedback = 0
            reward_total_discounted = 0
            reward_total_not_discounted = 0
            percentage_tile_seen = 0

            n_feedback_this_ep = 0

            self_kill_trial = 0

            rendered_images = []

            # Do we store images of state and q function associated with it ?
            if save_images and (num_episode in save_images_at):
                save_images_and_q_this_ep = True
            else:
                save_images_and_q_this_ep = False

            while not done:

                # Render state, and compute q values to visualize them later
                if save_images_and_q_this_ep:
                    array_rendered = render_state_and_q_values(model=model, game=game, state=state)
                    rendered_images.append(array_rendered)

                    # Save only the last frames, to avoid overloading tensorboard
                    if len(rendered_images) > MAX_STATE_TO_REMEMBER:
                        rendered_images.pop(0)

                action = model.select_action(state['state'])
                next_state, reward, done, info = game.step(action=action.item())

                reward = torch.FloatTensor([reward])

                if done:
                    next_state['state'] = None
                else:
                    next_state['state'] = torch.FloatTensor(next_state['state']).unsqueeze(0)
                next_state['gave_feedback'] = torch.FloatTensor([next_state['gave_feedback']])

                model.push(state['state'].to('cpu'), action, next_state['state'], reward, next_state['gave_feedback'])
                model.optimize(total_iter=total_iter)

                state = next_state

                total_iter += 1
                iter_this_ep = iter_this_ep + 1

                percentage_tile_seen = max(info.get('percentage_road_visited', 0), percentage_tile_seen)
                n_feedback_this_ep += info['gave_feedback']
                self_kill_trial += info['tried_destruct']

                assert max(next_state['gave_feedback']) == info['gave_feedback'], "Problem, info should contain the same info as state"

                reward_total_discounted += reward * (discount_factor ** iter_this_ep)
                reward_total_not_discounted += reward

                reward_wo_feedback += reward - info['gave_feedback'] * reward_when_falling


                #=======================
                # LOG STATS HERE
                if total_iter % log_stats_every == 0:
                    reward_discount_mean = np.mean(reward_discount_list)
                    reward_undiscount_mean = np.mean(reward_undiscount_list)

                    last_rewards_discount = np.mean(last_reward_undiscount_list)
                    last_rewards_undiscount = np.mean(last_reward_discount_list)

                    last_reward_wo_feedback = np.mean(reward_wo_feedback_list)

                    iter_this_ep_mean = np.mean(iter_this_ep_list)

                    last_feedback_mean = np.mean(feedback_per_ep_list)

                    if "racing" in full_config["env_name"].lower():
                        writer.add_scalar("data/percentage_tile_seen", np.mean(percentage_tile_seen_list), total_iter)

                    writer.add_scalar("data/number_of feedback", last_feedback_mean, total_iter)

                    # writer.add_scalar("data/reward_discounted", last_rewards_discount, total_iter)
                    # writer.add_scalar("data/reward_not_discounted", last_rewards_undiscount, total_iter)

                    writer.add_scalar("data/reward_wo_feedback(unbiaised)", last_reward_wo_feedback, total_iter)
                    writer.add_scalar("data/n_episodes", num_episode, total_iter)

                    #writer.add_scalar("data/self_destruct_trial", np.mean(self_destruct_trial_list), total_iter)
                    writer.add_scalar("data/self_destruct", np.mean(self_destruct_list), total_iter)

                    # writer.add_scalar("data/running_mean_reward_discounted", reward_discount_mean, total_iter)
                    # writer.add_scalar("data/running_mean_reward_not_discounted", reward_undiscount_mean, total_iter)
                    writer.add_scalar("data/iter_per_ep", iter_this_ep_mean, total_iter)
                    writer.add_scalar("data/epsilon", model.current_eps, total_iter)
                    # writer.add_scalar("data/model_update", model.num_update_target, total_iter)
                    writer.add_scalar("data/n_episode_since_last_log", len(last_reward_discount_list), total_iter)
                    # writer.add_scalar("data/model_update_ep", model.num_update_target, num_episode)

                    if last_rewards_undiscount > best_undiscount_reward:
                        best_undiscount_reward = reward_discount_mean
                        torch.save(model.policy_net.state_dict(), os.path.join(expe_path, "best_model.pth"))

                    torch.save(model.policy_net.state_dict(), os.path.join(expe_path, "last_model.pth"))

                    # Reset feedback and percentage
                    feedback_per_ep_list = []
                    percentage_tile_seen_list = []
                    last_reward_undiscount_list = []
                    last_reward_discount_list = []
                    iter_this_ep_list = []
                    reward_wo_feedback_list = []

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

            last_reward_undiscount_list.append(reward_total_not_discounted.item())
            last_reward_discount_list.append(reward_total_discounted.item())

            feedback_per_ep_list.append(n_feedback_this_ep)
            percentage_tile_seen_list.append(percentage_tile_seen)
            iter_this_ep_list.append(iter_this_ep)

            self_destruct_list.append(info['self_destruct'])
            self_destruct_trial_list.append(self_kill_trial)
            reward_wo_feedback_list.append(reward_wo_feedback)


            print("End of ep #{}, n_timesteps (estim) {}, iter_this_ep : {}, current_eps {}".format(
                num_episode, total_iter, np.mean(iter_this_ep_list[-3:]), model.current_eps))

            print("(Estim) Discounted rew : {} undiscounted : {}, n_feedback {} \n\n".format(
                np.mean(last_reward_discount_list[-3:]), np.mean(last_reward_undiscount_list[-3:]), np.mean(feedback_per_ep_list[-3:])))

            if reward_total_discounted > score_success:
                success_count += 1
                if success_count > 5:
                    early_stopping = True
            else:
                success_count = 0

            num_episode += 1

        print("Experiment over")

    # Enforce cleaning
    del model.memory
    del model
    del game
    torch.cuda.empty_cache()
    return True


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser('Log Parser arguments!')

    parser.add_argument("-env_config", type=str)
    parser.add_argument("-env_ext", type=str)
    parser.add_argument("-model_config", type=str)
    parser.add_argument("-model_ext", type=str)
    parser.add_argument("-exp_dir", type=str, default="out", help="Directory all results")
    parser.add_argument("-seed", type=int, default=42, help="Random seed used")
    parser.add_argument("-local_test", type=bool, default=False, help="If env is run on my PC or a headless server")

    args = parser.parse_args()

    ray.init(num_gpus=1, local_mode=args.local_test)
    a = ray.get(train.remote(args.env_config,
                             args.env_ext,
                             args.model_config,
                             args.model_ext,
                             args.exp_dir,
                             args.seed,
                             args.local_test,
                             override_expe=False,
                             save_images=True))

    print(a)

    # ext_test = {'dqn_params': {'feedback_percentage_in_buffer': 0.1, 'learning_rate': 0.001, 'consistency_loss_weight': 1,
    #                            'classification_margin': 0.01, 'classification_loss_weight': 0, 'classification_max_loss_weight': 0.7
    #                            },
    #             'name': 'feedback_percentage_in_buffer:0.1-learning_rate:0.001-consistency_loss_weight:1-classification_margin:0.01-classification_loss_weight:0-classification_max_loss_weight:0.7'
    #             }
    #
    # ray.get(train.remote(args.env_config, args.env_ext, args.model_config, ext_test, args.exp_dir, args.seed, args.local_test))
