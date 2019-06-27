from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from gym_minigrid.envs.safe_crossing import SafeCrossing
from env_tools.wrapper import MinigridFrameStacker

import numpy as np

import gym

seed = 42
full_config = {"reward_when_out" : 0,
               "size_env" : 7,
               "feedback_when_wall_hit" : False,
               "proba_reset" : 0,
               "use_lava" : False,
               "n_zone" : 3,
               "good_zone_action_proba" : 1,
               "bad_zone_action_proba" : 0,
               "obstacle_type" : 'none',
               "n_frameskip" : 3}


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
                    proba_reset=proba_reset,
                    feedback_when_wall_hit=feedback_when_wall_hit,
                    use_lava=use_lava,
                    n_zone=n_zone,
                    good_zone_action_proba=good_zone_action_proba,
                    bad_zone_action_proba=bad_zone_action_proba,
                    obstacle_type=obstacle_type,
                    seed=seed)

games = lambda : MinigridFrameStacker(game, full_config["n_frameskip"])


#games = DummyVecEnv([games])
#games = SubprocVecEnv([games, games])

games = lambda : gym.make("CartPole-v0")
games = SubprocVecEnv([games, games])

n_episodes = 10
for ep in range(n_episodes):

    obs = games.reset()
    done = [False for i in range(2)]

    while not all(done):

        next_state, rew, done, info = games.step(np.random.randint(0,2,size=2))

        model.act
        print(done)

