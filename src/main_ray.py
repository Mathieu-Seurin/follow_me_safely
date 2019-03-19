from ray.rllib.agents import ppo, dqn
import argparse
import ray

from rllib_utils import env_basic_creator
from ray.tune import register_env

from ray.tune.logger import pretty_print

import numpy as np

parser = argparse.ArgumentParser('Log Parser arguments!')

parser.add_argument("-env_config",    type=str)
parser.add_argument("-env_ext",       type=str)
parser.add_argument("-model_config",  type=str)
parser.add_argument("-model_ext",     type=str)
parser.add_argument("-exp_dir",       type=str, default="out", help="Directory all results")
parser.add_argument("-seed",          type=int, default=0, help="Random seed used")
parser.add_argument("-n_cpu",         type=int, default=24, help="How many cpus do you want ?")
parser.add_argument("-n_gpu",         type=int, default=1, help="How many cpus do you want ?")

args = parser.parse_args()

ray.init(num_cpus=args.n_cpu,
         num_gpus=args.n_gpu,
         #local_mode=ray.PYTHON_MODE
         )


full_config = load_config(env_config_file=args.env_config,
                          model_config_file=args.model_config,
                          env_ext_file=args.env_ext,
                          model_ext_file=args.model_ext
                          )


register_env(full_config["env_config"]["env"], lambda env_config: env_basic_creator(env_config))


#full_config["callbacks"] = {"on_episode_end" : call_back_function(on_episode_end)}
full_config["algo_config"]["monitor"] = False

agent = select_agent(full_config)

#agent.set_weights(agent.get_weights())
#agent.set_weights({'default' : np.load('test_weight.npy')})

max_reward = 0

for i in range(20):
    result = agent.train()
    print(pretty_print(result))

    if result['episode_reward_mean'] > max_reward:
        np.save('impala_weight.npy',agent.get_weights()['default'])
        max_reward = result['episode_reward_mean']
    #
    # if i % 10 == 0:
    #     checkpoint = agent.save()
    #     print("checkpoint saved at", checkpoint)
