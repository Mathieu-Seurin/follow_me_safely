from main import train
import argparse

import json
import ray

import os

@ray.remote(num_gpus=0.3)
def dummy_train(a = 10):
    import torch
    print(a)
    print(torch.cuda.is_available())

@ray.remote
def dummy_train2(a = 10):
    import torch
    print(a)
    print(torch.cuda.is_available())


def read_multiple_config_file(config_path):

    json_config = json.load(open(os.path.join("config/multiple_run_config", config_path), "r"))

    all_expe_to_run = []
    for ext in json_config["model_ext"]:
        expe_config = {}
        expe_config["env_config"] = json_config["common"]["env_config"]
        expe_config["model_config"] = json_config["common"]["model_config"]
        expe_config["seed"] = json_config["common"]["seed"]

        expe_config["exp_dir"] = "out"
        expe_config["env_ext"] = ''
        expe_config["local_test"] = False

        expe_config["model_ext"] = ext

        all_expe_to_run.append(expe_config)

    return all_expe_to_run

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Log Parser arguments!')

    parser.add_argument("-multiple_run_config", type=str)
    parser.add_argument("-n_gpus", type=int, default=4)

    args = parser.parse_args()

    configs = read_multiple_config_file(args.multiple_run_config)
    print(configs)

    ray.init(num_gpus=args.n_gpus)
    ray.get([train.remote(**config) for config in configs])
