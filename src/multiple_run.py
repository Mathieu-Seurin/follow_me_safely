from main import train
import argparse

import json
import ray

import os
from config import read_multiple_ext_file, read_multiple_config_file, create_grid_search_config

@ray.remote(num_gpus=0.3)
def dummy_train_gpu(a):
    import torch
    print(a)
    print(torch.cuda.is_available())

@ray.remote
def dummy_train(a):
    import torch
    print(a)
    print(torch.cuda.is_available())

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Log Parser arguments!')

    parser.add_argument("-multiple_ext_config", type=str)
    parser.add_argument("-multiple_run_config", type=str)
    parser.add_argument("-grid_search_config", type=str)
    parser.add_argument("-n_gpus", type=int, default=4)

    args = parser.parse_args()

    configs = []
    if args.multiple_run_config:
        configs.extend(read_multiple_ext_file(args.multiple_run_config))

    if args.multiple_ext:
        configs.extend(read_multiple_ext_file(args.multiple_ext))

    if args.grid_search_config:
        configs.extend(create_grid_search_config(args.grid_search_config))

    ray.init(num_gpus=args.n_gpus)
    ray.get([dummy_train.remote(**config) for config in configs])
