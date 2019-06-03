from main import train
import argparse

import json
import ray

import os
from config import read_multiple_ext_file, read_multiple_config_file, create_grid_search_config, extend_multiple_seed, read_run_directory_again

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
    parser.add_argument("-grid_search_config2", type=str)
    parser.add_argument("-run_dir", type=str)

    parser.add_argument("-n_gpus", type=int, default=4)
    parser.add_argument("-n_seeds", type=int, default=1)

    parser.add_argument("-out_dir", type=str)


    args = parser.parse_args()

    configs = []
    if args.multiple_run_config:
        configs.extend(read_multiple_config_file(args.multiple_run_config))

    if args.multiple_ext_config:
        configs.extend(read_multiple_ext_file(args.multiple_ext_config))

    if args.grid_search_config:
        configs.extend(create_grid_search_config(args.grid_search_config))

    if args.grid_search_config2:
        configs.extend(create_grid_search_config(args.grid_search_config2))

    if args.run_dir:
        configs.extend(read_run_directory_again(args.run_dir))

    if args.n_seeds > 1:
        configs = extend_multiple_seed(configs, number_of_seed=args.n_seeds)

    ray.init(num_gpus=args.n_gpus)

    print("Number of expe to launch : {}".format(len(configs)))

    if args.out_dir:
        for config in configs:
            config["exp_dir"] = args.out_dir


    ray.get([train.remote(**config, override_expe=False, save_images=True) for config in configs])
    #ray.get([train.remote(**config) for config in configs[10:12]])
    #ray.get([dummy_train.remote(config) for config in configs[:1]])
