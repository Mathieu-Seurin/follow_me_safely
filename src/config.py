import json
import os

#from env.env_tool import env_basic_creator

# from ray.tune import register_env, Experiment, grid_search
# from ray.tune.schedulers import pbt

from ray.rllib.agents import ppo, dqn, impala
import random

#from neural_toolbox.algo_graph import FilmFrozenApex, VisionFrozenApex, FilmFrozenApexLoadedWeight, VisionFrozenApexLoadedWeight

import warnings

def override_config_recurs(config, config_extension):

    for key, value in config_extension.items():
        if type(value) is dict:
            config[key] = override_config_recurs(config[key], config_extension[key])
        else:
            assert key in config, "Warning, key defined in extension but not original : key is {}".format(key)
            config[key] = value

    return config

def create_experiment_directory(out, config_dict, seed):
    """
    Create all necessary directory to store experiment
    Storing :
        - Config files
        - Std output
        - Tensorboard logger (done in main loop)
    """





def load_single_config(config_file):
    with open(config_file, 'rb') as f_config:
        config_str = f_config.read().decode('utf-8')
        config = json.loads(config_str)
    return config

def load_config(env_config_file, model_config_file, seed,
                out_dir,
                env_ext_file=None,
                model_ext_file=None,
                ):
    def _check_json_intregrity(config_file_path, config_dict):
        config_file = open(config_file_path, 'r')
        config_dict_loaded = json.load(config_file)

        assert config_dict_loaded==config_dict, \
            """Error in config file handling, config_file on disk and this one must be the same !"
            config_dict : {}
            ========
            config_dict_loaded : {}
            """.format(config_dict, config_dict_loaded)


    # === Loading ENV config, extension and check integrity =====
    # ===========================================================
    env_config = load_single_config(os.path.join("config","env",env_config_file))

    # Override env file if specified
    if env_ext_file:
        env_ext_config = load_single_config(os.path.join("config", "env_ext", env_ext_file))
        env_config = override_config_recurs(env_config, env_ext_config)

    # create env_file if necessary
    env_name = env_config["env_name"]
    env_path = os.path.join(out_dir, env_name)

    if not os.path.exists(env_path):
        os.mkdir(env_path)

    env_config_path = os.path.join(env_path, "env_config.json")
    if not os.path.exists(env_config_path):
        config_file = open(env_config_path, 'w')
        json.dump(obj=env_config,fp=config_file)
    else:
        _check_json_intregrity(config_file_path=env_config_path,
                               config_dict=env_config)


    # === Loading MODEL config, extension and check integrity =====
    # ===========================================================
    model_config = load_single_config(os.path.join("config","model",model_config_file))
    # Override model file if specified
    # todo : check name
    if model_ext_file:
        model_ext_config = load_single_config(os.path.join("config","model_ext",model_ext_file))
        model_config = override_config_recurs(model_config, model_ext_config)

    # create model_file if necessary
    model_name = model_config["model_name"]
    model_path = os.path.join(env_path, model_name)

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model_config_path = os.path.join(model_path, "model_config.json")
    if not os.path.exists(model_config_path):
        config_file = open(model_config_path, 'w')
        json.dump(obj=model_config,fp=config_file)
    else:
        _check_json_intregrity(config_file_path=model_config_path,
                               config_dict=model_config)


    # Merge env and model config into one dict
    full_config = {**model_config, **env_config}

    # set seed
    set_seed(seed)

    return full_config

# =====================
# OTHER RANDOM FUNCTION
# =====================
def set_seed(seed):

    import torch
    import random
    import tensorflow
    import numpy as np

    if seed >= 0:
        print('Using seed {}'.format(seed))
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        tensorflow.set_random_seed(seed)
    else:
        raise NotImplementedError("Cannot set negative seed")




if __name__ == "__main__":
    pass
