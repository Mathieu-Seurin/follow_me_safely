import json
import os
import itertools as it

from copy import deepcopy
from random import shuffle

EXPE_DEFAULT_CONFIG = {
    "env_ext" : 'shorter.json',
    "model_ext" : '',
    "seed": 42,
    "exp_dir": "grid_out",
    "local_test": False}

def override_config_recurs(config, config_extension):

    for key, value in config_extension.items():
        if type(value) is dict:
            config[key] = override_config_recurs(config[key], config_extension[key])
        else:
            assert key in config, "Warning, key defined in extension but not original : new key is {}".format(key)

            # Don't override names, change add name extension to the original
            if key == "name":
                config["name"] = config["name"]+"_"+value
            else:
                config[key] = value

    return config

def load_single_config(config_path):
    return json.load(open(config_path, "r"))

def check_json_intregrity(config_file_path, config_dict):
    config_file = open(config_file_path, 'r')
    config_dict_loaded = json.load(config_file)

    assert config_dict_loaded == config_dict, \
        """
        Error in config file handling, config_file on disk and this one must be the same !"
        config_dict :        {}
        ========
        config_dict_loaded : {}
        
        """.format(config_dict, config_dict_loaded)

def load_config(env_config_file, model_config_file, seed,
                out_dir,
                env_ext_file=None,
                model_ext_file=None
                ):

    # === Loading ENV config, extension and check integrity =====
    # ===========================================================
    env_config = load_single_config(os.path.join("config", "env", env_config_file))

    # Override env file if specified
    if env_ext_file:
        env_ext_config = load_single_config(os.path.join("config","env_ext", env_ext_file))
        env_config = override_config_recurs(env_config, env_ext_config)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # create env_file if necessary
    env_name = env_config["name"]
    env_path = os.path.join(out_dir, env_name)

    if not os.path.exists(env_path):
        os.mkdir(env_path)

    env_config_path = os.path.join(env_path, "env_config.json")
    if not os.path.exists(env_config_path):
        config_file = open(env_config_path, 'w')
        json.dump(obj=env_config,fp=config_file)
    else:
        check_json_intregrity(config_file_path=env_config_path,
                              config_dict=env_config)


    # === Loading MODEL config, extension and check integrity =====
    # ===========================================================
    model_config = load_single_config(os.path.join("config","model",model_config_file))

    # Override model file if specified
    # Can be a dict of parameters or a str indicating the path to the extension
    if model_ext_file:
        if type(model_ext_file) is str:
            model_ext_config = load_single_config(os.path.join("config", "model_ext", model_ext_file))
        else:
            assert type(model_ext_file) is dict, "Not a dict problem, type : {}".format(type(model_ext_file))
            model_ext_config = model_ext_file

        model_config = override_config_recurs(model_config, model_ext_config)
    else:
        model_ext_config = {"name" : ''}



    # create model_file if necessary
    model_name = model_config["name"]
    model_path = os.path.join(env_path, model_name)

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model_config_path = os.path.join(model_path, "model_full_config.json")
    if not os.path.exists(model_config_path):
        config_file = open(model_config_path, 'w')
        json.dump(obj=model_config,fp=config_file)

        # Dump the extension file too, easier to visualize quickly
        model_ext_config_path = os.path.join(model_path, "model_ext_config.json")
        model_ext_config_file = open(model_ext_config_path, 'w')
        json.dump(obj=model_ext_config, fp=model_ext_config_file)

    else:
        check_json_intregrity(config_file_path=model_config_path,
                              config_dict=model_config)

    # Merge env and model config into one dict
    full_config = {**model_config, **env_config}
    full_config["model_name"] = model_config["name"]
    full_config["env_name"] = env_config["name"]
    del full_config["name"]

    # set seed
    set_seed(seed)
    path_to_expe = os.path.join(model_path, str(seed))

    if not os.path.exists(path_to_expe):
        os.mkdir(path_to_expe)

    return full_config, model_path

def read_multiple_ext_file(config_path):

    json_config = json.load(open(os.path.join("config/multiple_run_config", config_path), "r"))

    all_expe_to_run = []
    for ext in json_config["model_ext"]:

        expe_config = EXPE_DEFAULT_CONFIG
        expe_config.update(json_config["common"])

        expe_config["model_ext"] = ext

        all_expe_to_run.append(expe_config)

    return all_expe_to_run

def read_multiple_config_file(config_path):

    json_config = json.load(open(os.path.join("config/multiple_run_config", config_path), "r"))
    assert type(json_config) == list, "Should be a list"

    all_expe_to_run = []

    for config in json_config:

        expe_config = deepcopy(EXPE_DEFAULT_CONFIG)
        expe_config.update(config)
        all_expe_to_run.append(expe_config)

    return all_expe_to_run

def create_grid_search_config(grid_path):

    grid_config = json.load(open(os.path.join("config", "multiple_run_config", grid_path), 'r'))

    all_expe_to_run = []

    param_storage = dict()
    param_storage["key_order"] = []
    param_storage["lists"] = []

    for key in grid_config["dqn_params"].keys():
        param_storage["key_order"].append(key)
        param_storage["lists"].append(grid_config["dqn_params"][key])

    for param_tuple in it.product(*param_storage["lists"]):

        params_dict = dict(zip(param_storage["key_order"], param_tuple))
        expe_ext = {"dqn_params" : params_dict}

        # Join key and value in the name
        expe_ext["name"] = '-'.join([':'.join(map(str,items)) for items in params_dict.items()])

        expe_config = deepcopy(EXPE_DEFAULT_CONFIG)
        expe_config["model_ext"] = expe_ext
        expe_config["model_config"] = grid_config["model_config"]
        expe_config["env_config"] = grid_config["env_config"]
        expe_config["env_ext"] = grid_config["env_ext"]

        all_expe_to_run.append(expe_config)


    expe_to_filter = []

    # Filter list : delete expe who doesn't have at least one condition specified
    only_one_required = grid_config["only_one_required"]
    for num_expe, expe in enumerate(all_expe_to_run):
        for condition in only_one_required:
            value1 = expe["model_ext"]["dqn_params"][condition[0]]
            value2 = expe["model_ext"]["dqn_params"][condition[1]]

            # if both are 0 reject, if both are positive => reject
            if bool(value1) == bool(value2) :
                expe_to_filter.append(num_expe)

    # other filter ?

    # Delete expe that are in the filter
    for num_expe in reversed(expe_to_filter):
        all_expe_to_run.pop(num_expe)

    # Grid search become random search, yea.
    shuffle(all_expe_to_run)

    return all_expe_to_run


# =====================
# OTHER RANDOM FUNCTION
# =====================
def set_seed(seed):

    import torch
    import random
    import numpy as np

    if seed >= 0:
        print('Using seed {}'.format(seed))
        np.random.seed(seed)
        torch.manual_seed(seed)
        # random.seed(seed)
    else:
        raise NotImplementedError("Cannot set negative seed")




if __name__ == "__main__":
    pass
