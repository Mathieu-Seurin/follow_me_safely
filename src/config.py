import json
import os

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

def load_single_config(config_file):
    with open(config_file, 'rb') as f_config:
        config_str = f_config.read().decode('utf-8')
        config = json.loads(config_str)
    return config

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
                model_ext_file=None,
                ):

    # === Loading ENV config, extension and check integrity =====
    # ===========================================================
    env_config = load_single_config(os.path.join("config","env",env_config_file))

    # Override env file if specified
    if env_ext_file:
        env_ext_config = load_single_config(os.path.join("config", "env_ext", env_ext_file))
        env_config = override_config_recurs(env_config, env_ext_config)

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
    if model_ext_file:
        model_ext_config = load_single_config(os.path.join("config","model_ext",model_ext_file))
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
    else:
        print("Warning, experiment already exists, overriding it.")

    return full_config, model_path

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
        random.seed(seed)
    else:
        raise NotImplementedError("Cannot set negative seed")




if __name__ == "__main__":
    pass
