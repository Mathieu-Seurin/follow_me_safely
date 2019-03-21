import gym

from ray.rllib.agents import impala, ppo
from ray.rllib.models import ModelCatalog, Model

import tensorflow as tf
import sonnet as snt

from rl_agent.model_utils import get_init_conv, get_init_mlp

from env_tools.wrapper import ObsSpaceWrapper

def env_basic_creator(env_name):
    import gym_minigrid
    env = ObsSpaceWrapper(gym.make(env_name))
    return env

def select_agent(algo_name, env_name, algo_config):

    algo = algo_name.lower()

    if algo=="impala":
        return impala.ImpalaAgent(config=algo_config, env=env_name)
    else:
        raise NotImplementedError("Not implemented error")

class ConvPolicy(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):

        config = options["custom_options"]

        initializers_mlp = get_init_mlp()
        initializers_conv = get_init_conv()

        # Image part
        #============
        state = input_dict["obs"]["image"]

        feat_stem = state
        stem_config = config["vision"]["stem_config"]

        for layer in range(stem_config["n_layers"]):
            feat_stem = snt.Conv2D(output_channels=stem_config["n_channels"][layer],
                                   # the number of channel is marked as list, index=channel at this layer
                                   kernel_shape=stem_config["kernel"][layer],
                                   stride=stem_config["stride"][layer],
                                   padding=snt.VALID,
                                   initializers=initializers_conv)(feat_stem)

            feat_stem = tf.nn.relu(feat_stem)

            if len(stem_config["pooling"]) > layer:
                feat_stem = tf.keras.layers.MaxPool2D(stem_config["pooling"][layer])(feat_stem)

        feat_stem = snt.BatchFlatten(preserve_dims=1)(feat_stem)

        fc = snt.Linear(num_outputs, initializers=initializers_mlp)(feat_stem)

        return fc, feat_stem



ModelCatalog.register_custom_model("conv_policy", ConvPolicy)