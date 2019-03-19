import gym
from ray.rllib.models import ModelCatalog, Model

import tensorflow as tf
import sonnet as snt


def env_basic_creator(env_config):
    import gym_minigrid

    difficulty = env_config.get("difficulty", '')
    env_name = env_config["env"] + difficulty.capitalize() + "-v0"

    env = gym.make(env_name)

    return env

class ConvPolicy(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        pass


class FullConnectedPolicy(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):

        config = options["custom_options"]

        n_resblock = config["vision"]["resblock_config"]["n_resblock"]
        initializers_mlp = get_init_mlp()
        initializers_conv = get_init_conv()

        # Objective pipeline
        # ==================
        objective = input_dict["obs"]["mission"]

        # Embedding : if one-hot encoding for word -> no embedding
        embedded_obj = compute_embedding(objective=objective, config=config["text_objective_config"])


        # (+bi) lstm ( + layer_norm), specified in config
        last_ht_rnn = compute_dynamic_rnn(inputs=embedded_obj, config=config["text_objective_config"])

        # Image part
        #============
        state = input_dict["obs"]["image"]

        # Stem
        feat_stem = state
        stem_config = config["vision"]["stem_config"]

        for layer in range(stem_config["n_layers"]):
            feat_stem = snt.Conv2D(output_channels=stem_config["n_channels"][layer],
                                   # the number of channel is marked as list, index=channel at this layer
                                   kernel_shape=stem_config["kernel"][layer],
                                   stride=stem_config["stride"][layer],
                                   padding=snt.VALID,
                                   initializers=initializers_conv)(feat_stem)

        # Resblock and modulation
        resconfig = config["vision"]["resblock_config"]

        next_block = feat_stem
        for block in range(n_resblock):
            film_layer = FilmLayer()
            next_block = FilmedResblock(film_layer=film_layer,
                                        n_conv_channel=resconfig["n_channels"][block],
                                        kernel=resconfig["kernel"][block],
                                        stride=resconfig["stride"][block])({"state" : next_block,
                                                                            "objective" : last_ht_rnn})

        head_config = config["vision"]["head_config"]
        head_conv = snt.Conv2D(output_channels=head_config["n_channels"],
                                kernel_shape=head_config["kernel"],
                                stride=head_config["stride"],
                                padding=snt.SAME,
                                initializers=initializers_conv)(next_block)

        flatten_resblock = snt.BatchFlatten(preserve_dims=1)(head_conv)

        # Classifier
        # ===========
        out_mlp1 = tf.nn.relu(snt.Linear(config["last_layer_hidden"], initializers=initializers_mlp)(flatten_resblock))
        out_mlp2 = snt.Linear(num_outputs, initializers=initializers_mlp)(out_mlp1)

        return out_mlp2, out_mlp1