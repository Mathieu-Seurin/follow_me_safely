import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np


class FullyConnectedModel(nn.Module):
    def __init__(self, config, n_action, state_dim):
        super().__init__()

        self.output_size = n_action.n
        self.n_hidden_mlp = config["n_mlp_hidden"]

        self.n_input = np.product(state_dim.spaces["image"].shape)
        self.additionnal_input_size = 0

        self.hidden_layer = nn.Linear(self.n_input+self.additionnal_input_size, self.n_hidden_mlp)
        self.out_layer = nn.Linear(self.n_hidden_mlp, self.output_size)

    def forward(self, x):

        x = x["env_state"]

        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        x = F.relu(self.hidden_layer(x))
        x = self.out_layer(x)

        return x

class ConvModel(nn.Module):
    def __init__(self, config, n_action, state_dim):

        super().__init__()

        self.output_size = n_action.n
        self.n_hidden_mlp = config["n_mlp_hidden"]

        self.n_input = state_dim.spaces["image"]
        self.additionnal_input_size = 0

        self.conv_layers = nn.Sequential()

        in_channels = self.n_input.shape[2] # Because image are h,w, N_FEAT_MAP

        # todo : check size
        for layer in range(config["n_layers"]) :

            # CONV -> RELU -> (Optionnal MaxPooling)
            self.conv_layers.add_module(name="conv{}".format(layer + 1),
                                        module=nn.Conv2d(in_channels=in_channels,
                                                        out_channels=config["out_channels"][layer],
                                                        kernel_size=config["filter_size"][layer]))
            self.conv_layers.add_module(name="relu{}".format(layer + 1),
                                        module=nn.ReLU())

            if layer < len(config["pool"]):
                self.conv_layers.add_module(name="max_pool{}".format(layer + 1),
                                            module=nn.MaxPool2d(config["pooling"][layer]))

        if config["use_memory"]:
            raise NotImplementedError("Not available at the moment, need to to recurrent replay buffer and recurrent update")


        size_to_fc = self.conv_layers(torch.ones(1, *self.n_input)).view(1, -1).size(1)

        self.hidden_layer = nn.Linear(size_to_fc, self.n_hidden_mlp)
        self.output_layer = nn.Linear(self.n_hidden_mlp, n_action)

    def forward(self, x):

        x = self.conv_layers(x)
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)

        return x