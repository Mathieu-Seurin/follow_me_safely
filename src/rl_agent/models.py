import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import gym
from rl_agent.gpu_utils import TORCH_DEVICE

class FullyConnectedModel(nn.Module):
    def __init__(self, config, n_action, state_dim):
        super().__init__()

        self.output_size = n_action.n
        self.n_input = state_dim.shape[0]

        self.n_hidden_mlp = config["n_mlp_hidden"]

        self.hidden_layer = nn.Linear(self.n_input, self.n_hidden_mlp)
        self.out_layer = nn.Linear(self.n_hidden_mlp, self.output_size)

    def forward(self, x):

        x = F.relu(self.hidden_layer(x))
        x = self.out_layer(x)

        return x

class ConvModel(nn.Module):
    def __init__(self, config, n_action, state_dim, learn_feedback_classif):

        super().__init__()

        self.n_action = n_action.n
        self.output_size = self.n_action
        self.n_hidden_mlp = config["n_mlp_hidden"]

        if isinstance(state_dim, gym.spaces.Box):
            self.input_shape = state_dim.shape
        else:
            self.input_shape = state_dim.spaces["state"].shape

        self.additionnal_input_size = 0

        self.conv_layers = nn.Sequential()

        in_channels = self.input_shape[0] # Because image are (N_FEAT_MAP,h,w) as usual

        # todo : check size
        for layer in range(config["n_layers"]):

            # CONV -> RELU -> (Optionnal MaxPooling)
            out_channels = config["out_channels"][layer]
            self.conv_layers.add_module(name="conv{}".format(layer + 1),
                                        module=nn.Conv2d(in_channels=in_channels,
                                                        out_channels= out_channels,
                                                        kernel_size=config["kernel_size"][layer]))

            in_channels = out_channels

            self.conv_layers.add_module(name="relu{}".format(layer + 1),
                                        module=nn.ReLU())

            if layer < len(config["pooling"]):
                self.conv_layers.add_module(name="max_pool{}".format(layer + 1),
                                            module=nn.MaxPool2d(config["pooling"][layer]))

        if config["use_memory"]:
            raise NotImplementedError("Not available at the moment, need to to recurrent replay buffer and recurrent update")

        size_to_fc = self.conv_layers(torch.ones(1, *self.input_shape)).view(1, -1).size(1)

        self.hidden_layer = nn.Linear(size_to_fc, self.n_hidden_mlp)
        self.output_layer = nn.Linear(self.n_hidden_mlp, self.output_size)

        if learn_feedback_classif:
            num_class = 2 # Feedback or not
            self.classif_layer = nn.Linear(self.n_hidden_mlp+self.n_action, num_class)

    def forward(self, x):

        #x = x['state']
        self.last_x = x # To check when using compute_classif

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        self.last_activation_before_output = F.relu(self.hidden_layer(x))
        output = self.output_layer(self.last_activation_before_output)
        return output

    def compute_classif_forward(self, states, actions):

        assert torch.all(states == self.last_x)
        batch_size = self.last_x.size(0)

        # In your for loop
        action_onehot = torch.zeros(batch_size, self.n_action).to(TORCH_DEVICE)
        action_onehot.scatter_(1, actions, 1)

        states = torch.cat([self.last_activation_before_output, action_onehot], dim=1)
        logits = self.classif_layer(states)

        return logits
