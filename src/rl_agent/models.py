import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np


class FullyConnectedModel(nn.Module):
    def __init__(self, config, n_action, state_dim):
        super(FullyConnectedModel, self).__init__()

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
