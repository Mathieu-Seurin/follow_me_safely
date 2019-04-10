import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn

import logging


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return [self.memory[sample] for sample in np.random.choice(len(self), batch_size)]

    def __len__(self):
        return len(self.memory)


def freeze_as_np_dict(tensor_dict):
    out = {}
    for key in tensor_dict.keys():
        out[key] = tensor_dict[key].cpu().clone().numpy()
    return out

def check_params_changed(dict1, dict2):
    "Takes two parameters dict (key:torchtensor) and prints a warning if identical"

    dict1, dict2 = dict1, dict2

    for key in dict1.keys():
        if key.split('.')[-1] in ['running_mean', 'running_var']:
            # No message if this is for BatchNorm
            continue
        tmp1 = dict1[key]
        tmp2 = dict2[key]
        if torch.max(torch.abs(tmp1 - tmp2)).item() == 0:
            print('No change in params {}'.format(key))

def compute_slow_params_update(slow_params, fast_params, tau):

    slow_params_dict = slow_params.state_dict()
    fast_params_dict = fast_params.state_dict()

    for module_key in slow_params_dict.keys() :
        slow_params_dict[module_key] += tau*(fast_params_dict[module_key] - slow_params_dict[module_key])

    return slow_params_dict