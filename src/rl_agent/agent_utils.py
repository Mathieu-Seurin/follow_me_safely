import numpy as np
from collections import namedtuple

import torch
from rl_agent.gpu_utils import TORCH_DEVICE

import logging

import matplotlib.pyplot as plt
import copy

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'gave_feedback'))


class PrioritizedReplayMemory(object):

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
        return [self.memory[sample] for sample in np.random.choice(len(self), batch_size, replace=True)]

    def __len__(self):
        return len(self.memory)



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
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        samples = [self.memory[sample] for sample in np.random.choice(len(self), batch_size, replace=True)]
        return samples

    def __len__(self):
        return len(self.memory)


class ProportionReplayMemory(object):

    def __init__(self, capacity):

        self.capacity = capacity

        self.memory = []
        self.position = 0

        self.memory_feedback = []
        self.position_feedback = 0

    def push(self, *args):
        """Saves a transition."""

        if args[4]: # Gave feedback
            if len(self.memory_feedback) < self.capacity:
                self.memory_feedback.append(None)
            self.memory_feedback[self.position_feedback] = Transition(*args)
            self.position_feedback = (self.position_feedback + 1) % self.capacity
        else:
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = Transition(*args)
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, proportion=0):

        if proportion == 0:
            proportion = len(self.memory_feedback) / len(self)

        batch_size_feed = int(batch_size * proportion)

        # Sample from memory who stores non-feedback tuple
        samples = [self.memory[sample] for sample in np.random.choice(len(self.memory), batch_size - batch_size_feed, replace=True)]
        # Add sample from feedback tuple
        samples.extend([self.memory_feedback[sample] for sample in np.random.choice(len(self.memory_feedback), batch_size_feed, replace=True)])

        return samples

    def __len__(self):
        return len(self.memory)+len(self.memory_feedback)

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

def compute_slow_params_update(slow_network, fast_network, tau):

    slow_params_dict = slow_network.state_dict()
    fast_params_dict = fast_network.state_dict()

    for module_key in slow_params_dict.keys() :
        slow_params_dict[module_key] += tau*(fast_params_dict[module_key] - slow_params_dict[module_key])

    return slow_params_dict


def render_state_and_q_values(model, game, state):

    processed_state = model.preprocessor(copy.deepcopy(state['state']))

    q = model.get_q_values(processed_state)
    max_action = torch.max(q, dim=1)[1].item()

    fig = plt.figure()
    fig.add_subplot(121)

    plt.imshow(game._unconvert(state))

    f = plt.gcf()
    f.set_size_inches(9, 5)

    fig.add_subplot(122)

    plt.bar(list(range(game.action_space.n)), height=q[0, :].cpu(),
            color=[(0.1, 0.2, 0.8) if i != max_action else (0.8, 0.1, 0.1) for i in
                   range(game.action_space.n)], tick_label=[str(l) for l in game.env.action_map])

    plt.xticks(fontsize=10, rotation=70)

    plt.xlabel('action', fontsize=16)
    plt.ylabel('q_value', fontsize=16)

    plt.tight_layout()

    fig.canvas.draw()
    array_rendered = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    array_rendered = array_rendered.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()
    # X = np.array(fig.canvas)

    return array_rendered
