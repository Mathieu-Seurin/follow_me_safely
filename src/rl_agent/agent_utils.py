import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import random
import logging

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Flatten(nn.Module):
  def forward(self, x):
    N, C, H, W = x.size() # read in N, C, H, W
    return x.view(N, -1)

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward, error=0):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(state, action, next_state, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = np.min([batch_size, len(self.memory)])
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedMemory(object):

    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity

        self.memory = []
        self.errors = []
        self.indexes = []

        self.position = 0
        self.alpha = alpha

    def push(self, state, action, next_state, reward, error=0):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(state, action, next_state, reward)
        self.errors[self.position] = np.max(self.errors)

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = np.min([batch_size, len(self.memory)])

        #indexes = np.random.(0, len(self), batch_size, p=)




        return random.sample(self.memory, batch_size)

    def update(self, indexes):
        pass

    def __len__(self):
        return len(self.memory)


class ReplayMemoryRecurrent(object):

    def __init__(self, capacity, max_seq_length=8):

        self.capacity = capacity
        self.max_seq_length = max_seq_length
        self.memory = [[]]
        self.position = 0

    def push(self, *args):
        """
        Saves a transition in the replay buffer
        We deal with sequences so : 1 case in the memory buffer => a sequence until episode's end.
        """
        # To avoid storing everything on gpu
        args = [arg.cpu() for arg in args]
        self.memory[self.position].append(Transition(*args))

    def sample(self, batch_size):

        # Select randomly a sequence length
        min_seq_length = np.random.randint(4, self.max_seq_length)
        max_batch_size = batch_size//min_seq_length

        trajectories = []
        selected = set()

        check_inf_loop = 0

        while len(trajectories) != max_batch_size:
            check_inf_loop += 1
            if check_inf_loop > 100:
                break

            id_mem = np.random.randint(len(self.memory))

            # Don't sample the same trajectorie twice.
            if id_mem in selected:
                continue
            selected.add(id_mem)
            random_traj = self.memory[id_mem]

            if len(random_traj) < min_seq_length:
                continue
            elif len(random_traj) == min_seq_length:
                trajectories.append(random_traj)
            else:
                trajectories.append(random_traj[len(random_traj)-min_seq_length:])

                # limit_begin_id = len(random_traj) - min_seq_length + 1 # to include sup
                # random_begin = np.random.randint(0,limit_begin_id)
                # random_traj_cut = random_traj[random_begin:random_begin+min_seq_length]
                # trajectories.append(random_traj_cut)


        # trajectories are : [[ step 1, step 2, step 3], [step1, step2, step3]] etc..
        # we want : [[ step1, step1], [step2, step2]]
        batch_size = len(trajectories)
        return zip(*trajectories), batch_size

    def end_of_ep(self):
        """
        A episode ends
        """
        self.position = (self.position + 1) % self.capacity

        if len(self.memory) < self.capacity:
            self.memory.append([])
        else:
            self.memory[self.position] = []


    def __len__(self):
        return len(self.memory)


def freeze_as_np_dict(tensor_dict):
    out = {}
    for key in tensor_dict.keys():
        out[key] = tensor_dict[key].cpu().clone().numpy()
    return out

def check_params_changed(dict1, dict2):
    "Takes two parameters dict (key:torchtensor) and prints a warning if identical"
    for key in dict1.keys():
        if key.split('.')[-1] in ['running_mean', 'running_var']:
            # No message if this is for BatchNorm
            continue
        tmp1 = dict1[key]
        tmp2 = dict2[key]
        if np.max(np.abs(tmp1 - tmp2))==0:
            logging.debug('No change in params {}'.format(key))

def compute_slow_params_update(slow_params, fast_params, tau):

    slow_params_dict = slow_params.state_dict()
    fast_params_dict = fast_params.state_dict()

    for module_key in slow_params_dict.keys() :
        slow_params_dict[module_key] += tau*(fast_params_dict[module_key] - slow_params_dict[module_key])

    return slow_params_dict