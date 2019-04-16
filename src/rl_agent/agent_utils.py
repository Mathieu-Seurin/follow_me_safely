import numpy as np
from collections import namedtuple

import torch
from rl_agent.gpu_utils import TORCH_DEVICE

import logging


# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'gave_feedback'))


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


def feedback_loss(qs, action, feedback, margin, regression_loss):
    """
    Compute the expert loss

    ab is the "bad" action
    m is a margin function

    Can be written as :
    Minimize    Q(s,ab) - min_a [ Q(s,a) - m(ab, a) ]
                                          m(ab,b) = margin if ab = a
                                                  = 0 else
    """

    # Keep qs where a feedback from environment was given.
    n_action = qs.size(1)
    qs_where_bad = qs[feedback != 0]
    action_where_bad = action[feedback != 0]

    # Q(s, ab) => action taken was bad (feedback from env)
    qs_a_where_bad = qs_where_bad.gather(1, action_where_bad.view(-1,1))

    #  =====  Compute l(a_b, a) =====
    action_mask = torch.arange(n_action).unsqueeze(0).to(TORCH_DEVICE) != action_where_bad
    # action_mask is the same size as qs. for every row, there is a 0 in column of action, 1 elsewhere
    # Exemple : action = [0, 1, 0] action_mask = [[0,1],[1,0],[0,1]]

    margin_malus = action_mask.float() * margin

    # Compute Q(s,a) - l(a_b, a)
    ref_qs = qs_where_bad.detach() # You optimize with respect to this ref_qs minus the margin, so you need to detach
    min_qs_minus_margin, _ = torch.min(ref_qs - margin_malus, dim=1)

    # Actual classification loss
    loss = regression_loss(min_qs_minus_margin, qs_a_where_bad) # Bring bad action down under margin
    return loss



if __name__ == "__main__":

    import torch

    qs = torch.arange(12).view(4,3).float()
    actions = torch.Tensor([0,2,1,0]).long()
    feedback = torch.Tensor([1,1,1,0])

    margin = 0.1
    regr_loss = torch.nn.functional.smooth_l1_loss

    print(feedback_loss(qs, actions, feedback, margin, regr_loss))

