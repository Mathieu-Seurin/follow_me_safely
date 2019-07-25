import numpy as np

import torch
from torch import nn
from rl_agent.gpu_utils import TORCH_DEVICE

def ImagePreprocessor(state):

    if type(state) in (list, tuple):
        state = np.stack(state, axis=0)
        state = torch.FloatTensor(state)

    elif type(state) is np.ndarray:
        state = torch.FloatTensor(state).unsqueeze(0)

    else:
        raise NotImplementedError("State type is unknown")

    return state.to(TORCH_DEVICE)


def TextPreprocessor(state):
    pass