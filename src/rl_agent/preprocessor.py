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
    """
    Batchify and convert text state to torch Tensor
    :param state:
    :return:
    """

    processed_state = dict()
    if isinstance(state, dict):
        for key in state.keys():
            processed_state[key] = dict()
            processed_state[key]["seq"] = torch.tensor(state[key]).unsqueeze(0).to(TORCH_DEVICE)
            processed_state[key]["lengths"] = torch.LongTensor(state[key].shape)
    else:
        for key in state[0].keys():
            processed_state[key] = dict()
            processed_state[key]["lengths"] = torch.LongTensor([x[key].shape[0] for x in state])
            processed_state[key]["seq"] = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(x[key]) for x in state],
                                                                          padding_value=0, batch_first=True).to(TORCH_DEVICE)
    return processed_state