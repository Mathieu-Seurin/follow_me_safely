import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import gym
from rl_agent.gpu_utils import TORCH_DEVICE

class FullyConnectedModel(nn.Module):
    def __init__(self, config, action_space, obs_space):
        super().__init__()

        self.output_size = action_space.n
        self.n_input = obs_space.shape[0]

        self.n_hidden_mlp = config["n_mlp_hidden"]

        self.hidden_layer = nn.Linear(self.n_input, self.n_hidden_mlp)
        self.out_layer = nn.Linear(self.n_hidden_mlp, self.output_size)

    def forward(self, x):

        x = F.relu(self.hidden_layer(x))
        x = self.out_layer(x)

        return x

class TextModel(nn.Module):
    def __init__(self, config, action_space, obs_space):

        super().__init__()

        self.n_input = obs_space.shape[0]
        self.n_output_size = action_space.n

        vocab_size = obs_space.vocab_size
        embedding_dim = config["word_embedding_size"]

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim,
                                      padding_idx=0)

        if config["rnn_type"].lower() == 'lstm':
            RNN = nn.LSTM
        elif config["rnn_type"].lower() == 'gru':
            RNN = nn.GRU
        else:
            raise NotImplementedError("Wrong RNN type, is {}".format(config["rnn_type"]))

        self.encoders = dict()

        self.encoders['obs'] = RNN(input_size=embedding_dim,
                                       num_layers=1,
                                       hidden_size=config["obs_rnn_size"],
                                       dropout=0,
                                       bidirectional=config["bidir"],
                                       batch_first=True)

        self.add_module('obs_encoder', self.encoders['obs'])

        self.encoders['description'] = RNN(input_size=embedding_dim,
                                         num_layers=1,
                                         hidden_size=config["description_rnn_size"],
                                         bidirectional=config["bidir"],
                                         dropout=0,
                                         batch_first=True)

        self.add_module('description_encoder', self.encoders['description'])


        self.encoders['inventory'] = RNN(input_size=embedding_dim,
                                             num_layers=1,
                                             hidden_size=config["inventory_rnn_size"],
                                             bidirectional=config["bidir"],
                                             dropout=0,
                                             batch_first=True)

        self.add_module('inventory_encoder', self.encoders['inventory'])


        in_fc = config["obs_rnn_size"] + config["description_rnn_size"] + config["inventory_rnn_size"]

        self.fc_hidden = nn.Linear(in_fc, out_features=config["fc_hidden"])
        self.fc_output = nn.Linear(config["fc_hidden"], out_features=self.n_output_size)

    def encode_sequences(self, batch_seq):

        output_seqs = []
        for key in batch_seq.keys():
            seq = torch.nn.utils.rnn.pack_padded_sequence(batch_seq[key]["seq"], batch_seq[key]["lengths"],
                                                          batch_first=True,
                                                          enforce_sorted=False)
            output, (hidden, cn) = self.encoders[key](seq)
            # seq = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            output_seqs.append(hidden[0])

        return output_seqs


    def forward(self, x):

        output = dict()
        for key in x.keys():
            output[key] = dict()
            output[key]["seq"] = self.embedding(x[key]["seq"])
            output[key]["lengths"] = x[key]["lengths"]

        encoded_seq_list = self.encode_sequences(output)
        encoded_seq_concat = torch.cat(encoded_seq_list, dim=1)

        x = F.relu(self.fc_hidden(encoded_seq_concat))
        x = self.fc_output(x)
        return x




class ConvModel(nn.Module):
    def __init__(self, config, action_space, obs_space, learn_feedback_classif=False):

        super().__init__()

        self.n_action = action_space.n
        self.output_size = self.n_action
        self.n_hidden_mlp = config["n_mlp_hidden"]

        if isinstance(obs_space, gym.spaces.Box):
            self.input_shape = obs_space.shape
        else:
            self.input_shape = obs_space.spaces["state"].shape

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
        last_activation_before_output = F.relu(self.hidden_layer(x))
        output = self.output_layer(last_activation_before_output)
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


class ConvACModel(nn.Module):
    def __init__(self):
        super().__init__()

class FCACModel(nn.Module):
    def __init__(self):
        super().__init__()