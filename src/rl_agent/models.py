import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import logging

# GPU compatibility setup
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor



class FullyConnectedModel(nn.Module):
    def __init__(self, config, n_action, state_dim):
        super(FullyConnected, self).__init__()
        self.output_size = n_action




class DQN(nn.Module):

    def __init__(self, config, n_action, state_dim, is_multi_objective):
        super(DQN, self).__init__()
        self.output_size = n_action
        self.n_channels_total = state_dim['concatenated'][0]
        self.input_resolution = list(state_dim['concatenated'][1:])


        conv_layers = nn.ModuleList()
        dense_layers = nn.ModuleList()

        self.conv_shapes = config['conv_shapes']
        self.dense_shapes = config['dense_shapes'] + [self.output_size]
        self.use_batch_norm = config['use_batch_norm']



        self.is_multi_objective = is_multi_objective

        # At least 1 conv, then dense head
        for idx, shape in enumerate(self.conv_shapes):
            if idx == 0:
                conv_layers.append(nn.Conv2d(self.n_channels_total, shape, kernel_size=3, stride=2))
            else:
                conv_layers.append(nn.Conv2d(intermediate_channel, shape, kernel_size=5, stride=2))
            conv_layers.append(nn.ReLU())
            if self.use_batch_norm:
                conv_layers.append(nn.BatchNorm2d(shape))
            intermediate_channel = shape
        self.conv_layers = conv_layers

        # Infer shape after flattening
        shape_after_flatten = self._get_conv_output_size([self.n_channels_total, ] + self.input_resolution)

        for idx, shape in enumerate(self.dense_shapes):
            dense_layers.append(nn.Linear(shape_after_flatten, shape))
            if idx < len(self.dense_shapes)-1:
                dense_layers.append(nn.ReLU())
                if self.use_batch_norm:
                    logging.info('BatchNorm in dense head gives issues')
                    # dense_layers.append(nn.BatchNorm1d(shape))
            shape_after_flatten = shape
        self.dense_layers = dense_layers

        if config['optimizer'].lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
        elif config['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        else:
            assert False, 'Optimizer not recognized'


    def _get_conv_output_size(self, shape):
        bs = 1
        inpt = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_conv(inpt)
        total_size = output_feat.data.view(bs, -1).size(1)
        return total_size

    def _forward_conv(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x

    def _forward_dense(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = torch.cat((x['env_state'], x['objective']), dim=1)
        # print(x.data.cpu().numpy().shape)
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        return self._forward_dense(x)


class SoftmaxDQN(nn.Module):

    def __init__(self, config, n_action, state_dim, is_multi_objective):
        super(SoftmaxDQN, self).__init__()
        self.output_size = n_action

        conv_layers = nn.ModuleList()
        dense_layers = nn.ModuleList()

        self.input_resolution = config['input_resolution']
        self.n_channels = config['n_channels']
        self.conv_shapes = config['conv_shapes']
        self.dense_shapes = config['dense_shapes'] + [self.output_size]
        self.use_batch_norm = config['use_batch_norm']
        self.lr = config['learning_rate']
        self.use_first_block = config['use_first_block']

        prev_shape = self.n_channels

        if self.use_first_block:
            self.cnn1 = nn.Conv2d(in_channels=self.n_channels, out_channels=16, kernel_size=5,stride=1,padding=2)
            self.relu1=nn.ELU()
            nn.init.xavier_uniform(self.cnn1.weight)
            self.maxpool1=nn.MaxPool2d(kernel_size=2)
            self.cnn2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2)
            self.relu2=nn.ELU()
            self.dropout=nn.Dropout(0.2)
            nn.init.xavier_uniform(self.cnn2.weight)
            self.maxpool2=nn.MaxPool2d(kernel_size=2)

            prev_shape = self._forward_first_block(Variable(torch.rand(1,self.n_channels,28,28))).shape[1]
            print(prev_shape)


        # At least 1 conv, then dense head
        for idx, shape in enumerate(self.conv_shapes):
            conv_layers.append(nn.Conv2d(prev_shape, shape, kernel_size=5, stride=2))
            conv_layers.append(nn.ReLU())
            if self.use_batch_norm:
                conv_layers.append(nn.BatchNorm2d(shape))
            prev_shape = shape

        self.conv_layers = conv_layers

        # Infer shape after flattening
        tmp = self._get_conv_output_size([self.n_channels,] + self.input_resolution)

        for idx, shape in enumerate(self.dense_shapes):
            dense_layers.append(nn.Linear(tmp, shape))
            if idx < len(self.dense_shapes)-1:
                dense_layers.append(nn.ReLU())
                if self.use_batch_norm:
                    # dense_layers.append(nn.BatchNorm1d(shape))
                    print('BatchNorm in dense layers not working')
            else:
                dense_layers.append(nn.Softmax(dim=1))
            tmp = shape
        self.dense_layers = dense_layers

        logging.info('Model summary :')
        for l in self.conv_layers:
            logging.info(l)
        for l in self.dense_layers:
            logging.info(l)




    def _get_conv_output_size(self, shape):
        bs = 1
        inpt = Variable(torch.rand(bs, *shape))
        inpt = self._forward_first_block(inpt)
        output_feat = self._forward_conv(inpt)
        total_size = output_feat.data.view(bs, -1).size(1)
        return total_size

    def _forward_first_block(self, x):
        if not self.use_first_block:
            return x

        out=self.cnn1(x)
        out=self.relu1(out)
        out=self.maxpool1(out)
        out=self.dropout(out)
        out=self.cnn2(out)
        out=self.relu2(out)
        out=self.maxpool2(out)
        out=self.dropout(out)
        return out

    def _forward_conv(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x

    def _forward_dense(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = torch.cat((x['env_state'], x['objective']), dim=1)
        x = self._forward_first_block(x)
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        return self._forward_dense(x)