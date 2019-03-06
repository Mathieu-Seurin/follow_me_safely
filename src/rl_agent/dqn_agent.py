import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
import random
from .agent_utils import ReplayMemory, Transition, freeze_as_np_dict, compute_slow_params_update, ReplayMemoryRecurrent
from .dqn_models import DQN
import pickle

from rl_agent.FiLM_agent import FilmedNetText
from .film_utils import FilmedNet

import logging
import os

from .gpu_utils import use_cuda, FloatTensor, LongTensor, ByteTensor, Tensor
from image_text_utils import TextToIds

import os

class DQNAgent(object):
    def __init__(self, config, n_action, state_dim, is_multi_objective, objective_type):

        self.objective_is_text = 'text' in objective_type

        if config['agent_type'] == "resnet_dqn":
            config = config['resnet_dqn_params']

            if self.objective_is_text:
                model = FilmedNetText(config=config,
                                      n_actions=n_action,
                                      state_dim=state_dim,
                                      is_multi_objective=is_multi_objective)
                self.text_to_vect = TextToIds()

            else:
                model = FilmedNet(config=config,
                                  n_actions=n_action,
                                  state_dim=state_dim,
                                  is_multi_objective=is_multi_objective)

        elif config["agent_type"] == "dqn":
            config = config['dqn_params']
            model = DQN(config=config,
                        n_action=n_action,
                        state_dim=state_dim,
                        is_multi_objective=is_multi_objective)
        else:
            raise NotImplementedError("model_type not recognized")

        self.fast_model = model
        self.ref_model = deepcopy(self.fast_model)
        if use_cuda:
            self.fast_model.cuda()
            self.ref_model.cuda()
        self.n_action = n_action

        self.tau = config['tau']
        self.batch_size = config["batch_size"]
        self.soft_update = config["soft_update"]

        self.memory = ReplayMemory(config["memory_size"])

        if config["exploration_method"] == "eps_greedy":
            self.forward = self.select_action_eps_greedy
        elif config["exploration_method"] == "boltzmann":
            self.forward = self.select_action_boltzmann
        else:
            raise NotImplementedError("Wrong action selection method")

        self.discount_factor = self.fast_model.discount_factor

        # logging.info('Model summary :')
        # logging.info(self.forward_model.forward)

    def apply_config(self, config):
        pass

    def callback(self, epoch):
        if not self.soft_update and epoch % int(1/self.tau) == 0:
            self.ref_model.load_state_dict(self.fast_model.state_dict())

    def train(self):
        self.fast_model.train()
        self.ref_model.train()

    def eval(self):
        self.fast_model.eval()
        self.ref_model.eval()

    def format_state(self, state):

        # state is {"env_state" : img, "objective": img/text}
        var_state = dict()
        var_state['env_state'] = Variable(FloatTensor(state['env_state']).unsqueeze(0), volatile=True)

        if self.objective_is_text:
            objective = self.text_to_vect.sentence_to_matrix(state['objective'])
            objective = LongTensor(objective)  # Long expected for int input
        else:
            objective = FloatTensor(state['objective'])  # for image, use Float

        var_state['objective'] = Variable(objective.unsqueeze(0), volatile=True)
        return var_state

    def select_action_eps_greedy(self, state, epsilon=0.1):

        plop = np.random.rand()
        if plop < epsilon:
            idx = np.random.randint(self.n_action)
        else:
            var_state = self.format_state(state)
            idx = self.fast_model(var_state).data.max(1)[1].cpu().numpy()[0]
        return idx

    def select_action_boltzmann(self, state, epsilon=0.1):

        var_state = self.format_state(state=state)
        score = F.softmax(self.fast_model(var_state), dim=1)
        chosen_action = np.random.choice([i for i in range(self.n_action)], p=score.data.cpu().numpy()[0,:])

        return chosen_action


    def optimize(self, state, action, next_state, reward):

        # state is {"env_state" : img, "objective": img}
        state_loc = FloatTensor(state['env_state'])
        next_state_loc = FloatTensor(next_state['env_state'])

        if self.objective_is_text:
            objective = self.text_to_vect.sentence_to_matrix(state['objective'])
        else:
            objective = state['objective']

        objective = FloatTensor(objective).unsqueeze(0)

        # if self.concatenate_objective:
        #     state_loc = torch.cat((state_loc, FloatTensor(state['objective'])))
        #     next_state_loc = torch.cat((next_state_loc, FloatTensor(next_state['objective'])))

        state = state_loc.unsqueeze(0)
        next_state = next_state_loc.unsqueeze(0)
        action = LongTensor([int(action)]).view((1, 1,))
        reward = FloatTensor([reward])

        self.memory.push(state, action, next_state, reward, objective)

        if len(self.memory.memory) < self.batch_size:
            batch_size = len(self.memory.memory)
        else:
            batch_size = self.batch_size

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = Variable(torch.cat(batch.state).type(Tensor))

        if self.objective_is_text:
            # Batchify : all sequence must have the same size, so you pad the dialogue with token
            objective_batch = self.text_to_vect.pad_batch_sentence(batch.objective)
            objective_batch = Variable(torch.cat(objective_batch).type(LongTensor))

        else:
            objective_batch = Variable(torch.cat(batch.objective).type(Tensor))

        action_batch = Variable(torch.cat(batch.action).type(LongTensor))
        reward_batch = Variable(torch.cat(batch.reward).type(Tensor))

        if len(state_batch.shape) == 3:
            state_batch = state_batch.unsqueeze(0)
            objective_batch.unsqueeze(0)

        state_obj = dict()
        state_obj['env_state'] = state_batch
        state_obj['objective'] = objective_batch

        state_action_values = self.fast_model(state_obj).gather(1, action_batch)

        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]).type(Tensor), volatile=True)

        if self.objective_is_text:
            objective_batch = [batch.objective[i] for i, s in enumerate(batch.next_state) if s is not None]
            objective_batch = self.text_to_vect.pad_batch_sentence(objective_batch)
            non_final_state_corresponding_objective = torch.cat(objective_batch).type(LongTensor)
        else:
            non_final_state_corresponding_objective = torch.cat([batch.objective[i] for i, s in enumerate(batch.next_state) if s is not None]).type(Tensor)

        non_final_state_corresponding_objective = Variable(non_final_state_corresponding_objective, volatile=True)

        non_final_next_states_obj = dict()
        non_final_next_states_obj['env_state'] = non_final_next_states
        non_final_next_states_obj['objective'] = non_final_state_corresponding_objective

        if len(non_final_next_states.shape) == 3:
            non_final_next_states = non_final_next_states.unsqueeze(0)

        next_state_values = Variable(torch.zeros(batch_size).type(Tensor))
        next_state_values[non_final_mask] = self.ref_model(non_final_next_states_obj).max(1)[0]

        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        loss_q_learning = F.mse_loss(state_action_values, expected_state_action_values)

        if self.fast_model.use_predict_forward_model:
            forward_loss_weight = self.fast_model.forward_loss_weight
            loss_forward_model = self.loss_forward_model(state=state_batch, next_state=non_final_next_states, action=action_batch)

            # Weighting parameters to balance between forward model and q-learning
            loss = (1-forward_loss_weight)*loss_q_learning + forward_loss_weight*loss_forward_model
        else:
            loss = loss_q_learning

        #old_params = freeze_as_np_dict(self.forward_model.state_dict())
        self.fast_model.optimizer.zero_grad()
        loss.backward()
        for name, param in self.fast_model.named_parameters():
            #logging.debug(param.grad.data.sum())
            param.grad.data.clamp_(-1., 1.)
        self.fast_model.optimizer.step()

        # Update slowly ref model towards fast model, to stabilize training.
        if self.soft_update:
            self.ref_model.load_state_dict(compute_slow_params_update(self.ref_model, self.fast_model, self.tau))

        new_params = freeze_as_np_dict(self.fast_model.state_dict())
        #check_params_changed(old_params, new_params)


        return loss.data[0]

    def loss_forward_model(self, state, next_state, action):

        state_predicted_encoded = self.fast_model.compute_forward_dynamics(state, actions=action)
        next_state_encoded = self.fast_model.compute_conv(x=next_state, ignore_side_input=True).view(state.size(0), -1)

        loss = F.mse_loss(input=state_predicted_encoded, target=next_state_encoded)
        return loss

    def save_state(self):
        # Store the whole agent state somewhere
        state_dict = self.fast_model.state_dict()
        memory = deepcopy(self.memory)
        return state_dict, memory

    def load_state(self, state_dict, memory):
        self.fast_model.load_state_dict(state_dict)
        self.memory = deepcopy(memory)


class RDQNAgent(DQNAgent):

    def __init__(self, config, n_action, state_dim, is_multi_objective, objective_type):

        super(RDQNAgent, self).__init__(config, n_action, state_dim, is_multi_objective, objective_type)
        self.memory = ReplayMemoryRecurrent(config["memory_size"])

    def init_ht(self, batch_size):
        return Variable(torch.ones(batch_size, self.fast_model.recurrent_size))

    def callback(self, epoch):
        super(RDQNAgent, self).callback(epoch)
        # todo reset ht

        self.current_ht = self.init_ht(batch_size=1)

    def optimize_one_time_step_batch(self, batch, batch_size):
        """
        DRQN needs to be optimized for several timestep, updating at every time step h_t to train the recurrent part of
        the network.
        :return:
        """
        state_batch = Variable(torch.cat(batch.state).type(Tensor))

        if self.objective_is_text:
            # Batchify : all sequence must have the same size, so you pad the dialogue with token
            objective_batch = self.text_to_vect.pad_batch_sentence(batch.objective)
            objective_batch = Variable(torch.cat(objective_batch).type(LongTensor))

        else:
            objective_batch = Variable(torch.cat(batch.objective).type(Tensor))

        action_batch = Variable(torch.cat(batch.action).type(LongTensor))
        reward_batch = Variable(torch.cat(batch.reward).type(Tensor))

        if len(state_batch.shape) == 3:
            state_batch = state_batch.unsqueeze(0)
            objective_batch.unsqueeze(0)

        state_obj = dict()
        state_obj['env_state'] = state_batch
        state_obj['objective'] = objective_batch

        state_action_values = self.fast_model(state_obj).gather(1, action_batch)

        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]).type(Tensor),
                                         volatile=True)

        if self.objective_is_text:
            objective_batch = [batch.objective[i] for i, s in enumerate(batch.next_state) if s is not None]
            objective_batch = self.text_to_vect.pad_batch_sentence(objective_batch)
            non_final_state_corresponding_objective = torch.cat(objective_batch).type(LongTensor)
        else:
            non_final_state_corresponding_objective = torch.cat(
                [batch.objective[i] for i, s in enumerate(batch.next_state) if s is not None]).type(Tensor)

        non_final_state_corresponding_objective = Variable(non_final_state_corresponding_objective, volatile=True)

        non_final_next_states_obj = dict()
        non_final_next_states_obj['env_state'] = non_final_next_states
        non_final_next_states_obj['objective'] = non_final_state_corresponding_objective

        if len(non_final_next_states.shape) == 3:
            non_final_next_states = non_final_next_states.unsqueeze(0)

        next_state_values = Variable(torch.zeros(batch_size).type(Tensor))
        next_state_values[non_final_mask] = self.ref_model(non_final_next_states_obj).max(1)[0]

        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        loss = F.mse_loss(state_action_values, expected_state_action_values)
        return loss

    def optimize(self, state, action, next_state, reward):

        # state is {"env_state" : img, "objective": img}
        state_loc = FloatTensor(state['env_state'])
        next_state_loc = FloatTensor(next_state['env_state'])

        if self.objective_is_text:
            objective = self.text_to_vect.sentence_to_matrix(state['objective'])
        else:
            objective = state['objective']

        objective = FloatTensor(objective).unsqueeze(0)

        # if self.concatenate_objective:
        #     state_loc = torch.cat((state_loc, FloatTensor(state['objective'])))
        #     next_state_loc = torch.cat((next_state_loc, FloatTensor(next_state['objective'])))

        state = state_loc.unsqueeze(0)
        next_state = next_state_loc.unsqueeze(0)
        action = LongTensor([int(action)]).view((1, 1,))
        reward = FloatTensor([reward])

        self.memory.push(state, action, next_state, reward, objective)

        transitions, batch_size = self.memory.sample(self.batch_size)
        if not batch_size:
            # not enough sample at the moment
            return


        loss = 0
        self.fast_model.optimize_mode(optimize=True, batch_size=batch_size)
        self.ref_model.optimize_mode(optimize=True, batch_size=batch_size)

        for timestep_batch in transitions:

            batch = Transition(*zip(*timestep_batch))
            current_loss = self.optimize_one_time_step_batch(batch, batch_size)
            loss += current_loss

        #old_params = freeze_as_np_dict(self.forward_model.state_dict())
        self.forward_model.optimizer.zero_grad()
        loss.backward()
        for param in self.forward_model.parameters():
            logging.debug(param.grad.data.sum())
            param.grad.data.clamp_(-1., 1.)
        self.forward_model.optimizer.step()

        # Update slowly ref model towards fast model, to stabilize training.
        if self.soft_update:
            self.ref_model.load_state_dict(compute_slow_params_update(self.ref_model, self.forward_model, self.tau))

        new_params = freeze_as_np_dict(self.forward_model.state_dict())
        #check_params_changed(old_params, new_params)

        self.forward_model.optimize_mode(optimize=False)
        self.ref_model.optimize_mode(optimize=False)

        return loss.data[0]

