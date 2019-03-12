import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy

from rl_agent.agent_utils import ReplayMemory, Transition, freeze_as_np_dict, compute_slow_params_update, ReplayMemoryRecurrent
from rl_agent.gpu_utils import use_cuda, FloatTensor, LongTensor, ByteTensor, Tensor

from rl_agent.models import FullyConnectedModel

import logging


class DQNAgent(object):
    def __init__(self, config, n_action, state_dim):

        if config["agent_type"] == "dqn":
            config = config['dqn_params']
            model = FullyConnectedModel(config=config,
                                        n_action=n_action,
                                        state_dim=state_dim)
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

        self.clamp_grad = config["clamp_grad"]

        self.memory = ReplayMemory(config["memory_size"])

        if config["exploration_method"] == "eps_greedy":
            self.forward = self.select_action_eps_greedy
            self.epsilon_init = config["exploration_method"]["begin_eps"]
            self.current_eps = self.current_eps.copy()
            self.expected_exploration_steps = config["exploration_method"]["expected_step_explo"]
            self.n_step_eps = 0

        elif config["exploration_method"] == "boltzmann":
            self.forward = self.select_action_boltzmann
        else:
            raise NotImplementedError("Wrong action selection method")

        self.lr = config['learning_rate']
        self.gamma = config['discount_factor']

        if config['optimizer'].lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(self.fast_model.parameters(), lr=self.lr)
        elif config['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(self.fast_model.parameters(), lr=self.lr)
        else:
            assert False, 'Optimizer not recognized'

        # logging.info('Model summary :')
        # logging.info(self.forward_model.forward)

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
        var_state['env_state'] = FloatTensor(state['env_state']).unsqueeze(0)
        return var_state

    def select_action_eps_greedy(self, state):

        if self.fast_model.train:
            epsilon = 0
        else:
            epsilon = self.current_eps

        plop = np.random.rand()
        if plop < epsilon:
            idx = np.random.randint(self.n_action)
        else:
            var_state = self.format_state(state)
            idx = self.fast_model(var_state).data.max(1)[1].cpu().numpy()[0]

        if self.fast_model.train :
            self.current_eps = self.epsilon_init * np.exp(-1. * self.n_step_eps / self.expected_exploration_steps)
            self.n_step_eps += 1

        assert False, "Test epsilon decay here"

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

        state = state_loc.unsqueeze(0)
        next_state = next_state_loc.unsqueeze(0)
        action = LongTensor([int(action)]).view((1, 1,))
        reward = FloatTensor([reward])

        self.memory.push(state, action, next_state, reward)

        if len(self.memory.memory) < self.batch_size:
            batch_size = len(self.memory.memory)
        else:
            batch_size = self.batch_size

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).type(Tensor).requires_grad_(True)

        action_batch = torch.cat(batch.action).type(LongTensor).requires_grad_(True)
        reward_batch = torch.cat(batch.reward).type(Tensor).requires_grad_(True)

        if len(state_batch.shape) == 3:
            state_batch = state_batch.unsqueeze(0)

        state_obj = dict()
        state_obj['env_state'] = state_batch

        state_action_values = self.fast_model(state_obj).gather(1, action_batch)

        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).type(Tensor)
        # check if requires grad

        non_final_next_states_obj = dict()
        non_final_next_states_obj['env_state'] = non_final_next_states

        if len(non_final_next_states.shape) == 3:
            non_final_next_states = non_final_next_states.unsqueeze(0)

        next_state_values = torch.zeros(batch_size).type(Tensor).requires_grad_(True)
        next_state_values[non_final_mask] = self.ref_model(non_final_next_states_obj).max(1)[0]

        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        loss_q_learning = F.mse_loss(state_action_values, expected_state_action_values)
        loss = loss_q_learning

        #old_params = freeze_as_np_dict(self.forward_model.state_dict())
        self.fast_model.optimizer.zero_grad()
        loss.backward()

        if self.clamp_grad:
            for name, param in self.fast_model.named_parameters():
                #logging.debug(param.grad.data.sum())
                param.grad.data.clamp_(-1., 1.)

        self.fast_model.optimizer.step()

        # Update slowly ref model towards fast model, to stabilize training.
        if self.soft_update:
            self.ref_model.load_state_dict(compute_slow_params_update(self.ref_model, self.fast_model, self.tau))

        # new_params = freeze_as_np_dict(self.fast_model.state_dict())
        # check_params_changed(old_params, new_params)

        return loss.detach()[0]

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

        raise NotImplementedError("Hophophop, touche pas a ca petit con")
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

