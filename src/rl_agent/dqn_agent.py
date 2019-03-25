import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy

from rl_agent.agent_utils import ReplayMemory, PrioritizedMemory, Transition, \
    freeze_as_np_dict, compute_slow_params_update, ReplayMemoryRecurrent

from rl_agent.gpu_utils import TORCH_DEVICE

from rl_agent.models import FullyConnectedModel, ConvModel

import logging
from copy import copy


class FullRandomAgent(object):
    def __init__(self, config, n_action, state_dim):
        self.n_action = n_action.n
    def forward(self, *args, **kwargs):
        return np.random.randint(0, self.n_action, size=None)
    def optimize(self, *args, **kwargs):
        pass
    def callback(self, epoch):
        pass

class DQNAgent(object):
    def __init__(self, config, n_action, state_dim, discount_factor):

        if config["model_type"] == "fc":
            model = FullyConnectedModel(config=config["model_params"],
                                        n_action=n_action,
                                        state_dim=state_dim)
        else:
            model = ConvModel(config=config["model_params"],
                                        n_action=n_action,
                                        state_dim=state_dim)


        self.fast_model = model.to(TORCH_DEVICE)
        self.ref_model = deepcopy(self.fast_model).to(TORCH_DEVICE).eval()

        self.n_action = n_action.n

        self.tau = config['tau']
        self.batch_size = config["batch_size"]
        self.soft_update = config["soft_update"]

        self.clamp_grad = config["clamp_grad"]

        if config["memory"] == "prioritized":
            self.memory = PrioritizedMemory(config["memory_size"])
        else:
            self.memory = ReplayMemory(config["memory_size"])

        if config["exploration_method"]["name"] == "eps_greedy":
            self.forward = self._select_action_eps_greedy
            self.epsilon_init = config["exploration_method"]["begin_eps"]
            self.current_eps = copy(self.epsilon_init)
            self.expected_exploration_steps = config["exploration_method"]["expected_step_explo"]
            self.minimum_epsilon = config["exploration_method"]["epsilon_minimum"]
            self.n_step_eps = 0

        else:
            raise NotImplementedError("Wrong action selection method, chosen : {}".format(
                config["exploration_method"]["name"]))

        self.lr = config['learning_rate']
        self.discount_factor = discount_factor

        if config['optimizer'].lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(self.fast_model.parameters(), lr=self.lr)
        elif config['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(self.fast_model.parameters(), lr=self.lr)
        else:
            assert False, 'Optimizer not recognized'

        # logging.info('Model summary :')
        # logging.info(self.forward_model.forward)

        self.num_update = 0

    def callback(self, epoch):
        if not self.soft_update and epoch % int(1/self.tau) == 0:
            self.ref_model.load_state_dict(self.fast_model.state_dict())
            self.num_update += 1

    def train(self):
        self.fast_model.train()
        self.ref_model.train()

    def eval(self):
        self.fast_model.eval()
        self.ref_model.eval()

    def format_state(self, state):

        # state is {"env_state" : img, "objective": img/text}
        var_state = dict()

        env_state = state['image'] if isinstance(state, dict) else state

        var_state['env_state'] = torch.FloatTensor(env_state).unsqueeze(0).to(TORCH_DEVICE)
        return var_state

    def _select_action_eps_greedy(self, state):

        if self.fast_model.train:
            epsilon = self.minimum_epsilon
        else:
            epsilon = self.current_eps

        random_number = np.random.rand()
        if random_number < epsilon:
            action = np.random.randint(self.n_action)
        else:
            var_state = self.format_state(state)
            with torch.no_grad():
                action = self.fast_model(var_state).data.max(1)[1].cpu().numpy()[0]

        # Formula for eps decay :
        # exp(- 2.5 * iter / expected iter)  2.5 is set by hand, just to have a smooth decay until end
        # init = 100%     end = 5%
        # 0% : eps=1      10% : eps=0.77    50% : eps=0.28     80% : eps=0.13
        if self.fast_model.train :
            self.current_eps = max(self.minimum_epsilon, self.epsilon_init * np.exp(- 2.5 * self.n_step_eps / self.expected_exploration_steps))
            self.n_step_eps += 1

        return action

    def push(self, state, action, next_state, reward):
        # with torch.no_grad():
        #
        #     current_est = self.fast_model(self.format_state(state)).detach()[0][action] # [0] is for batch_dimension
        #
        #     if next_state != None:
        #         next_state_val = self.ref_model(self.format_state(next_state)).detach().max()
        #         target_val = reward + self.discount_factor * next_state_val
        #     else:
        #         target_val = reward
        #
        #     error = abs(current_est - target_val)

        state = state['image'] if isinstance(state, dict) else state
        next_state = next_state['image'] if isinstance(next_state, dict) else next_state

        # state is {"env_state" : img, "objective": img}
        state = torch.FloatTensor(np.expand_dims(state, axis=0))

        if isinstance(next_state, np.ndarray) :
            next_state = torch.FloatTensor(np.expand_dims(next_state, axis=0))

        action = torch.LongTensor(np.expand_dims(np.array([int(action)]), axis=0))
        reward = torch.FloatTensor(np.array([reward]))

        error = 0
        self.memory.push(state, action, next_state, reward, error)

    def optimize(self):
        #  Optimize with respect to replay buffer
        # =======================================

        if len(self.memory) < self.batch_size:
            return
        else:
            batch_size = self.batch_size

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(TORCH_DEVICE)

        action_batch = torch.cat(batch.action).to(TORCH_DEVICE)
        reward_batch = torch.cat(batch.reward).to(TORCH_DEVICE)

        if len(state_batch.shape) == 3:
            state_batch = state_batch.unsqueeze(0)

        state_obj = dict()
        state_obj['env_state'] = state_batch

        state_action_values = self.fast_model(state_obj).gather(1, action_batch)

        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state))).to(TORCH_DEVICE)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(TORCH_DEVICE)
        # check if requires grad

        non_final_next_states_obj = dict()
        non_final_next_states_obj['env_state'] = non_final_next_states

        if len(non_final_next_states.shape) == 3:
            non_final_next_states = non_final_next_states.unsqueeze(0)

        # Don't rembember gradients when computing the Q-value reference.
        next_state_values = torch.zeros(batch_size).to(TORCH_DEVICE)
        next_state_values[non_final_mask] = self.ref_model(non_final_next_states_obj).max(1)[0]
        next_state_values.detach()

        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        loss_q_learning = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        loss = loss_q_learning

        #old_params = freeze_as_np_dict(self.forward_model.state_dict())
        self.optimizer.zero_grad()
        loss.backward()

        if self.clamp_grad:
            for name, param in self.fast_model.named_parameters():
                #logging.debug(param.grad.data.sum())
                param.grad.data.clamp_(-1., 1.)

        self.optimizer.step()

        # Update slowly ref model towards fast model, to stabilize training.
        if self.soft_update:
            self.ref_model.load_state_dict(compute_slow_params_update(self.ref_model, self.fast_model, self.tau))

        # new_params = freeze_as_np_dict(self.fast_model.state_dict())
        # check_params_changed(old_params, new_params)

        return loss.detach().item()

    def save_state(self):
        # Store the whole agent state somewhere
        state_dict = self.fast_model.state_dict()
        memory = deepcopy(self.memory)
        return state_dict, memory

    def load_state(self, state_dict, memory):
        self.fast_model.load_state_dict(state_dict)
        self.memory = deepcopy(memory)