from rl_agent.agent_utils import Transition, ReplayMemory, check_params_changed
import torch
import torch.nn.functional as F

from rl_agent.models import FullyConnectedModel, ConvModel
import numpy as np
from copy import copy, deepcopy

from rl_agent.gpu_utils import TORCH_DEVICE

class DQNAgent(object):

    def __init__(self, config, n_action, state_dim, discount_factor, biased_sampling):

        self.discount_factor = discount_factor
        self.n_action = n_action.n

        self.lr = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.batch_size = config["batch_size"]

        self.memory = ReplayMemory(config["memory_size"])

        self.update_every_n_ep = config["update_every_n_ep"]

        self.num_update = 0

        Model = FullyConnectedModel if config["model_type"] == "fc" else ConvModel

        self.policy_net = Model(config["model_params"], n_action=n_action, state_dim=state_dim).to(TORCH_DEVICE)
        self.target_net = Model(config["model_params"], n_action=n_action, state_dim=state_dim).to(TORCH_DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.target_net.eval()

        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if config["exploration_method"]["name"] == "eps_greedy":

            self.select_action = self._select_action_eps_greedy
            self.epsilon_init = config["exploration_method"]["begin_eps"]
            self.current_eps = copy(self.epsilon_init)
            self.expected_exploration_steps = config["exploration_method"]["expected_step_explo"]
            self.minimum_epsilon = config["exploration_method"]["epsilon_minimum"]
            self.n_step_eps = 0

            self.biased_sampling = biased_sampling
            self.action_proba = np.zeros(self.n_action)
            self.action_proba[1], self.action_proba[5], self.action_proba[9] = 1, 1, 1
            self.action_proba = self.action_proba * 14 + 1
            self.action_proba /= np.sum(self.action_proba)

        else:
            raise NotImplementedError("Boltzman explo not available ({})".format(config["exploration_method"]["name"]))

    def select_action_greedy(self, state):

        if np.random.random() > self.minimum_epsilon:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                qs = self.policy_net(state.to(TORCH_DEVICE))
                return qs.max(1)[1].view(1, 1).to('cpu')
        else:
            return torch.tensor([[np.random.randint(self.n_action)]], dtype=torch.long)

    def _select_action_eps_greedy(self, state):

        # Formula for eps decay :
        # exp(- 2.5 * iter / expected iter)  2.5 is set by hand, just to have a smooth decay until end
        # init = 100%     end = 5%
        # 0% : eps=1      10% : eps=0.77    50% : eps=0.28     80% : eps=0.13
        if self.policy_net.train:
            self.current_eps = max(self.minimum_epsilon, self.epsilon_init * np.exp(
                - 2.5 * self.n_step_eps / self.expected_exploration_steps))

        self.n_step_eps += 1

        if np.random.random() > self.current_eps:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                qs = self.policy_net(state.to(TORCH_DEVICE))
                return qs.max(1)[1].view(1, 1).to('cpu')
        else:
            if self.biased_sampling:
                act = np.random.choice(self.n_action, p=self.action_proba)
                return torch.tensor([[act]], dtype=torch.long)
            else:
                return torch.tensor([[np.random.randint(self.n_action)]], dtype=torch.long)


    def push(self, *args):
        self.memory.push(*args)

    def callback(self, epoch):

        if epoch % self.update_every_n_ep == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.num_update += 1

    def expert_loss(self):
        # todo : this
        pass

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=TORCH_DEVICE, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None]).to(TORCH_DEVICE)

        state_batch = torch.cat(batch.state).to(TORCH_DEVICE)
        action_batch = torch.cat(batch.action).to(TORCH_DEVICE)
        reward_batch = torch.cat(batch.reward).to(TORCH_DEVICE)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=TORCH_DEVICE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        expected_state_action_values = expected_state_action_values.unsqueeze(1)
        # Compute Huber loss
        assert state_action_values.size() == expected_state_action_values.size()
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        #old_params = deepcopy(self.policy_net.state_dict())

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        #check_params_changed(old_params, self.policy_net.state_dict())