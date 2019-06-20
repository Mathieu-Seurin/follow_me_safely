from rl_agent.agent_utils import Transition, ReplayMemory, ProportionReplayMemory, \
    feedback_bad_to_min_when_max, consistency_loss_dqfd, feedback_bad_to_percent_max, feedback_bad_to_min, feedback_frontier_margin

import torch
import torch.nn.functional as F

from rl_agent.models import FullyConnectedModel, ConvModel
import numpy as np
from copy import copy, deepcopy

from rl_agent.gpu_utils import TORCH_DEVICE

import tensorboardX

from sklearn.metrics import f1_score, accuracy_score

class DQNAgent(object):

    def __init__(self, config, n_action, state_dim, discount_factor, writer=None, log_stats_every=1e4):

        self.save_config = config
        self.log_stats_every = log_stats_every

        self.q_loss_logger = []
        self.feedback_loss_logger = []
        self.percent_feedback_logger = []
        self.supervised_score_logger = []
        self.random_score_logger = []

        self.discount_factor = discount_factor
        self.n_action = n_action.n

        self.lr = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.batch_size = config["batch_size"]
        self.clip_grad = config["clip_grad"]
        self.q_loss_weight = config["q_loss_weight"]

        if config["feedback_percentage_in_buffer"] == 0: # no proportionnal, use classical replay buffer
            self.memory = ReplayMemory(config["memory_size"])
        else:
            self.memory = ProportionReplayMemory(proportion=config["feedback_percentage_in_buffer"],
                                                 capacity=config["memory_size"])

        self.update_every_n_ep = config["update_every_n_ep"]

        self.num_update_target = 0
        self.num_optim_dqn = 0

        # ============ LOSSES : Bellman, feedback, regularization ========
        # ================================================================
        if config["regression_loss_func"] == "l2":
            self.regression_loss = F.mse_loss
        elif config["regression_loss_func"] == "l1":
            self.regression_loss = F.smooth_l1_loss
        else:
            raise NotImplementedError("Wrong regression loss func, is {}".format(config["regression_loss_func"]))

        self.use_nudging_loss = config["nudging_loss_weight"] > 0
        if self.use_nudging_loss:

            self.nudging_loss_weight = config["nudging_loss_weight"]
            self.nudging_loss_margin = config["nudging_margin"]

            loss_type = config["nudging_type"]
            if loss_type == "max":
                self.nudging_loss = feedback_bad_to_min_when_max
            elif loss_type == "min":
                self.nudging_loss = feedback_bad_to_min
            elif loss_type == "percent":
                self.nudging_loss = feedback_bad_to_percent_max
            elif loss_type == "frontier":
                self.nudging_loss = feedback_frontier_margin
            else:
                raise NotImplementedError("Wrong nudging loss, {}".format(loss_type))

        self.use_supervised_loss = config["supervised_loss_weight"] > 0
        if self.use_supervised_loss:
            self.supervised_loss_weight = config["supervised_loss_weight"]

        # Temporal Consistency loss : see https://arxiv.org/pdf/1805.11593.pdf
        # To avoid over generalization
        self.use_consistency_loss_dfqd = config["consistency_loss_weight"] > 0
        if self.use_consistency_loss_dfqd:
            self.consistency_loss = consistency_loss_dqfd
            self.consistency_loss_weight = config["consistency_loss_weight"]


        # ========== Model ===============

        Model = FullyConnectedModel if config["model_type"] == "fc" else ConvModel

        self.policy_net = Model(config["model_params"], n_action=n_action, state_dim=state_dim,
                                learn_feedback_classif=self.use_supervised_loss).to(TORCH_DEVICE)
        self.target_net = Model(config["model_params"], n_action=n_action, state_dim=state_dim,
                                learn_feedback_classif=self.use_supervised_loss).to(TORCH_DEVICE)
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

            self.biased_sampling = config["biased_sampling"]
            if self.biased_sampling:
                self.action_proba = np.zeros(self.n_action)
                self.action_proba[1], self.action_proba[5], self.action_proba[9] = 1, 1, 1
                self.action_proba = self.action_proba * 14 + 1
                self.action_proba /= np.sum(self.action_proba)
            else:
                self.action_proba = np.ones(self.n_action)
                self.action_proba /= np.sum(self.action_proba)

        else:
            raise NotImplementedError("Boltzman explo not available ({})".format(config["exploration_method"]["name"]))

        self.summary_writer = writer

    def select_action_greedy(self, state):

        if np.random.random() > 0.01:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                qs = self.policy_net(state.to(TORCH_DEVICE))
                return qs.max(1)[1].view(1, 1).to('cpu')
        else:
            return torch.tensor([[np.random.randint(self.n_action)]], dtype=torch.long)

    def get_q_values(self, state):
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            qs = self.policy_net(state.to(TORCH_DEVICE))
            return qs

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
            return self.get_q_values(state).max(1)[1].view(1, 1).to('cpu')
        else:
            act = np.random.choice(self.n_action, p=self.action_proba)
            return torch.tensor([[act]], dtype=torch.long)


    def push(self, *args):
        self.memory.push(*args)

    def callback(self, epoch):

        if epoch % self.update_every_n_ep == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.num_update_target += 1

    def auxiliary_classification_loss(self, state_batch, action_batch, feedback_batch):

        logits = self.policy_net.compute_classif_forward(states=state_batch,
                                                         actions=action_batch)

        feedback_batch_label = feedback_batch.detach().long()
        loss = F.nll_loss(logits, feedback_batch_label)
        return loss, logits


    def optimize(self, total_iter):
        if len(self.memory) < self.batch_size:
            if self.summary_writer and total_iter % self.log_stats_every == 0:
                self.summary_writer.add_scalar("data/feedback_loss", 0, total_iter)
                self.summary_writer.add_scalar("data/q_loss", 0, total_iter)
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
        feedback_batch = torch.cat(batch.gave_feedback).to(TORCH_DEVICE)


        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_values = self.policy_net(state_batch)

        supervised_feedback_loss_weighted = 0
        if self.use_supervised_loss:
            supervised_loss, logits = self.auxiliary_classification_loss(state_batch=state_batch,
                                                                         action_batch=action_batch,
                                                                         feedback_batch=feedback_batch)

            supervised_feedback_loss_weighted = supervised_loss * self.supervised_loss_weight

            output_class = torch.max(logits.cpu(), dim=1)[1]
            supervised_score = f1_score(y_true=feedback_batch.cpu(), y_pred=output_class)


            mean = feedback_batch.mean().item()
            y_pred = np.random.choice([0,1], p=[1-mean, mean], size=(feedback_batch.size()))
            random_f1 = f1_score(y_true=feedback_batch.cpu(), y_pred=y_pred)

            self.supervised_score_logger.append(supervised_score)
            self.random_score_logger.append(random_f1)


        state_action_values = state_values.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_action_values = self.target_net(non_final_next_states)
        next_state_action_values = next_state_action_values.detach()

        next_state_values = torch.zeros(self.batch_size, device=TORCH_DEVICE)

        # Double Q-learning : the action is selected by the policy_net not the target
        action_selected_by_policy = self.policy_net(non_final_next_states).detach()
        action_selected_by_policy = action_selected_by_policy.max(1)[1]

        next_state_values[non_final_mask] = next_state_action_values.gather(1, action_selected_by_policy.unsqueeze(1)).view(-1)

        # Q learning : argmax selected on target
        #next_state_values[non_final_mask] = next_state_action_values.max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        expected_state_action_values = expected_state_action_values.unsqueeze(1)
        # Compute Huber loss
        assert state_action_values.size() == expected_state_action_values.size()
        q_loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        nudging_loss_weighted = 0
        if self.use_nudging_loss:
            nudging_loss = self.nudging_loss(qs=state_values,
                                             action=action_batch,
                                             feedback=feedback_batch,
                                             margin=self.nudging_loss_margin,
                                             regression_loss=self.regression_loss
                                             )

            nudging_loss_weighted = nudging_loss * self.nudging_loss_weight

        if self.use_consistency_loss_dfqd:
            next_state_values_ref = next_state_action_values
            next_states_qs = self.policy_net(non_final_next_states)

            consistency_loss = self.consistency_loss(next_state_values_ref, next_states_qs, self.regression_loss)
            consistency_loss_weighted = consistency_loss * self.consistency_loss_weight
        else:
            consistency_loss_weighted = 0

        # Compute the sum of all losses
        loss = q_loss * self.q_loss_weight + nudging_loss_weighted + consistency_loss_weighted + supervised_feedback_loss_weighted

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        #old_params = deepcopy(self.policy_net.state_dict())

        if self.clip_grad:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

        try:
            nudging_loss_weighted = nudging_loss_weighted.item()
        except AttributeError:
            nudging_loss_weighted = 0

        # feedback_loss_weighted = feedback_loss_weighted.item() if isinstance(feedback_loss_weighted,
        #                                                                      torch.Tensor) else 0

        self.feedback_loss_logger.append(nudging_loss_weighted)
        self.q_loss_logger.append(q_loss.item())
        self.percent_feedback_logger.append(feedback_batch.mean().item())

        if self.summary_writer and total_iter % self.log_stats_every == 0:
            self.summary_writer.add_scalar("data/feedback_loss", np.mean(self.feedback_loss_logger), total_iter)
            self.summary_writer.add_scalar("data/q_loss", np.mean(self.q_loss_logger), total_iter)
            self.summary_writer.add_scalar("data/feedback_percentage_in_buffer", np.mean(self.percent_feedback_logger), self.num_optim_dqn)

            # self.summary_writer.add_histogram("data/reward_in_batch_replay_buffer", reward_batch.detach().cpu().numpy(), self.num_update, bins=4)
            # self.summary_writer.add_histogramm("data/q_values", state_values.mean, self.num_update, bins=4)

            if self.use_supervised_loss:
                self.summary_writer.add_scalar("data/supervised_f1_score",
                                               np.mean(self.supervised_score_logger), total_iter)

                self.summary_writer.add_scalar("data/random_f1_score",
                                               np.mean(self.random_score_logger), total_iter)

            self.supervised_score_logger = []
            self.feedback_loss_logger = []
            self.q_loss_logger = []
            self.percent_feedback_logger = []
            self.random_score_logger = []

        self.num_optim_dqn += 1

        #check_params_changed(old_params, self.policy_net.state_dict())