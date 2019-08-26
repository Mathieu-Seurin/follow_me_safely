from rl_agent.agent_utils import Transition, ReplayMemory, ProportionReplayMemory

from rl_agent.nudging_loss import feedback_bad_to_min_when_max, \
    feedback_bad_to_percent_max, feedback_bad_to_min, feedback_frontier_margin, \
    feedback_frontier_margin_learnt_feedback, compute_entropy_loss

import torch
import torch.nn.functional as F

from rl_agent.models import FullyConnectedModel, ConvModel, TextModel
import numpy as np
from copy import copy, deepcopy

from rl_agent.gpu_utils import TORCH_DEVICE

import tensorboardX

from sklearn.metrics import f1_score, accuracy_score

from rl_agent.preprocessor import TextPreprocessor, ImagePreprocessor

class DQNAgent(object):

    def __init__(self, config, action_space, obs_space, discount_factor, writer=None, log_stats_every=1e4):

        self.save_config = config
        self.log_stats_every = log_stats_every

        self.q_loss_logger = []
        self.feedback_loss_logger = []
        self.percent_feedback_logger = []
        self.action_classif_f1_logger = []
        self.action_classif_random_score_logger = []
        self.entropy_loss_logger = []

        self.discount_factor = discount_factor
        self.n_action = action_space.n

        self.lr = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.batch_size = config["batch_size"]
        self.clip_grad = config["clip_grad"]
        self.q_loss_weight = config["q_loss_weight"]
        self.boostrap_feedback = config["boostrap_feedback"]

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
        self.learn_feedback = False

        if self.use_nudging_loss:
            self.nudging_loss_weight = config["nudging_loss_weight"]
            self.nudging_loss_margin = config["nudging_margin"]
            self.certainty_ceil = config["certainty_ceil_classif"]

            loss_type = config["nudging_type"]
            if loss_type == "max":
                self.nudging_loss = feedback_bad_to_min_when_max
            elif loss_type == "min":
                self.nudging_loss = feedback_bad_to_min
            elif loss_type == "percent":
                self.nudging_loss = feedback_bad_to_percent_max
            elif loss_type == "frontier":
                self.nudging_loss = feedback_frontier_margin
            elif loss_type == "frontier_feedback_learnt":
                self.learn_feedback = True
                self.nudging_loss = feedback_frontier_margin_learnt_feedback
            else:
                raise NotImplementedError("Wrong nudging loss, {}".format(loss_type))

        self.use_entropy_loss = config["entropy_loss_weight"] > 0
        self.entropy_loss_weight = config["entropy_loss_weight"]

        # ========== Model ===============

        if config["model_type"] == "fc" :
            Model = FullyConnectedModel
        elif config["model_type"] == "text":
            Model = TextModel
            self.preprocessor = TextPreprocessor
        else:
            Model = ConvModel
            self.preprocessor = ImagePreprocessor

        self.policy_net = Model(config["model_params"], action_space=action_space, obs_space=obs_space).to(TORCH_DEVICE)
        self.target_net = Model(config["model_params"], action_space=action_space, obs_space=obs_space).to(TORCH_DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.classif_feedback_net = None
        if self.learn_feedback:
            self.classif_feedback_net = Model(config["model_params"], action_space=action_space, obs_space=obs_space).to(TORCH_DEVICE)
            self.classif_feedback_loss = torch.nn.SoftMarginLoss()
            self.classif_feedback_optim = torch.optim.RMSprop(self.classif_feedback_net.parameters(),
                                                              lr=config["classif_learning_rate"],
                                                              )

        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if config["exploration_method"] == "eps_greedy":

            self._select_action = self._select_action_eps_greedy
            self.epsilon_init = config["exploration_params"]["begin_eps"]
            self.current_eps = copy(self.epsilon_init)
            self.expected_exploration_steps = config["exploration_params"]["expected_step_explo"]
            self.minimum_epsilon = config["exploration_params"]["epsilon_minimum"]
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

        elif config["exploration_method"] == "boltzmann" :

            self._select_action = self._select_action_boltzmann
            self.epsilon_init = config["exploration_params"]["begin_eps"]
            self.current_eps = copy(self.epsilon_init)
            self.expected_exploration_steps = config["exploration_params"]["expected_step_explo"]
            self.minimum_epsilon = config["exploration_params"]["epsilon_minimum"]
            self.n_step_eps = 0


        self.select_action = lambda x : self._select_action(self.preprocessor(x))
        self.summary_writer = writer

    def select_action_greedy(self, state):

        if np.random.random() > 0.01:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                qs = self.policy_net(state)
                return qs.max(1)[1].view(1, 1).to('cpu')
        else:
            return np.random.randint(self.n_action)

    def get_q_values(self, state):
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            qs = self.policy_net(state)
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
            return self.get_q_values(state).max(1)[1].item()
        else:
            act = np.random.choice(self.n_action, p=self.action_proba)
            return act

    def _select_action_boltzmann(self, state):

        q = self.get_q_values(state)
        p = torch.nn.functional.softmax(q / self.current_eps, dim=1).view(-1)
        act = np.random.choice(self.n_action, p=p.cpu().type(torch.float16).numpy())

        self.current_eps = max(self.minimum_epsilon, self.epsilon_init * np.exp(
            - 2.5 * self.n_step_eps / self.expected_exploration_steps))
        self.n_step_eps += 1

        return act


    def push(self, *args):
        self.memory.push(*args)

    def callback(self, epoch):

        if epoch % self.update_every_n_ep == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.num_update_target += 1

    def train_feedback_classif(self, state_batch, action_batch, feedback_batch):

        output = self.classif_feedback_net.forward(state_batch)
        output = torch.tanh(output)

        feedback_per_action_logits = output.gather(1, action_batch)

        changed_label_feedback_batch = torch.ones_like(feedback_batch)
        changed_label_feedback_batch[feedback_batch == 0] = -1 # convert to pytorch label

        loss = self.classif_feedback_loss(feedback_per_action_logits, changed_label_feedback_batch)

        self.classif_feedback_optim.zero_grad()
        loss.backward()
        self.classif_feedback_optim.step()

        return loss.item(), feedback_per_action_logits.detach().view(-1).cpu(), output.detach().cpu()


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

        state_batch = self.preprocessor(batch.state)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(TORCH_DEVICE)
        reward_batch = torch.FloatTensor(batch.reward).to(TORCH_DEVICE)
        feedback_batch =  torch.FloatTensor(batch.gave_feedback).unsqueeze(1).to(TORCH_DEVICE)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=TORCH_DEVICE, dtype=torch.bool)

        if not self.boostrap_feedback:
            no_feed_batch = feedback_batch == 0
            non_final_mask = non_final_mask * no_feed_batch

        non_final_next_states = [s for i, s in enumerate(batch.next_state)
                                 if s is not None and non_final_mask[i]]

        non_final_next_states = self.preprocessor(non_final_next_states)


        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_values = self.policy_net(state_batch)

        if self.learn_feedback:
            ###########
            # Since the model is separated from the rl, can be learnt remotly
            # The train optimizes the model already
            ###########
            supervised_loss, feedback_classif_logits_per_action, feedback_classif_logits = self.train_feedback_classif(state_batch=state_batch,
                                                                                                                       feedback_batch=feedback_batch,
                                                                                                                       action_batch=action_batch)

            rounded_feedback_logits = np.zeros(feedback_classif_logits_per_action.shape[0])

            #rounded_feedback_logits[feedback_classif_logits_per_action <= 0] = 0
            rounded_feedback_logits[feedback_classif_logits_per_action.numpy() > 0] = 1

            y_true = feedback_batch.cpu().view(-1)
            random_f1 = feedback_batch.mean().item()

            supervised_f1_score = f1_score(y_true=y_true, y_pred=rounded_feedback_logits)
            random_f1 = f1_score(y_true=y_true, y_pred = np.random.choice([0,1], p=[1-random_f1, random_f1]))

            self.action_classif_f1_logger.append(supervised_f1_score)
            self.action_classif_random_score_logger.append(random_f1)

        else:
            feedback_classif_logits = None


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
        assert state_action_values.size() == expected_state_action_values.size(), "Q(s,a) : {} \n Q'(s,a) : {}".format(
            state_action_values.size(),expected_state_action_values.size())

        q_loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        nudging_loss_weighted = 0
        nudging_loss_to_log = 0
        if self.use_nudging_loss:
            nudging_loss = self.nudging_loss(qs=state_values,
                                             action=action_batch,
                                             feedback=feedback_batch,
                                             margin=self.nudging_loss_margin,
                                             regression_loss=self.regression_loss,
                                             feedback_logits=feedback_classif_logits,
                                             ceil=self.certainty_ceil
                                             )

            nudging_loss_weighted = nudging_loss * self.nudging_loss_weight

            if nudging_loss_weighted != 0:
                nudging_loss_to_log = nudging_loss_weighted.detach().item()
                assert nudging_loss_weighted.requires_grad == True



        entropy_loss_weighted = 0
        entropy_loss_to_log = 0
        # To avoid spikes in Q func, add an entropy regularizer term
        if self.use_entropy_loss:
            entropy_loss = compute_entropy_loss(state_values)
            entropy_loss_weighted = entropy_loss * self.entropy_loss_weight
            entropy_loss_to_log = entropy_loss_weighted.detach().item()

            if entropy_loss_weighted != 0:
                entropy_loss_to_log = entropy_loss_weighted.detach().item()
                assert entropy_loss_weighted.requires_grad == True


        # Compute the sum of all losses
        loss = q_loss * self.q_loss_weight + nudging_loss_weighted + entropy_loss_weighted

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        #old_params = deepcopy(self.policy_net.state_dict())

        if self.clip_grad:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        self.entropy_loss_logger.append(entropy_loss_to_log)
        self.feedback_loss_logger.append(nudging_loss_to_log)
        self.q_loss_logger.append(q_loss.item())
        self.percent_feedback_logger.append(feedback_batch.mean().item())

        if self.summary_writer and total_iter % self.log_stats_every == 0:
            self.summary_writer.add_scalar("data/feedback_loss", np.mean(self.feedback_loss_logger), total_iter)
            self.summary_writer.add_scalar("data/q_loss", np.mean(self.q_loss_logger), total_iter)
            self.summary_writer.add_scalar("data/feedback_percentage_in_buffer", np.mean(self.percent_feedback_logger), self.num_optim_dqn)

            # self.summary_writer.add_histogram("data/reward_in_batch_replay_buffer", reward_batch.detach().cpu().numpy(), self.num_update, bins=4)
            # self.summary_writer.add_histogramm("data/q_values", state_values.mean, self.num_update, bins=4)

            if self.learn_feedback:
                self.summary_writer.add_scalar("data/action_classif_f1_score",
                                               np.mean(self.action_classif_f1_logger), total_iter)

                self.summary_writer.add_scalar("data/action_classif_random_f1_score",
                                               np.mean(self.action_classif_random_score_logger), total_iter)

            if self.use_entropy_loss:
                self.summary_writer.add_scalar("data/entropy_loss_logger",
                                               np.mean(self.entropy_loss_logger), total_iter)

            self.action_classif_f1_logger = []
            self.feedback_loss_logger = []
            self.q_loss_logger = []
            self.percent_feedback_logger = []
            self.action_classif_random_score_logger = []
            self.entropy_loss_logger = []

        self.num_optim_dqn += 1

        #check_params_changed(old_params, self.policy_net.state_dict())