from rl_agent.agent_utils import Transition, ReplayMemory, ProportionReplayMemory

import torch
import torch.nn.functional as F

from rl_agent.models import ConvACModel, FCACModel
import numpy as np
from copy import copy, deepcopy

from rl_agent.gpu_utils import TORCH_DEVICE

import tensorboardX

from sklearn.metrics import f1_score, accuracy_score

class ACAgent(object):

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
        self.boostrap_feedback = config["boostrap_feedback"]

        # ============ LOSSES : Bellman, feedback, regularization ========
        # ================================================================
        if config["regression_loss_func"] == "l2":
            self.regression_loss = F.mse_loss
        elif config["regression_loss_func"] == "l1":
            self.regression_loss = F.smooth_l1_loss
        else:
            raise NotImplementedError("Wrong regression loss func, is {}".format(config["regression_loss_func"]))

        self.use_supervised_loss = config["supervised_loss_weight"] > 0
        if self.use_supervised_loss:
            self.supervised_loss_weight = config["supervised_loss_weight"]

        # ========== Model ===============

        Model = FCACModel if config["model_type"] == 'fc' else ConvACModel

        self.policy_net = Model(config["model_params"], n_action=n_action, state_dim=state_dim,
                                learn_feedback_classif=self.use_supervised_loss).to(TORCH_DEVICE)


        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)


        if config["exploration_method"] == "boltzmann" :
            pass


        self.summary_writer = writer

    def act(self, state):

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


    def push(self, *args):
        self.memory.push(*args)

    def callback(self, epoch):

        if epoch % self.update_every_n_ep == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.num_update_target += 1


    def optimize(self, total_iter):

        # Compute Huber loss
        q_loss = 0

        # Optimize the model
        self.optimizer.zero_grad()
        q_loss.backward()

        #old_params = deepcopy(self.policy_net.state_dict())

        if self.clip_grad:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()


        if self.summary_writer and total_iter % self.log_stats_every == 0:
            self.summary_writer.add_scalar("data/feedback_loss", np.mean(self.feedback_loss_logger), total_iter)
            self.summary_writer.add_scalar("data/q_loss", np.mean(self.q_loss_logger), total_iter)
            self.summary_writer.add_scalar("data/feedback_percentage_in_buffer", np.mean(self.percent_feedback_logger), self.num_optim)

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

        self.num_optim += 1

        #check_params_changed(old_params, self.policy_net.state_dict())