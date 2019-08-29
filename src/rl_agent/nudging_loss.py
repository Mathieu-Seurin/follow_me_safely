import torch.nn.functional as F
from rl_agent.gpu_utils import TORCH_DEVICE

import copy
import torch


def feedback_bad_to_min_when_max(qs, action, feedback, margin, regression_loss, feedback_logits=None, ceil=None):
    """
    Compute the expert loss when action is flagged as bad AND is the max q

    ab is the "bad" action
    m is a margin function

    Can be written as :
    Minimize    Q(s,ab) - min_a [ Q(s,a) - m(ab, a) ]
                                          m(ab,b) = margin if ab = a
                                                  = 0 else

    VERSION 2 : Applied only when Q(s,ab) is the max Q
    """

    # No action was flagged as 'bad' so it's okay
    if feedback.sum() == 0:
        return 0

    feedback = feedback.view(-1)

    # Ignore qs where the action is not flagged as bad
    n_action = qs.size(1)
    qs_where_bad = qs[feedback != 0]
    action_where_bad = action[feedback != 0]

    # Q(s, ab) => action taken was bad (feedback from env)
    qs_a_where_bad = qs_where_bad.gather(1, action_where_bad.view(-1,1))

    # Ignore q(s,ab) where q(s,ab) != max_a q(s,a)
    index_where_action_bad_max = qs_a_where_bad.view(-1) == torch.max(qs_where_bad, dim=1)[0]
    qs_a_where_bad = qs_a_where_bad[index_where_action_bad_max].view(-1)
    if qs_a_where_bad.size(0) == 0:
        return 0

    action_where_bad = action_where_bad[index_where_action_bad_max]
    qs_where_bad = qs_where_bad[index_where_action_bad_max]

    #  =====  Compute l(a_b, a) =====
    action_mask = torch.arange(n_action).unsqueeze(0).to(TORCH_DEVICE) != action_where_bad.view(-1, 1)
    # action_mask is the same size as qs. for every row, there is a 0 in column of action, 1 elsewhere
    # Exemple : action = [0, 1, 0] action_mask = [[0,1],[1,0],[0,1]]

    margin_malus = action_mask.float() * margin

    # Compute Q(s,a) - l(a_b, a)
    ref_qs = qs_where_bad.detach() # You optimize with respect to this ref_qs minus the margin, so you need to detach
    min_qs_minus_margin, _ = torch.min(ref_qs - margin_malus, dim=1)

    # Actual classification loss
    assert min_qs_minus_margin.size() == qs_a_where_bad.size(), \
        "Problem in loss, size 1 {}  size 2 : {}".format(min_qs_minus_margin.size(), qs_a_where_bad.size())

    assert qs_a_where_bad.requires_grad is True
    assert min_qs_minus_margin.requires_grad is False

    loss = regression_loss(min_qs_minus_margin, qs_a_where_bad) # Bring bad action down under margin
    return loss


def feedback_bad_to_min(qs, action, feedback, margin, regression_loss, feedback_logits=None, ceil=0.1):
    """
    Compute the expert loss

    ab is the "bad" action
    m is a margin function

    Can be written as :
    Minimize    Q(s,ab) - min_a [ Q(s,a) - m(ab, a) ]
                                          m(ab,b) = margin if ab = a
                                                  = 0 else
    """

    # Keep qs where a feedback from environment was given.

    if feedback.sum() == 0:
        return 0

    feedback = feedback.view(-1)

    n_action = qs.size(1)
    qs_where_bad = qs[feedback != 0]
    action_where_bad = action[feedback != 0]

    # Q(s, ab) => action taken was bad (feedback from env)
    qs_a_where_bad = qs_where_bad.gather(1, action_where_bad.view(-1,1)).squeeze(1)

    #  =====  Compute l(a_b, a) =====
    action_mask = torch.arange(n_action).unsqueeze(0).to(TORCH_DEVICE) != action_where_bad.view(-1, 1)
    # action_mask is the same size as qs. for every row, there is a 0 in column of action, 1 elsewhere
    # Exemple : action = [0, 1, 0] action_mask = [[0,1],[1,0],[0,1]]

    margin_malus = action_mask.float() * margin

    # Compute Q(s,a) - l(a_b, a)
    ref_qs = qs_where_bad.detach() # You optimize with respect to this ref_qs minus the margin, so you need to detach
    min_qs_minus_margin, _ = torch.min(ref_qs - margin_malus, dim=1)

    assert qs_a_where_bad.requires_grad is True
    assert min_qs_minus_margin.requires_grad is False

    # Actual classification loss
    assert min_qs_minus_margin.size() == qs_a_where_bad.size(),\
        "Problem in loss, size 1 {}  size 2 : {}".format(min_qs_minus_margin.size(), qs_a_where_bad.size())
    loss = regression_loss(min_qs_minus_margin, qs_a_where_bad) # Bring bad action down under margin
    return loss

def feedback_bad_to_percent_max(qs, action, feedback, margin, regression_loss, feedback_logits=None, ceil=None):
    """
    VERSION 4 : Applied only when Q(s,ab) is close to the max Q by a certain margin in % instead of absolute value

    Compute the expert loss when action is flagged as bad AND is too close to max q

    ab is the "bad" action
    m is a margin function

    Can be written as :
    Minimize    Q(s,ab) - max_margin * max_a Q(s,a)

    """

    # No action was flagged as 'bad' so it's okay
    if feedback.sum() == 0:
        return 0

    n_action = qs.size(1)
    LIM_INF = -1e10

    feedback = feedback.view(-1)

    # Ignore qs where the action is not flagged as bad
    qs_where_bad = qs[feedback != 0]
    action_where_bad = action[feedback != 0]

    # Q(s, ab) => action taken was bad (feedback from env)
    qs_a_where_bad = qs_where_bad.gather(1, action_where_bad.view(-1,1))

    masking_bad_actions = torch.arange(n_action).unsqueeze(0).to(TORCH_DEVICE) == action_where_bad.view(-1, 1)
    qs_where_bad_masked = qs_where_bad + masking_bad_actions.float() * LIM_INF # Float doesn't work, fuck this shit

    # Keep q(s,ab) where q(s,ab) is close to max_a q(s,a) (except bad action) by a certain margin
    max_qs = torch.max(qs_where_bad_masked, dim=1)[0]

    max_qs_and_margin = max_qs - torch.abs(max_qs) * (1 - margin)

    index_where_action_too_close_to_max = qs_a_where_bad.view(-1) >= max_qs_and_margin
    qs_a_where_bad = qs_a_where_bad[index_where_action_too_close_to_max].view(-1)

    if qs_a_where_bad.size(0) == 0:
        return 0

    max_qs_and_margin = max_qs_and_margin[index_where_action_too_close_to_max]
    max_qs_and_margin = max_qs_and_margin.detach()

    # Actual classification loss
    assert max_qs_and_margin.size() == qs_a_where_bad.size(), \
        "Problem in loss, size 1 {}  size 2 : {}".format(max_qs_and_margin.size(), qs_a_where_bad.size())

    assert max_qs_and_margin.requires_grad is False
    assert qs_a_where_bad.requires_grad is True

    loss = regression_loss(max_qs_and_margin, qs_a_where_bad) # Bring bad action down under margin
    return loss


def feedback_frontier_margin(qs, action, feedback, margin, regression_loss, testing=False, feedback_logits=None, ceil=None):
    """
    Compute the expert loss

    ab is the "bad" action
    m is a margin function
    """
    # if feedback.mean() in [0,1]:
    #     return 0

    n_action = qs.size(1)

    qs_a = qs.gather(1, action)

    qs_a_where_good = qs_a[feedback == 0]
    qs_a_where_bad = qs_a[feedback == 1]

    # Compute frontier with margin
    min_good = torch.min(qs_a_where_good) - margin
    max_bad = torch.max(qs_a_where_bad) + margin

    min_good = min_good.item()
    max_bad = max_bad.item()

    # Bring good actions above the max of bad actions
    qs_a_where_good_below_max = qs_a_where_good[qs_a_where_good < max_bad]
    if qs_a_where_good_below_max.size()[0] == 0:
        loss_good = 0
    else:
        max_bad_vec = torch.ones_like(qs_a_where_good_below_max)
        max_bad_vec[:] = max_bad

        if not testing:
            assert max_bad_vec.requires_grad == False
            assert qs_a_where_good_below_max.requires_grad == True

        loss_good = regression_loss(qs_a_where_good_below_max, max_bad_vec)

    # Bring bad actions below the min of good actions
    qs_a_where_bad_above_min = qs_a_where_bad[qs_a_where_bad > min_good]
    if qs_a_where_bad_above_min.size()[0] == 0:
        loss_bad = 0
    else:
        min_good_vec = torch.ones_like(qs_a_where_bad_above_min)
        min_good_vec[:] = min_good

        if not testing:
            assert min_good_vec.requires_grad == False
            assert qs_a_where_bad_above_min.requires_grad == True

        loss_bad = regression_loss(qs_a_where_bad_above_min, min_good_vec)

    return loss_good + loss_bad

def compute_entropy_loss(x):
    loss = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    return loss.mean()

class FeedbackFrontierMarginLearnFeedback(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, *input):
        pass

def feedback_frontier_margin_learnt_feedback(qs, action=None, feedback=None, margin=None, regression_loss=None, testing=False, feedback_logits=None, ceil=0.5):
    """
    Compute the expert loss

    ab is the "bad" action
    m is a margin function
    """

    # todo : stats on percent change

    assert feedback_logits is not None, "Need logits from the classification network"
    n_actions = qs.size(1)

    almost_sure_feedback = torch.zeros(*feedback_logits.size()).to(TORCH_DEVICE)
    almost_sure_no_feedback = torch.zeros(*feedback_logits.size()).to(TORCH_DEVICE)

    # Feedback (or no feedback) that is almost sure (using a classification network), use them.
    almost_sure_feedback[feedback_logits > ceil] = 1
    almost_sure_no_feedback[feedback_logits < - ceil] = 1

    # check if at least there are "sure" predicted feedback per line
    at_least_one_feedback_per_line = almost_sure_feedback.sum(dim=1).type(torch.uint8)
    at_least_one_nofeedback_per_line = almost_sure_no_feedback.sum(dim=1).type(torch.uint8)

    # Statistics about the number of action that are considered
    certainty_percentage = ((at_least_one_nofeedback_per_line + at_least_one_feedback_per_line).float() / n_actions).mean()
    certainty_percentage_feed = (at_least_one_feedback_per_line.float() / n_actions).mean()
    certainty_percentage_no_feed = (at_least_one_nofeedback_per_line.float() / n_actions).mean()

    both_per_line = at_least_one_feedback_per_line + at_least_one_nofeedback_per_line > 1

    qs = qs[both_per_line, :]
    if qs.size(0) == 0:
        return 0

    almost_sure_no_feedback = almost_sure_no_feedback[both_per_line, :]
    almost_sure_feedback = almost_sure_feedback[both_per_line, :]

    qs_feedback = qs.clone().detach() # Q(s,a) for action flagged as "gives feedback" aka bad actions by classification algorithm
    qs_no_feedback = qs.clone().detach() #Q(s,a) for action flagged as "don't give feedback" aka good actions by classification algorithm

    # Don't know if it's a feedback yet ? Temporarily set to the minimum qs so the optim doesn't touch it
    qs_feedback[almost_sure_feedback == 0] = torch.min(qs).item() - margin

    # Don't know if it's NOT a feedback yet ? Temporarily set to the maximum qs so the optim doesn't touch it
    qs_no_feedback[almost_sure_no_feedback == 0] = torch.max(qs).item() + margin

    min_no_feedback_per_line = qs_no_feedback.min(dim=1)[0].repeat([n_actions, 1]).t() - margin
    max_feedback_per_line = qs_feedback.max(dim=1)[0].repeat([n_actions, 1]).t() + margin

    sure_feedback_and_above_min = almost_sure_feedback.byte() * (qs > min_no_feedback_per_line) # '*' is logical and
    sure_no_feedback_and_below_max = almost_sure_no_feedback.byte() * (qs < max_feedback_per_line)

    qs_target = qs.clone().detach()
    qs_feedback_target = torch.where(sure_feedback_and_above_min, min_no_feedback_per_line, qs_target)
    qs_no_feedback_target = torch.where(sure_no_feedback_and_below_max, max_feedback_per_line, qs_target)

    if not testing:
        assert qs_no_feedback_target.requires_grad == False
        assert qs_feedback_target.requires_grad == False
        assert qs.requires_grad == True
        reduction = 'mean'
    else:
        reduction = 'sum'

    # ===== Get good action values above the best bad action
    loss_no_feedback = regression_loss(qs, qs_no_feedback_target, reduction=reduction)
    loss_feedback = regression_loss(qs, qs_feedback_target, reduction=reduction)

    return loss_no_feedback + loss_feedback


if __name__ == "__main__":

    import torch
    TORCH_DEVICE = 'cpu'
    regr_loss = torch.nn.functional.smooth_l1_loss
    margin = 0.1

    # Test 1
    qs = torch.arange(12).view(4,3).float()
    actions = torch.Tensor([0,0,0,0]).long()
    feedback = torch.Tensor([1,1,1,0])

    assert feedback_bad_to_min_when_max(qs, actions, feedback, margin, regr_loss) == 0

    # Test 2
    qs = torch.arange(12).view(4,3).float()
    actions = torch.Tensor([0,0,0,0]).long()
    feedback = torch.Tensor([1,1,1,0])
    loss1 = feedback_bad_to_min_when_max(qs, actions, feedback, margin, regr_loss)

    qs = torch.arange(12).view(4,3).float()
    actions = torch.Tensor([0,1,2,0]).long()
    feedback = torch.Tensor([1,1,1,0])
    loss2 = feedback_bad_to_min_when_max(qs, actions, feedback, margin, regr_loss)

    assert loss1 < loss2

    # # Test 3
    # qs = torch.arange(12).view(4, 3).float()
    #
    # max_margin = 0.50 # If bad action is 50% of the max : Put it down niggah
    # actions = torch.Tensor([0, 1, 2, 0]).long()
    # feedback = torch.Tensor([1, 1, 1, 0])
    # loss1 = feedback_bad_to_percent_max(qs, actions, feedback, regr_loss, max_margin)
    #
    # max_margin = 0.90 # If bad action is 90% of the max : Put it down niggah
    # qs = torch.arange(12).view(4, 3).float()
    # actions = torch.Tensor([0, 1, 2, 0]).long()
    # feedback = torch.Tensor([1, 1, 1, 0])
    # loss2 = feedback_bad_to_percent_max(qs, actions, feedback, regr_loss, max_margin)
    #
    # # assert loss1 > loss2, "loss1 {},  loss2 {}".format(loss1, loss2)
    #
    # max_margin = 0.90  # If bad action is 90% of the max : Put it down niggah
    # qs = - torch.arange(12).view(4, 3).float()
    # actions = torch.Tensor([0, 1, 2, 0]).long()
    # feedback = torch.Tensor([1, 1, 1, 0])
    # loss3 = feedback_bad_to_percent_max(qs, actions, feedback, regr_loss, max_margin)
    #
    # #=================================================
    #
    # qs = torch.arange(12).view(4,3).float()
    # actions = torch.Tensor([1,2,1,0]).long()
    # feedback = torch.Tensor([1,1,1,0])
    # loss1 = feedback_bad_to_min(qs, actions, feedback, margin, regr_loss)

    #==================================================
    #
    # qs = torch.arange(21).view(7, 3).float()
    # actions = torch.Tensor([1, 2, 1, 0, 2 , 1, 0]).long().view(-1, 1)
    # feedback = torch.Tensor([1, 0, 1, 0, 1, 0, 1])
    # loss1 = feedback_frontier_margin(qs, actions, feedback, margin, regr_loss, testing=True)
    #
    # actions = torch.Tensor([[1, 2, 1, 0, 2, 1, 0]]).long().view(-1, 1)
    # feedback = torch.Tensor([1, 0, 1, 0, 1, 0, 1])
    #
    # qs = - torch.arange(21).view(7, 3).float()
    # loss2s = feedback_frontier_margin(qs, actions, feedback, margin, regr_loss, testing=True)

    #=======================================================

    qs = torch.arange(21).view(7, 3).float()
    logits = torch.ones(7,3)
    logits[:, 2] *= -1
    logits[:, 1] = 0


    loss1 = feedback_frontier_margin_learnt_feedback(qs, margin=margin, regression_loss=regr_loss, feedback_logits=logits,
                                                     testing=True)
    assert loss1 == 0

    # =======================================================
    qs = torch.arange(21).view(7, 3).float()
    logits = torch.ones(7, 3)
    logits[:, 0] = 0
    logits[:, 1] = 0

    loss2 = feedback_frontier_margin_learnt_feedback(qs, margin=margin, regression_loss=regr_loss,
                                                     feedback_logits=logits,
                                                     testing=True)
    assert loss2 == 0

    #=======================================================
    qs = -torch.arange(21).view(7, 3).float()
    logits = torch.ones(7, 3)
    logits[:, 2] *= -1
    logits[:, 1] = 0

    loss3 = feedback_frontier_margin_learnt_feedback(qs, margin=margin, regression_loss=regr_loss,
                                                     feedback_logits=logits,
                                                     testing=True)

    #========================================================

    qs = torch.arange(21).view(7, 3).float()
    logits = torch.ones(7, 3)
    logits[:, 0] *= -1
    logits[:, 1] = 0

    loss4 = feedback_frontier_margin_learnt_feedback(qs, margin=margin, regression_loss=regr_loss,
                                                     feedback_logits=logits,
                                                     testing=True)

    assert loss3 == loss4

    # ========================================================

    qs = torch.arange(21).view(7, 3).float()
    logits = torch.ones(7, 3)
    logits[:, 0] *= -1
    logits[:, 1] = 0
    logits[-1, :] = 0

    loss5 = feedback_frontier_margin_learnt_feedback(qs, margin=margin, regression_loss=regr_loss,
                                                     feedback_logits=logits,
                                                     testing=True)

    assert loss5 != 0
    assert loss5 < loss4

    # =========================================================
    qs = torch.arange(21).view(7, 3).float()
    logits = torch.zeros(7, 3)
    logits[:,0] = 1

    loss6 = feedback_frontier_margin_learnt_feedback(qs, margin=margin, regression_loss=regr_loss,
                                                     feedback_logits=logits,
                                                     testing=True)

    assert loss6 == 0

    print("Tests okay !")
