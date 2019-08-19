import torch
from rl_agent.gpu_utils import TORCH_DEVICE


def consistency_loss_dqfd(next_qs_target, next_qs, regression_loss):
    """
    This loss enforce next_q to be consistent with target network, to avoid over-generalization
    """
    assert next_qs_target.requires_grad == False, "qs_target should be a reference, requires_grad should be false, is {}".format(next_qs_target.requires_grad)
    assert next_qs.requires_grad == True, "next_qs requires_grad should be True, is {}".format(next_qs.requires_grad)

    assert next_qs_target.size() == next_qs.size()

    return regression_loss(next_qs_target, next_qs)

def feedback_bad_to_min_when_max(qs, action, feedback, margin, regression_loss, classif_net=None):
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


def feedback_bad_to_min(qs, action, feedback, margin, regression_loss, classif_net=None):
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

def feedback_bad_to_percent_max(qs, action, feedback, margin, regression_loss, classif_net=None):
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


def feedback_frontier_margin(qs, action, feedback, margin, regression_loss, testing=False, classif_net=None):
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


def feedback_frontier_margin_learnt_feedback(qs, action, feedback, margin, regression_loss, testing=False, classif_net=None):
    """
    Compute the expert loss

    ab is the "bad" action
    m is a margin function
    """
    # if feedback.mean() in [0,1]:
    #     return 0

    return 0

    assert classif_net != None, "Need a classification network in 'learnt_feedback_loss'"

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



def learn_feedback_loss(classif_net, feedback_batch):
    pass



if __name__ == "__main__":

    import torch
    TORCH_DEVICE = 'cpu'
    regr_loss = torch.nn.functional.smooth_l1_loss
    margin = 0.1

    # # Test 1
    # qs = torch.arange(12).view(4,3).float()
    # actions = torch.Tensor([0,0,0,0]).long()
    # feedback = torch.Tensor([1,1,1,0])
    #
    # assert feedback_bad_to_min_when_max(qs, actions, feedback, margin, regr_loss) == 0
    #
    # # Test 2
    # qs = torch.arange(12).view(4,3).float()
    # actions = torch.Tensor([0,0,0,0]).long()
    # feedback = torch.Tensor([1,1,1,0])
    # loss1 = feedback_bad_to_min_when_max(qs, actions, feedback, margin, regr_loss)
    #
    # qs = torch.arange(12).view(4,3).float()
    # actions = torch.Tensor([0,1,2,0]).long()
    # feedback = torch.Tensor([1,1,1,0])
    # loss2 = feedback_bad_to_min_when_max(qs, actions, feedback, margin, regr_loss)
    #
    # assert loss1 < loss2
    #
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

    qs = torch.arange(21).view(7, 3).float()
    actions = torch.Tensor([1, 2, 1, 0, 2 , 1, 0]).long().view(-1, 1)
    feedback = torch.Tensor([1, 0, 1, 0, 1, 0, 1])
    loss1 = feedback_frontier_margin(qs, actions, feedback, margin, regr_loss, testing=True)

    actions = torch.Tensor([[1, 2, 1, 0, 2, 1, 0]]).long().view(-1, 1)
    feedback = torch.Tensor([1, 0, 1, 0, 1, 0, 1])

    qs = - torch.arange(21).view(7, 3).float()
    loss2s = feedback_frontier_margin(qs, actions, feedback, margin, regr_loss, testing=True)


    print("Tests okay !")
