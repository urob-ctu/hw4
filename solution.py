import torch

def policy_gradient_loss_simple(logp, tensor_r):
    """given the policy (T, N) log-probabilities and (T, N) rewards,
    compute the scalar loss for pytorch based on the policy gradient"""
    policy_loss = torch.tensor(0.)  # placeholder

    with torch.no_grad():
        # TODO: compute returns of the trajectories from the reward tensor

    # TODO: compute the policy loss with trajectory returns
    return policy_loss


def discount_cum_sum(rewards, gamma):
    """rewards is (T, N) tensor, gamma is scalar, output should be (T, N) tensor.
    Here we want to compute the discounted trajectory returns at each timestep.
    At each timestep, produce the exponentially weighted sum of (only) the following rewards on a given trajectory
    i.e. $R(\tau_i, t) = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}$"""
    T = rewards.shape[0]
    returns = torch.zeros_like(rewards)  # placeholder

    # TODO: implement the discounted cummulative sum, i.e. the discounted returns computed from rewards and gamma
    return returns


def policy_gradient_loss_discounted(logp, tensor_r, gamma):
    """given the policy (T, N) log-probabilities, (T, N) rewards
    and the discount factor gamma, compute the scalar loss for pytorch based
    on the policy gradient with discounted returns"""
    policy_loss = torch.tensor(0.)  # placeholder

    with torch.no_grad():
        # TODO: compute discounted returns of the trajectories from the reward tensor

    # TODO: compute the policy loss with discounted returns
    return policy_loss


def policy_gradient_loss_advantages(logp, advantage_estimates):
    """given the policy (T, N) log-probabilities and (T, N) advantage estimates,
    compute the scalar loss for pytorch based on the policy gradient with advantages"""
    policy_loss = torch.tensor(0.)  # placeholder

    # TODO: compute the policy gradient estimate using the advantage estimate weighting
    return policy_loss

def value_loss(values, value_targets):
    """ given (T, N) values, (T, N) value targets, compute the scalar regression loss for pytorch"""
    value_loss = torch.tensor(0.)  # placeholder

    # TODO: compute the value function L2 loss
    return value_loss

def PPO_loss(p_ratios, advantage_estimates, epsilon):
    """ given (T, N) p_ratios probability ratios, (T, N) advantage_estimates, and epsilon clipping ratio,
    compute the scalar loss for pytorch based on the PPO surrogate objective"""
    ppo_loss = torch.tensor(0.)  # placeholder

    # TODO: compute the PPO loss
    return ppo_loss

