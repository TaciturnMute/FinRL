import torch
from torch.distributions import Normal
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union,Iterable
from finrl.models.utils import sum_independent_dims

class SquashedDiagGaussianDistribution():

    def __init__(self,
                 actions_dim: int,
                 epsilon: float = 1e-6):
        '''

        :param actions_dim:
        :param epsilon:

        method:
        _prob_distribution: use mean_actions and log_std params to update distribution
        get_actions_from_params: use params to generate squashed actions
        get_actions_log_prob_from_params: use params to generate squashed actions and its log_likelihood, return both
        get_log_prob_from_params: use params to generate squashed actions and its log_likelihood, return the latter
        get_log_prob_from_actions: get actions'/gaussian actions' log_likelihood
        '''
        self.actions_dim = actions_dim
        self.gaussian_actions = None
        self.epsilon = epsilon

    def _prob_distribution(self,
                           mean_actions: torch.Tensor,
                           log_std: torch.Tensor) -> "SquashedDiagGaussianDistribution":
        # Update the proba distribution with new params
        self.dist = Normal(mean_actions, log_std.exp())
        return self

    def get_actions_from_params(self,
                                mean_actions: torch.Tensor,
                                log_std: torch.Tensor) -> torch.Tensor:
        # only get squashed actions
        # gaussian_actions : action that sample directly from the gaussian distribution
        self._prob_distribution(mean_actions, log_std)  # update distribution with new params
        self.gaussian_actions = self.dist.rsample()  # original action  (without batch dim)
        actions = torch.tanh(self.gaussian_actions)
        return actions  # reparameterization trick; squashed to [-1,1]

    def get_actions_log_prob_from_params(self,
                                         mean_actions: torch.Tensor,
                                         log_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        select action and get log(pi(a|s)) form dist params
            1. update dist with params
            2. sample gaussian_action from dist, and get squashed action
            3. calculate log_prob of squashed action

        about 3 : from gaussian_actions log_prob to actual squashed actions log_prob
            Because actor squashed gaussian action into [-1,1] with tanh function,
            so we must transform log_prob(gaussian_actions) into log_prob(actions).
            This is just like, now we have random variable X and its pdf, we want to get the pdf of variable Y(X).
            Gaussian_actions sampled form gaussian distribution, and Y(X) = tanh(gaussian_actions) = actions.

        :param mean_actions: mean of normal dist
        :param log_std: log std of normal dist
        :return:
        '''

        # actions = self.get_actions_from_params(mean_actions,log_std) # get squashed actions
        self._prob_distribution(mean_actions, log_std)  # update distribution with new params
        self.gaussian_actions = self.dist.rsample()  # original action  (without batch dim)
        actions = torch.tanh(self.gaussian_actions)
        gaussian_action_log_probs = sum_independent_dims(self.dist.log_prob(self.gaussian_actions))
        log_probs = gaussian_action_log_probs - torch.sum(torch.log(1 - actions**2 + self.epsilon), dim=1)
        log_probs = log_probs.reshape(-1, 1)  # (batch_size, 1)
        return (actions,log_probs)

    # def get_log_prob_from_params(self,
    #                              mean_actions:torch.Tensor,
    #                              log_std:torch.Tensor) -> torch.Tensor:
    #     # only get log(pi(a|s)) from params.
    #     # update dist, rsample fom the dist, and calculate log_prob of the sample action
    #     return self.get_actions_log_prob_from_params(mean_actions, log_std)[1]

    def get_log_prob_from_actions(self,
                                  mean_actions: torch.Tensor,
                                  log_std: torch.Tensor,
                                  actions = None,
                                  gaussian_actions = None,
                                  ):
        # PPO calculate likelihood log(pi(a|s)) with current policy
        assert actions is not None or gaussian_actions is not None, \
            'both of actions and gaussian_actions can not be None'
        self._prob_distribution(mean_actions, log_std)  # update distribution with new params
        if gaussian_actions is None:
            # inverse squashed actions to gaussian_actions
            eps = torch.finfo(actions.dtype).eps
            actions = torch.clamp(actions, 1 - eps, 1 + eps) # clip actions avoid NAN
            gaussian_actions = 0.5 * (torch.log1p(actions) - torch.log1p(-actions))
            self._prob_distribution(mean_actions, log_std)  # update distribution with new params # 多余？
        log_probs = sum_independent_dims(self.dist.log_prob(gaussian_actions)) # without sampling actions
        log_probs -= torch.sum(torch.log(1 - actions**2 + self.epsilon), dim=1)
        return log_probs
