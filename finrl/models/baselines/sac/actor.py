import torch
from torch import nn
from typing import List
from finrl.models.distributions import *

# CAP the standard deviation of the actor. exp(2) is 7.38, exp(-20) is 2e-9.
# std will be in [2e-9,7.38] which is practical.
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            activation_fn: Type[nn.Module] = None,
            dropout: float = 0,
    ):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation_fn = activation_fn
        self.actions_dist = SquashedDiagGaussianDistribution(action_dim)
        self._setup_model()

    def _setup_model(self):
        # model
        self.latent_pi = nn.Sequential(
            *[
                nn.Linear(self.state_dim, 256,bias=True),
                self.activation_fn(),
                nn.Linear(256,256,bias=True),
                self.activation_fn(),
            ]
        )
        self.mu = nn.Linear(256, self.action_dim,bias=True)  # 单层
        self.log_std = nn.Linear(256, self.action_dim,bias=True)  # 单层
        self.latten = nn.Flatten(1,-1)

    def _get_actions_dist_params(self, obs:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #get mean and std params of the actions distribution
        if len(obs.shape)==3:
            obs = self.latten(obs)
        latent_pi = self.latent_pi(obs)    # common neural networks outputs
        mean_actions = self.mu(latent_pi)  #neural networks outputs
        log_std = self.log_std(latent_pi)  #neural networks outputs
        # Original Implementation to cap the standard deviation
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)  #otherwise, std will be extreme, like 0 or +inf
        return (mean_actions, log_std)

    def forward(self, obs: torch.Tensor):
        # only get actions
        if len(obs.shape)==3:
            obs = self.latten(obs)
        mean_actions, log_std = self._get_actions_dist_params(obs)
        actions = self.actions_dist.get_actions_from_params(mean_actions, log_std)  #reparameterization trick
        return actions  # range [-1,1]

    def get_actions(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if len(obs.shape)==3:
            obs = self.latten(obs)
        if deterministic:
            mean_actions, log_std = self._get_actions_dist_params(obs)
            actions = torch.tanh(mean_actions)
        else:
            actions = self(obs)
        return actions

    def actions_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # select action and calculate log(prob(action))
        if len(obs.shape)==3:
            obs = self.latten(obs)
        mean_actions, log_std = self._get_actions_dist_params(obs)
        actions, log_probs = self.actions_dist.get_actions_log_prob_from_params(mean_actions,log_std)
        return (actions, log_probs)


