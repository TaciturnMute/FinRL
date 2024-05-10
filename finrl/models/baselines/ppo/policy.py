import torch
from torch import nn
from typing import Type
from finrl.models.distributions import SquashedDiagGaussianDistribution

class MlpExtractor(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 activation_fn) -> None:
        super(MlpExtractor,self).__init__()
        self.shared_net = nn.Sequential()
        self.policy_net = nn.Sequential(
            *[
                nn.Linear(state_dim, 64, bias=True),
                activation_fn(),
                nn.Linear(64,64,bias=True),
                activation_fn()
            ]
        )
        self.value_net = nn.Sequential(
            *[
                nn.Linear(state_dim, 64, bias=True),
                activation_fn(),
                nn.Linear(64,64,bias=True),
                activation_fn()
            ]
        )
        self.flatten = nn.Flatten(1,-1)
    
    def forward(self,obs):
        if len(obs.shape)==3:
            obs = self.flatten(obs)
        shared_latent = self.shared_net(obs)
        latent_vf = self.value_net(shared_latent)
        latent_pi = self.policy_net(shared_latent)
        return (latent_pi,latent_vf)
    
    def forward_critic(self,obs):
        if len(obs.shape)==3:
            obs = self.flatten(obs)
        shared_latent = self.shared_net(obs)
        latent_vf = self.value_net(shared_latent)
        return latent_vf

    def forward_actor(self,obs):
        if len(obs.shape)==3:
            obs = self.flatten(obs)
        shared_latent = self.shared_net(obs)
        latent_pi = self.policy_net(shared_latent)
        return latent_pi

class Policy(nn.Module):

    def __init__(self,
                 state_dim: int = None,
                 action_dim: int = None,
                 activation_fn: Type[nn.Module] = None,
                 log_std_init: float = 0.0,
                 ):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self._setup_model()
        self.dist = SquashedDiagGaussianDistribution(actions_dim = self.action_dim)

    def _setup_model(self):
        self.mlp_extractor = MlpExtractor(self.state_dim,self.action_dim,self.activation_fn)
        self.value_net = nn.Linear(64, 1)  # critic
        self.action_net = nn.Linear(64, self.action_dim) # actor
        self.log_std = nn.Parameter(torch.ones(self.action_dim) * self.log_std_init, requires_grad = True)  # a number  
        self.flatten = nn.Flatten(1,-1)

    def predict_values(self, obs: torch.Tensor):
        # critic part
        # only get value estimation, PPO collect_rollout function use
        if len(obs)==3:
            obs = self.flatten(obs)
        latent_vf = self.mlp_extractor.forward_critic(obs)
        values = self.value_net(latent_vf)
        return values

    def get_actions(self, obs: torch.Tensor, deterministic=False):
        if len(obs)==3:
            obs = self.flatten(obs)
        if deterministic:
            latent_pi = self.mlp_extractor.forward_actor(obs)
            mean_actions = self.action_net(latent_pi)
            actions = torch.tanh(mean_actions)
        else:
            # actor part
            latent_pi = self.mlp_extractor.forward_actor(obs)
            mean_actions = self.action_net(latent_pi)
            actions = self.dist.get_actions_from_params(mean_actions, self.log_std)
        return actions

    def get_gaussian_actions(self):
        return self.dist.gaussian_actions

    def forward(self, obs: torch.Tensor):
        # get actions, values, log_prob
        if len(obs)==3:
            obs = self.flatten(obs)
        latent_pi, latent_vf = self.mlp_extractor(obs)
        values = self.value_net(latent_vf)
        mean_actions = self.action_net(latent_pi)
        actions, log_prob = self.dist.get_actions_log_prob_from_params(mean_actions, self.log_std)
        return actions, values, log_prob  #torch.tensor,torch.tensor,torch.tensor

    def evaluate_actions(self,
                         obs: torch.Tensor,
                         actions: torch.Tensor,
                         gaussian_actions: torch.Tensor):
        # Evaluate actions according to the current policy, given the observations.
        if len(obs)==3:
            obs = self.flatten(obs)
        latent_pi, latent_vf = self.mlp_extractor(obs)
        values = self.value_net(latent_vf)  # estimate current state values
        mean_actions = self.action_net(latent_pi)
        # get current policy log(pi(at|st)) 1. get current policy from given params(each state==> each mean_actions==> given params) 2. get log prob for given action
        log_prob = self.dist.get_log_prob_from_actions(actions=actions,
                                                       gaussian_actions=gaussian_actions,
                                                       mean_actions=mean_actions,
                                                       log_std=self.log_std)
        log_prob = log_prob.reshape(-1, 1)
        entropy = self.dist.dist.entropy()
        entropy = entropy.reshape(-1, 1)
        return values, log_prob, entropy   # (batch_size, 1)


