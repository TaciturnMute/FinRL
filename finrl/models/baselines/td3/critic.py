import torch 
from torch import nn
from typing import Tuple,Type


class Critic(nn.Module):
    def __init__(self,
                 state_dim: int = None,
                 action_dim: int = None,
                 activation_fn: Type[nn.Module] = None,
                 n_critics: int = 2,
                 dropout: float = 0,
                 ):

        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation_fn = activation_fn
        self.n_critics = n_critics
        self._setup_model()

    def _setup_model(self):
        for idx in range(1, self.n_critics+1):
            qf = nn.Sequential(
                *[
                   nn.Linear(self.state_dim+self.action_dim,400),
                   self.activation_fn(),
                   nn.Linear(400,300),
                   self.activation_fn(),
                   nn.Linear(300,1)
                ]
            )
            self.add_module(f'qf{idx}',qf)
        self.flatten = nn.Flatten(1,-1)

    def get_qf1_value(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:

        if len(obs.shape)==3:
            obs = self.flatten(obs)

        inputs = torch.cat((obs, actions), dim=1)

        return self.qf1(inputs)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(obs.shape)==3:
            obs = self.flatten(obs)
        input_ = torch.cat((obs, actions), dim=1)
        q1 = self.qf1(input_)
        q2 = self.qf2(input_)
        return (q1, q2)

