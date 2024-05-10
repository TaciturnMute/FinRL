import torch
from torch import nn
from typing import Type,Tuple


class Critic(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 activation_fn: Type[nn.Module],
                 n_critics:int = 2,
                 dropout: float = 0,
                 ):
        super(Critic,self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_critics = n_critics
        self.activation_fn = activation_fn
        self._setup_model()

    def _setup_model(self):
        # critic model
        for idx in range(1, self.n_critics+1):
            qf = nn.Sequential(
                *[
                    nn.Linear(self.state_dim+self.action_dim,256,bias=True),
                    self.activation_fn(),
                    nn.Linear(256,256,bias=True),
                    self.activation_fn(),
                    nn.Linear(256,1,bias=True)
                ]
            )
            self.add_module(f'qf{idx}',qf)
        self.flatten = nn.Flatten(1,-1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if len(obs.shape)==3:
            obs = self.flatten(obs)
        inputs = torch.cat((obs,actions), dim=1)
        return (self.qf1(inputs), self.qf2(inputs))

    def get_q_values(self, obs: torch.Tensor, actions: torch.Tensor):
        if len(obs.shape)==3:
            obs = self.flatten(obs)
        return self(obs, actions)

