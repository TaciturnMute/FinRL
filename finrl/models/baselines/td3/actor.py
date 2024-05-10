import torch
from torch import nn
from typing import Type

class Actor(nn.Module):
    '''
    Sequential MLP actor
    '''
    def __init__(self,
                 dropout: float = 0,
                 state_dim: int = None,
                 action_dim: int = None,
                 activation_fn: Type[nn.Module] = None,
                 ):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation_fn = activation_fn
        self.dropout = dropout
        self._setup_model()

    def _setup_model(self):
        self.mu = nn.Sequential(
            *[nn.Linear(self.state_dim,400,bias=True),
             self.activation_fn(),
             nn.Linear(400,300,bias=True),
             self.activation_fn(),
             nn.Linear(300,self.action_dim,bias=True),
             nn.Tanh()]
        )
        self.flatten = nn.Flatten(1,-1)

        # print('xavier_uniform!')
        # for name, param in self.mu.named_parameters():
        #     if 'weight' in name:
        #         nn.init.xavier_uniform_(param)

    def get_actions(self, obs: torch.Tensor, deterministic=True):
        if obs.shape == 3:
            obs = nn.Flatten(start_dim=1,end_dim=-1)
        return self(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if len(obs.shape) == 3:
            obs = self.flatten(obs)
        return self.mu(obs)

