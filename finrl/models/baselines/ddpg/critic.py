import torch
from torch import nn
from typing import List, Type


class Critic(nn.Module):
    '''
    Sequential MLP
    '''
    def __init__(self,
                 dropout: float = 0,
                 state_dim: int = None,
                 action_dim: int = None,
                 activation_fn: Type[nn.Module] = None,
                 ):
        super(Critic, self).__init__() 
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation_fn = activation_fn
        self.dropout = dropout
        self._setup_model()

    def _setup_model(self):
        self.qf1 = nn.Sequential(
            *[
                nn.Linear(self.state_dim+self.action_dim,400,bias=True),
                self.activation_fn(),
                nn.Linear(400,300,bias=True),
                self.activation_fn(),
                nn.Linear(300,1,bias=True)
            ]
        )
        self.flatten = nn.Flatten(1,-1)
        # print('he init!')
        # for name, param in self.qf1.named_parameters():
        #     if 'weight' in name:
        #         nn.init.kaiming_normal_(param, a=0, nonlinearity='relu')

        # print('xavier_uniform!')
        # for name, param in self.qf1.named_parameters():
        #     if 'weight' in name:
        #         nn.init.xavier_uniform_(param)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if len(obs.shape) == 3:
            obs = self.flatten(obs)
        inputs = torch.cat((obs, actions), dim=1)
        return self.qf1(inputs)
