import torch
from torch import nn


class D2RL(nn.Module):
    def __init__(self,
                 env_state_dim,
                 hidden_dim,
                 activation_fn,
                 ):
        super(D2RL,self).__init__()
        self.env_state_dim = env_state_dim
        self.hidden_dim = hidden_dim
        self.activation_fn = activation_fn
        self._setup()
    
    def _setup(self):
        self.layer1 = nn.Linear(self.env_state_dim,self.hidden_dim,bias=True)
        self.layer2 = nn.Linear(self.hidden_dim+self.env_state_dim, self.hidden_dim, bias=True)
        self.last_hidden_dim = self.hidden_dim + self.env_state_dim
        self.flatten = nn.Flatten(1,-1)

    def forward(self,obs):
        if len(obs.shape)==3:
            obs = self.flatten(obs)
        x = self.activation_fn(self.layer1(obs))
        x = torch.concat([x,obs],dim=1)
        x = self.activation_fn(self.layer2(x))
        x = torch.concat([x,obs],dim=1)
        x = self.activation_fn(x)
        return x        
        

class OFENet(nn.Module):
    def __init__(self,
                 env_state_dim,
                 hidden_dim,
                 activation_fn,
                 ):
        super(OFENet,self).__init__()
        self.env_state_dim = env_state_dim
        self.hidden_dim = hidden_dim
        self.activation_fn = activation_fn
        self._setup()
    
    def _setup(self):
        self.layer1 = nn.Linear(self.env_state_dim, self.hidden_dim, bias=True)
        self.layer2 = nn.Linear(self.env_state_dim + self.hidden_dim, self.hidden_dim, bias=True)
        self.flatten = nn.Flatten(1,-1)
        self.last_hidden_dim = self.hidden_dim*2 + self.env_state_dim
    
    def forward(self,obs):
        if len(obs.shape)==3:
            obs = self.flatten(obs)
        # 第一层
        out = self.layer1(obs)
        out = self.activation_fn(out)

        input = torch.concat([out,obs],dim=1)

        # 第二层
        out = self.layer2(input)
        out = self.activation_fn(out)

        out = torch.concat([out,input],dim=1)
        out = self.activation_fn(out)
        return out
        


class DenseNet(nn.Module):
    def __init__(self,
                 env_state_dim,
                 hidden_dim,
                 activation_fn,
                 ):
        super(DenseNet,self).__init__()
        self.env_state_dim = env_state_dim
        self.hidden_dim = hidden_dim
        self.activation_fn = activation_fn
        self._setup()
    
    def _setup(self):
        self.layer1 = nn.Linear(self.env_state_dim, self.hidden_dim, bias=True)
        self.layer2 = nn.Linear(self.env_state_dim + self.hidden_dim, self.hidden_dim, bias= True)
        self.flatten = nn.Flatten(1,-1)
        self.last_hidden_dim = (self.env_state_dim + self.hidden_dim)*2
        
    
    def forward(self,obs):
        if len(obs.shape)==3:
            obs = self.flatten(obs)
        # 第一层
        out = self.layer1(obs)
        out = self.activation_fn(out)  # hidden_dim
        input2 = torch.concat([out,obs],dim=1)  # env_state_dim + hidden_dim
        out = self.layer2(input2)  
        out = self.activation_fn(out) # hidden_dim
        out = torch.concat([out,input2,obs],dim=1)  # 2*env_state_dim + 2*hidden_dim
        out = self.activation_fn(out)
        return out

        