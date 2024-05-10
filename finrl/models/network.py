import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self,activation):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(14,14), stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=3)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(7,7), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3,3), stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(32)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3,3), stride=2)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(32)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(3,3), stride=2)

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(64)
        self.activation = activation()

    
    def forward(self,x):
        # 支持 torch.float32
        # 每一个模块：卷积 BN 池化 激活
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.maxpool1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.maxpool2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.maxpool3(out)
        out = self.activation(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.maxpool4(out)
        out = self.activation(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.activation(out)

        return out
    
class LSTM(nn.Module):
    def __init__(self,input_size,units):
        super(LSTM,self).__init__()
        self.units = units
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size,units,num_layers=1,batch_first=True)
        self.layernorm = nn.LayerNorm(units)
    
    def forward(self,x):
        lstm_states,(hn,cn) = self.lstm(x)
        lstm_states = self.layernorm(lstm_states)
        hn = self.layernorm(hn.permute(1,0,2)) # 转换batch_size和num_layer维度
        cn = self.layernorm(cn.permute(1,0,2))

        return lstm_states,(hn,cn)
    

class DuelingMLP(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 dropout,
                 ):
        super(DuelingMLP,self).__init__()
        self.action_dim = action_dim
        self.fc_adv = nn.Linear(state_dim,256)
        self.adv_head = nn.Linear(256,action_dim)
        self.fc_v = nn.Linear(state_dim,256)
        self.v_head = nn.Linear(256,1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()


    def forward(self,state):
        adv = self.fc_adv(state)
        adv = self.activation(adv)
        adv = self.dropout(adv)
        adv = self.adv_head(adv)
        bs = state.shape[0]
        assert adv.shape == (bs,self.action_dim)

        v = self.fc_v(state)
        v = self.activation(v)
        v = self.dropout(v)
        v = self.v_head(v)

        adv = adv - torch.max(adv,dim=1,keepdim=True)[0]
        q = adv + v

        return q
        
      