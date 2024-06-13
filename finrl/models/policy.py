import torch
from torch import nn
from finrl.models.network import CNN,LSTM,DuelingMLP
from finrl.models.srl import *
import torch.nn.functional as F
import numpy as np
from typing import Tuple


'''
按照论文设置, 图像尺寸为 4, 400, 240

图片首先送入CNN, 输出维度为 64, 11, 4

其次, 将特征图案列排列，多个通道的列合并，得到维度 4, 704

lstm处理, 最后时刻隐状态维度为: , 1024。上述过程如果添加batch size, 则最后lstm输出维度为(batch size, 1024)

三类图片的lstm隐状态合并, 维度为 3072.
'''

class DDPG_Policy_MCTS(nn.Module):
    # 用于第四章REDQ方法的policy
    def __init__(
            self,
            N,
            M,
            cnn_activation,
            lstm_input_size,
            lstm_hidden_size,
            env_obs_dim,
            action_dim,
            mlp_activation,
            srl_aliase,
            srl_hidden_dim,
            ):
        super(DDPG_Policy_MCTS,self).__init__()
        self.N = N
        self.M = M
        # self.if_redq = if_redq  # 是否进行集成
        self.cnn_activation = cnn_activation
        self.mlp_activation = mlp_activation
        self.lstm_input_size = lstm_input_size    # istm输入尺度
        self.lstm_hidden_size = lstm_hidden_size  # lstm隐状态维度
        self.env_obs_dim = env_obs_dim  # 环境状态的维度
        self.action_dim = action_dim    # 动作维度
        self.srl_aliase = srl_aliase
        self.srl_hidden_dim = srl_hidden_dim
        self._setup_model()


    def _setup_model(self):
        ACTIVATION_FUNCTIONS = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
            # 你可以在这里添加更多的激活函数
        }
        self.cnn_activation = ACTIVATION_FUNCTIONS[self.cnn_activation]
        self.mlp_activation = ACTIVATION_FUNCTIONS[self.mlp_activation]
        # 创建CNN，LSTM，actor和critic模块
        self.cnn1 = CNN(self.cnn_activation)  
        self.lstm1 = LSTM(self.lstm_input_size, self.lstm_hidden_size)  
        self.cnn2 = CNN(self.cnn_activation)
        self.lstm2 = LSTM(self.lstm_input_size,self.lstm_hidden_size)
        self.cnn3 = CNN(self.cnn_activation)
        self.lstm3 = LSTM(self.lstm_input_size,self.lstm_hidden_size)

        # srl部分
        if self.srl_aliase == 'd2rl':
            self.srl = D2RL(self.env_obs_dim, self.srl_hidden_dim, self.mlp_activation())
        elif self.srl_aliase == 'ofenet':
            self.srl = OFENet(self.env_obs_dim, self.srl_hidden_dim, self.mlp_activation())
        else:
            assert self.srl_aliase == 'densenet'
            self.srl = DenseNet(self.env_obs_dim, self.srl_hidden_dim, self.mlp_activation())
        
        self.state_dim = self.srl.last_hidden_dim
        self.flatten = nn.Flatten(1,-1)
        self.f_pre_obs = nn.Linear(self.state_dim, self.action_dim, bias=True)
        self.f_pre_r = nn.Linear(self.state_dim, 1, bias=True)

        # critic 部分
        self.critic_mlp_list = []
        self.critic_net_list = []
        critic_feature_dim = self.lstm_hidden_size * 3 + 300

        for idx in range(1, self.N + 1):
            critic_mlp = nn.Sequential(
                *[
                    nn.Linear(self.state_dim + self.action_dim, 400, bias=True),
                    self.mlp_activation(),
                    nn.Linear(400, 300, bias=True),
                    self.mlp_activation(),
                ]
            )
            critic_net = nn.Linear(critic_feature_dim,1,bias=True)
            self.critic_mlp_list.append(critic_mlp)
            self.critic_net_list.append(critic_net)
            self.add_module(f'critic_mlp{idx}', critic_mlp)
            self.add_module(f'critic_net{idx}', critic_net)

        # actor部分
        self.action_mlp = nn.Sequential(
            *[
                nn.Linear(self.state_dim, 400, bias=True),
                self.mlp_activation(),
                nn.Linear(400,300,bias=True),
                self.mlp_activation(),
            ]
        )

        action_feature_dim = self.lstm_hidden_size * 3+300

        self.action_net = nn.Sequential(
            *[
                nn.Linear(action_feature_dim,self.action_dim,bias=True),
                nn.Tanh()
            ]
        )

    def precess_cnn_output(self,cnn_output) -> torch.Tensor:
        cnn_output = cnn_output.reshape(cnn_output.shape[0], cnn_output.shape[-1], -1)
        return cnn_output

    def get_figure_feature(self,figure1,figure2,figure3) -> torch.Tensor:
        # 处理图片部分
        bs = figure1.shape[0]
        cnn_output1 = self.precess_cnn_output(self.cnn1(figure1))   # (bs, 64, 11, 4) -> (bs,4,704)
        cnn_output2 = self.precess_cnn_output(self.cnn1(figure2))   
        cnn_output3 = self.precess_cnn_output(self.cnn1(figure3))   
        _,(hn1,cn1) = self.lstm1(cnn_output1) # hn: (bs, 1, 1024)
        _,(hn2,cn2) = self.lstm2(cnn_output2)
        _,(hn3,cn3) = self.lstm3(cnn_output3)
        hn1,hn2,hn3 = hn1[:,0],hn2[:,0],hn3[:,0]  # (bs, 1, 1024) -> (bs, 1024)
        hidden = torch.concat([hn1,hn2,hn3],dim=1)
        return hidden

    def get_action_only(self,figure1,figure2,figure3,env_state) -> torch.Tensor:
        # 环境交互时用到
        figure_hidden = self.get_figure_feature(figure1,figure2,figure3)
        if len(env_state.shape) == 3:
            env_state = self.flatten(env_state)
        env_state = self.srl(env_state) # 一定有srl部分
        env_state_hidden = self.action_mlp(env_state)
        action_feature = torch.cat([figure_hidden,env_state_hidden],dim=1)
        action = self.action_net(action_feature)
        return action

    def get_q_value_only(self,figure1,figure2,figure3,env_state,action,indexs=None,states=None) -> torch.Tensor:
        # 计算q value pre时用到。
        figure_hidden = self.get_figure_feature(figure1,figure2,figure3)
        if states is None:
            env_state = self.srl(env_state).detach()   
        else:
            env_state = states

        q_values_list = []
        for i in indexs:
            env_state_action_hidden = self.critic_mlp_list[i](torch.cat([env_state,action],dim=1))
            critic_feature = torch.cat([figure_hidden,env_state_action_hidden],dim=1)
            q_values_list.append(self.critic_net_list[i](critic_feature))  # ()
        return q_values_list,figure_hidden

    
    def forward(
        self,
        figure1,
        figure2,
        figure3,
        env_state=None,
        figure_hidden=None,
        indexs=None,
        states=None,
        ) -> Tuple[torch.Tensor]:

        # 用于计算TD Target和actor loss。
        # 前者需要传入M indexs，后者使用全部的q函数。

        # 公共特征
        if figure_hidden is None:  # 减少不必要的运算
            figure_hidden = self.get_figure_feature(figure1,figure2,figure3)
        if states is None:
            env_state = self.srl(env_state)  # srl内部会flatten
        else:
            env_state = states

        # action部分
        env_state_hidden = self.action_mlp(env_state)
        action_feature = torch.cat([figure_hidden,env_state_hidden],dim=1)
        action_pre = self.action_net(action_feature)

        # critic部分
        q_values_list = []
        for i in indexs:
            env_state_action_hidden = self.critic_mlp_list[i](torch.cat([env_state,action_pre],dim=1)) # mlp部分特征
            critic_feature = torch.cat([figure_hidden,env_state_action_hidden],dim=1) 
            q_values_list.append(self.critic_net_list[i](critic_feature))

        return action_pre,q_values_list




class DDPG_Policy(nn.Module):
    def __init__(
            self,
            cnn_activation,
            lstm_input_size,
            lstm_hidden_size,
            env_obs_dim,
            action_dim,
            mlp_activation,
            if_srl,
            srl_aliase,
            srl_hidden_dim,
            ):
        super(DDPG_Policy,self).__init__()
        self.cnn_activation = cnn_activation
        self.mlp_activation = mlp_activation
        self.lstm_input_size = lstm_input_size    # istm输入尺度
        self.lstm_hidden_size = lstm_hidden_size  # lstm隐状态维度
        self.env_obs_dim = env_obs_dim  # 环境状态的维度
        self.action_dim = action_dim    # 动作维度
        self.if_srl = if_srl
        self.srl_aliase = srl_aliase
        self.srl_hidden_dim = srl_hidden_dim
        self._setup_model()

    def _setup_model(self):
        ACTIVATION_FUNCTIONS = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
            # 你可以在这里添加更多的激活函数
        }
        self.cnn_activation = ACTIVATION_FUNCTIONS[self.cnn_activation]
        self.mlp_activation = ACTIVATION_FUNCTIONS[self.mlp_activation]
        
        # 创建CNN，LSTM，actor和critic模块
        self.cnn1 = CNN(self.cnn_activation)  
        self.lstm1 = LSTM(self.lstm_input_size, self.lstm_hidden_size)  
        self.cnn2 = CNN(self.cnn_activation)
        self.lstm2 = LSTM(self.lstm_input_size,self.lstm_hidden_size)
        self.cnn3 = CNN(self.cnn_activation)
        self.lstm3 = LSTM(self.lstm_input_size,self.lstm_hidden_size)

        if not self.if_srl:
            self.action_mlp = nn.Sequential(
                *[
                    nn.Linear(self.env_obs_dim,400,bias=True),
                    self.mlp_activation(),
                    nn.Linear(400,300,bias=True),
                    self.mlp_activation(),
                ]
            )
            self.critic_mlp = nn.Sequential(
                *[
                    nn.Linear(self.env_obs_dim+self.action_dim,400,bias=True),
                    self.mlp_activation(),
                    nn.Linear(400,300,bias=True),
                    self.mlp_activation(),
                ]
            )

            action_feature_dim = self.lstm_hidden_size*3+300
            self.action_net = nn.Sequential(
                *[
                    nn.Linear(action_feature_dim,self.action_dim,bias=True),
                    nn.Tanh()
                ]
            )

            critic_feature_dim = self.lstm_hidden_size*3+300
            self.critic_net = nn.Linear(critic_feature_dim,1,bias=True)

            self.flatten = nn.Flatten(1,-1)
        
        else:
            if self.srl_aliase == 'd2rl':
                self.srl = D2RL(self.env_obs_dim, self.srl_hidden_dim, self.mlp_activation())
            elif self.srl_aliase == 'ofenet':
                self.srl = OFENet(self.env_obs_dim, self.srl_hidden_dim, self.mlp_activation())
            else:
                assert self.srl_aliase == 'densenet'
                self.srl = DenseNet(self.env_obs_dim, self.srl_hidden_dim, self.mlp_activation())
            
            self.state_dim = self.srl.last_hidden_dim

            self.action_mlp = nn.Sequential(
                *[
                    nn.Linear(self.state_dim, 400, bias=True),
                    self.mlp_activation(),
                    nn.Linear(400,300,bias=True),
                    self.mlp_activation(),
                ]
            )
            self.critic_mlp = nn.Sequential(
                *[
                    nn.Linear(self.state_dim + self.action_dim, 400, bias=True),
                    self.mlp_activation(),
                    nn.Linear(400, 300, bias=True),
                    self.mlp_activation(),
                ]
            )

            action_feature_dim = self.lstm_hidden_size * 3+300
            self.action_net = nn.Sequential(
                *[
                    nn.Linear(action_feature_dim,self.action_dim,bias=True),
                    nn.Tanh()
                ]
            )

            critic_feature_dim = self.lstm_hidden_size * 3+300
            self.critic_net = nn.Linear(critic_feature_dim,1,bias=True)

            self.flatten = nn.Flatten(1,-1)

            self.f_pre_obs = nn.Linear(self.state_dim, self.action_dim, bias=True)
            self.f_pre_r = nn.Linear(self.state_dim, 1, bias=True)


    def precess_cnn_output(self,cnn_output) -> torch.Tensor:
        cnn_output = cnn_output.reshape(cnn_output.shape[0], cnn_output.shape[-1], -1)
        return cnn_output

    def get_figure_feature(self,figure1,figure2,figure3) -> torch.Tensor:
        # 处理图片部分
        bs = figure1.shape[0]
        cnn_output1 = self.precess_cnn_output(self.cnn1(figure1))   # (bs, 64, 11, 4) -> (bs,4,704)
        cnn_output2 = self.precess_cnn_output(self.cnn1(figure2))   
        cnn_output3 = self.precess_cnn_output(self.cnn1(figure3))   
        _,(hn1,cn1) = self.lstm1(cnn_output1) # hn: (bs, 1, 1024)
        _,(hn2,cn2) = self.lstm2(cnn_output2)
        _,(hn3,cn3) = self.lstm3(cnn_output3)
        hn1,hn2,hn3 = hn1[:,0],hn2[:,0],hn3[:,0]  # (bs, 1, 1024) -> (bs, 1024)
        hidden = torch.concat([hn1,hn2,hn3],dim=1)
        return hidden

    def get_action_only(self,figure1,figure2,figure3,env_state) -> torch.Tensor:
        # actor单独的通道，输出动作。在智能体和环境交互时用到。
        figure_hidden = self.get_figure_feature(figure1,figure2,figure3)
        if len(env_state.shape) == 3:
            env_state = self.flatten(env_state)
        if self.if_srl:
            env_state = self.srl(env_state)
        env_state_hidden = self.action_mlp(env_state)
        action_feature = torch.cat([figure_hidden,env_state_hidden],dim=1)
        action = self.action_net(action_feature)
        return action

    def get_q_value_only(self,figure1,figure2,figure3,env_state,action,states=None) -> torch.Tensor:
        # 计算q value预测值时用到。
        figure_hidden = self.get_figure_feature(figure1,figure2,figure3)
        if len(env_state.shape) == 3:
            env_state = self.flatten(env_state)
        if self.if_srl:
            if states is None:
                env_state = self.srl(env_state)
            else:
                env_state = states
        env_state_action_hidden = self.critic_mlp(torch.cat([env_state,action],dim=1))
        critic_feature = torch.cat([figure_hidden,env_state_action_hidden],dim=1)
        q = self.critic_net(critic_feature)

        return q,figure_hidden
    
    def forward(self,figure1,figure2,figure3,env_state,figure_hidden=None,states=None) -> Tuple[torch.Tensor]:
        # 计算actor loss用到。分两步，首先利用actor计算action pre，其次利用critic计算action loss。
        # 返回action pre（实际上用不到）和q值（actor loss）

        # 公共特征
        if figure_hidden is None:  # 减少不必要的运算
            figure_hidden = self.get_figure_feature(figure1,figure2,figure3)

        if len(env_state.shape) == 3:
            env_state = self.flatten(env_state)
        if self.if_srl:
            if states is None:
                env_state = self.srl(env_state)
            else:
                env_state = states
        # action部分
        env_state_hidden = self.action_mlp(env_state)
        action_feature = torch.cat([figure_hidden,env_state_hidden],dim=1)
        action_pre = self.action_net(action_feature)

        # critic部分
        env_state_action_hidden = self.critic_mlp(torch.cat([env_state,action_pre],dim=1)) # mlp部分特征
        critic_feature = torch.cat([figure_hidden,env_state_action_hidden],dim=1) 
        q = self.critic_net(critic_feature)

        return action_pre,q
