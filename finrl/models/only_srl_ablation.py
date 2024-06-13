import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict,Tuple
import numpy as np
from finrl.models.ReplayBuffer import ReplayBuffer
from finrl.models.logger import episode_logger
from finrl.models.utils import polyak_update
from datetime import datetime
from finrl.models.metrics import *
from finrl.models.noise import get_noise
from finrl.models.mcts import *
from finrl.models.network import CNN,LSTM,DuelingMLP
from finrl.models.srl import *
from itertools import chain
import numpy as np


class DDPG_Policy_Only_SRL(nn.Module):
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
        super(DDPG_Policy_Only_SRL,self).__init__()

        self.mlp_activation = mlp_activation
        self.env_obs_dim = env_obs_dim  # 环境状态的维度
        self.action_dim = action_dim    # 动作维度
        self.srl_hidden_dim = srl_hidden_dim
        self._setup_model()

    def _setup_model(self):

        self.srl = D2RL(self.env_obs_dim, self.srl_hidden_dim, self.mlp_activation())
        
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

        action_feature_dim = 300
        self.action_net = nn.Sequential(
            *[
                nn.Linear(action_feature_dim,self.action_dim,bias=True),
                nn.Tanh()
            ]
        )

        critic_feature_dim = 300
        self.critic_net = nn.Linear(critic_feature_dim,1,bias=True)

        self.flatten = nn.Flatten(1,-1)

        self.f_pre_obs = nn.Linear(self.state_dim, self.action_dim, bias=True)
        self.f_pre_r = nn.Linear(self.state_dim, 1, bias=True)


    def get_action_only(self,env_state) -> torch.Tensor:
        # actor单独的通道，输出动作。在智能体和环境交互时用到。
        if len(env_state.shape) == 3:
            env_state = self.flatten(env_state)
        env_state = self.srl(env_state)
        action_feature = self.action_mlp(env_state)
        action = self.action_net(action_feature)
        return action

    def get_q_value_only(self,env_state,action) -> torch.Tensor:
        # critic单独的通道
        if len(env_state.shape) == 3:
            env_state = self.flatten(env_state)
        env_state = self.srl(env_state)
        critic_feature = self.critic_mlp(torch.cat([env_state,action],dim=1))
        q = self.critic_net(critic_feature)
        return q
    
    def forward(self,env_state) -> Tuple[torch.Tensor]:
        # 获得公共图片特征，据此获取动作价值预测，获取action_pre，获取critic预测的动作价值

        if len(env_state.shape) == 3:
            env_state = self.flatten(env_state)
        env_state = self.srl(env_state)
        # action部分
        action_feature = self.action_mlp(env_state)
        action_pre = self.action_net(action_feature)

        # critic部分
        critic_feature = self.critic_mlp(torch.cat([env_state,action_pre],dim=1)) # mlp部分特征
        q = self.critic_net(critic_feature)

        return action_pre,q


class DDPG():

    def __init__(
            self,
            env_train=None,
            env_validation=None,
            env_test=None,
            episodes: int = None,
            n_updates: int = None,
            buffer_size: int = None,
            batch_size: int = None,
            n_steps: int = None,
            if_prioritized: int = None, 
            tau: float = None,
            gamma: float = None,
            target_update_interval: float = None,
            policy_lr: float = None,  # 统一的学习率
            training_start: int = None,
            policy_kwargs: dict = None,
            noise_aliase: str = None,
            noise_kwargs: dict = None,
            print_interval: int = None,
            figure_path: str = None,
            device: str = None,
            task: str = None,
            train_time: int = '0',
            if_clip=False,
    ):

        self.env_train = env_train
        self.env_validation = env_validation
        self.env_test = env_test
        self.episodes = episodes
        self.n_updates = n_updates
        self.if_prioritized = if_prioritized
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.tau = tau
        self.gamma = gamma
        self.print_interval = print_interval
        self.training_start = training_start
        self.target_update_interval = target_update_interval
        self.figure_path = figure_path
        self.device = device
        self.task = task
        self.buffer = ReplayBuffer(buffer_capacity=buffer_size,
                                   batch_size=batch_size,
                                   n_steps=n_steps,
                                   gamma=gamma,
                                   if_prioritized=if_prioritized,
                                   device=device)

        self.policy = DDPG_Policy_Only_SRL(**policy_kwargs).to(self.device)
        self.policy_target = DDPG_Policy_Only_SRL(**policy_kwargs).to(self.device)
        self.policy_target.load_state_dict(self.policy.state_dict())

        # 优化器和学习率

        # 分开优化。
        self.parameters_srl = chain(self.policy.srl.parameters(), self.policy.f_pre_obs.parameters(),self.policy.f_pre_r.parameters()) 
        self.policy.optim_srl = torch.optim.Adam(self.parameters_srl,policy_lr)

        self.policy.optim_actor = torch.optim.Adam(
            torch.nn.ModuleList([
            self.policy.action_mlp, 
            self.policy.action_net,
            ]).parameters(),
            policy_lr
        )

        self.policy.optim_critic = torch.optim.Adam(
            torch.nn.ModuleList([
            self.policy.critic_mlp, 
            self.policy.critic_net,
            ]).parameters(),
            policy_lr
        )

        lambda1 = lambda epoch: 1
        lambda2 = lambda epoch: 1
        lambda3 = lambda epoch: 1
        self.policy.lr_scheduler_srl = torch.optim.lr_scheduler.LambdaLR(self.policy.optim_srl, lr_lambda=lambda1)
        self.policy.lr_scheduler_actor = torch.optim.lr_scheduler.LambdaLR(self.policy.optim_actor, lr_lambda=lambda2)
        self.policy.lr_scheduler_critic = torch.optim.lr_scheduler.LambdaLR(self.policy.optim_critic, lr_lambda=lambda3)

        self.noise = get_noise(noise_aliase, noise_kwargs)
        self.action_space_range = [self.env_train.action_space.low, self.env_train.action_space.high]
        # self.total_save = {}
        self.best_val_result = {'best val Sharpe ratio':-np.inf}
        self.best_test_result = {'best test Sharpe ratio':-np.inf}
        self.train_time = train_time
        self.if_clip = if_clip

    def replay(self) -> None:
        self.actor_loss_list_once_replay = []
        self.critic_loss_list_once_replay = []
        self.srl_loss_list_once_replay = []
        for _ in range(self.n_updates):
            # 准备数据
            data = self.buffer.sample()
            # figures1, figures2, figures3 = get_figure(self.figure_path, data.dates, self.device)
            # next_figures1, next_figures2, next_figures3  = get_figure(self.figure_path, data.next_dates, self.device)

            # 训练
            # srl训练
            if self.task == 'trading':
                next_obs_part = data.next_observations[:, 1:1 + self.env_train.action_dim]  # trading任务预测价格。
            else:
                next_obs_part = data.next_observations[:,self.env_train.action_dim,:]   # portfolio任务预测macd。
            state = self.policy.srl(data.observations)
            srl_loss = 0.5 * nn.functional.mse_loss(self.policy.f_pre_obs(state), next_obs_part) +\
                        0.5 * nn.functional.mse_loss(self.policy.f_pre_r(state), data.rewards)
            self.policy.optim_srl.zero_grad()
            srl_loss.backward()
            self.policy.optim_srl.step()
            self.policy.lr_scheduler_srl.step()

            # rl训练
            with torch.no_grad():
                next_actions,next_q_values = self.policy_target(data.next_observations)
                targets = data.rewards + self.gamma * (1 - data.dones) * next_q_values
            q_value_pres = self.policy.get_q_value_only(data.observations, data.actions)

            if self.if_prioritized:
                td_errors = (q_value_pres - targets).detach().numpy()
                alpha = self.buffer.alpha
                self.buffer.update_priorities([*zip(data.sample_idx, (abs(td_errors)**alpha))])
                weights = (data.sample_probs / min(data.sample_probs))**(-self.buffer.beta())
                assert weights.requires_grad == False
                critic_loss = (((q_value_pres - targets)**2) * (weights / 2)).mean()
            else:
                critic_loss = nn.functional.mse_loss(q_value_pres, targets)
                self.policy.optim_critic.zero_grad()
                critic_loss.backward()
                self.policy.optim_critic.step()
                self.policy.lr_scheduler_critic.step()  

            action_pres, actor_loss = self.policy(data.observations)
            actor_loss = -actor_loss.mean() 
            self.policy.optim_actor.zero_grad() 
            actor_loss.backward()
            self.policy.optim_actor.step()
            self.policy.lr_scheduler_actor.step()  

            # loss = critic_loss + actor_loss + srl_loss if self.if_srl else critic_loss + actor_loss
            # loss.backward()
            if self.if_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            # self.policy.optim.step()
            # self.policy.lr_scheduler.step()
            self.logger.total_updates_plus(1)
            self.actor_loss_list_once_replay.append(actor_loss.cpu().detach().numpy())
            self.critic_loss_list_once_replay.append(critic_loss.cpu().detach().numpy())
            self.srl_loss_list_once_replay.append(srl_loss.cpu().detach().numpy())

    def train(self):
        self.logger = episode_logger()
        for ep in range(1, self.episodes + 1):
            self.logger.reset()
            s = self.env_train.reset()
            done = False
            while not done:
                # 获取状态
                s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)  # (,dim) -> (1,dim)
                date,next_date = str(self.env_train.date),str(self.env_train.next_date) 
                # figure1,figure2,figure3 = get_figure(self.figure_path, [date], self.device)   # (1, 4, 400, 240)
                # 获取动作
                a = self.policy.get_action_only(s_tensor).cpu().detach().numpy().reshape(-1)  # (action_dim,)
                a = np.clip(a + self.noise.__call__(), self.action_space_range[0], self.action_space_range[1])
                s_, r, done, _ = self.env_train.step(a)   
                self.buffer.add(s, a, r, s_, date, next_date, done)   # 添加数据
                s = s_
                self.logger.timesteps_plus()
                # training and update
                if self.logger.total_timesteps > self.batch_size and self.logger.total_timesteps > self.training_start:
                    self.replay()
                    if self.logger.total_timesteps // self.target_update_interval == 0:
                        polyak_update(self.policy.parameters(), self.policy_target.parameters(), self.tau)

                    date_list = self.env_train.date.split('-')
                    year, month, day = date_list[0], date_list[1], date_list[2]
                    self.logger.record(reward=r,
                                       asset=self.env_train.asset_memory[-1],
                                       time=f'{year[2:]}_{month}_{day}',
                                       actor_loss=np.mean(self.actor_loss_list_once_replay),
                                       critic_loss=np.mean(self.critic_loss_list_once_replay),
                                       srl_loss=np.mean(self.srl_loss_list_once_replay)
                                       )
                else:  # before training:
                    date_list = self.env_train.date.split('-')
                    year, month, day = date_list[0], date_list[1], date_list[2]
                    self.logger.record(reward=r,
                                       asset=self.env_train.asset_memory[-1],
                                       time=f'{year[2:]}_{month}_{day}',
                                       actor_loss=0,
                                       critic_loss=0,
                                       srl_loss=0,
                                       )
                if self.logger.timesteps % self.print_interval == 0:
                    self.logger.show(interval=self.print_interval)
            self.save_train_memory('train',ep,self.train_time)
            self.examine('validation')
            self.examine('test')
        print("当前时间：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # self.save_model()

    def save_train_memory(self,mode:str,episode,train_time):
        with open(train_time+mode+'_date.txt','w') as f:
            f.write(str(episode)+'\n\n')
            for line in self.env_train.date_memory:
                f.write(str(line)+'\n')
            f.write('\n')
        with open(train_time+mode+'_asset.txt','w') as f:
            f.write(str(episode)+'\n\n')
            for line in self.env_train.asset_memory:
                f.write(str(line)+'\n')
            f.write('\n')
        with open(train_time+mode+'_reward.txt','w') as f:
            f.write(str(episode)+'\n\n')
            for line in self.env_train.rewards_memory:
                f.write(str(line)+'\n')
            f.write('\n')
            
        with open(train_time+mode+'_actor_loss.txt','w') as f:
            f.write(str(episode)+'\n\n')
            star_index = self.logger.record_dict['actor_loss'].index('*')
            self.logger.record_dict['actor_loss'].pop(star_index)
            for line in self.logger.record_dict['actor_loss']:
                f.write(str(line)+'\n')
            f.write('\n')
        with open(train_time+mode+'_critic_loss.txt','w') as f:
            f.write(str(episode)+'\n\n')
            star_index = self.logger.record_dict['critic_loss'].index('*')
            self.logger.record_dict['critic_loss'].pop(star_index)
            for line in self.logger.record_dict['critic_loss']:
                f.write(str(line)+'\n')
            f.write('\n')
        with open(train_time+mode+'_srl_loss.txt','w') as f:
            f.write(str(episode)+'\n\n')
            star_index = self.logger.record_dict['srl_loss'].index('*')
            self.logger.record_dict['srl_loss'].pop(star_index)
            for line in self.logger.record_dict['srl_loss']:
                f.write(str(line)+'\n')
            f.write('\n')
                        
    def save_model(self):
        # 保存模型，保存环境的date,action_memory,reward_memory,asset_memory
        checkpoint = {'policy_state_dict': self.policy.state_dict()}
        name = self.train_time + '_' + 'best' + '_' +   'model.pth'
        torch.save(checkpoint,name)

    def load_actor(self,path):
        return self.policy.load_state_dict(torch.load(path)['policy_state_dict'])
    
    def examine(self,mode:str):
        self.policy.eval()
        if mode == 'validation':
            env = self.env_validation
        else:
            env = self.env_test
        s = env.reset()
        done = False
        while not done:
            # interact
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
            date,next_date = str(env.date),str(env.next_date) 
            # figure1,figure2,figure3 = get_figure(self.figure_path, [date], self.device)   # (1, 4, 400, 240)
            a = self.policy.get_action_only(s_tensor).cpu().detach().numpy().reshape(-1)
            s_, r, done, _ = env.step(a)   # ndarray,float,bool,dict
            s = s_
        returns = env.returns
        total_asset_ = env.asset_memory[-1]
        cummulative_returns_ = cum_returns_final(returns)
        annual_return_ = annual_return(returns)
        sharpe_ratio_ = sharpe_ratio(returns)
        max_drawdown_ = max_drawdown(returns)

        # 不管test还是validation，都打印
        print(f'++++++++++++++ {mode} result +++++++++++++++')
        print(f'{mode} date range: {env.DATE_START} -- {env.DATE_END}')
        print(f'Total asset: {total_asset_}')
        print(f'Cumulative returns: {cummulative_returns_}')
        print(f'Annual return: {annual_return_}')
        print(f'Sharpe ratio: {sharpe_ratio_}')
        print(f'Max drawdown: {max_drawdown_}')
        print('++++++++++++++++++++++++++++++++++++++++++++++++')

        if mode == 'validation':  # 保存最好的验证集结果对应的模型，并保存相应的最好验证集结果
            if self.best_val_result['best val Sharpe ratio'] < sharpe_ratio_:
                self.best_val_result.update({
                    'best val Total asset':total_asset_,
                    'best val Cumulative returns':cummulative_returns_,
                    'best val Annual return':annual_return_,
                    'best val Sharpe ratio':sharpe_ratio_,
                    'best val Max drawdown':max_drawdown_,
                    'best val episode':self.logger.episode,
                })
                # self.save_model()
            print(f'+++++++++++++ best validation result +++++++++++++')
            print(self.best_val_result)
            print('++++++++++++++++++++++++++++++++++++++++++++++++')


        if mode == 'test':  
            if self.best_test_result['best test Sharpe ratio'] < sharpe_ratio_:
                self.best_test_result.update({
                    'best test Total asset':total_asset_,
                    'best test Cumulative returns':cummulative_returns_,
                    'best test Annual return':annual_return_,
                    'best test Sharpe ratio':sharpe_ratio_,
                    'best test Max drawdown':max_drawdown_,
                    'best test episode':self.logger.episode,
                })
                self.save_model()
            print(f'+++++++++++++ best test result +++++++++++++')
            print(self.best_test_result)
            print('++++++++++++++++++++++++++++++++++++++++++++++++')



        if mode == 'validation':
            self.env_validation = env
        else:
            self.env_test = env
        self.policy.train()