import torch
from torch import nn
import numpy as np
from finrl.models.baselines.sac.actor import Actor
from finrl.models.baselines.sac.critic import Critic
from finrl.models.replay_buffer import ReplayBuffer
from finrl.models.logger import episode_logger
from finrl.models.metrics import *
from finrl.models.utils import polyak_update
from datetime import datetime


class SAC():
    def __init__(
            self,
            env_train,
            env_validation,
            env_test,
            episodes: int = 10,
            n_updates: int = None,
            buffer_size: int = 100000,
            batch_size: int = 100,
            actor_lr: float = 3e-4,
            critic_lr: float = 3e-4,
            training_start: int = 100,
            target_copy_interval: int = 1,
            policy_update_delay: int = 1,
            gamma: float = 0.99,
            tau: float = 0.005,
            init_value: float = 1.0,
            ent_coef_lr: float = 3e-4,
            auto_ent_coef: bool = True,
            critic_kwargs: dict = None,
            actor_kwargs: dict = None,
            print_interval: int = 100,
            train_time: str = '0',
            device: str = 'cuda',
    ):
        self.env_train = env_train
        self.env_validation = env_validation
        self.env_test = env_test
        self.gamma = gamma
        self.batch_size = batch_size
        self.episodes = episodes
        self.n_updates = n_updates
        self.tau = tau
        self.training_start = training_start
        self.policy_update_delay = policy_update_delay  # actor update delay
        self.target_copy_interval = target_copy_interval  # target_net update delay
        self.init_value = init_value
        self.auto_ent_coef = auto_ent_coef
        self.print_interval = print_interval

        self.critic = Critic(**critic_kwargs).to(device)
        self.critic_target = Critic(**critic_kwargs).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optim1 = torch.optim.Adam(self.critic.qf1.parameters(),critic_lr)
        self.critic.optim2 = torch.optim.Adam(self.critic.qf2.parameters(),critic_lr)
        self.critic.lr_scheduler1 = torch.optim.lr_scheduler.ConstantLR(self.critic.optim1,factor=1)
        self.critic.lr_scheduler2 = torch.optim.lr_scheduler.ConstantLR(self.critic.optim2,factor=1)
        self.actor = Actor(**actor_kwargs).to(device)
        self.actor.optim = torch.optim.Adam(self.actor.parameters(),actor_lr)
        self.actor.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.actor.optim,factor=1)
        self.buffer = ReplayBuffer(buffer_capacity=buffer_size,
                                   batch_size=batch_size,
                                   gamma=gamma,
                                   device=device)
        self.action_range = [self.env_train.action_space.low,self.env_train.action_space.high]
        self.n_updates_now = 0

        if self.auto_ent_coef:
            self.target_entropy = -np.prod(self.env_train.action_space.shape).astype(np.float32)  # float
            # 注意这里的device放置的位置。
            self.log_ent_coef = torch.log(torch.ones(1,device=device) * self.init_value).requires_grad_(True) # torch.float
            self.ent_coef_optim = torch.optim.Adam([self.log_ent_coef], ent_coef_lr)
        else:
            self.log_ent_coef = torch.log(torch.ones(1) * init_value).to(device)  # torch.float
            self.target_entropy, self.ent_coef_optim = None, None
        self.best_val_result = {'best val Sharpe ratio':-np.inf}
        self.best_test_result = {'best test Sharpe ratio':-np.inf}
        self.train_time = train_time
        self.device = device

    def replay(self) -> None:
        self.actor_loss_list_once_replay = []
        self.critic_loss_list_once_replay = []
        self.ent_coef_loss_list_once_replay = []
        self.ent_coef_list_once_replay = []
        for _ in range(self.n_updates):
            # get one batch samples:
            data = self.buffer.sample()
            # I:update ent_coef
            if self.auto_ent_coef:
                actions_pi, log_prob = self.actor.actions_log_prob(data.observations) # pi suffix, selected by current policy
                # assert actions_pi.shape==(self.batch_size, self.actor.action_dim)
                # assert log_prob.shape==(self.batch_size, 1)
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optim.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optim.step()
            else:
                ent_coef = torch.exp(self.log_ent_coef.detach())
            # II:update critic
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.actions_log_prob(data.next_observations)
                next_q_values = torch.cat(self.critic_target(data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob
                targets = data.rewards + self.gamma * (1 - data.dones) * (next_q_values)
            current_q_values = self.critic(data.observations, data.actions)
            critic_loss = 0.5 * sum(nn.functional.mse_loss(current_q_value, targets) for current_q_value in current_q_values)
            self.critic.optim1.zero_grad()
            self.critic.optim2.zero_grad()
            critic_loss.backward()
            self.critic.optim1.step()
            self.critic.lr_scheduler1.step()
            self.critic.optim2.step()
            self.critic.lr_scheduler2.step()

            # III:update actor
            if self.logger.total_updates % self.policy_update_delay == 0:
                q_values_pi = torch.cat(self.critic(data.observations, actions_pi), dim=1)
                min_q_values_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
                # assert min_q_values_pi.shape == log_prob.shape == (self.batch_size, 1)
                actor_loss = (ent_coef * log_prob - min_q_values_pi).mean()
                self.actor.optim.zero_grad()
                actor_loss.backward()
                self.actor.optim.step()
                self.actor_loss_list_once_replay.append(actor_loss.cpu().detach().numpy())
            if self.logger.total_updates % self.target_copy_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            self.critic_loss_list_once_replay.append(critic_loss.cpu().detach().numpy())
            self.ent_coef_list_once_replay.append(ent_coef.cpu().numpy())
            self.ent_coef_loss_list_once_replay.append(ent_coef_loss.cpu().detach().numpy())
            self.logger.total_updates_plus()

    def train(self):
        self.logger = episode_logger()
        for ep in range(self.episodes):
            self.logger.reset()
            done = False
            s = self.env_train.reset()
            while not done:
                s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
                a = self.actor.get_actions(s_tensor, False).cpu().detach().numpy().reshape(-1, )  
                s_, r, done, _ = self.env_train.step(a)   
                self.buffer.add(s, a, r, s_, None, None, done)   # np.ndarray,np.ndarray,scale,list,bool
                s = s_
                self.logger.timesteps_plus()

                if self.logger.total_timesteps > self.training_start and self.logger.total_timesteps > self.batch_size:
                    self.replay()
                    date_list = self.env_train.date.split('-')
                    year, month, day = date_list[0], date_list[1], date_list[2]
                    self.logger.record(reward=r,
                                       asset=self.env_train.asset_memory[-1],
                                       time=f'{year[2:]}_{month}_{day}',
                                       critic_loss=np.mean(self.critic_loss_list_once_replay),
                                       )
                    if len(self.actor_loss_list_once_replay) > 0:
                        # in case in one replay, actor can not be updated
                        self.logger.record(actor_loss=np.mean(self.actor_loss_list_once_replay))
                else:
                    date_list = self.env_train.date.split('-')
                    year, month, day = date_list[0], date_list[1], date_list[2]
                    self.logger.record(reward=r,
                                       asset=self.env_train.asset_memory[-1],
                                       time=f'{year[2:]}_{month}_{day}',
                                       actor_loss=0,
                                       critic_loss=0,
                                       )
                if  self.logger.timesteps % self.print_interval == 0 and self.logger.timesteps > 0:
                    self.logger.show(interval=self.print_interval)
            self.save_train_memory('train',ep, self.train_time)
            self.examine('validation')
            self.examine('test')
        print("当前时间：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def save_train_memory(self,mode: str, episode: int, train_time: str):
        with open(train_time + mode + '_date.txt','a') as f:
            f.write(str(episode)+'\n\n')
            for line in self.env_train.date_memory:
                f.write(str(line)+'\n')
            f.write('\n')
        with open(train_time + mode +'_asset.txt','a') as f:
            f.write(str(episode)+'\n\n')
            for line in self.env_train.asset_memory:
                f.write(str(line)+'\n')
            f.write('\n')
        with open(train_time + mode + '_reward.txt','a') as f:
            f.write(str(episode)+'\n\n')
            for line in self.env_train.rewards_memory:
                f.write(str(line)+'\n')
            f.write('\n')
        with open(train_time + mode + '_actor_loss.txt','a') as f:
            f.write(str(episode)+'\n\n')
            star_index = self.logger.record_dict['actor_loss'].index('*')
            self.logger.record_dict['actor_loss'].pop(star_index)
            for line in self.logger.record_dict['actor_loss']:
                f.write(str(line)+'\n')
            f.write('\n')
        with open(train_time + mode + '_critic_loss.txt','a') as f:
            f.write(str(episode)+'\n\n')
            star_index = self.logger.record_dict['critic_loss'].index('*')
            self.logger.record_dict['critic_loss'].pop(star_index)
            for line in self.logger.record_dict['critic_loss']:
                f.write(str(line)+'\n')
            f.write('\n')
                        
    def save_model(self):
        # 保存模型，保存环境的date,action_memory,reward_memory,asset_memory
        checkpoint = {'actor_state_dict': self.actor.state_dict(),
                      'critic_state_dict': self.critic.state_dict()}
        name = self.train_time + '_' + 'best' + '_' +   'model.pth'
        torch.save(checkpoint,name)
        # self.logger.save(self.filename)

    def load_actor(self, path):
        return self.actor.load_state_dict(torch.load(path,map_location=torch.device('cpu'))['actor_state_dict'])
    
    def examine(self,mode:str):
        self.actor.eval()
        if mode == 'validation':
            env = self.env_validation
        else:
            env = self.env_test
        s = env.reset()
        done = False
        while not done:
            # interact
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)  # add batch dim
            a = self.actor.get_actions(s_tensor,True).cpu().detach().numpy().reshape(-1)  # (action_dim,)
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
        self.actor.train()

