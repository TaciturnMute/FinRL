import torch
import numpy as np
from torch import nn
from finrl.models.baselines.ddpg.actor import *
from finrl.models.baselines.ddpg.critic import *
from finrl.models.noise import get_noise
from finrl.models.utils import polyak_update
from finrl.models.replay_buffer import ReplayBuffer
from finrl.models.logger import episode_logger
from datetime import datetime
from finrl.models.metrics import *


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
            actor_lr: float = None,
            critic_lr: float = None,
            training_start: int = None,
            actor_kwargs: dict = None,
            critic_kwargs: dict = None,
            noise_aliase: str = None,
            noise_kwargs: dict = None,
            print_interval: int = None,
            train_time: int = '0',
            device: str = 'cuda',
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
        self.buffer = ReplayBuffer(buffer_capacity=buffer_size,
                                   batch_size=batch_size,
                                   n_steps=n_steps,
                                   gamma=gamma,
                                   if_prioritized=if_prioritized,
                                   device=device)
        self.actor = Actor(**actor_kwargs).to(device)
        self.actor_target = Actor(**actor_kwargs).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(**critic_kwargs).to(device)
        self.critic_target = Critic(**critic_kwargs).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor.optim = torch.optim.Adam(self.actor.parameters(),actor_lr)
        self.actor.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.actor.optim,factor=1)
        self.critic.optim = torch.optim.Adam(self.critic.parameters(),critic_lr)
        self.critic.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.critic.optim,factor=1)  # 之前弄错了，设置为了self.actor.optim
        self.noise = get_noise(noise_aliase, noise_kwargs)
        self.action_space_range = [self.env_train.action_space.low, self.env_train.action_space.high]
        # self.total_save = {}
        self.best_val_result = {'best val Sharpe ratio':-np.inf}
        self.best_test_result = {'best test Sharpe ratio':-np.inf}
        self.train_time = train_time
        self.device = device

    def replay(self) -> None:
        self.actor_loss_list_once_replay = []
        self.critic_loss_list_once_replay = []
        for _ in range(self.n_updates):
            data = self.buffer.sample()
            # I: update critic
            with torch.no_grad():
                next_actions = self.actor_target(data.next_observations)
                next_q_values = self.critic_target(data.next_observations, next_actions) # (batch_size, 1)
                targets = data.rewards + self.gamma * (1 - data.dones) * next_q_values
            q_value_pres = self.critic(data.observations, data.actions)

            if self.if_prioritized:
                td_errors = (q_value_pres - targets).detach().numpy()
                alpha = self.buffer.alpha
                self.buffer.update_priorities([*zip(data.sample_idx, (abs(td_errors)**alpha))])
                weights = (data.sample_probs / min(data.sample_probs))**(-self.buffer.beta())
                assert weights.requires_grad == False
                critic_loss = (((q_value_pres - targets)**2) * (weights / 2)).mean()
                # nn.functional.mse_loss(x,y) == ((x - y)**2).mean()
            else:
                critic_loss = nn.functional.mse_loss(q_value_pres, targets)

            self.critic.optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic.optim.step()
            self.critic.lr_scheduler.step()

            # II: update actor
            action_pres = self.actor(data.observations)
            actor_loss = -self.critic(data.observations, action_pres).mean()
            self.actor.optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor.optim.step()
            self.actor.lr_scheduler.step()

            self.logger.total_updates_plus()
            self.actor_loss_list_once_replay.append(actor_loss.cpu().detach().numpy())
            self.critic_loss_list_once_replay.append(critic_loss.cpu().detach().numpy())

    def train(self):
        self.logger = episode_logger()
        for ep in range(1, self.episodes + 1):
            self.logger.reset()
            s = self.env_train.reset()  
            done = False
            while not done:
                # interact
                s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)  # add batch dim
                a = self.actor(s_tensor).cpu().detach().numpy().reshape(-1)  # (action_dim,)
                a = np.clip(a + self.noise.__call__(), self.action_space_range[0], self.action_space_range[1])
                s_, r, done, _ = self.env_train.step(a)   # ndarray,float,bool,dict
                self.buffer.add(s, a, r, s_, None, None, done)
                s = s_
                self.logger.timesteps_plus()
                # training and update
                if self.logger.total_timesteps > self.batch_size and self.logger.total_timesteps > self.training_start:
                    self.replay()
                    polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                    polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

                    date_list = self.env_train.date.split('-')
                    year, month, day = date_list[0], date_list[1], date_list[2]
                    self.logger.record(reward=r,
                                       asset=self.env_train.asset_memory[-1],
                                       time=f'{year[2:]}_{month}_{day}',
                                       actor_loss=np.mean(self.actor_loss_list_once_replay),
                                       critic_loss=np.mean(self.critic_loss_list_once_replay),
                                       )
                else:  # before training:
                    date_list = self.env_train.date.split('-')
                    year, month, day = date_list[0], date_list[1], date_list[2]
                    self.logger.record(reward=r,
                                       asset=self.env_train.asset_memory[-1],
                                       time=f'{year[2:]}_{month}_{day}',
                                       actor_loss=0,
                                       critic_loss=0,
                                       )
                if self.logger.timesteps % self.print_interval == 0:
                    self.logger.show(interval=self.print_interval)
            # self.logger.episode_results_sample()
            # self.total_save[f'episode{ep}'] = {'date':self.logger.record_dict['time'],'asset':self.logger.record_dict['asset'],'reward':self.logger.record_dict['reward'],'actor_loss':self.logger.record_dict['actor_loss'],'critic_loss':self.logger.record_dict['critic_loss']}
            self.save_train_memory('train',ep,self.train_time)
            self.examine('validation')
            self.examine('test')
        print("当前时间：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # self.save_model()

    def save_train_memory(self,mode:str,episode,train_time):
        with open(train_time+mode+'_date.txt','a') as f:
            f.write(str(episode)+'\n\n')
            for line in self.env_train.date_memory:
                f.write(str(line)+'\n')
            f.write('\n')
        with open(train_time+mode+'_asset.txt','a') as f:
            f.write(str(episode)+'\n\n')
            for line in self.env_train.asset_memory:
                f.write(str(line)+'\n')
            f.write('\n')
        with open(train_time+mode+'_reward.txt','a') as f:
            f.write(str(episode)+'\n\n')
            for line in self.env_train.rewards_memory:
                f.write(str(line)+'\n')
            f.write('\n')
            
        with open(train_time+mode+'_actor_loss.txt','a') as f:
            f.write(str(episode)+'\n\n')
            star_index = self.logger.record_dict['actor_loss'].index('*')
            self.logger.record_dict['actor_loss'].pop(star_index)
            for line in self.logger.record_dict['actor_loss']:
                f.write(str(line)+'\n')
            f.write('\n')
        with open(train_time+mode+'_critic_loss.txt','a') as f:
            f.write(str(episode)+'\n\n')
            star_index = self.logger.record_dict['critic_loss'].index('*')
            self.logger.record_dict['critic_loss'].pop(star_index)
            for line in self.logger.record_dict['critic_loss']:
                f.write(str(line)+'\n')
            f.write('\n')
                        
    def save_model(self):
        # 保存模型，保存环境的date,action_memory,reward_memory,asset_memory
        checkpoint = {'actor_state_dict': self.actor.state_dict(),
                      'critic_state_dict': self.critic.state_dict(),
                      'actor_optim_state_dict': self.actor.optim.state_dict(),
                      'critic_optim_state_dict': self.critic.optim.state_dict()}
        name = self.train_time + '_' + 'best' + '_' +   'model.pth'
        torch.save(checkpoint,name)
        # self.logger.save(self.filename)

    def load_actor(self, path):
        return self.actor.load_state_dict(torch.load(path,map_location=torch.device(self.device))['actor_state_dict'])
    
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
            a = self.actor.get_actions(s_tensor).cpu().detach().numpy().reshape(-1)  # (action_dim,)
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
