import torch
from torch import nn
import numpy as np
from datetime import datetime
from finrl.models.replay_buffer import ReplayBuffer
from finrl.models.baselines.td3.actor import Actor
from finrl.models.baselines.td3.critic import Critic
from finrl.models.noise import get_noise
from finrl.models.logger import episode_logger
from finrl.models.metrics import *
from finrl.models.utils import polyak_update


class TD3():
    def __init__(
            self,
            env_train = None,
            env_validation = None,
            env_test = None,
            buffer_size: int = None,
            batch_size:int = None,
            episodes: int = None,
            n_updates: int = None,
            tau: float = None,
            gamma: float = None,
            policy_update_delay: int = None,  # 2
            target_copy_interval: int = None,
            training_start: int = None,
            actor_lr: float = None,
            critic_lr: float = None,
            critic_kwargs: dict = None,
            actor_kwargs: dict = None,
            noise_aliase: str = None,
            noise_kwargs: dict = None,
            smooth_noise_aliase: str = None,
            smooth_noise_kwargs: dict = None,
            print_interval: int = None,
            train_time: str = None,
            device: str = 'cuda',
            ):
        self.env_train = env_train
        self.env_validation = env_validation
        self.env_test = env_test
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_capacity=buffer_size,
                                   batch_size=batch_size,
                                   gamma=gamma,
                                   device=device)
        self.episodes = episodes
        self.n_updates = n_updates
        self.tau = tau
        self.policy_update_delay = policy_update_delay
        self.target_copy_interval = target_copy_interval
        self.gamma = gamma
        self.training_start = training_start
        self.print_interval = print_interval

        self.actor = Actor(**actor_kwargs).to(device)
        self.actor_target = Actor(**actor_kwargs).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor.optim = torch.optim.Adam(self.actor.parameters(),actor_lr)
        self.actor.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.actor.optim,factor=1)
        self.critic = Critic(**critic_kwargs).to(device)
        self.critic_target = Critic(**critic_kwargs).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optim1 = torch.optim.Adam(self.critic.qf1.parameters(),critic_lr)
        self.critic.optim2 = torch.optim.Adam(self.critic.qf2.parameters(),critic_lr)
        self.critic.lr_scheduler1 = torch.optim.lr_scheduler.ConstantLR(self.critic.optim1,factor=1)
        self.critic.lr_scheduler2 = torch.optim.lr_scheduler.ConstantLR(self.critic.optim2,factor=1)

        self.noise = get_noise(noise_aliase, noise_kwargs)
        smooth_noise_kwargs['batch_size'] = self.batch_size
        self.smooth_noise = get_noise(smooth_noise_aliase, smooth_noise_kwargs)
        self.action_space_range = [self.env_train.action_space.low, self.env_train.action_space.high]
        self.best_val_result = {'best val Sharpe ratio':-np.inf}
        self.best_test_result = {'best test Sharpe ratio':-np.inf}
        self.train_time = train_time
        self.device = device

    def replay(self) -> None:
        self.actor_loss_list_once_replay = []
        self.critic_loss_list_once_replay = []
        for _ in range(self.n_updates):
            data = self.buffer.sample()
            # I update critic:
            with torch.no_grad():
                next_actions = self.actor_target(data.next_observations) + self.smooth_noise.__call__().to(self.device)
                next_actions = next_actions.clamp(torch.tensor(self.action_space_range[0]).to(self.device), torch.tensor(self.action_space_range[1]).to(self.device))
                next_q1, next_q2 = self.critic_target(data.next_observations, next_actions)
                td_targets = data.rewards + self.gamma * (1 - data.dones) * torch.min(next_q1, next_q2)
            q_value_pres1, q_value_pres2 = self.critic(data.observations, data.actions)  # tensor
            critic_loss = nn.functional.mse_loss(q_value_pres1, td_targets) + nn.functional.mse_loss(q_value_pres2,td_targets)
            # optim respectively
            self.critic.optim1.zero_grad()
            self.critic.optim2.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic.optim1.step()
            self.critic.lr_scheduler1.step()
            self.critic.optim2.step()
            self.critic.lr_scheduler2.step()

            # II update actor:
            if self.logger.total_updates % self.policy_update_delay == 0:
                action_pres = self.actor(data.observations)  # tensor, cumpute by learning net actor
                actor_loss = -self.critic.get_qf1_value(data.observations, action_pres).mean()  # take derivative of the actor_loss, get ddpg policy gradient
                self.actor.optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor.optim.step()
                self.actor.lr_scheduler.step()
                self.actor_loss_list_once_replay.append(actor_loss.cpu().detach().numpy())
            self.critic_loss_list_once_replay.append(critic_loss.cpu().detach().numpy())
            # note: in td3 paper, target_copy_delay is equal to policy_update_delay
            # it means no matter actor training frequency or actor/critic soft update, the frequency is lag behind
            if self.logger.total_updates % self.target_copy_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
            self.logger.total_updates_plus()

    def train(self):
        self.logger = episode_logger()
        for ep in range(1, self.episodes+1):
            s = self.env_train.reset()
            done = False
            self.logger.reset()
            while not done:
                s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
                a = self.actor(s_tensor).cpu().detach().numpy().reshape(-1)
                a = np.clip(a + self.noise.__call__(), self.action_space_range[0], self.action_space_range[1])
                s_, r, done, _ = self.env_train.step(a)  # ndarray,float,bool,empty_dict
                self.buffer.add(s, a, r, s_, None, None, done)  # ndarray,ndarray,scale,list,bool
                s = s_
                self.logger.timesteps_plus()

                if self.logger.total_timesteps > self.batch_size and self.logger.total_timesteps > self.training_start:
                    self.replay()
                    date_list = self.env_train.date.split('-')
                    year, month, day = date_list[0], date_list[1], date_list[2]
                    self.logger.record(reward=r, asset=self.env_train.asset_memory[-1], time=f'{year[2:]}_{month}_{day}',
                                       critic_loss=np.mean(self.critic_loss_list_once_replay))
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
                if self.logger.timesteps % self.print_interval == 0:
                    self.logger.show(interval=self.print_interval)
                    # self.logger.show(policy_update_delay=self.policy_update_delay,
                    #                  interval=self.print_interval,
                    #                  n_updates=self.n_updates)
            self.save_train_memory('train',ep,self.train_time)
            self.examine('validation')
            self.examine('test')
        print("当前时间：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

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



