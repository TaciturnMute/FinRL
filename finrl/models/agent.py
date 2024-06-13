import torch
from torch import nn
from typing import Dict
import numpy as np
from finrl.models.policy import *
from finrl.models.utils import get_figure,polyak_update
from finrl.models.replay_buffer import ReplayBuffer
from finrl.models.logger import episode_logger
from datetime import datetime
from finrl.models.metrics import *
from finrl.models.noise import get_noise
from finrl.models.mcts import *
from itertools import chain


class DDPG_MCTS():

    """第四章代码，使用MCTS和集成Q学习。"""

    def __init__(
            self,
            env_train=None,
            env_validation=None,
            env_test=None,
            n_updates: int = None,
            buffer_size: int = None,
            batch_size: int = None,
            n_steps: int = None,
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
            total_updates_times_maximum: int = None,
            mcts_gamma: float = None,
            expand_length: str = None,
            children_maximum: int = None,
            q_target_mode: str = 'redq',
            N: int = None,
            M: int = None,
            mcts_C: float = None,
            random_select_prob: float = 0.5,
    ):
        # 一些基本参数
        self.env_train = env_train
        self.env_validation = env_validation
        self.env_test = env_test
        self.n_updates = n_updates
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
        self.q_target_mode = q_target_mode
        self.N = N
        self.M = M
        # 经验池
        self.buffer = ReplayBuffer(buffer_capacity=buffer_size,
                                   batch_size=batch_size,
                                   gamma=gamma,
                                   device=device)
        # 初始化Policy和Target Policy
        self.policy = DDPG_Policy_MCTS(**policy_kwargs).to(self.device)
        self.policy_target = DDPG_Policy_MCTS(**policy_kwargs).to(self.device)
        self.policy_target.load_state_dict(self.policy.state_dict())
        # 优化器。表征学习、actor、critic均分开优化。
        self.parameters_srl = chain(self.policy.srl.parameters(), self.policy.f_pre_obs.parameters(),self.policy.f_pre_r.parameters()) 
        self.policy.optim_srl = torch.optim.Adam(self.parameters_srl,policy_lr)
        self.policy.optim_actor = torch.optim.Adam(
            torch.nn.ModuleList([
            self.policy.action_mlp, 
            self.policy.action_net,
            self.policy.cnn1,
            self.policy.cnn2,
            self.policy.cnn3,
            self.policy.lstm1,
            self.policy.lstm2,
            self.policy.lstm3,
            ]).parameters(),
            policy_lr
        )
        # 每个critic都各自优化。
        self.optim_critic_list = []
        for i in range(self.N):
            optim_critic = torch.optim.Adam(
                torch.nn.ModuleList([
                self.policy.critic_mlp_list[i], 
                self.policy.critic_net_list[i],
                self.policy.cnn1,
                self.policy.cnn2,
                self.policy.cnn3,
                self.policy.lstm1,
                self.policy.lstm2,
                self.policy.lstm3,
                ]).parameters(),
                policy_lr
            )
            self.optim_critic_list.append(optim_critic)
        # 学习率
        lambda1 = lambda epoch: 1
        lambda2 = lambda epoch: 1
        lambda3 = lambda epoch: 1
        self.policy.lr_scheduler_srl = torch.optim.lr_scheduler.LambdaLR(self.policy.optim_srl, lr_lambda=lambda1)
        self.policy.lr_scheduler_actor = torch.optim.lr_scheduler.LambdaLR(self.policy.optim_actor, lr_lambda=lambda2)
        self.lr_scheduler_critic_list = []
        for i in range(self.N):
            lr_scheduler_critic = torch.optim.lr_scheduler.LambdaLR(self.optim_critic_list[i], lr_lambda=lambda3)
            self.lr_scheduler_critic_list.append(lr_scheduler_critic)
        # 其他参数
        self.noise = get_noise(noise_aliase, noise_kwargs)
        self.action_space_range = [self.env_train.action_space.low, self.env_train.action_space.high]
        self.best_val_result = {'best val Sharpe ratio':-np.inf}
        self.best_test_result = {'best test Sharpe ratio':-np.inf}
        self.train_time = train_time
        self.if_clip = if_clip
        assert self.N == policy_kwargs['N'] and self.M == policy_kwargs['M']
        # MCTS部分
        self.total_updates_times_maximum = total_updates_times_maximum
        self.children_maximum = children_maximum
        # 蒙特卡洛树
        self.mcts = MCTS(
            mcts_gamma=mcts_gamma,
            expand_length=expand_length,
            C=mcts_C,
            random_select_prob=random_select_prob,
        )

    def replay(self) -> None:

        """一次DDPG训练"""

        self.actor_loss_list_once_replay = []
        self.critic_loss_list_once_replay = []
        self.srl_loss_list_once_replay = []

        for _ in range(self.n_updates):
            # 小批量抽取
            data = self.buffer.sample()
            figures1, figures2, figures3 = get_figure(self.figure_path, data.dates, self.device)
            next_figures1, next_figures2, next_figures3  = get_figure(self.figure_path, data.next_dates, self.device)
            # srl训练
            if self.task == 'trading':
                next_obs_part = data.next_observations[:, 1:1 + self.env_train.stock_dim]  # trading任务预测价格。
            else:
                next_obs_part = data.next_observations[:,self.env_train.stock_dim,:]   # portfolio任务预测macd。
            states = self.policy.srl(data.observations)  
            srl_loss = 0.5 * nn.functional.mse_loss(self.policy.f_pre_obs(states), next_obs_part) +\
                        0.5 * nn.functional.mse_loss(self.policy.f_pre_r(states), data.rewards)
            self.policy.optim_srl.zero_grad()
            srl_loss.backward()
            self.policy.optim_srl.step()
            self.policy.lr_scheduler_srl.step()
            # critic训练
            M = self._get_probabilistic_num_min(self.M)
            M_indexs = np.random.choice(self.N, M, replace=False).tolist()  # [0,N-1]
            with torch.no_grad():
                targets = self._cal_target(next_figures1, next_figures2, next_figures3, data.rewards, data.next_observations, data.dones, M_indexs)
                targets = targets.expand((-1, self.N)) if targets.shape[1] == 1 else targets
            indexs = [i for i in range(self.N)]
            q_value_pres_list,figure_hidden = self.policy.get_q_value_only(figures1, figures2, figures3,data.observations, data.actions,indexs)
            q_values_pre_all = torch.cat(q_value_pres_list, dim=1)
            critic_loss = nn.functional.mse_loss(q_values_pre_all, targets) 
            for index in range(self.N):
                self.optim_critic_list[index].zero_grad()  # 优化器梯度清零
            critic_loss.backward() # loss反向传播
            for index in range(self.N):
                self.optim_critic_list[index].step()  # 优化器更新参数
                self.lr_scheduler_critic_list[index].step()  # 学习率步进
            # actor训练
            # 计算TD Target和计算actor loss所使用的policy功能是类似的，都是forward函数。
            action_pres, q_values_list = self.policy(figures1, figures2, figures3, data.observations,figure_hidden=None,indexs=[i for i in range(self.N)])  
            actor_loss = torch.mean(torch.cat(q_values_list, dim=1), dim=1, keepdim=True) # 对N求平均！！
            actor_loss = -actor_loss.mean()  # 对batch size求mean
            self.policy.optim_actor.zero_grad() 
            actor_loss.backward()
            self.policy.optim_actor.step()
            self.policy.lr_scheduler_actor.step()  
            # 记录结果
            self.logger.total_updates_plus(1)
            self.actor_loss_list_once_replay.append(actor_loss.cpu().detach().numpy())
            self.critic_loss_list_once_replay.append(critic_loss.cpu().detach().numpy())
            self.srl_loss_list_once_replay.append(srl_loss.cpu().detach().numpy())

    def init(self):
        """初始化蒙特卡洛树的根结点"""
        self.env_train.reset()
        # 创建根节点，此时visit=0
        self.mcts.root = Node(
            name=str(self.mcts.node_num),  # 结点在树中的编号
            day=self.env_train.day,       # 共有
            children_maximum=self.children_maximum,   # 共有
            asset_memory=self.env_train.asset_memory,  # 共有
            date_memory=self.env_train.date_memory,   # 共有
            rewards_memory=self.env_train.rewards_memory ,  # 共有
            actions_memory=self.env_train.actions_memory,  # 共有
            rollout_length=len(self.env_train.df.date.unique()), # 共有
            parent=None, # 共有
            cost=self.env_train.cost, # 共有
            trades=self.env_train.trades, # 共有
            x_prev=self.noise.x0 if self.noise.x0 is not None else np.zeros_like(self.noise.mu),
            state_memory=self.env_train.state_memory if self.task=='trading' else None, # trading
            previous_state=self.env_train.state if self.task == 'trading' else None, # trading
            turbulence=self.env_train.turbulence if self.task=='trading' else None, # trading
            last_weights=self.env_train.last_weights if self.task == 'portfolio' else None, # portfolio
            portfolio_value=self.env_train.portfolio_value if self.task == 'portfolio' else None, # portfolio
            portfolio_value_vector=self.env_train.portfolio_value_vector if self.task == 'portfolio' else None, # portfolio
            portfolio_return_memory=self.env_train.portfolio_return_memory if self.task == 'portfolio' else None, # portfolio
        )
        print("训练开始，当前时间：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.logger = episode_logger()

    def train(self):

        """训练主函数"""

        # 初始化MCTS
        self.init()
        # 循环训练
        while True:
            if self.logger.total_updates >= self.total_updates_times_maximum:  # 判断
                break
            # 新的一幕的开始
            # 选择
            self.logger.reset()
            node,layer = self.mcts.search()
            if not node.is_terminal:  # 还可以扩展+预演
                expand_length = self.mcts.expand_length
                # 环境恢复断点
                self.env_train.from_given_state(
                    day=node.day,
                    cost=node.cost,
                    trades=node.trades,
                    asset_memory=node.asset_memory,
                    date_memory=node.date_memory,
                    rewards_memory=node.rewards_memory,
                    actions_memory=node.actions_memory,

                    previous_state=node.previous_state,
                    state_memory=node.state_memory,
                    turbulence=node.turbulence,

                    last_weights=node.last_weights,
                    portfolio_value=node.portfolio_value,
                    portfolio_value_vector=node.portfolio_value_vector,
                    portfolio_return_memory=node.portfolio_return_memory,
                )   
                self.noise.from_given_state(node.x_prev)

                done = self.env_train.day >= len(self.env_train.df.index.unique()) - 1
                s = self.env_train.state
                # 扩展，一段和环境交互的过程，不会到达幕终点。
                expand_times = -1
                for i in range(expand_length):
                    expand_times+=1
                    s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to('cuda')
                    date,next_date = str(self.env_train.date),str(self.env_train.next_date) 
                    figure1,figure2,figure3 = get_figure(self.figure_path, [date], self.device) 
                    a = self.policy.get_action_only(figure1,figure2,figure3,s_tensor).cpu().detach().numpy().reshape(-1)
                    a = np.clip(a + self.noise.__call__(), self.action_space_range[0], self.action_space_range[1])
                    s_, r, done, _ = self.env_train.step(a)
                    self.buffer.add(s, a, r, s_, date, next_date, done)
                    s = s_

                    self.logger.timesteps_plus()

                    # 执行一轮训练。
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

                    if self.logger.total_updates >= self.total_updates_times_maximum:  # 不继续扩展，退出扩展的循环。
                        break

                if expand_times < expand_length-1:  # 说明此次扩展无法完成，即到达最大更新次数，退出最外层while循环。
                    break

                # 生成新node
                self.mcts.node_num += 1
                child = Node(
                    name=str(self.mcts.node_num),  # 结点在树中的编号
                    day=self.env_train.day,       # 共有
                    children_maximum=self.children_maximum,   # 共有
                    asset_memory=self.env_train.asset_memory,  # 共有
                    date_memory=self.env_train.date_memory,   # 共有
                    rewards_memory=self.env_train.rewards_memory ,  # 共有
                    actions_memory=self.env_train.actions_memory,  # 共有
                    rollout_length=len(self.env_train.df.date.unique())-int(self.env_train.day), # 共有
                    parent=node, # 共有
                    cost=self.env_train.cost, # 共有
                    trades=self.env_train.trades, # 共有
                    x_prev=self.noise.x_prev,   # 共有
                    state_memory=self.env_train.state_memory if self.task=='trading' else None, # trading
                    previous_state=self.env_train.state if self.task == 'trading' else None, # trading
                    turbulence=self.env_train.turbulence if self.task=='trading' else None, # trading
                    last_weights=self.env_train.last_weights if self.task == 'portfolio' else None, # portfolio
                    portfolio_value=self.env_train.portfolio_value if self.task == 'portfolio' else None, # portfolio
                    portfolio_value_vector=self.env_train.portfolio_value_vector if self.task == 'portfolio' else None, # portfolio
                    portfolio_return_memory=self.env_train.portfolio_return_memory if self.task == 'portfolio' else None, # portfolio
                    )

                if len(self.env_train.df.date.unique())-child.day < expand_length + 50:
                    child.is_terminal = True # 该结点不再继续扩展。
                node.children.append(child)

            else:  # 不可以扩展，直接rollout。
                pass

            # rollout, 到达幕终点。当更新次数达到上限时，退出循环。      
            rollout_time = 0
            while not done and self.logger.total_updates < self.total_updates_times_maximum:
                rollout_time += 1
                s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
                date,next_date = str(self.env_train.date),str(self.env_train.next_date) 
                figure1,figure2,figure3 = get_figure(self.figure_path, [date], self.device) 
                a = self.policy.get_action_only(figure1,figure2,figure3,s_tensor).cpu().detach().numpy().reshape(-1)
                a = np.clip(a + self.noise.__call__(), self.action_space_range[0], self.action_space_range[1])
                s_, r, done, _ = self.env_train.step(a)   
                self.buffer.add(s, a, r, s_, date, next_date, done)
                s = s_

                self.logger.timesteps_plus()

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
            print(f'rollout长度为{rollout_time}')
            
            # back prop
            asset = self.env_train.asset_memory[-1]
            reward = self.mcts.cal_mcts_reward(asset)
            self.mcts.back_prop(child,reward)
            self.save_train_memory('train',ep,self.train_time)
            self.examine('validation')
            self.examine('test')
        self.examine('validation')
        self.examine('test')
        self.save_mcts()
        print("训练结束，当前时间：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def _cal_target(self, next_figures1, next_figures2, next_figures3, rewards, next_observations, dones, M_indexs):

        """计算集成Q学习的标签"""

        with torch.no_grad():
            if self.q_target_mode == 'redq':
                """Q target is min of a subset of Q values"""
                next_actions, next_q_values_list = self.policy_target(next_figures1, next_figures2, next_figures3,next_observations,None,M_indexs)
                next_q_values,_ = torch.min(torch.cat(next_q_values_list, dim=1), dim=1, keepdim=True)  # min！！
                targets = rewards + self.gamma * (1 - dones) * next_q_values

            if self.q_target_mode == 'average':
                """Q target is average of a subset of Q values"""
                M_indexs = [i for i in range(self.N)] # all
                next_actions, next_q_values_list = self.policy_target(next_figures1, next_figures2, next_figures3,next_observations,None,M_indexs)
                assert len(next_q_values_list) == self.N
                next_q_values = torch.mean(torch.cat(next_q_values_list, dim=1), dim=1, keepdim=True)  # mean！！
                targets = rewards + self.gamma * (1 - dones) * next_q_values

            if self.q_target_mode == 'weighted':
                """Q target is expectation of REDQ Q target"""
                M_indexs = [i for i in range(self.N)] # all
                next_actions, next_q_values_list = self.policy_target(next_figures1, next_figures2, next_figures3,next_observations,None,M_indexs)
                sorted_values_of_all_samples = torch.cat(next_q_values_list, axis=1).sort()[0]
                weights = [math.factorial(self.N - i) / \
                           (math.factorial(self.M - 1) * math.factorial(self.N - i - self.M + 1)) \
                           for i in range(1, self.N - self.M + 2)] + [0] * (self.M - 1)
                weighted_sum_values = sorted_values_of_all_samples.mul_(torch.tensor(weights).to(self.device))
                N_select_M = math.factorial(self.N) / (math.factorial(self.M) * math.factorial(self.N - self.M))
                next_q_values = (weighted_sum_values.sum(axis=1) / N_select_M).reshape(-1, 1)
                targets = rewards + self.gamma * (1 - dones) * next_q_values

            if self.q_target_mode == 'maxmin':
                """Q target is min of all Q values"""
                M_indexs = [i for i in range(self.N)] # all
                next_actions, next_q_values_list = self.policy_target(next_figures1, next_figures2, next_figures3,next_observations,None,M_indexs)
                next_q_values,_ = torch.min(torch.cat(next_q_values_list, dim=1), dim=1, keepdim=True)  # min!!
                targets = rewards + self.gamma * (1 - dones) * next_q_values

        return targets
    
    def _get_probabilistic_num_min(self, num_mins):
        
        """允许M设置为小数"""

        floored_num_mins = np.floor(num_mins)
        if num_mins - floored_num_mins > 0.001:
            prob_for_higher_value = num_mins - floored_num_mins
            if np.random.uniform(0, 1) < prob_for_higher_value:
                return int(floored_num_mins+1)
            else:
                return int(floored_num_mins)
        else:
            return num_mins

    def save_train_memory(self,mode:str,episode,train_time):

        """保存训练过程结果，每一幕保存一次"""

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

        """保存模型"""

        checkpoint = {'policy_state_dict': self.policy.state_dict()}
        name = self.train_time + '_' + 'best' + '_' +   'model.pth'
        torch.save(checkpoint,name)
    
    def save_mcts(self):
        
        """保存MCTS过程有意义的结果，包括buffer中的日期分布，树结构。"""
        
        with open(self.train_time + '_mcts_dates.txt','w') as f:
            for data in self.buffer.buffer:
                f.write(data[-3] + '\n')
        
        def add_nodes_edges(graph, node):
            node_name = node.name
            graph.node(node_name, node_name)  # 创建当前节点

            for child in node.children:
                child_name = child.name if child.name else 'Unnamed'
                graph.node(child_name, child_name)  # 创建子节点
                graph.edge(node_name, child_name)  # 在当前节点和子节点之间创建边
                add_nodes_edges(graph, child)  # 对子节点递归执行同样的操作

        dot = Digraph(comment='Tree Structure')
        add_nodes_edges(dot, self.mcts.root)
        dot.render(self.train_time + '_tree', view=False, format='pdf')

    def load_actor(self,path):

        """导入actor"""

        return self.policy.load_state_dict(torch.load(path)['policy_state_dict'])
    
    def examine(self,mode:str):

        """执行验证集或测试集推断"""

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
            figure1,figure2,figure3 = get_figure(self.figure_path, [date], self.device)   # (1, 4, 400, 240)
            a = self.policy.get_action_only(figure1,figure2,figure3,s_tensor).cpu().detach().numpy().reshape(-1)
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


class DDPG():

    """第三章代码，使用多模态数据和表征学习模型。"""

    def __init__(
            self,
            env_train=None,
            env_validation=None,
            env_test=None,
            episodes: int = None,
            n_updates: int = None,
            buffer_size: int = None,
            batch_size: int = None,
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
                                   gamma=gamma,
                                   device=device)

        self.policy = DDPG_Policy(**policy_kwargs).to(self.device)
        self.policy_target = DDPG_Policy(**policy_kwargs).to(self.device)
        self.policy_target.load_state_dict(self.policy.state_dict())

        self.if_srl = self.policy.if_srl
        if self.if_srl:
            self.parameters_srl = chain(self.policy.srl.parameters(), self.policy.f_pre_obs.parameters(),self.policy.f_pre_r.parameters()) 
            self.policy.optim_srl = torch.optim.Adam(self.parameters_srl,policy_lr)

        self.policy.optim_actor = torch.optim.Adam(
            torch.nn.ModuleList([
            self.policy.action_mlp, 
            self.policy.action_net,
            self.policy.cnn1,
            self.policy.cnn2,
            self.policy.cnn3,
            self.policy.lstm1,
            self.policy.lstm2,
            self.policy.lstm3,
            ]).parameters(),
            policy_lr
        )

        self.policy.optim_critic = torch.optim.Adam(
            torch.nn.ModuleList([
            self.policy.critic_mlp, 
            self.policy.critic_net,
            self.policy.cnn1,
            self.policy.cnn2,
            self.policy.cnn3,
            self.policy.lstm1,
            self.policy.lstm2,
            self.policy.lstm3,
            ]).parameters(),
            policy_lr
        )

        lambda1 = lambda epoch: 1
        lambda2 = lambda epoch: 1
        lambda3 = lambda epoch: 1
        if self.if_srl:
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
            figures1, figures2, figures3 = get_figure(self.figure_path, data.dates, self.device)
            next_figures1, next_figures2, next_figures3  = get_figure(self.figure_path, data.next_dates, self.device)

            # 训练
            # srl训练
            if self.if_srl:
                if self.task == 'trading':
                    next_obs_part = data.next_observations[:, 1:1 + self.env_train.stock_dim]  # trading任务预测价格。
                else:
                    next_obs_part = data.next_observations[:,self.env_train.stock_dim,:]   # portfolio任务预测macd。
                states = self.policy.srl(data.observations)
                srl_loss = 0.5 * nn.functional.mse_loss(self.policy.f_pre_obs(states), next_obs_part) +\
                            0.5 * nn.functional.mse_loss(self.policy.f_pre_r(states), data.rewards)
                self.policy.optim_srl.zero_grad()
                srl_loss.backward()
                self.policy.optim_srl.step()
                self.policy.lr_scheduler_srl.step()

            else:
                self.srl_loss_list_once_replay = [0]

            # rl训练
            with torch.no_grad():
                next_actions,next_q_values = self.policy_target(next_figures1, next_figures2, next_figures3, data.next_observations)
                targets = data.rewards + self.gamma * (1 - data.dones) * next_q_values
            q_value_pres,figure_hidden = self.policy.get_q_value_only(figures1, figures2, figures3,data.observations, data.actions)

            critic_loss = nn.functional.mse_loss(q_value_pres, targets)
            self.policy.optim_critic.zero_grad()
            critic_loss.backward()
            self.policy.optim_critic.step()
            self.policy.lr_scheduler_critic.step()  

            action_pres, actor_loss = self.policy(figures1, figures2, figures3, data.observations,figure_hidden=None,states=None)
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
            if self.if_srl:
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
                figure1,figure2,figure3 = get_figure(self.figure_path, [date], self.device)   # (1, 4, 400, 240)
                # 获取动作
                a = self.policy.get_action_only(figure1,figure2,figure3,s_tensor).cpu().detach().numpy().reshape(-1)  # (action_dim,)
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
            figure1,figure2,figure3 = get_figure(self.figure_path, [date], self.device)   # (1, 4, 400, 240)
            a = self.policy.get_action_only(figure1,figure2,figure3,s_tensor).cpu().detach().numpy().reshape(-1)
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

