import torch
from torch import nn
from finrl.models.rollout_buffer import RolloutBuffer
from finrl.models.baselines.ppo.policy import Policy
import warnings
from finrl.models.logger import rollout_logger
from finrl.models.metrics import *
from datetime import datetime


class PPO():
    def __init__(
            self,
            env_train=None,
            env_validation=None,
            env_test=None,
            total_timesteps: int = None,
            n_updates: int = None,
            n_rollout_steps: int = None, # equal to buffer size
            max_grad_norm: float = None,
            batch_size: int = None,
            lambda_coef: float = 0.95,
            gamma: float = 0.99,
            clip_range: float = 0.2,
            policy_lr: float = None,
            ent_coef: float = 0,
            value_coef: float = 0.5,
            policy_kwargs: dict = None,
            train_time: int= '0',
            device: str = 'cuda',
    ):

        self.env_train = env_train
        self.env_validation = env_validation
        self.env_test = env_test
        self.total_timesteps = total_timesteps
        self.n_updates = n_updates
        self.buffer_size = n_rollout_steps
        self.n_rollout_steps = n_rollout_steps
        self.max_grad_norm = max_grad_norm
        self.buffer = RolloutBuffer(self.buffer_size, batch_size, lambda_coef, gamma, self.env_train.observation_space.shape, self.env_train.stock_dim, device)
        self.policy = Policy(**policy_kwargs).to(device)   # no target net
        self.policy.optim = torch.optim.Adam(self.policy.parameters(),policy_lr)
        self.policy.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.policy.optim,factor=1)
        self.last_state = self.env_train.reset()  # ndarray
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self._last_episode_starts = True
        self.best_val_result = {'best val Sharpe ratio':-np.inf}
        self.best_test_result = {'best test Sharpe ratio':-np.inf}
        self.train_time = train_time
        self.device = device

        if n_rollout_steps % batch_size > 0:
            untruncated_batches = n_rollout_steps // batch_size
            warnings.warn(
                f"You have specified a mini-batch size of {batch_size},"
                f" but because the `RolloutBuffer` is of size `n_rollout_steps = {n_rollout_steps}`,"
                f" after every {untruncated_batches} untruncated mini-batches,"
                f" there will be a truncated mini-batch of size {n_rollout_steps % batch_size}\n"
                f"We recommend using a `batch_size` that is a factor of `n_rollout_steps`.\n"
                f"Info: (n_rollout_steps={self.n_rollout_steps})"
            )

    def collect_rollout(self)->None:
        '''
        Collect a whole rollout data.
        When one episode ends but rollout is not complete
        Env will be reset.
        '''
        n_steps = 0
        self.buffer.reset()
        while n_steps < self.n_rollout_steps:
            last_state_tensor = torch.tensor(self.last_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():  # value is going to be used to compute return, which does not need gradient
                action, value, log_prob = self.policy(last_state_tensor)  # tensor,tensor,tensor log_prob is log(pi_ref(at|st))
                gaussian_action = self.policy.get_gaussian_actions()

            action = action.cpu().detach().numpy().reshape(-1,)  # ndarray
            gaussian_action = gaussian_action.cpu().detach().numpy().reshape(-1,)
            value = value.cpu().detach().numpy()
            log_prob = log_prob.cpu().detach().numpy()

            next_state, reward, done, _ = self.env_train.step(action)   # ndarray, float, bool, dict
            self.buffer.add(self.last_state,
                            action,
                            gaussian_action,
                            reward,
                            self._last_episode_starts,  # if self.last_state is the beginning of an episode
                            value,
                            log_prob)  # ndarray, ndarray, float, bool, ndarray, ndarray

            if done:
                self.last_state = self.env_train.reset()
                self._last_episode_starts = True
            else:
                self.last_state = next_state
                self._last_episode_starts = done

            n_steps += 1
            self.logger.record(reward=reward, asset=self.env_train.asset_memory[-1])
            self.logger.timesteps_plus()
            if done:
                self.logger.episode_start()
                # 每一幕结束之后做测试。
                self.examine('validation')
                self.examine('test')

        with torch.no_grad(): # last_value also does not need gradient
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            last_value = self.policy.predict_values(next_state_tensor)   # tensor
        self.buffer.compute_return_and_advantages(last_value.cpu().detach().numpy(), done)

    def replay(self):

        for _ in range(self.n_updates):
            self.rollout_data = self.buffer.get_rollout_samples()
            for data in self.rollout_data:
                states = data.states
                actions = data.actions
                gaussian_actions = data.gaussian_actions
                value_targets = data.returns
                log_prob_old = data.log_prob_old # pi_theta_old(a|s)
                advantages = data.advantages
                values, log_prob, entropy = self.policy.evaluate_actions(obs=states,
                                                                         actions=actions,
                                                                         gaussian_actions=gaussian_actions)

                # I: update critic
                # assert values.shape==value_targets.shape==(self.buffer.batch_size, 1)
                value_loss = nn.functional.mse_loss(value_targets, values)

                # # II: update actor
                ratio = torch.exp(log_prob - log_prob_old)
                # assert log_prob.shape == ratio.shape == (self.buffer.batch_size, 1)
                # assert entropy.shape == (self.env.action_dim, 1)
                policy_loss = -torch.min(torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages,
                                         ratio * advantages)
                policy_loss = policy_loss.mean()
                entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.value_coef * value_loss
                
                self.policy.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optim.step()
                self.policy.lr_scheduler.step()
                self.logger.total_updates_plus()
            self.logger.record(value_loss=value_loss.cpu(), policy_loss=policy_loss.cpu(), entropy_loss=entropy_loss.cpu(), loss=loss.cpu())


    def train(self):
        self.logger = rollout_logger()
        self.logger.episode_start()
        replay_times = 1
        while self.logger.total_timesteps < self.total_timesteps:
            self.collect_rollout()
            self.replay()
            self.logger.show()
            self.save_train_memory('train',replay_times,self.train_time)
            self.logger.reset()
            replay_times+=1
        self.logger.print_elapsed_time()
        print("当前时间：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def save_train_memory(self,mode: str, replay_times: int, train_time: str):
        with open(train_time + mode + '_asset.txt','a') as f:
            f.write(str(replay_times)+'\n\n')
            for line in self.env_train.asset_memory:
                f.write(str(line)+'\n')
            f.write('\n')
        with open(train_time + mode+'_reward.txt','a') as f:
            f.write(str(replay_times)+'\n\n')
            for line in self.env_train.rewards_memory:
                f.write(str(line)+'\n')
            f.write('\n')
        with open(train_time + mode+'_policy_loss.txt','a') as f:
            f.write(str(replay_times)+'\n\n')
            for line in self.logger.record_dict['policy_loss']:
                f.write(str(line)+'\n')
            f.write('\n')
        with open(train_time + mode+'_value_loss.txt','a') as f:
            f.write(str(replay_times)+'\n\n')
            for line in self.logger.record_dict['value_loss']:
                f.write(str(line)+'\n')
            f.write('\n')
        with open(train_time + mode+'_entropy_loss.txt','a') as f:
            f.write(str(replay_times)+'\n\n')

            for line in self.logger.record_dict['entropy_loss']:
                f.write(str(line)+'\n')
            f.write('\n')
        with open(train_time + mode+'_loss.txt','a') as f:
            f.write(str(replay_times)+'\n\n')
            for line in self.logger.record_dict['loss']:
                f.write(str(line)+'\n')
            f.write('\n')

                        
    def save_model(self):
        # 保存模型，保存环境的date,action_memory,reward_memory,asset_memory
        checkpoint = {'policy_state_dict': self.policy.state_dict()}
        name = self.train_time + '_' + 'best' + '_' +   'model.pth'
        torch.save(checkpoint,name)
        # self.logger.save(self.filename)

    def load_actor(self, path):
        return self.policy.load_state_dict(torch.load(path,map_location=torch.device('cpu'))['policy_state_dict'])
    
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
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)  # add batch dim
            a = self.policy.get_actions(s_tensor,True).cpu().detach().numpy().reshape(-1)  # (action_dim,)
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
