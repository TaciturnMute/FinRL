import random
from collections import namedtuple
import numpy as np
from typing import Tuple
import torch
import collections
from finrl.models.utils import CompleteShape
from finrl.models.utils import trans_tensor
from finrl.models.utils import linear_schedule

# proportional variant
class SumTree:
    write = 0
    def __init__(self, capacity):
        '''
        :param capacity: buffer size, which is also the number of leaf nodes
        '''
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.n_entries = 0
        self.pending_idx = set()

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx
        # if left >= self.write + self.capacity - 1:
        #     return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p):
        idx = self.write + self.capacity - 1
        self.pending_idx.add(idx)
        self.write += 1
        self.update(idx, p)

        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority for one sample
    def update(self, idx, p):
        if idx not in self.pending_idx:
            return
        self.pending_idx.remove(idx)
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        self.pending_idx.add(idx)   # data sampled will update its priority, so idx will be removed
        return (idx, self.tree[idx], dataIdx)


class ReplayBuffer():
    '''
    sumtree has bug.
    when update sumtree node's value, round-off error will occur, which makes sample index out of deque range.
    expedient is use try except block, so when error occur, we can resample.
    '''
    def __init__(
            self,
            keys: list = ['observation','action','reward','next_observation','date','next_date','done'],
            buffer_capacity: int = None,
            batch_size: int = None,
            n_steps: int = None,
            gamma: int = None,
            if_prioritized: bool = False,
            alpha: float = 0.5,
            beta_annealing = linear_schedule(start_point=0.4, end_point=1.0, end_time=7600),
            device: str = 'cpu',
    ):
        '''

        :param keys: used to generate corresponding deque
        :param buffer_capacity:
        :param batch_size:
        :param n_steps: n-steps TD learning
        :param gamma:
        :param if_prioritized:
        :param alpha: prioritized replay params, to control how much prioritization is used. alpha is zero means randomly sample
        :param beta_annealing: to apply importance sampling, beta_annealing is a schedule, generate beta series.
        '''
        self.keys = keys
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta_annealing
        self.if_prioritized = if_prioritized
        if if_prioritized:
            # 初始化sumtree
            self.sumtree = SumTree(self.buffer_capacity)
            self.max_priority = 1
        self.weights = np.array([self.gamma ** i for i in range(0, self.n_steps)])
        self.pos = -self.n_steps
        self._size = -self.n_steps + 1
        for key in self.keys:
            setattr(self, key + '_temp', collections.deque(maxlen=self.n_steps))
        self.buffer = collections.deque(maxlen=self.buffer_capacity)
        while len(self.done_temp) < self.done_temp.maxlen:
            self.done_temp.append(False)
        self.ReplayBufferSamples = namedtuple('Transition', ['{}s'.format(key) for key in self.keys])
        self.PrioritizedReplayBufferSamples = namedtuple('Transition', ['{}s'.format(key) for key in self.keys] + ['sample_idx', 'sample_probs'])
        self.device = torch.device('cuda') if device == 'cuda' else torch.device('cpu')
        self.size = 0

    def add(self, observation, action, reward, next_observation, date, next_date, done) -> None:
        self.size += 1
        if not self.is_full():
            self.pos += 1 
            self._size += 1

        if self.n_steps == 1:
            transition = [observation, action, reward, next_observation, date, next_date, done]
            self.buffer.append(transition)
            if self.if_prioritized:
                self.sumtree.add(self.max_priority)
        else:
            transition = []

            self.observation_temp.append(observation)
            if len(self.observation_temp) == self.observation_temp.maxlen:
                transition.append(self.observation_temp[0])

            self.action_temp.append(action)
            if len(self.action_temp) == self.action_temp.maxlen:
                transition.append(self.action_temp[0])

            self.reward_temp.append(reward)
            if len(self.reward_temp) == self.reward_temp.maxlen:
                if np.array(self.done_temp).any():
                    index = np.where(self.done_temp)[0][0]
                    if index == 0:
                        index = self.n_steps
                        # index += 1
                    reward_cum = np.sum(np.array(self.reward_temp)[:index] * self.weights[:index])
                else:
                    reward_cum = np.sum(np.array(self.reward_temp) * self.weights)
                transition.append(reward_cum)

            self.next_observation_temp.append(next_observation)
            if len(self.next_observation_temp) == self.next_observation_temp.maxlen:

                if np.array(self.done_temp).any():
                    index = np.where(self.done_temp)[0][0]
                    if index > 0:
                        value = self.last_next_observation
                    else:
                        value = self.next_observation_temp[-1]
                else:
                    value = self.next_observation_temp[-1]
                transition.append(value)

            self.date_temp.append(date)
            if len(self.date_temp) == self.date_temp.maxlen:
                transition.append(self.date_temp[0])

            self.next_date_temp.append(next_date)
            if len(self.next_date_temp) == self.next_date_temp.maxlen:
                if np.array(self.done_temp).any():
                    index = np.where(self.done_temp)[0][0]
                    if index > 0:
                        value = self.last_next_date
                    else:
                        value = self.next_date_temp[-1]
                else:
                    value = self.next_date_temp[-1]
                transition.append(value)

            self.done_temp.append(done)
            if self.pos >= 0: # done无法用 len(self.done_temp) == self.done_temp.maxlen条件。但是pos初始化为-n_steps，所以transition添加done和添加其他元素的开始时间是一致的。
                if np.array(self.done_temp).any():
                    value = True
                else:
                    value = False
                transition.append(value)

            if done:
                self.last_next_observation = self.next_observation_temp[-1]
                self.last_next_date = self.next_date_temp[-1]


            if self.pos >= 0:
                self.buffer.append(transition)
                if self.if_prioritized:
                    self.sumtree.add(self.max_priority) # 确保当前样本在未被抽样时有较大概率被抽取到

    def update_priorities(self, info):
        for idx, priority in info:
            self.max_priority = max(self.max_priority, priority)
            self.sumtree.update(idx, priority)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if not self.if_prioritized:
            batch_samples = random.sample(self.buffer, batch_size)
            observations, actions, rewards, next_observations, dates, next_dates, dones = map(np.asarray, zip(*batch_samples))
            dones = np.where(dones, 1, 0)

            observations, rewards, next_observations, dones = \
                map(CompleteShape, (observations, rewards, next_observations, dones))
            observations, actions, rewards, next_observations, dones = \
                map(trans_tensor, (observations, actions, rewards, next_observations, dones))
            
            data = (observations.to(self.device), actions.to(self.device), rewards.to(self.device), next_observations.to(self.device), dates, next_dates, dones.to(self.device))
            replay_data = self.ReplayBufferSamples(*tuple(data))
        else:
            batch_samples = []
            sample_idx = []
            sample_probs = []
            segment = self.sumtree.total() / batch_size
            i = 0
            while len(batch_samples) < batch_size:
                a = i * segment
                b = (i + 1) * segment
                rnum = np.random.uniform(a, b)
                try:
                    (idx, priority, data_index) = self.sumtree.get(rnum)
                    batch_samples.append(self.buffer[data_index])
                    sample_idx.append(idx)
                    sample_probs.append(priority / self.sumtree.total())
                    i += 1
                except:
                    print('index is out of range')
                finally:
                    pass

            observations, actions, rewards, next_observations, dones = map(np.asarray, zip(*batch_samples))
            dones = np.where(dones, 1, 0)
            observations, rewards, next_observations, dones, sample_probs = \
                map(CompleteShape, (observations, rewards, next_observations, dones, np.array(sample_probs)))
            observations, actions, rewards, next_observations, dones, sample_probs= \
                map(trans_tensor, (observations, actions, rewards, next_observations, dones, sample_probs))
            data = (observations.to(self.device), actions.to(self.device), rewards.to(self.device), next_observations.to(self.device), dones.to(self.device), sample_idx.to(self.device), sample_probs.to(self.device))
            replay_data = self.PrioritizedReplayBufferSamples(*tuple(data))

        # for d in replay_data:
        #     if torch.is_tensor(d):
        #         d = d.to(self.device)
        
        return replay_data

    def is_full(self):
        return self._size == self.buffer_capacity
