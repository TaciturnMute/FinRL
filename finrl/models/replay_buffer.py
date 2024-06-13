import random
from collections import namedtuple
import numpy as np
from typing import Tuple
import torch
import collections
from finrl.models.utils import CompleteShape
from finrl.models.utils import trans_tensor
from finrl.models.utils import linear_schedule


class ReplayBuffer():

    def __init__(
            self,
            keys: list = ['observation','action','reward','next_observation','date','next_date','done'],
            buffer_capacity: int = None,
            batch_size: int = None,
            gamma: int = None,
            device: str = 'cpu',
    ):
        self.keys = keys
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.buffer = collections.deque(maxlen=self.buffer_capacity)
        self.ReplayBufferSamples = namedtuple('Transition', ['{}s'.format(key) for key in self.keys])
        self.device = torch.device('cuda') if device == 'cuda' else torch.device('cpu')
        self.size = 0

    def add(self, observation, action, reward, next_observation, date, next_date, done) -> None:
        self.size += 1
        transition = [observation, action, reward, next_observation, date, next_date, done]
        self.buffer.append(transition)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
            
        batch_samples = random.sample(self.buffer, batch_size)
        observations, actions, rewards, next_observations, dates, next_dates, dones = map(np.asarray, zip(*batch_samples))
        dones = np.where(dones, 1, 0)

        observations, rewards, next_observations, dones = \
            map(CompleteShape, (observations, rewards, next_observations, dones))
        observations, actions, rewards, next_observations, dones = \
            map(trans_tensor, (observations, actions, rewards, next_observations, dones))
        
        data = (observations.to(self.device), actions.to(self.device), rewards.to(self.device), next_observations.to(self.device), dates, next_dates, dones.to(self.device))
        replay_data = self.ReplayBufferSamples(*tuple(data))
        
        return replay_data

    def is_full(self):
        return self.size == self.buffer_capacity
