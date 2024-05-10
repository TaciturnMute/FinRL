import torch
import numpy as np
import random
from collections import deque
from typing import Tuple, Union, Generator, NamedTuple
from finrl.models.utils import CompleteShape

class RolloutBufferSamples(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    gaussian_actions: torch.Tensor
    old_values: torch.Tensor
    log_prob_old: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

class RolloutBuffer():

    def __init__(
            self,
            buffer_size: int,
            batch_size: Union[int, None],
            lambda_coef: float,
            gamma: float = 0.99,
            obs_shape: tuple = None,
            action_dim: int = None,
            device: str = 'cpu',
            ):

        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
        self.batch_size = batch_size
        self.pos = 0
        self.lambda_coef = lambda_coef
        self.gamma = gamma
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device

    def reset(self) -> None:
        self.pos = 0
        self.states = np.zeros(((self.buffer_size,) + self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.gaussian_actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, 1), dtype=np.float32)  # states values, vt
        self.log_probs = np.zeros((self.buffer_size, 1), dtype=np.float32)  # log(pi(at))
        self.advantages = np.zeros((self.buffer_size, 1), dtype=np.float32)  # advantage_t
        # self.generator_ready = False

    def compute_return_and_advantages(self,
                                      last_value: np.ndarray,
                                      done: bool) -> None:
        '''
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_value: value estimate of the next state of self.states[-1]
        :return:
        '''

        assert self.full == True, 'buffer is not full'
        last_gae_lambda = 0
        for step in reversed(range(self.buffer_size)):
            '''
            next_non_terminal:  If the following td error need to cut
                                For example, when time step is T
                                Then gae_lambda is delta(T), the following td error will be cut.

            delta:              Td error of state in the time step of step.

            next_value:         Next state value estimation.
            '''
            if step == self.buffer_size - 1:
                next_non_terminal = 1 - np.where(done, 1, 0)
                next_value = last_value
            else:
                next_non_terminal = 1 - self.episode_starts[step + 1]
                next_value = self.values[step + 1]

            delta = self.rewards[step] + self.gamma * next_value - self.values[step]
            last_gae_lambda = delta + self.gamma * self.lambda_coef * next_non_terminal * last_gae_lambda
            self.advantages[step] = last_gae_lambda
        self.returns = self.advantages + self.values

    def add(self,
            state: np.ndarray,
            action: np.ndarray,
            gaussian_action: np.ndarray,
            reward: float,
            episode_starts: bool,
            value: np.ndarray,
            log_prob: np.ndarray) -> None:
        # without batch dim
        episode_starts = np.where(episode_starts, 1, 0)
        self.states[self.pos] = np.array(state).copy()  # copy?
        self.actions[self.pos] = np.array(action).copy()
        self.gaussian_actions[self.pos] = np.array(gaussian_action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_starts).copy()
        self.values[self.pos] = np.array(value).copy()
        self.log_probs[self.pos] = np.array(log_prob).copy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get_rollout_samples(self) -> Generator[RolloutBufferSamples, None, None]:

        '''
        After a whole rollout data is collected
        Model will use this generator to generate mini-batch samples to train
        Rollout data will be disordered before it is sampled, according to the disordered index sequence indices
        indices is [66,2,5,8,32,14,56,89,,73,68......], len(indices) == buffer_size
        one mini-batch will be sampled from the rollout data with the disordered index in indices, like [66,2,5,8,32],[14,56,89,73,68],...

        :return: one mini-batch samples of the rollout data.
        '''

        if self.batch_size == None:  # return all things, don't create mini-batch
            self.batch_size = self.buffer_size

        indices = np.random.permutation(self.buffer_size)

        start_idx = 0
        while start_idx < self.buffer_size:
            batch_index = indices[start_idx: start_idx + self.batch_size]

            states = self.states[batch_index]
            actions = self.actions[batch_index]
            gaussian_actions = self.gaussian_actions[batch_index]
            values = self.values[batch_index]
            log_probs = self.log_probs[batch_index]
            advantages = self.advantages[batch_index]
            returns = self.returns[batch_index]

            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            gaussian_actions = torch.tensor(gaussian_actions, dtype=torch.float32)
            values = torch.tensor(values, dtype=torch.float32)
            log_probs = torch.tensor(log_probs, dtype=torch.float32)
            advantages = torch.tensor(advantages, dtype=torch.float32)
            returns = torch.tensor(returns, dtype=torch.float32)

            data = (
                states.to(self.device),
                actions.to(self.device),
                gaussian_actions.to(self.device),
                values.to(self.device),
                log_probs.to(self.device),
                advantages.to(self.device),
                returns.to(self.device),
            )  # corresponding to RolloutBufferSamples

            rollout_data = RolloutBufferSamples(*tuple(data))  # namedtuple, * is used for args
            yield rollout_data
            start_idx += self.batch_size