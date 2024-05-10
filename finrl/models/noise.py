import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union,Iterable


class OrnsteinUhlenbeckActionNoise():

    def __init__(self,
                 mu: np.ndarray,
                 theta: float,
                 sigma: Union[float, int],
                 dt: float,
                 x0: np.ndarray = None,
                 seed = 1,
                 randomness = False):
        '''
        # mu和x0在动作是一维情况下，形状都是向量，即(dim,)形状。输出的噪声形状亦是如此。
        :param mu: mean level of the noise
        :param theta: how quickly noise will be draw back to the mean level, or how strongly the system will react to the perturbation
        :param sigma: the variation or the size of the noise
        :param dt: time interval size
        :param x0: initial value
        :param seed: remove randomness
        :param randomness: if True, then noise can not be reproduction, seed is useless.
        '''
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
        if not randomness:
            np.random.seed(seed)
        self.seeds = np.random.randint(0, 10000000,100000)  # fix seeds series, so ou noise will randomly generate with the fix seed obtain from seed series
        self.seed_order = 0
        self.randomness = randomness
        print(f'------noise------')
        print(f'ou noise, randomness is {self.randomness} !')
        print(f'-----------------')

    def __call__(self) -> np.ndarray:
        # set random seed
        if not self.randomness:
            np.random.seed(self.seeds[self.seed_order])
            self.seed_order += 1
            # Xt+1 = Xt + theta * (mean - Xt) * dt + sigma * (dt)^0.5 * N(0,1)
            x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
                self.dt) * np.random.normal(size=self.mu.shape)
            self.x_prev = x
        # do not set random seed, so each ou class instantiation will generate different noise
        else:
            x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
                self.dt) * np.random.normal(size=self.mu.shape)
            self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def from_given_state(self,x_prev):
        self.x_prev = x_prev

    def Plot(self, duration: int, reset=True,label:str = None):
        plt.figure(figsize=(6, 3))
        if reset == True:
            self.reset()
        noise = []
        for i in range(duration):
            noise.append(self.__call__())

        plt.plot(noise,label = label,linewidth = 0.8, color = 'tab:cyan')
        plt.title('OU Noise')
        plt.grid()
        plt.legend()
        plt.show()


class NormalNoise():

    def __init__(self,
                 loc: np.ndarray,  # shape is like action
                 std: float,
                 seed: int = 1,
                 randomness: bool = False):
        self.loc = loc
        self.std = std
        self.seed = seed
        self.randomness = randomness
        if not randomness:
            np.random.seed(seed)
        # fix seeds series, so noise will randomly generate with the fix seed obtain from seed series
        self.seeds = np.random.randint(0, 10000000, 1000000)
        self.seed_order = 0
        self.randomness = randomness
        print(f'------noise------')
        print(f'normal noise, randomness is {self.randomness} !')
        print(f'-----------------')

    def __call__(self) -> np.ndarray:
        # set random seed
        if self.randomness == False:
            np.random.seed(self.seeds[self.seed_order])
            self.seed_order += 1
            x = np.random.randn(self.loc.shape[0]) * self.std + self.loc  # generate one row
        # do not set random seed, so each ou class instantiation will generate different noise
        else:
            x = np.random.randn(self.loc.shape[0]) * self.std + self.loc
        return x

    def reset(self):
        self.seed_order = 0

    def Plot(self, duration: int, reset=True, label: str = None):
        plt.figure(figsize=(6, 3))
        if reset == True:
            self.reset()
        noise = []
        for i in range(duration):
            noise.append(self.__call__())
        plt.plot(noise,label = label,linewidth = 0.8, color = 'tab:cyan')
        plt.title('Normal Noise')
        plt.grid()
        plt.legend()
        plt.show()


class SmoothNoise():
    '''
    generate one batch of noise, adding to next_actions.(usually used for TD3)
    '''
    def __init__(self,
                 loc: np.ndarray,  # shape is like action
                 std: float,
                 seed: int = 1,
                 randomness: bool = False,
                 clip: float = 0.5,
                 batch_size: int = 1):
        self.loc = torch.tensor(loc)
        self.std = std
        self.seed = seed
        self.randomness = randomness
        self.batch_size = batch_size
        self.clip = torch.tensor(clip)
        if not randomness:
            np.random.seed(seed)
        # fix seeds series, so noise will randomly generate with the fix seed obtain from seed series
        self.seeds = np.random.randint(0, 10000000, 1000000)
        self.seed_order = 0
        self.randomness = randomness
        print(f'------smooth noise------')
        print(f'normal smooth noise, randomness is {self.randomness} !')
        print(f'-----------------')

    def __call__(self) -> np.ndarray:

        if self.randomness:
            # do not set random seed, so each ou class instantiation will generate different noise
            x = torch.randn(self.batch_size, self.loc.shape[0]) * self.std + self.loc  # (batch_size, action_dim)
            x = x.clamp(-self.clip, self.clip)
        else:
            torch.manual_seed(self.seeds[self.seed_order])
            self.seed_order += 1
            x = torch.randn(self.batch_size, self.loc.shape[0]) * self.std + self.loc  # generate (batch_size,action_dim)
            x = x.clamp(-self.clip, self.clip)

        return x

    def reset(self):
        self.seed_order = 0

    def Plot(self, duration: int, reset=True, label: str = None):
        plt.figure(figsize=(6, 3))
        if reset == True:
            self.reset()
        noise = []
        for i in range(duration):
            noise.append(self.__call__())

        plt.plot(noise,label = label,linewidth = 0.8, color = 'tab:cyan')
        plt.title('Smooth Noise')
        plt.grid()
        plt.legend()
        plt.show()


def get_noise(noise_aliase: str = None, kwargs: dict = None):

    noise_aliases: Dict = {
        'ou': OrnsteinUhlenbeckActionNoise,
        'normal': NormalNoise,
        'smooth': SmoothNoise
}

    noise_type = noise_aliases[noise_aliase]
    noise = noise_type(**kwargs)

    return noise

