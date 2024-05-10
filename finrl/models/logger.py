import numpy as np
import torch
import time
from collections import defaultdict


class episode_logger():
    # for off policy algorithms, like ddpg\td3\sac\
    def __init__(self):
        self.episode = 0
        self.timesteps = 0
        self.total_timesteps = 0
        self.start_time = time.time()
        self.record_dict = defaultdict(lambda :['*'])
        # self._plot_record_dict = {}  # record total training process
        self.total_updates = 0

    def record(self,**kwargs):
        # record result in corresponding dict with the parameter name
        for key,value in kwargs.items():
            self.record_dict[key].append(value.detach().numpy() if torch.is_tensor(value) else value)

    def reset(self):
        # reset the record dict for each episode
        self.episode += 1
        print(f'******************************   Episode : {self.episode}  ********************************')
        self.record_dict = defaultdict(lambda :['*'])
        self.timesteps = 0

    def show(self, interval=100):

        print(f' ---------------------------------------')
        print(f'| Episode {self.episode}, Timesteps {self.timesteps}')
        print(f'| in the last {interval} timesteps :')
        for key in sorted(self.record_dict.keys()):
            name = key
            if key=='time':
                continue
            else:
                value = self.record_dict[key]
                star_index = value.index('*')
                if name == 'asset':
                    mean = format(value[-1], '.4f')
                else:
                    mean = format(np.mean(value[star_index+1:]), '.4f')
                    # print(len(value[last_star_index+1:]))
                print('| mean of ' + name + ' is ' + ' '*(14-len(name)) + '| ' + mean)
                self.record_dict[key].pop(star_index)
                self.record_dict[key].append('*')
        print(f'| total_timesteps is        | {self.total_timesteps}')
        print(f'| total_updates is          | {self.total_updates}')
        print(f' ---------------------------------------')

    def total_updates_plus(self,n_updates=1):
        # training times
        self.total_updates += n_updates

    def timesteps_plus(self):
        # interact times with env
        self.timesteps += 1
        self.total_timesteps += 1

    def print_elapsed_time(self):
        self.end_time = time.time()
        print(f'elapsed time is {(self.end_time - self.start_time)//60} min')


class rollout_logger():
    # for on-policy algorithms, like ppo\a2c
    def __init__(self):
        self.episode = 0
        self.timesteps = 0
        self.total_timesteps = 0
        self.start_time = time.time()
        self.record_dict = defaultdict(list)
        self._plot_record_dict = {}
        self.total_updates = 1
        self.have_inferenced = False

    def reset(self):
        # reset the record dict for each rollout

        self.record_dict = defaultdict(list)

    def record(self,**kwargs):
        # record result in corresponding dict according to the parameter name
        for key,value in kwargs.items():
            self.record_dict[key].append(value.detach().numpy() if torch.is_tensor(value) else value)

    def show(self):
        # print mean results
        print(f' ---------------------------------------')
        print(f'| Episode {self.episode}, Timesteps {self.timesteps}, Total_Timesteps {self.total_timesteps}')
        print(f'| in the last rollout training process :')
        for key in sorted(self.record_dict.keys()):
            if isinstance(self.record_dict[key][0], str):
                continue
            else:
                name = key
                value = self.record_dict[key]
                if name == 'asset':
                    mean = mean = format(value[-1], '.4f')
                else:
                    mean = format(np.mean(value), '.4f')
                print('| mean of ' + name + ' is ' + ' '*(14-len(name)) + '| ' + mean)
        print(f' ---------------------------------------')

    def episode_start(self):
        self.have_inferenced = False
        self.episode += 1
        self._plot_record_dict[self.episode] = defaultdict(list)
        self.timesteps = 0
        print(f'******************************   Episode : {self.episode}  ********************************')

    def print_elapsed_time(self):
        self.end_time = time.time()
        print(f'elapsed time is {(self.end_time - self.start_time)//60} min')

    def timesteps_plus(self):
        self.timesteps += 1
        self.total_timesteps += 1

    def total_updates_plus(self):
        # training times
        self.total_updates += 1

    def save(self, name):
        np.save('training_results'+ name + '.npy', self._plot_record_dict)
