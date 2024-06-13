import numpy as np

class LongOnlyStrategy():

    def __init__(self,action_dim: int):
        self.action_dim = action_dim

    def cal_signal(self):
        return np.array(self.action_dim*[1])
