import sys
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from PIL import Image
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from typing import Iterable,Union,Dict
from copy import deepcopy
sys.path.append('/mnt/')


def load_image(image_path: str) -> torch.Tensor:
    """ 加载单张图片 """
    transform = transforms.ToTensor()
    with Image.open(image_path) as img:
        # print(np.array(img))
        return transform(img)

def load_images_batch(image_paths: list) -> torch.Tensor:
    """ 使用多线程高效地加载图片批次 """
    with ThreadPoolExecutor() as executor:
        images = torch.stack(list(executor.map(load_image, image_paths)))  # 一个batch的tensor
    return images

def get_figure(path: str, dates: list, device) -> torch.Tensor:
    # path = 'C:/Users/Administrator/Desktop/毕设/finrl/experiments/chart_generate/figures/'
    # 一定保证图片文件夹下的命名为  figure1_preprocess，figure2_preprocess，figure3_preprocess
    l = []
    for i in range(1,4):
        image_paths = [path + f'figure{i}_preprocess/' + date + '.png' for date in dates]
        batch = load_images_batch(image_paths)
        l.append(batch)

    return torch.stack(l).to(device)

def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.

    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo

def polyak_update(params:Iterable[torch.Tensor],target_params:Iterable[torch.Tensor],tau)->None:
    #execute soft update, tau=1 means hard update. in general, tau is 0.005
    with torch.no_grad():
        for param,target_param in zip_strict(params,target_params):
            target_param.data.mul_(1 - tau)
            torch.add(input = target_param.data, other = param.data, alpha = tau, out = target_param.data)

def get_daily_return(df: pd.DataFrame, value_col_name: str = "account_value"):
    '''
    get daily return rate with account_value dataframe
    :param df: account_value column
    :param value_col_name: target column
    :return:
    '''
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)

def CompleteShape(data: np.ndarray):

    batch_size = data.shape[0]  # first dim of data is batch_size by default
    if len(data.shape) == 2:    # each data in the batch is a vector or a scalar
        # if data.shape[1] > 1:     # vector
        #     dim1 = data.shape[1]
        #     data = data.reshape(batch_size, 1, dim1)
        #     return data
        # else:
        #     return data           # scalar and has complete shape
        return data
    elif len(data.shape) == 1:   # each data in this batch is scalar but has no complete shape
        return data.reshape(batch_size, 1)
    else: # each data in the batch is at least two or higher dimensionality, do not need to process
        return data

def trans_tensor(data):
    return torch.tensor(data, dtype=torch.float32)

class linear_schedule():
    def __init__(self, start_point: float, end_point: float, end_time: int):
        self.start_point = start_point
        self.end_point = end_point
        self.end_time = end_time
        self.current = start_point
        self.inc = (self.end_point - self.start_point) / self.end_time
        self.bound = min if end_point > start_point else max

    def _reset(self):
        self.current = self.start_point

    def __call__(self):
        value = self.bound(self.current + self.inc, self.end_point)
        self.current = value
        return value

    def Plot(self):
        self._reset()
        epsilons = []
        for i in range(int(self.end_time * 1.3)):
            epsilons.append(self.__call__())
        plt.figure(figsize=(10, 5))
        plt.plot(epsilons, label='linear_schedule')
        plt.legend()
        plt.show()
        
def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor

def data_split(df: pd.Series, start, end, target_date_col="date"):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(pd.to_datetime(df[target_date_col]) >= pd.to_datetime(start)) & (pd.to_datetime(df[target_date_col]) < pd.to_datetime(end))]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data
    
