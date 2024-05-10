import numpy as np
import pandas as pd


class UniformCRP():

    def __init__(
            self,
            df_train_validation: pd.DataFrame,
            df_trade: pd.DataFrame,
    ):
        self.w = None
        self.m = len(df_train_validation.tic.unique())

    def decide(self):
        if self.w is  None:
            self.w = np.array([1] * self.m) / self.m
        return self.w







