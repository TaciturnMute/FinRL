import numpy as np
import pandas as pd


class UniformBAH():
    def __init__(
            self,
            df_train_validation: pd.DataFrame,
            df_trade: pd.DataFrame,
    ):
        self.w = None
        self.m = len(df_train_validation.tic.unique())

    def decide(self, last_weights):
        # keep the weight unchanged, as same as the last portfolio weight
        # set default weight if needed
        if self.w is None:
            self.w = np.array([1] * self.m) / self.m

        else:
            self.w = last_weights
        return self.w


