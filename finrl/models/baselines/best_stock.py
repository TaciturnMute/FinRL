import numpy as np
import pandas as pd

class BestStock():

    def __init__(
            self,
            df_train_validation: pd.DataFrame = None,
            df_trade: pd.DataFrame = None,
    ):
        self.w = None
        self.m = len(df_train_validation.tic.unique())

        df_cumulative_returns = pd.DataFrame()
        tickers = df_train_validation.tic.unique()
        for ticker in tickers:
            df_ticker = df_train_validation[df_train_validation.tic == ticker].close
            df_cumulative_returns[ticker] = (df_ticker / df_ticker.shift(1)).dropna().cumprod()
        df_cumulative_returns.index = df_train_validation.date.unique()[1:]
        self.best_index = np.argmax(df_cumulative_returns.iloc[-1])

    def decide(self):

        if self.w is None:
            self.w = np.zeros(self.m)
            self.w[self.best_index] = 1
        return self.w