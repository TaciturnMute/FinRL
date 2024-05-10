import numpy as np
import pandas as pd


class UniversalPortfolio():
    def __init__(
            self,
            df_train_validation: pd.DataFrame = None,
            df_trade: pd.DataFrame = None,
            evaluation_point: int = 4,
    ):
        self.m = len(df_train_validation.tic.unique())  # 资产总数
        self.W = np.matrix(self.binnings(evaluation_point, self.m) / evaluation_point)
        self.S = np.matrix(np.ones(self.W.shape[0]).reshape(-1, 1))  # not normalization yet

        self.data = []
        for tic, df_ticker in df_trade.groupby('tic'):
            self.data.append((df_ticker.close / df_ticker.shift(1).close).dropna().values)
        self.data = np.array(self.data)
        self.data = pd.DataFrame(self.data.T)
        self.data.index = df_trade.date.unique()[1:]

        self.w = None

    def binnings(self, n, k, cache={}):
        if n == 0:
            return np.zeros((1, k))
        if k == 0:
            return np.empty((0, 0))
        args = (n, k)
        if args in cache:
            return cache[args]
        a = self.binnings(n - 1, k, cache)
        a1 = a + (np.arange(k) == 0)
        b = self.binnings(n, k - 1, cache)
        b1 = np.hstack((np.zeros((b.shape[0], 1)), b))
        result = np.vstack((a1, b1))
        cache[args] = result
        return result

    def decide(self):
        # 对每期新观测到的vt，计算每个分配权重的权重系数
        trade_day = 1
        if self.w is None:
            self.w = np.ones(self.m) / self.m
        else:
            x = np.matrix(self.data.iloc[trade_day].values.reshape(self.m, 1))
            self.S = np.multiply(self.W * x, self.S)
            self.w = self.W.T * self.S
            self.w = self.w / np.sum(self.w)
            trade_day += 1
        self.w = np.ravel(self.w)
        return self.w