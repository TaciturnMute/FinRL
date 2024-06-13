import numpy as np
import pandas as pd
from scipy.optimize import minimize

class BestCRP():

    def __init__(
            self,
            df_train_validation: pd.DataFrame,
            df_trade: pd.DataFrame,
    ):

        df_cumulative_returns = pd.DataFrame()
        for tic, df_tic in df_train_validation.groupby('tic'):
            df_cumulative_returns[tic] = (df_tic.close / df_tic.shift(1).close).dropna()
        X = df_cumulative_returns
        self.w = self.opt_weights(X)

    def opt_weights(self, X, max_leverage=1):
        x_0 = max_leverage * np.ones(X.shape[1]) / float(X.shape[1])
        objective = lambda w: -np.prod(X.dot(w))
        cons = ({'type': 'eq', 'fun': lambda w: max_leverage - np.sum(w)},)
        bnds = [(0., max_leverage)] * len(x_0)
        res = minimize(objective, x_0, bounds=bnds, constraints=cons, method='slsqp', options={'ftol': 1e-07})
        return res.x

    def decide(self):
        return self.w