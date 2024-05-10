import pandas as pd
import numpy as np


class ImmediateStrategy():

    def __init__(self,
                 w: float,
                 ):
        '''
        when w is 0, it degenerate to sgn(returns)
        :param w: intermediate weighting

        Returns:
        pd.Series: series of position sizes
        '''
        self.w = w

    def calc_returns(self,
                     srs: pd.Series,
                     day_offset: int = 1) -> pd.Series:
        """for each element of a pandas time-series srs,
        calculates the returns over the past number of days
        specified by offset
        Args:
            srs (pd.Series): time-series of prices
            day_offset (int, optional): number of days to calculate returns over. Defaults to 1.
        Returns:
            pd.Series: series of returns
        """
        returns = srs / srs.shift(day_offset).fillna(method='bfill') - 1.0
        return returns


    def calc_trend_intermediate_signal(self,
                                       srs: pd.Series,
                                       ) -> pd.Series:
        """Calculate intermediate strategy
        Args:
            srs (pd.Series): series of prices
            w (float): weight, w=0 is Moskowitz TSMOM
            volatility_scaling (bool, optional): [description]. Defaults to True.
        Returns:
            pd.Series: series of signal
        """
        monthly_returns = self.calc_returns(srs, 21)
        annual_returns = self.calc_returns(srs, 252)


        return self.w * np.sign(monthly_returns) + (1 - self.w) * np.sign(annual_returns)

    def scale_signal(self, y: pd.Series) -> np.ndarray:
        return np.sign(y)



