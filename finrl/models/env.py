from __future__ import annotations
import gym
import numpy as np
import pandas as pd
from gym import spaces
from typing import Tuple, List
from gym.utils import seeding
from finrl.models.utils import get_daily_return
from finrl.models.metrics import *
import copy


class StockTradingEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: List[int],
        buy_cost_pct: List[float],
        sell_cost_pct: List[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: List[str],
        turbulence_threshold: float = None,
        risk_indicator_col: str = "turbulence",
        day: int = 0,
        initial=True,
        previous_state=[],
        reward_aliase: str =  None,
        DATE_START: str = None,
        DATE_END: str = None,
        cash_norm_factor: float = None,
        num_share_norm_factor: float = None,
        if_price_norm: bool = None,
        if_indicator_norm: bool = None,
        if_num_share_norm: bool = None,
        max_price: list = None,
        min_price: list = None,
    ):
        self.reward_aliases = ['asset_diff', 'sharpe_rario_diff']
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_share_norm_factor = num_share_norm_factor  # 持股数归一化
        self.num_stock_shares = num_stock_shares 
        self.initial_amount = initial_amount
        self.cash_norm_factor = cash_norm_factor
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.state_dim = self.state_space
        self.action_dim = self.action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,)) # 测试时采样有用
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))
        self.data = self.df.loc[self.day, :]
        self.date = self._get_date()
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.reward_aliase = reward_aliase
        self.if_price_norm = if_price_norm  
        self.if_indicator_norm = if_indicator_norm
        self.if_num_share_norm = if_num_share_norm
        self.max_price = max_price
        self.min_price = min_price
        self.state = self._initiate_state()
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = []
        # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        real_price = np.array(self.state[1 : 1 + self.stock_dim]) if not self.if_price_norm else self._inverse_norm_price(np.array(self.state[1 : 1 + self.stock_dim]))
        real_num_shares = np.array(self.num_stock_shares) if not self.if_num_share_norm else np.array(self.num_stock_shares) / self.num_share_norm_factor
        self.asset_memory.append(self.initial_amount + np.sum(real_num_shares * real_price)) 
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = [self.state] # 初始state，即s0
        self.date_memory = [self._get_date()]
        self._seed()
        self.reward = np.nan   # 也许在使用sharpe ratio reward时有用到
        self.DATE_START = DATE_START
        self.DATE_END = DATE_END

    def _sell_stock(self, index: int, action: int):
        def _do_sell_normal():
            if (self.state[index + 2 * self.stock_dim + 1] != True):      # stock's macd
              # check if the stock is able to sell, for simlicity we just add it in techical index
                # if self.state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0: # 持股大于0
                    # Sell only if current asset is > 0
                    sell_num_shares = min(abs(action), self.state[index + self.stock_dim + 1] / self.num_share_norm_factor)   # 真实的卖出股数
                    real_price = self.state[index + 1] if not self.if_price_norm else self._inverse_norm_price_single(self.state[index + 1],index)
                    sell_amount = (real_price * sell_num_shares * (1 - self.sell_cost_pct[index])) # (扣除手续费)出售赚得的钱
                    # update balance
                    self.state[0] += sell_amount * self.cash_norm_factor  # 注意缩放
                    # update holding shares
                    self.state[index + self.stock_dim + 1] -= (sell_num_shares * self.num_share_norm_factor)  # 注意缩放
                    self.cost += (real_price * sell_num_shares * self.sell_cost_pct[index])  # 也需要缩放
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0: # 价格>0
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0: # 持股 > 0
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1] / self.num_share_norm_factor # 全部卖出，不受hmax限制
                        real_price = self.state[index + 1] if not self.if_price_norm else self._inverse_norm_price_single(self.state[index + 1],index)
                        sell_amount = (real_price * sell_num_shares * (1 - self.sell_cost_pct[index]))
                        # update balance
                        self.state[0] += sell_amount * self.cash_norm_factor  ## 注意缩放
                        self.state[index + self.stock_dim + 1] = 0 # 持股数=0
                        self.cost += (real_price * sell_num_shares * self.sell_cost_pct[index]) # 缩放
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if (self.state[index + 2 * self.stock_dim + 1] != True):  # check if the stock is able to buy
                # if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                real_price = self.state[index + 1] if not self.if_price_norm else self._inverse_norm_price_single(self.state[index + 1],index)
                available_amount = (self.state[0] / self.cash_norm_factor) // (
                    real_price * (1 + self.buy_cost_pct[index])
                )  
                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    real_price  # price
                    * buy_num_shares  # shares
                    * (1 + self.buy_cost_pct[index])  # add cost
                )

                self.state[0] -= buy_amount * self.cash_norm_factor

                self.state[index + self.stock_dim + 1] += (buy_num_shares * self.num_share_norm_factor)  # 需要缩放

                self.cost += (
                    real_price * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _cal_total_asset(self):
        # get total asset value at present
        # 返回真实cash
        real_price = np.array(self.state[1: (self.stock_dim + 1)]) if not self.if_price_norm else self._inverse_norm_price(np.array(self.state[1: (self.stock_dim + 1)]))
        total_asset = (self.state[0] / self.cash_norm_factor) + sum(real_price * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]) / self.num_share_norm_factor)
        return total_asset

    def _cal_reward(self):
        # calculate reward
        if self.reward_aliase == 'asset_diff':
            self.reward = (self.end_total_asset - self.begin_total_asset) * self.reward_scaling
        elif self.reward_aliase == 'sharpe_ratio_diff':
            # 根据历史return，计算夏普率
            returns = pd.DataFrame(self.asset_memory)
            returns.insert(0,"date", list(self.date_memory))
            returns.dropna()
            returns.columns = ['date', 'account_value']
            if np.isnan(self.reward) or len(self.date_memory) <= 10:
                self.reward = np.random.randn(1)[0] * 2
            else:
                self.reward = (sharpe_ratio(get_daily_return(returns)) - sharpe_ratio(
                    get_daily_return(returns.iloc[:-1]))) * self.reward_scaling
        else:
            assert self.reward_aliase in self.reward_aliases, \
                f"invalid reward type, supported reward types are{self.reward_aliases}"

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if not self.terminal:
            actions = (actions * self.hmax).astype(int)
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)  # sell all stocks(all actions is minus)
            self.begin_total_asset = self._cal_total_asset()  # get total asset before this trade start

            # get buy stocks index and sell stocks index
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
            for index in buy_index:
                actions[index] = self._buy_stock(index, actions[index])
            self.actions_memory.append(actions)  # actual executed actions

            # state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.next_data = self.df.loc[self.day+1, :] if self.day < len(self.df.index.unique()) - 1 else self.data
            self.date = self._get_date()
            self.next_date = self._get_next_date()
            if self.turbulence_threshold is not None:   # update turbulence
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()       # state transfer, get St+1

            self.end_total_asset = self._cal_total_asset()

            self.asset_memory.append(self.end_total_asset)
            self.date_memory.append(self._get_date())
            self.state_memory.append(self.state)

            self._cal_reward()
            self.rewards_memory.append(self.reward)

            return np.array(self.state), self.reward, self.terminal, {}
        else:
            self.returns = pd.DataFrame(self.asset_memory)
            self.returns.insert(0,"date", list(self.df.date.unique()))
            self.returns.dropna()
            self.returns.columns = ['date', 'account_value']
            self.returns = get_daily_return(self.returns)    # daily return
            return np.array(self.state), self.reward, self.terminal, {}

    def reset(self) -> np.ndarray:

        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.next_data = self.df.loc[self.day+1, :]
        self.date = self._get_date()
        self.next_date = self._get_next_date()
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.episode += 1
        self.reward = np.nan

        self.state = self._initiate_state()
        self.state_memory = [self.state]

        # 价格归一化
        real_price = np.array(self.state[1 : 1 + self.stock_dim]) if not self.if_price_norm else self._inverse_norm_price(np.array(self.state[1 : 1 + self.stock_dim]))

        if self.initial:
            self.asset_memory = [
                self.initial_amount  # 不需要缩放，记录真实cash即可
                + np.sum(
                    np.array(self.num_stock_shares)   # 这里的持股数不缩放
                    * real_price
                )
            ]
        else:
            # self.previous_state传入的时候，cash是真实的，但是在_initiate_state中创建state时，需要缩放。
            previous_total_asset = self.previous_state[0] + sum(
                real_price
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]   # 持股数不缩放
                )
            )
            self.asset_memory = [previous_total_asset]

        return np.array(self.state)

    def _initiate_state(self) -> list:
        if self.initial:   # default
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.initial_amount * self.cash_norm_factor]
                    + self.data.close.values.tolist()
                    + [n*self.num_share_norm_factor for n in self.num_stock_shares]   # 持股数缩放
                    + sum( (self.data[tech].values.tolist() for tech in self.tech_indicator_list), [], )
                )  # append initial stocks_share to initial state, instead of all zero
            else:
                # for single stock
                state = (                             
                    [self.initial_amount * self.cash_norm_factor]
                    + [self.data.close]
                    + [0] * self.stock_dim * self.num_share_norm_factor
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            # Using Previous State
            # 持股数和cash都需要缩放。
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0] * self.cash_norm_factor]
                    + self.data.close.values.tolist()
                    + [n*self.num_share_norm_factor for n in self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]]
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [self.previous_state[0] * self.cash_norm_factor]
                    + [self.data.close]
                    + [n*self.num_share_norm_factor for n in self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        return state

    def from_given_state(self,
                         day,
                         cost=None,
                         trades=None,
                         asset_memory=None,
                         date_memory=None,
                         rewards_memory=None,
                         actions_memory=None,

                         previous_state=None,
                         state_memory=None,
                         turbulence=None,

                         last_weights=None,
                         portfolio_value=None,
                         portfolio_value_vector=None,
                         portfolio_return_memory=None,
                         ):
        # 根据断点信息恢复状态
        self.day = day
        self.data = self.df.loc[day,:]
        self.next_data = self.df.loc[self.day+1, :]  ### 
        self.date = self._get_date()
        self.next_date = self._get_next_date()
        self.tubulence = turbulence
        self.cost = cost
        self.trades = trades
        # if len(self.df.tic.unique()) > 1:
        #         state = (
        #             [previous_state[0] * self.cash_norm_factor]
        #             + self.data.close.values.tolist()
        #             + [n*self.num_share_norm_factor for n in previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]]
        #             + sum(
        #                 (
        #                     self.data[tech].values.tolist()
        #                     for tech in self.tech_indicator_list
        #                 ),
        #                 [],
        #             )
        #         )
        # else:
        #         state = (
        #             [previous_state[0] * self.cash_norm_factor]
        #             + [self.data.close]
        #             + [n*self.num_share_norm_factor for n in previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]]
        #             + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
        #         )
        
        # 根据断点信息恢复memory
        self.asset_memory = copy.deepcopy(asset_memory)
        self.state_memory = copy.deepcopy(state_memory)
        self.rewards_memory = copy.deepcopy(rewards_memory)
        self.actions_memory = copy.deepcopy(actions_memory)
        self.date_memory = copy.deepcopy(date_memory)
        # self.state = state
        self.state = copy.deepcopy(previous_state)
        return 

    
    def _update_state(self) -> list:
        # 注意，update state函数调用是在t+1时刻调用，得到st+1，此时的state已经部分得到了更新，所以可以利用持股数目和cash的信息。
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])   # 已经缩放。
                + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
            )

        else:
            # for single stock
            state = (
                [self.state[0] * self.cash_norm_factor]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])   # 已经缩放
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:  # 股票数大于1，那么self.data行数等于股票数
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date
    
    def _get_next_date(self):
    
        if len(self.df.tic.unique()) > 1:  # 股票数大于1，那么self.data行数等于股票数
            next_date = self.next_data.date.unique()[0]
        else:
            next_date = self.next_data.date
        return next_date 

    
    def _inverse_norm_price(self, price: np.ndarray) -> np.ndarray:
        # 输入price部分向量，输出逆归一化后的向量，即真实价格
        return price * (np.array(self.max_price) - np.array(self.min_price)) + np.array(self.min_price)
    
    def _inverse_norm_price_single(self, price: float, index: int) -> float:
        # 输入某个价格状态以及对应的股票索引，输出真实价格
        return price * (self.max_price[index] - self.min_price[index]) + self.min_price[index]
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



class StockPortfolioEnv(gym.Env):
    '''
    Pt = sum(portfolio_value_vector * price_pct)
    '''
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame = None,
        stock_dim: int = None,
        hmax: int = None,
        initial_amount: int = None,
        buy_cost_pct: list = None,
        sell_cost_pct: list = None,
        reward_scaling: float = None,
        state_space: int = None,
        action_space: int = None,
        tech_indicator_list: list = None,   # indicators env used
        turbulence_threshold=None,
        lookback: int = 252,
        day: int = 0,
        reward_aliase: str = None,
        DATE_START: str = None,
        DATE_END: str = None,

    ):
        self.reward_aliases = ['asset_diff', 'asset_diff_dji', 'sharpe_ratio_diff']
        self.day = day
        self.lookback = lookback  # 暂时没用到
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_dim = state_space
        self.action_dim = action_space
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        # action_space normalization and shape is self.stock_dim
        # self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))  # continuous
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        # 状态空间为二维
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(self.state_space + len(self.tech_indicator_list), self.state_space),
                                            )

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.date = self._get_date()
        self.covs = self.data["cov_list"].values[0]
        # 二维array
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0,
        )
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.last_weights = np.array([1] * self.stock_dim) / self.stock_dim  # 向量，当前资产分配权重。后续可以参照设置previous_state。
        self.portfolio_value = self.initial_amount # 标量，当前资产总价值
        self.portfolio_value_vector = self.portfolio_value * self.last_weights# 向量

        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]  # 在reward 为asset之差时，记录的就是reward。
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        self.cost = 0
        self.trades = 0
        self.reward_aliase = reward_aliase
        self.reward = np.nan
        self.rewards_memory = [0]
        self.date_range = [self.df.date.unique()[0],self.df.date.unique()[-1]]  # 数据集的日期范围
        if self.reward_aliase == 'asset_diff_dji':
            self._dji_bah()
        self.DATE_START = DATE_START
        self.DATE_END = DATE_END

    def _dji_bah(self):
        self.dji = self.df.drop_duplicates(subset=['date'])[['DJI','date']]
        self.dji_shares = self.initial_amount / self.dji.iloc[0]['DJI']
        self.dji_assets = copy.deepcopy(self.dji)
        self.dji_assets.set_index(keys='date', drop=True, inplace=True)
        self.dji_assets['DJI'] = self.dji_assets['DJI'] * self.dji_shares

    def _transaction(self, weights):
        def _sell_asset(index, sell_value) -> float:
            self.portfolio_value -= sell_value
            self.portfolio_value_vector[index] -= sell_value
            # the sold value divided into two parts: Cash and Transaction Cost
            self.cost += sell_value * self.sell_cost_pct[index]  # add sell transaction cost
            cash_obtain_in_this_sell = sell_value * (1 - self.sell_cost_pct[index])  # obtain less
            self.trades += 1
            return cash_obtain_in_this_sell

        def _buy_asset(index, buy_value, cash) -> Tuple[float, bool]:
            # check if the asset is able to buy, cash should > 0
            if_terminal = False
            avaliable_value = cash / (1 + self.buy_cost_pct[index])  # max asset value that can purchase
            if avaliable_value < buy_value:
                buy_value = avaliable_value
                if_terminal = True
            cash -= buy_value * (1 + self.buy_cost_pct[index])  # 最多是cash。
            # cash used is divided into two parts: Transaction cost and Bought Asset Value.
            self.cost += buy_value * self.buy_cost_pct[index]
            self.portfolio_value_vector[index] += buy_value
            self.portfolio_value += buy_value
            self.trades+=1
            return cash, if_terminal

        # assert abs(sum(self.portfolio_value_vector) - self.portfolio_value) < 0.01
        new_portfolio_value_vector = self.portfolio_value * weights # 新的资产分配方案

        actions = new_portfolio_value_vector - self.portfolio_value_vector  # get change value  资产差额，需要调整的部分

        argsort_actions = np.argsort(actions)
        # attention: buy order is: the bigger amount, the higher up the stock will appear
        buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]  # get buy asset index
        sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]  # get sell asset index

        # sell first
        cash = 0
        for index in sell_index:
            cash += _sell_asset(index, abs(actions[index]))

        # then buy, the cost can't be larger than cash
        for index in buy_index:
            cash, terminal = _buy_asset(index, abs(actions[index]), cash)
            if terminal:
                break

    def _cal_reward(self):

        # calculate reward
        if self.reward_aliase == 'asset_diff':
            self.reward = (self.portfolio_value - self.last_portfolio) * self.reward_scaling
        elif self.reward_aliase == 'sharpe_ratio_diff':
            returns = pd.DataFrame(self.asset_memory)
            returns.insert(0,"date", list(self.date_memory))
            returns.dropna()
            returns.columns = ['date', 'account_value']
            if np.isnan(self.reward) or len(self.date_memory) <= 10:
                self.reward = np.random.randn(1)[0] * 2
            else:

                self.reward = (sharpe_ratio(get_daily_return(returns)) - sharpe_ratio(
                    get_daily_return(returns.iloc[:-1]))) * self.reward_scaling
        elif self.reward_aliase == 'asset_diff_dji':
            self.reward = (self.portfolio_value - self.dji_assets.loc[self.date].values[0]) * self.reward_scaling
        else:
            assert self.reward_aliase in self.reward_aliases, \
                f"invalid reward type, supported reward types are{self.reward_aliases}"

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        assert len(actions.shape) == 1
        # terminal
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if not self.terminal:

            # save portfolio before action is executed
            self.last_portfolio = self.portfolio_value   # 交易费也是动作带来的影响，所以计入奖励内。

            # get action
            weights = self.softmax(actions)

            # adjust the portfolio allocation and consider transaction cost
            self._transaction(weights)

            last_data = self.data  # vt-1

            # state transferring
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.next_data = self.df.loc[self.day + 1, :] if self.day < len(self.df.index.unique()) - 1 else self.data
            self.date = self._get_date()   # next date
            self.next_date = self._get_next_date()
            self.covs = self.data["cov_list"].values[0]
            self.state = np.append(
                np.array(self.covs),
                [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
                axis=0,
            )

            # calcualte portfolio return
            # individual stocks' return * weight
            # after next state is observed, get Pt
            # update portfolio value
            price_pct_vector = (self.data.close.values / last_data.close.values)  # vt/vt-1
            portfolio_return = sum((price_pct_vector - 1) * weights)
            # new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value_vector = self.portfolio_value_vector * price_pct_vector  # Pt distribution
            self.portfolio_value = sum(self.portfolio_value_vector) # Pt
            self.last_weights = self.portfolio_value_vector / self.portfolio_value
            # save into memory
            self.actions_memory.append(weights)
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(self.portfolio_value)

            self._cal_reward()
            self.rewards_memory.append(self.reward)

            return self.state, self.reward, self.terminal, {}
        else:
            self.date = self._get_date()  # data没变，所以终止时刻的date也不变。
            self.next_date = self._get_next_date()
            self.returns = pd.DataFrame(self.asset_memory)
            self.returns.insert(0,"date", list(self.df.date.unique()))
            self.returns.dropna()
            self.returns.columns = ['date', 'account_value']
            self.returns = get_daily_return(self.returns)    # daily return

            return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.next_data = self.df.loc[self.day+1, :]
        self.date = self._get_date()
        self.next_date = self._get_next_date()
        # load states
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0,
        )
        self.last_weights = np.array([1] * self.stock_dim)/self.stock_dim
        self.portfolio_value = self.initial_amount
        self.portfolio_value_vector = self.portfolio_value * self.last_weights
        self.cost = 0
        self.trades = 1
        self.terminal = False
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        self.reward = np.nan
        self.rewards_memory = [0]
        return self.state

    def from_given_state(self,
                         day,
                         cost=None,
                         trades=None,
                         asset_memory=None,
                         date_memory=None,
                         rewards_memory=None,
                         actions_memory=None,

                         previous_state=None,
                         state_memory=None,
                         turbulence=None,

                         last_weights=None,
                         portfolio_value=None,
                         portfolio_value_vector=None,
                         portfolio_return_memory=None,
                         ):
        # 根据断点信息恢复状态
        self.day = day
        self.data = self.df.loc[self.day,:]
        self.next_data = self.df.loc[self.day+1, :]
        self.date = self._get_date()
        self.next_date = self._get_next_date()
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0,
        )
        self.last_weights = last_weights
        self.portfolio_value = portfolio_value
        self.portfolio_value_vector = portfolio_value_vector
        self.cost = cost
        self.trades = trades

        # 根据断点信息恢复memory
        self.asset_memory = copy.deepcopy(asset_memory)
        # self.state_memory = state_memory.deepcopy()
        self.rewards_memory = copy.deepcopy(rewards_memory)
        self.actions_memory = copy.deepcopy(actions_memory)
        self.date_memory = copy.deepcopy(date_memory)
        self.portfolio_return_memory = copy.deepcopy(portfolio_return_memory)

        return 

    def render(self, mode="human"):
        return self.state

    def softmax(self, actions):
        e_x = np.exp(actions - np.max(actions))
        return e_x / e_x.sum(axis=0)  # 对于多维数组，axis=0表示按列求和

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date
    
    def _get_next_date(self):
    
        if len(self.df.tic.unique()) > 1:  # 股票数大于1，那么self.data行数等于股票数
            next_date = self.next_data.date.unique()[0]
        else:
            next_date = self.next_data.date
        return next_date 
