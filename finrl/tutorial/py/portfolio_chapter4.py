import sys
sys.path.append('/mnt/')
import argparse
import torch
import numpy as np
from torch import nn
from finrl.models.network import *
from finrl.models.agent import DDPG_MCTS
from finrl.data.generate import portfolio_data_generate
from finrl.models.constants import *
from finrl.models.utils import data_split
from finrl.models.env import StockPortfolioEnv


def parser_args():

    parser = argparse.ArgumentParser(description='全部参数组')

    data_group = parser.add_argument_group('数据参数组')
    data_group.add_argument('--stocks_dir', type=str, default='/mnt/finrl/data/csv/DOW_30/', help='股票csv文件路径')
    data_group.add_argument('--vix_data_dir', type=str, default='/mnt/finrl/data/csv/^VIX/', help='VIX csv文件路径')
    data_group.add_argument('--dji_dir', type=str, default='/mnt/finrl/data/csv/DJI/DJI.csv', help='DJIA csv文件路径')
    data_group.add_argument('--train_start_date', type=str, default='2010-01-01', help='训练集起始日期')
    data_group.add_argument('--train_end_date', type=str, default='2021-12-31', help='训练集终止日期')
    data_group.add_argument('--validation_start_date', type=str, default='2022-01-01', help='验证集起始日期')
    data_group.add_argument('--validation_end_date', type=str, default='2022-12-31', help='验证集终止日期')
    data_group.add_argument('--test_start_date', type=str, default='2023-01-01', help='测试集起始日期')
    data_group.add_argument('--test_end_date', type=str, default='2024-01-30', help='测试集终止日期')
    data_group.add_argument('--indicators', type=str, default="macd,rsi_30,cci_30,dx_30", help='使用的技术指标')
    data_group.add_argument('--if_indicator_norm', action='store_true', help='是否对技术指标执行归一化, 默认为False')

    env_group = parser.add_argument_group('环境参数组')
    env_group.add_argument('--hmax', type=int, default=100, help='最大购买/卖出股数上限')
    env_group.add_argument('--initial_amount', type=float, default=1000000.0, help='初始现金')
    env_group.add_argument('--reward_scaling', type=float, default=1e-4, help='计算奖励的缩放因子')
    env_group.add_argument('--buy_cost_pct', type=float, default=1e-3, help='购买手续费比例')
    env_group.add_argument('--sell_cost_pct', type=float, default=1e-3, help='卖出手续费比例')

    policy_group = parser.add_argument_group('策略参数组')
    policy_group.add_argument('--M', type=int, default=2, help='MLP结构激活函数')
    policy_group.add_argument('--N', type=int, default=3, help='MLP结构激活函数')
    policy_group.add_argument('--cnn_activation', type=str, default='relu', help='CNN激活函数')
    policy_group.add_argument('--lstm_input_size', type=int, default=704, help='LSTM输入向量维度')
    policy_group.add_argument('--lstm_hidden_size', type=int, default=1024, help='LSTM隐向量维度')
    policy_group.add_argument('--mlp_activation', type=str, default='relu', help='MLP结构激活函数')    
    policy_group.add_argument('--srl_aliase', type=str, default='d2rl', help='表征学习模型别称') 
    policy_group.add_argument('--srl_hidden_dim', type=int, default=512, help='表征学习模型隐藏层维度') 

    ou_noise_group = parser.add_argument_group('OU噪声参数组')
    ou_noise_group.add_argument('--sigma', type=float, default=0.05, help='控制OU噪声回归速度的参数')
    ou_noise_group.add_argument('--theta', type=float, default=0.10, help='控制OU噪声波动程度的参数')
    ou_noise_group.add_argument('--dt', type=float, default=0.1, help='控制OU噪声平滑程度和波动程度的参数')
    ou_noise_group.add_argument('--randomness', action='store_false', help='每次模拟的噪声是否固定随机性，默认True')

    agent_group = parser.add_argument_group('agent参数组')
    agent_group.add_argument('--buffer_size', type=int, default=int(1e4), help='经验池大小')
    agent_group.add_argument('--batch_size', type=int, default=4, help='小批量大小')
    agent_group.add_argument('--n_updates', type=int, default=5, help='每次小批量抽取的样本的训练次数')
    agent_group.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    agent_group.add_argument('--tau', type=float, default=0.005, help='软更新参数')
    agent_group.add_argument('--policy_lr', type=float, default=5e-8, help='policy学习率')
    agent_group.add_argument('--noise_aliase', type=str, default='ou', help='使用的噪声')
    agent_group.add_argument('--training_start', type=int, default=200, help='训练前预热次数')
    agent_group.add_argument('--target_update_interval', type=int, default=1, help='目标网络参数拷贝间隔')
    agent_group.add_argument('--print_interval', type=int, default=200, help='logger打印间隔')
    agent_group.add_argument('--figure_path', type=str, default='/mnt/finrl/data/figure/figures_DJIA/', help='图片路径')
    agent_group.add_argument('--device', type=str, default='cuda', help='cpu/gpu')
    agent_group.add_argument('--if_clip', action='store_true', help='是否进行梯度裁剪, 默认False')
    agent_group.add_argument('--task', type=str, default='portfolio', help='投资任务')
    agent_group.add_argument('--q_target_mode', type=str, default='redq', help='集成学习方式')
    agent_group.add_argument('--total_updates_times_maximum', type=int, default=138400, help='总训练次数')
    agent_group.add_argument('--mcts_gamma', type=float, default=0.99, help='Backprop阶段更新参数')
    agent_group.add_argument('--expand_length', type=int, default=50, help='Expand阶段交互长度')
    agent_group.add_argument('--children_maximum', type=int, default=5, help='结点最大孩子数')
    agent_group.add_argument('--mcts_C', type=float, default=0.5, help='平衡探索和利用的参数')
    agent_group.add_argument('--random_select_prob', type=float, default=0.2, help='Select阶段，控制继续深入或生成新结点的参数')

    args = parser.parse_args()

    args.indicators = args.indicators.split(',')

    return args

def get_df(args):

    df = portfolio_data_generate(
        data_dir=args.stocks_dir,
        start_date=args.train_start_date,
        end_date=args.test_end_date,
        use_technical_indicator=True,
        use_turbulence=True,
        user_defined_feature=False,
        use_vix=True,
        tech_indicator_list=args.indicators,
        vix_data_dir=args.vix_data_dir,
        dji_dir=args.dji_dir,
    )

    # 切分数据
    df_train = data_split(df, args.train_start_date, args.train_end_date)
    df_validation = data_split(df, args.validation_start_date, args.validation_end_date)
    df_test = data_split(df, args.test_start_date, args.test_end_date)
    df_train_validation = data_split(df, args.train_start_date, args.validation_end_date)


    # 技术指标归一化。
    min_indicator,max_indicator = [],[]
    if args.if_indicator_norm:
        print('正在进行技术指标归一化')
        tic_list = df_train_validation.tic.unique().tolist()
        min_indicator,max_indicator = [],[]

        for indicator in INDICATORS:
            df1 = df_train_validation[['tic',indicator]]
            for tic in tic_list:
                minimum = df1[df1.tic==tic][indicator].min()
                maximum = df1[df1.tic==tic][indicator].max()
                min_indicator.append(minimum)
                max_indicator.append(maximum)
        
        for i in range(df_train.shape[0]):
            tic_ = df_train.iloc[i]['tic']
            tic_index = tic_list.index(tic_)
            for indicator in INDICATORS:
                indicator_index = INDICATORS.index(indicator)
                indicator_column_index = df_train.columns.get_loc(indicator)
                df_train.iat[i,indicator_column_index] = (df_train.iat[i,indicator_column_index] - min_indicator[indicator_index*len(tic_list)+tic_index]) / (max_indicator[indicator_index*len(tic_list)+tic_index] - min_indicator[indicator_index*len(tic_list)+tic_index])

        for i in range(df_train_validation.shape[0]):
            tic_ = df_train_validation.iloc[i]['tic']
            tic_index = tic_list.index(tic_)
            for indicator in INDICATORS:
                indicator_index = INDICATORS.index(indicator)
                indicator_column_index = df_train_validation.columns.get_loc(indicator)
                df_train_validation.iat[i,indicator_column_index] = (df_train_validation.iat[i,indicator_column_index] - min_indicator[indicator_index*len(tic_list)+tic_index]) / (max_indicator[indicator_index*len(tic_list)+tic_index] - min_indicator[indicator_index*len(tic_list)+tic_index])

        for i in range(df_validation.shape[0]):
            tic_ = df_validation.iloc[i]['tic']
            tic_index = tic_list.index(tic_)
            for indicator in INDICATORS:
                indicator_index = INDICATORS.index(indicator)
                indicator_column_index = df_validation.columns.get_loc(indicator)
                df_validation.iat[i,indicator_column_index] = (df_validation.iat[i,indicator_column_index] - min_indicator[indicator_index*len(tic_list)+tic_index]) / (max_indicator[indicator_index*len(tic_list)+tic_index] - min_indicator[indicator_index*len(tic_list)+tic_index])

        for i in range(df_test.shape[0]):
            tic_ = df_test.iloc[i]['tic']
            tic_index = tic_list.index(tic_)
            for indicator in INDICATORS:
                indicator_index = INDICATORS.index(indicator)
                indicator_column_index = df_test.columns.get_loc(indicator)
                df_test.iat[i,indicator_column_index] = (df_test.iat[i,indicator_column_index] - min_indicator[indicator_index*len(tic_list)+tic_index]) / (max_indicator[indicator_index*len(tic_list)+tic_index] - min_indicator[indicator_index*len(tic_list)+tic_index])

    return df_train, df_validation, df_test, df_train_validation

def init_env(args, df_train, df_validation, df_test, df_train_validation):

    # 创建环境
    stock_dim = len(df_train_validation.tic.unique())
    action_dim = stock_dim  # 29

    args.stock_dim = stock_dim
    args.action_dim = action_dim
    args.state_dim = stock_dim * (stock_dim + len(args.indicators))

    env_train = StockPortfolioEnv(
        df=df_train,
        stock_dim=stock_dim,
        hmax=args.hmax,
        initial_amount=args.initial_amount,
        reward_scaling=args.reward_scaling,
        buy_cost_pct=[args.buy_cost_pct]*args.action_dim,
        sell_cost_pct=[args.sell_cost_pct]*args.action_dim,
        tech_indicator_list=args.indicators,
        DATE_START=args.train_start_date,
        DATE_END=args.train_end_date,
    )

    env_validation = StockPortfolioEnv(
        df=df_validation,
        stock_dim=stock_dim,
        hmax=args.hmax,
        initial_amount=args.initial_amount,
        reward_scaling=args.reward_scaling,
        buy_cost_pct=[args.buy_cost_pct]*args.action_dim,
        sell_cost_pct=[args.sell_cost_pct]*args.action_dim,
        tech_indicator_list=args.indicators,
        DATE_START=args.validation_start_date,
        DATE_END=args.validation_end_date,
    )

    env_test = StockPortfolioEnv(
        df=df_test,
        stock_dim=stock_dim,
        hmax=args.hmax,
        initial_amount=args.initial_amount,
        reward_scaling=args.reward_scaling,
        buy_cost_pct=[args.buy_cost_pct]*args.action_dim,
        sell_cost_pct=[args.sell_cost_pct]*args.action_dim,
        tech_indicator_list=args.indicators,
        DATE_START=args.test_start_date,
        DATE_END=args.test_end_date,
    )

    return env_train, env_validation, env_test, args

def main():
    args = parser_args()
    print('preparing data for training... \n')
    df_train, df_validation, df_test, df_train_validation = get_df(args)
    print('data is reday \n')
    env_train, env_validation, env_test, args = init_env(args, df_train, df_validation, df_test, df_train_validation)

    ou_noise_kwargs = {
        'mu':np.array([0]*args.action_dim),
        'sigma':args.sigma,
        'theta':args.theta,
        'dt':args.dt,
        'randomness':args.randomness
    }
    policy_kwargs = {
        'M': args.M,
        'N': args.N,
        'cnn_activation': args.cnn_activation,
        'lstm_input_size': args.lstm_input_size,
        'lstm_hidden_size': args.lstm_hidden_size,
        'env_obs_dim': args.state_dim,
        'action_dim': args.action_dim,
        'mlp_activation': args.mlp_activation,
        'srl_aliase': args.srl_aliase,
        'srl_hidden_dim': args.srl_hidden_dim
    }

    print('begin training')
    for i in range(1,6):
        agent = DDPG_MCTS(
            env_train=env_train,
            env_validation=env_validation,
            env_test=env_test,
            n_updates=args.n_updates,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            target_update_interval=args.target_update_interval,
            policy_lr=args.policy_lr,
            training_start=args.training_start,
            policy_kwargs=policy_kwargs,
            noise_aliase=args.noise_aliase,
            noise_kwargs=ou_noise_kwargs,
            print_interval=args.print_interval,
            figure_path=args.figure_path,
            device=args.device,
            task=args.task,
            train_time=str(i),
            if_clip=args.if_clip,
            M=args.M,
            N=args.N,
            q_target_mode=args.q_target_mode,
            total_updates_times_maximum=args.total_updates_times_maximum,
            mcts_gamma=args.mcts_gamma,
            expand_length=args.expand_length,
            children_maximum=args.children_maximum,
            mcts_C=args.mcts_C,
            random_select_prob=args.random_select_prob
        )
        agent.train()

if __name__ == "__main__":
    main()
