import sys
sys.append('/mnt/')
import json
import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
from finrl.data.photo_indicator import *
from finrl.data.generate import stock_trade_data_generate
from finrl.models.constants import *
from finrl.models.utils import data_split
from finrl.models.env import StockTradingEnv
from finrl.data.photo_plot import *
from datetime import timedelta


if __name__ == "__main__":

    path = '/mnt/finrl/models/kwargs.json'
    with open(path,'r',encoding='utf-8') as f:
        config = json.load(f)
    print(config)

    ### 导入参数
    path = '/mnt/finrl/models/kwargs.json'
    with open(path,'r',encoding='utf-8') as f:
        config = json.load(f)
    print(config)

    TRAIN_START_DATE,TRAIN_END_DATE='2005-01-01','2021-12-31'
    VALIDATE_START_DATE,VALIDATE_END_DATE='2022-01-01','2022-12-31'
    TEST_START_DATE,TEST_END_DATE='2023-01-01','2024-04-16'
    data_dir = '/mnt/finrl/data/csv/DOW_30/'
    vix_data_dir = '/mnt/finrl/data/csv/^VIX/'
    dji_dir = '/mnt/finrl/data/csv/DJI/DJI.csv'  # .csv

    # 生成数据
    df = stock_trade_data_generate(
        data_dir=data_dir,
        start_date=TRAIN_START_DATE,
        end_date=TEST_END_DATE,
        use_technical_indicator=True,
        use_turbulence=True,
        user_defined_feature=False,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        vix_data_dir=vix_data_dir,
        dji_dir=dji_dir,
    )

    # 图片涵盖的日期
    total_plot_time_range = df.date.unique().tolist()
    # DJIA数据生成图片，获取DJIA数据(尽可能大，防止出现nan)
    djia = pd.read_csv('/mnt/finrl/data/csv/DJI/DJI.csv')
    # 日期设为index
    djia['Date'] = pd.to_datetime(djia['Date'])
    djia = djia.set_index('Date',drop=True)
    # djia的每一列设为float
    def str_float(element):
        return float(element.replace(',',''))
    for column in djia.columns:
        djia[column] = djia[column].apply(str_float,1)/100 # 由于djia的价格都比较大，所以进行缩放。缩放到30支股票的平均价格的尺度上去。平均价格为两位数到三位数，所以将djia的价格除100
    print(djia.shape)


    # 批量生成第一种类型图片
    for d in tqdm.tqdm(total_plot_time_range):
        nowadays = d
        row = djia.index.get_loc(nowadays)
        data1 = djia.iloc[row-140:row]  # 需要当前时刻t过去140条数据
        time_range = data1.index
        fig,type_ = figure1_plot(data1)
        save_path = f'/mnt/finrl/data/figure/figures_DJIA/figure1/{nowadays}_{type_}.png'
        fig.savefig(save_path)
        plt.close(fig)

    # 批量裁剪处理第一种类型图片
    for dir in tqdm.tqdm(os.listdir('/mnt/finrl/data/figure/figures_DJIA/figure1/')):
        path = '/mnt/finrl/data/figure/figures_DJIA/figure1/' + dir
        type_ = int(dir.split('_')[1][0])
        date = dir.split('_')[0]
        fig,_ = figure1_preprocess(path,type_)
        fig.save(f'/mnt/finrl/data/figure/figures_DJIA/figure1_preprocess/{date}.png')

    # 第二类图片
    djia_with_dmi = calculate_dmi(djia,14)
    for d in tqdm.tqdm(df.date.unique()):
        nowadays = d
        row = djia_with_dmi.index.get_loc(nowadays)
        data2 = djia_with_dmi.iloc[row:row+20]
        fig = figure2_plot(data2)
        save_path = f'/mnt/finrl/data/figure/figures_DJIA/figure2/{nowadays}.png'
        fig.savefig(save_path)
        plt.close(fig)

    for dir in tqdm.tqdm(os.listdir('/mnt/finrl/data/figure/figures_DJIA/figure2')):
        path = '/mnt/finrl/data/figure/figures_DJIA/figure2' + dir
        date = dir.split('.')[0]
        fig,_ = figure2_preprocess(path)
        fig.save(f'/mnt/finrl/data/figure/figures_DJIA/figure2_preprocess/{date}.png')

    # 第三类图片
    djia_with_sso = stochastics(djia, 'Low', 'High', 'Close', 14, 3 )
    for d in tqdm.tqdm(df.date.unique()):
        nowadays = d
        row = djia_with_sso.index.get_loc(nowadays)
        data3 = djia_with_sso.iloc[row:row+20]
        fig = figure3_plot(data3)
        save_path = f'/mnt/finrl/data/figure/figures_DJIA/figure3/{nowadays}.png'
        fig.savefig(save_path)
        plt.close(fig)

    for dir in tqdm.tqdm(os.listdir('/mnt/finrl/data/figure/figures_DJIA/figure3')):
        path = '/mnt/finrl/data/figure/figures_DJIA/figure3' + dir
        date = dir.split('.')[0]
        fig,_ = figure3_preprocess(path)
        fig.save(f'/mnt/finrl/data/figure/figures_DJIA/figure3_preprocess/{date}.png')
