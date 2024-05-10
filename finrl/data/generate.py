import os
from finrl.data.yahoo import YahooDownloader
from finrl.data.feature import FeatureEngineer,INDICATORS
import pandas as pd
import itertools
from typing import List


def stock_trade_data_generate(
        data_dir: str,
        start_date: str,
        end_date: str,
        use_technical_indicator: bool = True,
        use_turbulence: bool = True,
        user_defined_feature: bool = False,
        tech_indicator_list: List = None,
        use_vix: bool = True,
        vix_data_dir: str = None,
        dji_dir: str = None,

):
    '''
    生成Trading任务的dataframe数据。

    :param data_dir: constituent stocks data saving fold dir
    :param start_date: start date of the total data
    :param end_date: end date of the total data
    :param use_technical_indicator: if add technical indicators in tech_indicator_list or not
    :param use_turbulence: if add turbulence feature or not
    :param user_defined_feature: if add user defined feature or not
    :param tech_indicator_list: technical indicators added
    :param use_vix: if add vix feature or not
    :param vix_data_dir: vix data saving fold dir
    :return: processed_full dataframe
    '''

    # get constituent stocks data df
    constituent_data_dir = data_dir
    constituent_csv_list = os.listdir(data_dir)
    df=YahooDownloader(
        start_date=start_date,
        end_date=end_date,
        data_dir=constituent_data_dir,
        csv_list=constituent_csv_list
    ).fetch_data()

    # get technical indicators and turbulence if needed
    csv_list = os.listdir(vix_data_dir)
    my_fe = FeatureEngineer(
        use_technical_indicator=use_technical_indicator,
        tech_indicator_list = tech_indicator_list,
        use_vix=use_vix,
        use_turbulence=use_turbulence,
        user_defined_feature = user_defined_feature,
        vix_data_dir=vix_data_dir,
        csv_list=csv_list
    )
    processed = my_fe.preprocess_data(df)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed,on=["date", "tic"], how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date', 'tic'])
    processed_full = processed_full.fillna(0)

    df_base = pd.read_csv(dji_dir)
    df_base.columns = [
        "date",
        "open",
        "high",
        "low",
        "DJI",
        "adjcp",
        "volume",
    ]
    df_base["DJI"] = df_base["adjcp"]
    df_base = df_base.drop(labels="adjcp", axis=1)
    processed_full = pd.merge(left=processed_full,
             right=df_base[['date', 'DJI']],
             how='left',
             left_on='date',
             right_on='date',
             )
    return processed_full


def portfolio_data_generate(
        data_dir: str,
        start_date: str,
        end_date: str,
        use_technical_indicator: bool = True,
        use_turbulence: bool = False,
        user_defined_feature: bool = False,
        tech_indicator_list: List = None,
        use_vix: bool = False,
        vix_data_dir: str  = None,
        dji_dir: str = None,

):
    '''
    生成Portfolio任务的数据。

    :param data_dir: constituent stocks data saving fold dir
    :param start_date: start date of the used data
    :param end_date: end date of the total data
    :param use_technical_indicator: if add technical indicators in tech_indicator_list or not
    :param use_turbulence: if add turbulence feature or not
    :param user_defined_feature: if add user defined feature or not
    :param tech_indicator_list: technical indicators added
    :param use_vix: if add vix feature or not
    :param vix_data_dir: vix data saving fold dir
    :return: df
    '''

    # get constituent data
    constituent_csv_list=os.listdir(data_dir)
    df=YahooDownloader(start_date=start_date, end_date=end_date, data_dir=data_dir, csv_list=constituent_csv_list
                          ).fetch_data()

    # add features
    csv_list = os.listdir(vix_data_dir)
    fe = FeatureEngineer(use_technical_indicator=use_technical_indicator,
                            use_turbulence=use_turbulence,
                            user_defined_feature=user_defined_feature,
                            tech_indicator_list=tech_indicator_list,

                            use_vix=use_vix,
                            vix_data_dir=vix_data_dir,
                            csv_list=csv_list,
                            )
    df = fe.preprocess_data(df)

    # add covariance matrix as states
    df = df.sort_values(['date','tic'], ignore_index=True)
    df.index = df.date.factorize()[0] 
    cov_list = []
    return_list = []
    # look back is one year
    lookback = 252
    for i in range(lookback, len(df.index.unique())):
        data_lookback = df.loc[i - lookback:i, :] 
        price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)
        covs = return_lookback.cov().values
        cov_list.append(covs)

    # merge
    df_cov = pd.DataFrame({'date': df.date.unique()[lookback:], 'cov_list': cov_list, 'return_list': return_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    df_base = pd.read_csv(dji_dir)
    df_base.columns = [
        "date",
        "open",
        "high",
        "low",
        "DJI",
        "adjcp",
        "volume",
    ]
    df_base["dji"] = df_base["adjcp"]
    df_base = df_base.drop(labels="adjcp", axis=1)
    df = pd.merge(left=df,
             right=df_base[['date', 'DJI']],
             how='left',
             left_on='date',
             right_on='date',
             )
    return df
