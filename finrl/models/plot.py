import copy
from copy import deepcopy
import pandas as pd
from finrl.data.yahoo import YahooDownloader
import os


def get_daily_return(df: pd.DataFrame, value_col_name: str = "account_value"):
    '''
    get daily return rate with account_value dataframe
    :param df: account_value column
    :param value_col_name: target column
    :return:
    '''
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)


def backtest_stats(account_value: pd.DataFrame,
                   value_col_name: str = "account_value"):
    '''
    use pyfolio(timeseries) to get criterias, and print
    :param account_value: account value dataframe
    :param value_col_name: target column
    :return:
    '''
    dr_test = get_daily_return(account_value, value_col_name=value_col_name)
    perf_stats_all = timeseries.perf_stats(
        returns=dr_test,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )
    print(perf_stats_all)
    return perf_stats_all


def my_get_baseline(data_dir: str, start_date: str, end_date: str, csv_list:str):
    '''
    baseline original data is in csv_list file, we get its path, and use My_YahooDownloader to process data.
    :param data_dir:
    :param start_date:
    :param end_date:
    :param csv_list:
    :return:
    '''
    return My_YahooDownloader(start_date=start_date, end_date=end_date, data_dir = data_dir, csv_list = csv_list ).fetch_data()


def my_backtest_plot(
    account_value: pd.DataFrame,
    data_dir: str,
    baseline_start: str = None,
    baseline_end: str = None,
    value_col_name: str = "account_value",
):
    df = deepcopy(account_value)
    df["date"] = pd.to_datetime(df["date"])
    test_returns = get_daily_return(df, value_col_name=value_col_name)

    baseline_df = my_get_baseline(data_dir = data_dir, start_date = baseline_start, end_date = baseline_end ,csv_list = os.listdir(data_dir))

    baseline_df["date"] = pd.to_datetime(baseline_df["date"], format="%Y-%m-%d")
    baseline_df = pd.merge(df[["date"]], baseline_df, how="left", on="date")
    baseline_df = baseline_df.fillna(method="ffill").fillna(method="bfill")
    baseline_returns = get_daily_return(baseline_df, value_col_name="close")

    with pyfolio.plotting.plotting_context(font_scale=1.1):
        pyfolio.create_full_tear_sheet(returns=test_returns, benchmark_rets=baseline_returns, set_context=False)