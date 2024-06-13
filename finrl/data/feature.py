from stockstats import StockDataFrame as Sdf
from finrl.data.yahoo import YahooDownloader
import pandas as pd
import numpy as np

INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

class FeatureEngineer:

    """
    特征引擎，添加给dataframe金融/技术指标，清洗数据。

    :param use_technical_indicator (bool): 是否使用技术指标。
    :param tech_indicator_list (list): 技术指标的种类列表。
    :param use_vix (bool): 是否使用 VIX 波动率指数，这是一个反映市场状况的重要指标。
    :param use_turbulence (bool): 是否使用 turbulence，用于监控市场风险。
    :param user_defined_feature (bool): 是否使用用户自定义特征。
    :param vix_data_dir (str): VIX 指标数据的存储目录。
    :param csv_list (list): VIX 数据的 CSV 文件列表。
    """

    def __init__(
            self,
            tech_indicator_list=INDICATORS,
            use_technical_indicator=True,
            use_turbulence=False,
            user_defined_feature=False,
            use_vix=False,
            vix_data_dir=None,
            csv_list=None
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature
        self.vix_data_dir = vix_data_dir
        self.csv_list = csv_list

    def preprocess_data(self, df):

        """
        执行构造特征的主要函数。

        :param df (dataframe): 处理前的dataframe
        :return : 处理后的dataframe
        """

        df = self.clean_data(df)

        # add technical indicators using stockstats
        if self.use_technical_indicator:
            df = self.add_technical_indicator(df)
            print("Successfully added technical indicators")

        # add vix for multiple stock
        if self.use_vix:
            df = self.add_vix(df, self.vix_data_dir, self.csv_list)
            print("Successfully added vix")

        # add turbulence index for multiple stock
        if self.use_turbulence:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")

        # add user defined feature
        if self.user_defined_feature:
            df = self.add_user_defined_feature(df)
            print("Successfully added user defined features")

        # fill the missing values at the beginning and the end
        df = df.fillna(method="ffill").fillna(method="bfill")
        return df

    def clean_data(self, data):
        """
        清理原始数据。因为股票可能从市场成分股名单中被去除，或时间上不一致。

        :param data (dataframe): pandas dataframe
        :return (dataframe): pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(["date", "tic"], ignore_index=True)
        df.index = df.date.factorize()[0]   # 将处于同一天的数据的索引设置为同一值
        merged_closes = df.pivot_table(index="date", columns="tic", values="close")
        merged_closes = merged_closes.dropna(axis=1)  # 删除列, DOW这支成分股缺失严重，所以将其剔除。
        tics = merged_closes.columns  # 获取剩余股票代码
        df = df[df.tic.isin(tics)]  # 获取剩余股票代码的数据

        return df

    def add_technical_indicator(self, data):

        """添加所有的技术指标特征"""

        df = data.copy()
        df = df.sort_values(by=["tic", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):  # 计算所有成分股对应的indicator
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]  # 利用stockstats API计算indicator
                    temp_indicator = pd.DataFrame(temp_indicator)  # series变为dataframe
                    temp_indicator["tic"] = unique_ticker[i]  # 赋tic
                    temp_indicator["date"] = df[df.tic == unique_ticker[i]][  # 赋date
                        "date"
                    ].to_list()
                    # indicator_df = indicator_df.append(
                    #     temp_indicator, ignore_index=True
                    # )
                    indicator_df = pd.concat([indicator_df,temp_indicator],ignore_index=True)
                except Exception as e:
                    print(e)
            # 将计算结果合并到df中去
            df = df.merge(
                indicator_df[["tic", "date", indicator]], on=["tic", "date"], how="left"
            )
        df = df.sort_values(by=["date", "tic"])
        return df


    def add_user_defined_feature(self, data):

        """修改并执行这个函数，可以添加用户自定义特征"""

        df = data.copy()
        df["daily_return"] = df.close.pct_change(1)
        # df['return_lag_1']=df.close.pct_change(2)
        # df['return_lag_2']=df.close.pct_change(3)
        # df['return_lag_3']=df.close.pct_change(4)
        # df['return_lag_4']=df.close.pct_change(5)
        return df

    def add_vix(self, data, vix_data_dir, csv_list):

        """添加VIX特征"""

        df = data.copy()

        df_vix = YahooDownloader(start_date=df.date.min(), end_date=df.date.max(), data_dir=vix_data_dir,
                                    csv_list=csv_list).fetch_data()

        vix = df_vix[["date", "close"]]
        vix.columns = ["date", "vix"]

        df = df.merge(vix, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, data):

        """计算turbulence指标"""

        # can add other market assets
        # use returns to calculate turbulence，使用returns计算turbulence
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        df_price_pivot = df_price_pivot.pct_change()  # 使用return计算turbulence，而不是price。
        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [0] * start # 开始的一年都是0。
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            # 使用当前日期过去一年的滑动数据
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
                ]

            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                                  hist_price.isna().sum().min():
                                  ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()  # 计算剩余数据的协方差
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )   # 当前日期股票价格减去过去一年股票价格均值，即(yt-u)
            # cov_temp = hist_price.cov()
            # current_temp=(current_price - np.mean(hist_price,axis=0))

            # turbulence 核心计算公式
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:  # turbulence要被记录。
                count += 1
                if count > 2:   # 避免异常点，所以过两次再计入
                    turbulence_temp = temp[0][0]  # 取两次索引是因为计算出的结果被两层列表包裹，并不意味着结果是向量。
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)
        try:
            turbulence_index = pd.DataFrame(
                {"date": df_price_pivot.index, "turbulence": turbulence_index}
            )
        except ValueError:
            raise Exception("Turbulence information could not be added.")
        return turbulence_index

    def add_turbulence(self, data):
        
        """添加turbulence指标"""

        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df