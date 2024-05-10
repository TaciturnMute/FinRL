import pandas as pd


class YahooDownloader():

    """
    聚合并简单处理所有原始csv数据。

    :param start_date: 待处理数据的起始日期
    :param end_date: 终止日期
    :param data_dir: 成分股原始数据所在路径
    :param csv_list: 所有csv文件名称的列表
    """

    def __init__(self,
                 start_date: str,
                 end_date: str,
                 data_dir: str,
                 csv_list: list):

        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir  
        self.csv_list = csv_list  

    def fetch_data(self) -> pd.DataFrame:

        data_df = pd.DataFrame()
        for tic in self.csv_list: 
            temp_df = pd.read_csv(self.data_dir + tic)  
            temp_df.Date = pd.to_datetime(temp_df.Date)
            temp_df = temp_df[(temp_df.Date >= pd.to_datetime(self.start_date)) & (temp_df.Date <= pd.to_datetime(self.end_date))]
            temp_df["tic"] = tic[:-4] 
            data_df = data_df.append(temp_df)
        data_df = data_df.reset_index(drop=True)

        try:
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tic",
            ]
            data_df["close"] = data_df["adjcp"]
            data_df = data_df.drop(labels="adjcp", axis=1)
        except NotImplementedError:
            print("the features are not supported currently")

        data_df["day"] = data_df["date"].dt.dayofweek
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        return data_df

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df
