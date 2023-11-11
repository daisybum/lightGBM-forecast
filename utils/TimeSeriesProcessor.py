import bisect
from datetime import datetime
from tqdm import tqdm


class TimeSeriesProcessor:
    def __init__(self, dataframe, dt_col):
        self.df = dataframe
        self.dt_col = dt_col

    @staticmethod
    def closest_time_index(items, pivot):
        """이진 검색을 사용하여 가장 가까운 datetime index를 검색"""
        index = bisect.bisect_left(items, pivot)
        if index == 0:
            return 0
        if index == len(items):
            return index - 1
        before = items[index - 1]
        after = items[index]
        if after - pivot < pivot - before:
            return index
        else:
            return index - 1

    def find_closest_indices(self, date_hour_list):
        """각각의 정각 시간에 가장 가까운 데이터의 인덱스들을 검색"""
        date_times = [datetime.strptime(str(date_hour), '%Y-%m-%d %H:%M:%S')
                      for date_hour in date_hour_list]
        return [self.closest_time_index(self.df[self.dt_col], dt) for dt in tqdm(date_times)]

    def extract_rounded_hour(self):
        self.df['date_hour'] = self.df[self.dt_col].apply(lambda dt: dt.replace(minute=0, second=0, microsecond=0))
        date_hour_list = self.df['date_hour'].unique()

        closest_indices = self.find_closest_indices(date_hour_list)

        df_hourly = self.df.loc[closest_indices].reset_index(drop=True)
        return df_hourly
