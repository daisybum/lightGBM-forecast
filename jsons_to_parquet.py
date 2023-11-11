import json
import pandas as pd
import bisect
from glob import glob
from tqdm import tqdm
from datetime import datetime

from utils.TimeSeriesProcessor import TimeSeriesProcessor


def file_name_to_datetime(file_path):
    date_str = file_path.split("_")[-2]
    time_str = file_path.split("_")[-1].split(".json")[0]
    date_time_str = date_str + " " + time_str
    date_time = datetime.strptime(date_time_str, '%Y%m%d %H%M%S')

    return file_path, date_time


def file_names_to_df(source_str):
    src_paths = glob(source_str)
    path_ds_list = [file_name_to_datetime(path) for path in src_paths]
    return pd.DataFrame(path_ds_list, columns=['path', 'ds'])


def find_closest_indices(date_hour_list, ds_series):
    """각각의 정각 시간에 가장 가까운 데이터의 인덱스들을 검색."""
    date_times = [datetime.strptime(str(date_hour), '%Y-%m-%d %H:%M:%S')
                  for date_hour in date_hour_list]
    return [closest_time_index(ds_series, dt) for dt in tqdm(date_times)]


def closest_time_index(items, pivot):
    """이진 검색을 사용하여 가장 가까운 날짜/시간 인덱스를 검색."""
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


def extract_data_from_path(src_path):
    with open(src_path, "rt") as f:
        data_dict = json.load(f)
    target_keys = {'temp': 'temperature', 'humi': 'humidity', 'pressure': 'pressure'}
    data_dict = {val: data_dict.get(key, None) for key, val in target_keys.items()}

    return pd.Series([data_dict['temperature'], data_dict['humidity'], data_dict['pressure']])


def json_batch_to_parquet(source, save_path):
    src_path_list = glob(source)
    dataframes = [pd.read_json(path) for path in tqdm(src_path_list)]
    df = pd.concat(dataframes, ignore_index=True)
    df.to_parquet(save_path)


if __name__ == '__main__':
    mvw_cols_list = ['temperature', 'humidity', 'pressure']
    kma_api_list = ['STN', 'STATE', 'ADR', 'LAT', 'LON', 'ALT', 'temperature', 'humidity', 'pressure']

    # 모바휠 자체 데이터 전처리 코드
    source = '.\\data\\서구괴곡동\\*.json'
    save_path = 'data/parquets/ASOS.parquet'

    df = file_names_to_df(source)

    processor = TimeSeriesProcessor(df, dt_col='ds')
    df_hourly = processor.extract_rounded_hour()

    df_whether = df_hourly['path'].apply(extract_data_from_path)
    df_whether.columns = mvw_cols_list

    df_hourly[mvw_cols_list] = df_whether
    df_hourly.to_parquet(".\\data\\MVW_상안보대로.parquet")

    # 기상청 API 크롤링 데이터 전처리
    # source = '.\\data\\ASOS\\*.json'
    # save_path = '.\\data\\parquet\\ASOS.parquet'
    # json_batch_to_parquet(source, save_path)
