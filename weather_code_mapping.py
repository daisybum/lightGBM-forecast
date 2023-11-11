import pandas as pd
import json


def code_to_natural_language(code, code_to_word):
    if len(code) != 8:
        raise ValueError("Code must be 8 characters long")

    if code == '12345678':
        return 'clear'

    # 코드를 두 글자씩 4개 코드로 분할
    sub_codes = [code[i:i + 2] for i in range(0, 8, 2)]

    # 코드를 자연어로 변환
    words = [code_to_word['codes'][sub_code] for sub_code in sub_codes]

    # 각 자연어 코드
    return ' '.join(filter(None, words)).strip()


def find_all_in_list(string, lst):
    found = [item for item in lst if item in string]
    return ' '.join(filter(None, found)).strip()


if __name__ == '__main__':
    code_path = 'configs/weather_code.json'
    with open(code_path, 'rt', encoding='utf-8') as file:
        codes_to_word = json.load(file)

    df = pd.read_parquet('data/parquets/기상데이터_대전_v0.0.1.parquet')
    index_nan = df['현상번호(국내식)'].isna()
    df.loc[index_nan] = 12345678
    weather_code = df['현상번호(국내식)'].astype('int').astype('str').str.zfill(8)

    mvw_name_list = ['clear', 'foggy', 'rainy', 'snowy']
    weather_name = weather_code.apply(lambda x: code_to_natural_language(x, codes_to_word))
    weather_name_mvw = weather_name.apply(lambda x: find_all_in_list(x, mvw_name_list))
