import os
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm


def raw_text_to_df(raw_lines, columns):
    # Find the start of the dataset
    start_index = 0
    for i, line in enumerate(raw_lines):
        if "#" not in line:
            start_index = i
            break

    # Map each field name to its corresponding value for each row in the dataset
    # Extract dataset rows
    rows = [line.split() for line in raw_lines[start_index:-1]]

    # Convert the structured data to JSON format and save
    return pd.DataFrame(rows, columns=columns)


def kma_to_df(start_dt, stn, category):
    if category == 'ASOS':
        domain = 'https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php?'
        columns = []
        strs = ["# ", ". "]
    else:
        domain = 'https://apihub.kma.go.kr/api/typ01/url/awsh.php?'
        columns = ['ds', 'STN']
        strs = ["#  ", " : "]

    end_dt = datetime.now()
    start_str = start_dt.strftime('%Y%m%d%H%M')
    end_str = end_dt.strftime('%Y%m%d%H%M')

    tm = "tm1=" + start_str + "&tm2=" + end_str + "&"
    stn_id = "stn=" + str(stn) + "&"
    option = "help=1&authKey="
    auth = "1uR-QZWITxSkfkGViN8U1Q"

    url = domain + tm + stn_id + option + auth

    response = requests.get(url)  # GET 요청
    response_text = response.text

    # Split the text into lines
    lines = response_text.strip().split("\n")

    # Extract field names from metadata

    for line in lines:
        if line.startswith(strs[0]) and strs[1] in line:
            parts = line.split(":", 1)
            column_name = parts[0].split()[-1]
            columns.append(column_name)

    return raw_text_to_df(lines, columns)


def save_as_json(df_kma: pd.DataFrame, category: str):
    for _, row in tqdm(df_kma.iterrows()):
        save_path = os.path.join(os.getcwd(), "data", category)
        base_ds = datetime(2018, 1, 1, 0, 0, 0)
        stn = row['지점']

        stn_name = row['지점명']
        if "(" in stn_name:
            stn_name = stn_name.split("(")[0] + "_" + stn_name.split("(")[1].split(")")[0]

        try:
            df = kma_to_df(base_ds, stn=stn, category=category)
            df['STN_NAME'] = stn_name
            df['LON'] = row['경도']
            df['LAT'] = row['위도']
            df['ALT'] = row['노장해발고도(m)']

            if pd.notna(row['지점주소']):
                df.loc[:, ['STATE', 'ADR']] = row['지점주소'].split(" ")[:2]

        except ValueError:
            continue

        df.to_json(os.path.join(save_path, stn_name + '.json'))
