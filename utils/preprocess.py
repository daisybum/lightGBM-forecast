from datetime import timedelta
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import LabelEncoder

from utils.TimeSeriesProcessor import TimeSeriesProcessor


def preprocessing(opt, args, df):
    tsp = TimeSeriesProcessor(df, opt.Model.dt_col)
    df_hourly = tsp.extract_rounded_hour()
    df_hourly = add_datetime_cols(df_hourly, opt.Model.dt_col, ['minute', 'second'])

    df_target = df_hourly[opt.Model.columns.feature_cols]
    df_target = fill_nan(df_target, df_target.columns.tolist())

    shift_cols = opt.Model.columns.shifting_cols
    df_shifted = attach_previous_ts_data(df_target, shift_cols, ts=opt.Model.ts)

    train_start_ds = min(df[opt.Model.dt_col])
    df_train, df_valid = split_train_test(opt, args, df_shifted, train_start_ds)

    df_train = df_train.set_index([opt.Model.dt_col])
    X_train = df_train.drop(columns=opt.Model.columns.shifting_cols)
    y_train = df_train['temperature'].copy()

    df_valid = df_valid.set_index([opt.Model.dt_col])
    X_valid = df_valid.drop(columns=opt.Model.columns.shifting_cols)
    y_valid = df_valid['temperature'].copy()

    X = X_train.to_numpy()
    y = y_train.to_numpy()

    # Convert into numpy arrays
    X_val = X_valid.to_numpy()
    y_val = y_valid.to_numpy()

    return X, y, X_val, y_val


def attach_previous_ts_data(df_shift, target_cols, ts):
    for n in range(1, ts + 1):
        for column in target_cols:
            new_column = column + '_' + str(n)
            df_shift = df_shift.copy()
            df_shift[new_column] = df_shift[column].shift(n).astype('float16')
    return df_shift


def fill_nan(df, columns):
    for col in columns:
        df.loc[:, col] = df.loc[:, col].interpolate()
    return df


def split_train_test(opt, args, df, train_start_ds):
    train_end_ds = train_start_ds + relativedelta(months=args.train_months)
    test_start_ds = train_end_ds + timedelta(hours=1)
    test_end_ds = test_start_ds + relativedelta(months=args.eval_months)

    dt_col = opt.Model.dt_col
    df_train = df[df[dt_col].between(train_start_ds, train_end_ds)]
    df_test = df[df[dt_col].between(test_start_ds, test_end_ds)]
    print('train_start_time: ', min(df_train[dt_col]), 'train_end_time:', max(df_train[dt_col]))
    print('test_start_time: ', min(df_test[dt_col]), 'test_end_time:', max(df_test[dt_col]))

    return df_train, df_test


def add_datetime_cols(df, dt_col, drop_cols):
    df['year'] = df[dt_col].dt.year.astype('int16')
    df['month'] = df[dt_col].dt.month.astype('int8')
    df['day'] = df[dt_col].dt.day.astype('int8')
    df['hour'] = df[dt_col].dt.hour.astype('int8')
    df['minute'] = df[dt_col].dt.minute.astype('int8')
    df['second'] = df[dt_col].dt.second.astype('int8')
    return df.drop(drop_cols, axis=1)
