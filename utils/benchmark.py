import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def benchmark(y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    std = (y_test - y_pred).std()
    quantile_95 = np.quantile((y_test - y_pred), 0.95)
    result = pd.DataFrame(
        [{'RMSE': rmse, 'MAE': mae, 'STD': std, '상위95%': quantile_95}]
    )
    return result


def metric_with_times(y_test, y_pred, train_time, inf_time):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    std = (y_test - y_pred).std()
    quantile_95 = np.quantile((y_test - y_pred), 0.95)
    result = pd.DataFrame(
        [{'RMSE': rmse,
          'MAE': mae,'STD': std,
          '상위95%': quantile_95,
          '학습 시간(s)': train_time,
          '추론 시간(s)': inf_time}]
    )
    return result