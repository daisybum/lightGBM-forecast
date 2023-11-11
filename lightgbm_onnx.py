import os
import time
import logging
import numpy as np
import pandas as pd
import lightgbm
from lightgbm import LGBMRegressor

from run.Inference import _args, load_config
from utils.benchmark import metric_with_times
from utils.preprocess import preprocessing
from utils.onnx_converter import lightgbm_to_onnx

if __name__ == '__main__':
    ts = 5
    logger = logging.getLogger()
    categorical_feats = ['year', 'month', 'day', 'hour', 'longitude', 'latitude']
    col_features = ['date_time', 'year', 'month', 'day', 'hour', 'y', 'pressure',
                    'humidity', 'longitude', 'latitude', 'altitude'
                    ]

    df = pd.read_parquet('data/parquets/MVW_원자력.parquet')

    args = _args()
    args.train_months = 8
    args.eval_months = 3
    opt = load_config(args.config)

    X, y, X_val, y_val = preprocessing(opt, args, df)

    # Convert into list of (X, y) tuple pairs
    valid_list = [(np.expand_dims(X_val[i], axis=0), np.expand_dims(y_val[i], axis=0))
                  for i in range(len(y_val))]
    callbacks = [lightgbm.early_stopping(stopping_rounds=100, verbose=True)]

    params = {'n_estimators': 5000,
              'objective': 'regression',
              'metric': 'rmse',
              'learning_rate': 0.01,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'max_depth': 7,
              'num_leaves': 31,
              'random_state': 42
              }

    start = time.time()
    booster = LGBMRegressor().set_params()
    booster.fit(X, y, eval_set=valid_list, callbacks=callbacks)
    train_time = time.time() - start

    start = time.time()
    pred = booster.predict(X_val)
    inf_time = time.time() - start
    print(metric_with_times(y_val, pred, train_time, inf_time))

    onnx_model_path = os.path.join(os.getcwd(), "model\\lgb_model_id3.onnx")
    lightgbm_to_onnx(lgb_model=booster, X_array=X, model_path=onnx_model_path)
