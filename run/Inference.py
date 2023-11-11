import os
import logging
import numpy as np
import pandas as pd
import yaml
import argparse
from easydict import EasyDict as ed

from utils.benchmark import benchmark
from utils.preprocess import preprocessing
from utils.onnx_converter import load_ort_session


def load_config(config_dir, easy=True):
    cfg = yaml.load(open(config_dir), yaml.FullLoader)
    if easy is True:
        cfg = ed(cfg)
    return cfg


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='configs/LightGBM_ts5_ONNX.yaml')
    parser.add_argument('--data', '-p', type=str, default='MVW_원자력.parquet')
    parser.add_argument('--model', '-m', type=str, default='lgb_model_id3.onnx')
    parser.add_argument('--train-months', '-t', type=int)
    parser.add_argument('--eval-months', '-e', type=int)
    return parser.parse_args()


def inference(opt, args, logger):
    # init model
    onnx_model_path = os.path.join(os.getcwd(), opt.Test.checkpoint_dir, args.model)
    sess = load_ort_session(onnx_model_path)

    # prepare random number input
    num_features = sess._inputs_meta[0].shape[-1]
    input_x = np.random.rand(1, num_features).astype(np.float32)

    # 모델 추론값 도출
    pred = sess.run(None, {"X": input_x})[0].ravel()
    logger.info("무작위 입력값에 대한 추론값: " + str(pred))

    return pred


if __name__ == "__main__":
    args = _args()
    args.train_months = 8
    args.eval_months = 3
    opt = load_config(args.config)
    result = inference(opt, args)
    a = 1
