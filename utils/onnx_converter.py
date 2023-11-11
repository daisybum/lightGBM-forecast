import os
import numpy as np
import warnings
import packaging.version as pv

import onnx
from lightgbm import LGBMRegressor
from onnxruntime import InferenceSession
from onnxmltools import __version__ as oml_version
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
    convert_lightgbm)
from skl2onnx import to_onnx, update_registered_converter
from skl2onnx.common.shape_calculator import (
    calculate_linear_regressor_output_shapes)


def lightgbm_to_onnx(lgb_model, X_array, model_path):
    """
    LightGBM 모델을 ONNX 파일로 변환
    """
    update_registered_converter(
        LGBMRegressor,
        "LightGbmLGBMRegressor",
        calculate_linear_regressor_output_shapes,
        skl2onnx_convert_lightgbm,
        options={"split": None},
    )

    model_onnx = to_onnx(
        lgb_model, X_array[:1].astype(np.float32), target_opset={"": 14, "ai.onnx.ml": 2}
    )
    onnx.save_model(model_onnx, model_path)


def skl2onnx_convert_lightgbm(scope, operator, container):
    options = scope.get_options(operator.raw_operator)
    if "split" in options:
        if pv.Version(oml_version) < pv.Version("1.9.2"):
            warnings.warn(
                "Option split was released in version 1.9.2 but %s is "
                "installed. It will be ignored." % oml_version
            )
        operator.split = options["split"]
    else:
        operator.split = None
    convert_lightgbm(scope, operator, container)

def load_ort_session(model_path):
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    sess = InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"]
    )

    return sess
