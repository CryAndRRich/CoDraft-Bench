import numpy as np
import xgboost as xgb

from config.config_model import CONFIG_MODEL


def get_model_xgboost() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=CONFIG_MODEL.NUM_CLASSES,
        device=CONFIG_MODEL.DEVICE,
        tree_method="hist",
        **CONFIG_MODEL.MODEL_CONFIG["xgboost"]["model_args"],
    )


def get_xgboost_sample_weights(y_train: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    return np.array([class_weights[int(label)] for label in y_train])
