import numpy as np
from sklearn.utils import class_weight


def compute_class_weight(y_train: np.ndarray) -> np.ndarray:
    return class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
