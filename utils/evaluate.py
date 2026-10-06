import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import EvalPrediction, Trainer

from config.config_data import CONFIG_DATA
from config.config_model import CONFIG_MODEL
from model.models.BiEncoder import get_model_bi_encoder_baseline
from preprocess.data_loader import PairSiameseDataset, collate_siamese


def get_preds_multi(
    trainer: Trainer,
    test_ds: object,
    df_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = trainer.predict(test_ds).predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if len(predictions) != 2 * len(df_test):
        raise ValueError(
            f"Got {len(predictions)} prediction rows for {len(df_test)} test pairs; expected "
            f"twice that, one row per masked copy of every pair."
        )
    avg_logits = predictions.reshape(-1, 2, predictions.shape[-1]).mean(axis=1)
    return np.argmax(avg_logits, axis=-1), df_test["label_score"].values, avg_logits


def build_result_df(
    df_test: pd.DataFrame, test_true: np.ndarray, test_preds: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Pair ID": df_test["Pair ID"].values,
            "label": test_true,
            "pred": test_preds,
            "Class 1": df_test["Class 1"].values,
            "Class 2": df_test["Class 2"].values,
        }
    )


def get_preds_cross_encoder(model: object, df_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pairs = [[str(a), str(b)] for a, b in zip(df_test["input_text_1"], df_test["input_text_2"])]
    return np.argmax(model.predict(pairs), axis=1), df_test["label_score"].values


def get_preds_siamese(
    test_df: pd.DataFrame,
    model_path: str,
    model_name: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = get_model_bi_encoder_baseline(model_name, num_classes=CONFIG_MODEL.NUM_CLASSES)
    state = torch.load(os.path.join(model_path, "siamese_state.pth"), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    loader = DataLoader(
        PairSiameseDataset(test_df, model.encoder.tokenizer, max_len=CONFIG_DATA.MAX_LEN),
        batch_size=CONFIG_MODEL.MODEL_CONFIG["siamese"]["batch_size"],
        shuffle=False,
        collate_fn=collate_siamese,
    )
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting Baseline"):
            logits = model(
                batch["ids1"].to(device),
                batch["mask1"].to(device),
                batch["ids2"].to(device),
                batch["mask2"].to(device),
            )
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    return np.array(preds), test_df["label_score"].values


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    logits, labels = eval_pred
    if isinstance(labels, tuple):
        labels = labels[0]
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "qwk": float(cohen_kappa_score(labels, preds, weights="quadratic")),
        "mae": float(mean_absolute_error(labels, preds)),
    }
