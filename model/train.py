import gc
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import xgboost as xgb
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config_model import CONFIG_MODEL
from model.models.BiEncoder import get_model_bi_encoder_baseline
from model.models.Xgboost import get_model_xgboost, get_xgboost_sample_weights


def train_xgboost(
    X_train: sp.spmatrix,
    y_train: np.ndarray,
    X_val: sp.spmatrix,
    y_val: np.ndarray,
    class_weights: np.ndarray,
) -> xgb.XGBClassifier:
    model = get_model_xgboost()
    model.fit(
        X_train,
        y_train,
        sample_weight=get_xgboost_sample_weights(y_train, class_weights),
        eval_set=[(X_train, y_train), (X_val, y_val)],
        **CONFIG_MODEL.MODEL_CONFIG["xgboost"]["train_args"],
    )
    return model


def train_bi_encoder_baseline(
    input_model_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: np.ndarray,
    output_path: str,
    device: torch.device,
    epochs: int,
) -> None:
    model = get_model_bi_encoder_baseline(input_model_path, num_classes=CONFIG_MODEL.NUM_CLASSES)
    model.to(device)
    try:
        model.encoder[0].auto_model.gradient_checkpointing_enable()
    except Exception as e:
        print(f"Gradient checkpointing is not available for this encoder: {e}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG_MODEL.MODEL_CONFIG["siamese"]["lr"])
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(device)
    )
    best_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            logits = model(
                batch["ids1"].to(device),
                batch["mask1"].to(device),
                batch["ids2"].to(device),
                batch["mask2"].to(device),
            )
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch["ids1"].to(device),
                    batch["mask1"].to(device),
                    batch["ids2"].to(device),
                    batch["mask2"].to(device),
                )
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(batch["label"].numpy())
        acc = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds, average="macro")
        print(
            f"Epoch {epoch + 1} | Train Loss: {total_loss / len(train_loader):.4f} | "
            f"Val Acc: {acc:.4f} | Val F1: {f1:.4f}"
        )
        if f1 > best_f1:
            best_f1 = f1
            model.save(output_path)
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()


def train_cross_encoder(
    model: CrossEncoder,
    train_dataloader: DataLoader,
    evaluator: CESoftmaxAccuracyEvaluator,
    output_path: str,
    epochs: int,
) -> CrossEncoder:
    model.model.gradient_checkpointing_enable()
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        loss_fct=model.loss_fct,
        save_best_model=True,
        optimizer_params={"lr": CONFIG_MODEL.LEARNING_RATE},
        weight_decay=CONFIG_MODEL.WEIGHT_DECAY,
        epochs=epochs,
        warmup_steps=int(len(train_dataloader) * 0.1),
        output_path=output_path,
        show_progress_bar=True,
        evaluation_steps=CONFIG_MODEL.MODEL_CONFIG["cross_encoder"]["evaluation_steps"],
    )
    if not os.path.isfile(os.path.join(output_path, "config.json")):
        print("No checkpoint was saved; predicting with the last epoch.")
        return model
    best = model.best_score
    print(f"Reloading the best checkpoint (val accuracy {best:.4f}) from {output_path}")
    model = CrossEncoder(output_path, max_length=model.max_length)
    model.best_score = best
    return model
