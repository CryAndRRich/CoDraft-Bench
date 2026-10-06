import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from datasets import Dataset
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer

from config.config_data import CONFIG_DATA
from config.config_model import CONFIG_MODEL


class PairSiameseDataset(TorchDataset):
    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizer, max_len: int) -> None:
        self.term1 = df["input_text_1"].tolist()
        self.term2 = df["input_text_2"].tolist()
        self.labels = df["label_score"].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.term1)

    def _encode(self, text: str) -> dict:
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

    def __getitem__(self, idx: int) -> dict:
        enc1, enc2 = self._encode(self.term1[idx]), self._encode(self.term2[idx])
        return {
            "ids1": enc1["input_ids"].squeeze(0),
            "mask1": enc1["attention_mask"].squeeze(0),
            "ids2": enc2["input_ids"].squeeze(0),
            "mask2": enc2["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_siamese(batch: list[dict]) -> dict:
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
    for ids, mask in (("ids1", "mask1"), ("ids2", "mask2")):
        n = int(out[mask].sum(dim=1).max())
        out[ids] = out[ids][:, :n]
        out[mask] = out[mask][:, :n]
    return out


def create_ml_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int = 5000,
) -> tuple:
    vectorizer = TfidfVectorizer(max_features=max_features, lowercase=True)
    vectorizer.fit(train_df["input_text_1"].tolist() + train_df["input_text_2"].tolist())

    def extract_features(df: pd.DataFrame) -> tuple[sp.spmatrix, np.ndarray]:
        tfidf1 = vectorizer.transform(df["input_text_1"].fillna(""))
        tfidf2 = vectorizer.transform(df["input_text_2"].fillna(""))
        cosine_sim = 1 - paired_cosine_distances(tfidf1, tfidf2)
        X = sp.hstack([tfidf1, tfidf2, abs(tfidf1 - tfidf2), cosine_sim.reshape(-1, 1)])
        return X, df["label_score"].values.astype(int)

    return (*extract_features(train_df), *extract_features(val_df), *extract_features(test_df))


def create_siamese_dataloader(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
) -> tuple[DataLoader, DataLoader]:
    cfg = CONFIG_MODEL.MODEL_CONFIG["siamese"]
    train_loader = DataLoader(
        PairSiameseDataset(train_df, tokenizer, CONFIG_DATA.MAX_LEN),
        batch_size=cfg["physical_batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        collate_fn=collate_siamese,
    )
    val_loader = DataLoader(
        PairSiameseDataset(val_df, tokenizer, CONFIG_DATA.MAX_LEN),
        batch_size=cfg["physical_batch_size"],
        shuffle=False,
        num_workers=2,
        collate_fn=collate_siamese,
    )
    return train_loader, val_loader


def create_patterns(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    class_to_token: dict,
    class_to_id: dict,
) -> pd.DataFrame:
    mask = tokenizer.mask_token
    rows = []
    for c1, t1, c2, t2, label in zip(
        df["Class 1"], df["input_text_1"], df["Class 2"], df["input_text_2"], df["label_score"]
    ):
        rows.append(
            {
                "text1": f"{mask} {t1}",
                "text2": f"{class_to_token.get(c2, '')} {t2}",
                "labels": label,
                "aux_labels": class_to_id.get(c1, 0),
            }
        )
        rows.append(
            {
                "text1": f"{class_to_token.get(c1, '')} {t1}",
                "text2": f"{mask} {t2}",
                "labels": label,
                "aux_labels": class_to_id.get(c2, 0),
            }
        )
    return pd.DataFrame(rows)


def preprocess_dataset(examples: dict, tokenizer: PreTrainedTokenizer) -> dict:
    tokenized = tokenizer(
        examples["text1"],
        examples["text2"],
        truncation=True,
        max_length=CONFIG_DATA.MAX_LEN,
        padding=False,
    )
    tokenized["labels"] = examples["labels"]
    tokenized["aux_labels"] = examples["aux_labels"]
    return tokenized


def create_dataloader_cross_encoder(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> tuple[DataLoader, CESoftmaxAccuracyEvaluator]:
    def examples(df: pd.DataFrame) -> list[InputExample]:
        return [
            InputExample(texts=[str(a), str(b)], label=int(y))
            for a, b, y in zip(df["input_text_1"], df["input_text_2"], df["label_score"])
        ]

    train_dataloader = DataLoader(examples(df_train), shuffle=True, batch_size=32)
    evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(
        examples(df_val), name="Ordinal_Check"
    )
    return train_dataloader, evaluator


def to_dataset(df_aug: pd.DataFrame, tokenizer: PreTrainedTokenizer) -> Dataset:
    ds = Dataset.from_pandas(df_aug).map(
        preprocess_dataset,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=df_aug.columns.tolist(),
    )
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "aux_labels"])
    return ds
