import os

import pandas as pd
from transformers import PreTrainedTokenizer

from config.config_data import CONFIG_DATA
from preprocess.data_loader import (
    create_dataloader_cross_encoder,
    create_ml_data,
    create_patterns,
    create_siamese_dataloader,
    to_dataset,
)
from preprocess.preprocess_data import preprocess

PIPELINES = {"ml", "multi_task", "cross_encoder", "siamese"}


class DataManager:
    def __init__(
        self,
        input_root: str,
        variant: str,
        build_for: str,
        tokenizer: PreTrainedTokenizer | None = None,
        binary: bool = False,
    ) -> None:
        if build_for not in PIPELINES:
            raise ValueError(f"Unknown build_for {build_for!r}. Use one of {sorted(PIPELINES)}.")
        root = os.path.join(input_root, variant)
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"{root} does not exist. Expected train.csv, val.csv and test.csv in it."
            )
        splits = [
            pd.read_csv(os.path.join(root, f"{s}.csv"), low_memory=False)
            for s in ("train", "val", "test")
        ]
        print(f"[{variant}] train={len(splits[0])} val={len(splits[1])} test={len(splits[2])}")
        classes = set()
        for d in splits:
            classes |= set(d["Class 1"]) | set(d["Class 2"])
        nice = sorted(CONFIG_DATA.NICE_CLASS_MAP)
        unknown = sorted(classes - set(nice))
        if unknown:
            raise ValueError(f"Classes outside the NICE map: {unknown}")
        self.class_to_token = {c: f"[CLASS_{c}]" for c in nice}
        self.class_to_id = {c: i for i, c in enumerate(nice)}
        self.NUM_PRODUCT_CLASSES = len(nice)
        self.df_train, self.df_val, self.df_test = (preprocess(d) for d in splits)
        if binary:
            for d in (self.df_train, self.df_val, self.df_test):
                d["label_5"] = d["label_score"]
                d["label_score"] = (d["label_score"] >= 1).astype(int)

        if build_for == "ml":
            self.ml_data = create_ml_data(self.df_train, self.df_val, self.df_test)
        elif build_for == "multi_task":
            self.datasets = tuple(
                to_dataset(
                    create_patterns(d, tokenizer, self.class_to_token, self.class_to_id), tokenizer
                )
                for d in (self.df_train, self.df_val, self.df_test)
            )
        elif build_for == "cross_encoder":
            self.loaders = create_dataloader_cross_encoder(self.df_train, self.df_val)
        else:
            self.loaders = create_siamese_dataloader(self.df_train, self.df_val, tokenizer)

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.df_train, self.df_val, self.df_test
