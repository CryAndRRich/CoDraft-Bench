import copy

import torch


class CONFIG_MODEL:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 5
    NUM_PRODUCT_CLASSES = 45
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.02

    MODEL_CONFIG = {
        "cross_encoder": {
            "epochs": 5,
            "evaluation_steps": 500,
        },
        "siamese": {
            "num_epochs_cls": 5,
            "lr": LEARNING_RATE / 2,
            "batch_size": 32,
            "physical_batch_size": 32,
            "num_workers": 2,
        },
        "multi_task": {
            "training_args": {
                "output_dir": "./output/multi_task",
                "learning_rate": LEARNING_RATE,
                "num_train_epochs": 10,
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "gradient_checkpointing": True,
                "dataloader_num_workers": 2,
                "per_device_eval_batch_size": 16,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
                "weight_decay": WEIGHT_DECAY,
                "eval_strategy": "epoch",
                "save_strategy": "epoch",
                "save_total_limit": 1,
                "save_only_model": True,
                "logging_steps": 10,
                "report_to": "none",
                "load_best_model_at_end": True,
                "metric_for_best_model": "f1_macro",
                "greater_is_better": True,
                "fp16": torch.cuda.is_available(),
                "seed": 42,
                "data_seed": 42,
                "remove_unused_columns": False,
            },
            "loss_args": {
                "alpha": 0.47,
                "aux_weight": 0.26,
            },
        },
        "xgboost": {
            "model_args": {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 500,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "early_stopping_rounds": 20,
            },
            "train_args": {
                "verbose": 10,
            },
        },
    }

    @classmethod
    def multi_task_args(cls, seed: int, output_dir: str) -> dict:
        cfg = copy.deepcopy(cls.MODEL_CONFIG["multi_task"])
        cfg["training_args"].update(seed=seed, data_seed=seed, output_dir=output_dir)
        return cfg
