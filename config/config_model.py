import torch

class CONFIG_MODEL:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 5
    NUM_PRODUCT_CLASSES = 45
    MAX_LEN = 256
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.02
    BATCH_SIZE = 32
    
    MODEL_CONFIG = {
        "cross_encoder": {
            "output_path" : "./output/cross_encoder",
            "evaluation_step" : 500,
            "show_progress_bar" : True,
            "save_best_model" : True,
            "epochs": 5
        },
        "siamese" : {
            "output_path": "./output/siamese",
            "num_epochs_simcse": 1,
            "num_epochs_cls": 5,
            "lr": LEARNING_RATE/2,
            "batch_size": BATCH_SIZE,
            "physical_batch_size": 32,
            "warmup_steps": 100,
            "accumulation_step": 4,
            "num_workers": 2,
            "path_simcse": "./output/siamese/simcse_model",
            "path_final": "./output/siamese/final_similarity_model"

        },
        "multi_task": {
            "training_args":{
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
                # Checkpoints hold the model only, not the optimizer state: 2.2 GB instead of
                # ~6.8 GB each. Training is unaffected; runs simply cannot be resumed midway.
                "save_only_model": True,
                "logging_steps": 10,
                "report_to": "none",
                "load_best_model_at_end": True,
                "metric_for_best_model": "f1_macro", 
                "greater_is_better": True,
                "fp16": torch.cuda.is_available(),
                "seed": 42,
                "data_seed": 42,
                "remove_unused_columns": False
            },
            "loss_args": {
                "alpha": 0.47,
                "aux_weight": 0.26
            }
        },
        "xgboost": {
            "output_path": "./output/xgboost",
            "max_features": 5000,
            "model_args": {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 500,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "early_stopping_rounds": 20
            },
            "train_args": {
                "verbose": 10
            }
        },
        "svm": {
            "output_path": "./output/svm",
            "max_features": 5000,
            "model_args": {
                "kernel": "linear",
                "C": 1.0,
                "probability": True,
                "class_weight": "balanced",
                "random_state": 42
            }
        }
    }
    # The paper trains with an effective batch of 32 (8 per device x 4 accumulation
    # steps on one GPU). Trainer multiplies the per-device batch by the GPU count, so
    # on two GPUs the same settings would give 64 and the numbers would not be
    # comparable. EFFECTIVE_BATCH is what multi_task_args keeps fixed.
    EFFECTIVE_BATCH = 32

    @classmethod
    def multi_task_args(cls, seed=None, output_dir=None, n_gpu=None):
        """Copy of the multi_task config with the seed, output directory and GPU
        count applied.

        Multi-seed runs need one directory per seed, otherwise save_total_limit=1
        makes each run overwrite the previous run's checkpoint.

        n_gpu: how many GPUs Trainer will shard the batch across. The accumulation
        steps are divided down so the effective batch stays at EFFECTIVE_BATCH; on
        two GPUs that halves the number of optimizer steps and roughly halves the
        wall-clock time. Defaults to the visible device count.
        """
        import copy
        cfg = copy.deepcopy(cls.MODEL_CONFIG["multi_task"])
        args = cfg["training_args"]

        if seed is not None:
            args["seed"] = seed
            args["data_seed"] = seed
            if output_dir is None:
                output_dir = f"./output/multi_task_seed{seed}"
        if output_dir is not None:
            args["output_dir"] = output_dir

        if n_gpu is None:
            n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 1
        n_gpu = max(1, int(n_gpu))

        per_device = args["per_device_train_batch_size"]
        accum, rem = divmod(cls.EFFECTIVE_BATCH, per_device * n_gpu)
        if rem or accum < 1:
            raise ValueError(
                f"effective batch {cls.EFFECTIVE_BATCH} is not divisible by "
                f"per_device_train_batch_size {per_device} x {n_gpu} GPUs"
            )
        args["gradient_accumulation_steps"] = accum
        args["dataloader_num_workers"] = max(2, 2 * n_gpu)
        print(f"[config] {n_gpu} GPU(s): per_device={per_device} x accum={accum} "
              f"-> effective batch {per_device * n_gpu * accum}")
        return cfg
