import argparse
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from collections.abc import Callable

from config.runs import RUNS, describe, get_run

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
KAGGLE_INPUT = "/kaggle/input"
RESULT_FILES = {
    "_preds.csv": "preds.csv",
    "_metrics.json": "metrics.json",
    "_logits.npy": "logits.npy",
    "_raw.jsonl": "raw.jsonl"
}
PRINT_LOCK = threading.Lock()


def run_seed(run_id: int) -> int:
    return int(RUNS[run_id].get("seed", SEED))


def run_tag(run_id: int, smoke: bool = False) -> str:
    return ("SMOKE_" if smoke else "") + f"{run_id:02d}_{RUNS[run_id]['name']}_seed{run_seed(run_id)}"


def _search_kaggle(ok: Callable[[str], bool]) -> str | None:
    for depth in ("*", "*/*", "*/*/*", "*/*/*/*"):
        for d in sorted(glob.glob(os.path.join(KAGGLE_INPUT, depth))):
            if ok(d):
                return d
    return None


def find_data_root(hint: str | None = None) -> str:
    def ok(d: str | None) -> bool:
        return bool(d) and os.path.isfile(os.path.join(d, "codraft", "train.csv"))

    for d in (hint, os.environ.get("CODRAFT_DATA"), os.path.join(REPO_DIR, "data")):
        if ok(d):
            return d
    found = _search_kaggle(ok)
    if found is None:
        raise FileNotFoundError(
            f"No dataset found (tried {hint!r} and {KAGGLE_INPUT}). "
            f"Expected a directory containing codraft/train.csv."
        )
    return found


def find_variant_root(data_root: str, variant: str, extra: tuple[str, ...] = ()) -> str:
    def ok(d: str | None) -> bool:
        return bool(d) and os.path.isfile(os.path.join(d, variant, "test.csv"))

    for d in (data_root, *extra):
        if ok(d):
            return d
    found = _search_kaggle(ok)
    if found is None:
        raise FileNotFoundError(
            f"Variant {variant!r} not found in {data_root} or under {KAGGLE_INPUT}. "
            f"Expected a directory containing {variant}/test.csv."
        )
    return found


def find_release_dir(source: int) -> str | None:
    """weights/<path> of a released run (the folder holding model/ and preds.csv), in this repo or
    anywhere under /kaggle/input (matched on the path suffix, so any dataset nesting works)."""
    path = RUNS[source]["path"]

    def ok(d: str) -> bool:
        return os.path.isfile(os.path.join(d, "model", "model.safetensors"))

    local = os.path.join(REPO_DIR, "weights", path)
    if ok(local):
        return local
    suffix = os.sep + os.path.join("weights", path)
    for root, dirs, _ in os.walk(KAGGLE_INPUT):
        if root.endswith(suffix) and ok(root):
            return root
        dirs[:] = [d for d in dirs if d not in ("model", "conditions")]
    return None


def find_checkpoint(source: int) -> str:
    path = RUNS[source]["path"]
    tag = run_tag(source)

    def ok(d: str | None) -> bool:
        return bool(d) and all(
            os.path.isfile(os.path.join(d, f)) for f in ("model.safetensors", "config.json")
        )

    if ok(os.environ.get("CODRAFT_CKPT")):
        return os.environ["CODRAFT_CKPT"]
    release = find_release_dir(source)
    if release is not None:
        return os.path.join(release, "model")
    found = []
    for root, _, files in os.walk(KAGGLE_INPUT):
        if {"model.safetensors", "added_tokens.json", "config.json"} <= set(files):
            try:
                with open(os.path.join(root, "config.json")) as fh:
                    if json.load(fh).get("num_product_classes"):
                        found.append(root)
            except Exception:
                pass
    named = [f for f in found if tag in f or f.rstrip("/").endswith(os.path.join(path, "model"))]
    if len(named) == 1:
        return named[0]
    if len(found) == 1:
        return found[0]
    candidates = f"; candidates: {found}" if found else ""
    raise FileNotFoundError(
        f"Checkpoint of run {source} ({path}) not found{candidates}. "
        f"Attach weights/{path}/model as a Kaggle dataset, or set CODRAFT_CKPT."
    )


def _smoke_data(data_root: str, variant: str, tmp_root: str) -> str:
    import pandas as pd

    dst = os.path.join(tmp_root, variant)
    os.makedirs(dst, exist_ok=True)
    for split, k in {"train": 40, "val": 10, "test": 10}.items():
        d = pd.read_csv(os.path.join(data_root, variant, f"{split}.csv"), low_memory=False)
        d = d.groupby("Similarity", sort=False, group_keys=False).head(k)
        d.sort_index().to_csv(os.path.join(dst, f"{split}.csv"), index=False)
    return tmp_root


def _setup_determinism(seed: int) -> object:
    import random
    import warnings

    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _environment() -> dict:
    import torch

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_gpu_visible": torch.cuda.device_count()
    }
    for pkg in ("transformers", "sentence_transformers", "xgboost", "sklearn", "datasets"):
        try:
            env[pkg] = __import__(pkg).__version__
        except Exception:
            env[pkg] = None
    return env


def _data_manager(ctx: dict, tokenizer: object, build_for: str) -> object:
    from preprocess.data_manager import DataManager

    dm = DataManager(
        input_root=ctx["data_root"],
        variant=ctx["spec"]["variant"],
        build_for=build_for,
        tokenizer=tokenizer,
        binary=ctx["spec"].get("binary", False),
        mask_hides_heading=ctx["spec"].get("mask_hides_heading", False)
    )
    for split, d in zip(("train", "val", "test"), dm.get_data()):
        for c in ("input_text_1", "input_text_2"):
            bad = d[c].astype(str).str.contains(r"(?:Nature|Use|Category): nan\b", regex=True)
            assert not bad.any(), (
                f'{split} {c}: {int(bad.sum())} inputs carry "nan", e.g. {d[c][bad].iloc[0]!r}'
            )
    print("Sample input:", dm.get_data()[2]["input_text_1"].iloc[0])
    for view in getattr(dm, "sample_views", []):
        print("Sample view:", view)
    return dm


def _run_xgboost(ctx: dict) -> tuple:
    from model.train import train_xgboost
    from utils.compute_weight import compute_class_weight

    dm = _data_manager(ctx, None, "ml")
    X_train, y_train, X_val, y_val, X_test, y_test = dm.ml_data
    model = train_xgboost(X_train, y_train, X_val, y_val, compute_class_weight(y_train))

    def save() -> None:
        model.get_booster().save_model(os.path.join(ctx["weights_dir"], "xgboost.json"))

    best = getattr(model, "best_iteration", None)
    extra = {"best_iteration": None if best is None else int(best)}
    return model.predict(X_test), y_test, dm.get_data()[2], save, extra, None


def _run_cross(ctx: dict) -> tuple:
    import torch

    from config.config_data import CONFIG_DATA
    from config.config_model import CONFIG_MODEL
    from model.models.CrossEncoder import get_model_cross_encoder
    from model.train import train_cross_encoder
    from utils.compute_weight import compute_class_weight
    from utils.evaluate import get_preds_cross_encoder

    dm = _data_manager(ctx, None, "cross_encoder")
    train_dl, evaluator = dm.loaders
    df_train, _, df_test = dm.get_data()
    weights = torch.tensor(
        compute_class_weight(df_train["label_score"].values),
        dtype=torch.float32,
        device=ctx["device"]
    )
    model = get_model_cross_encoder(
        ctx["spec"]["model"],
        num_classes=CONFIG_MODEL.NUM_CLASSES,
        max_len=CONFIG_DATA.MAX_LEN,
        weights_tensor=weights
    )
    epochs = 1 if ctx["smoke"] else CONFIG_MODEL.MODEL_CONFIG["cross_encoder"]["epochs"]
    model = train_cross_encoder(
        model, train_dl, evaluator, output_path=ctx["weights_dir"], epochs=epochs
    )
    preds, true = get_preds_cross_encoder(model, df_test)
    extra = {
        "epochs": epochs,
        "class_weights": weights.tolist(),
        "batch_size": CONFIG_MODEL.MODEL_CONFIG["cross_encoder"]["batch_size"],
        "best_val_accuracy": float(getattr(model, "best_score", float("nan")))
    }
    return preds, true, df_test, None, extra, None


def _run_bi(ctx: dict) -> tuple:
    from transformers import AutoTokenizer

    from config.config_model import CONFIG_MODEL
    from model.train import train_bi_encoder_baseline
    from utils.compute_weight import compute_class_weight
    from utils.evaluate import get_preds_siamese

    model_name = ctx["spec"]["model"]
    dm = _data_manager(ctx, AutoTokenizer.from_pretrained(model_name), "siamese")
    train_loader, val_loader = dm.loaders
    df_train, _, df_test = dm.get_data()
    cw = compute_class_weight(df_train["label_score"].values)
    epochs = 1 if ctx["smoke"] else CONFIG_MODEL.MODEL_CONFIG["siamese"]["num_epochs_cls"]
    out = ctx["weights_dir"]
    train_bi_encoder_baseline(
        model_name, train_loader, val_loader, cw, out, ctx["device"], epochs=epochs
    )
    preds, true = get_preds_siamese(df_test, out, model_name, ctx["device"])
    extra = {
        "epochs": epochs,
        "class_weights": list(map(float, cw)),
        "batch_size": CONFIG_MODEL.MODEL_CONFIG["siamese"]["batch_size"]
    }
    return preds, true, df_test, None, extra, None


def _check_class_tokens(tokenizer: object, tokens: list[str]) -> None:
    bad = [t for t in tokens if len(tokenizer.encode(t, add_special_tokens=False)) != 1]
    assert not bad, f"Class tokens split into subwords: {bad[:5]}"


def _trainer_tokenizer_kw() -> str:
    import inspect

    from transformers import Trainer

    return (
        "processing_class"
        if "processing_class" in inspect.signature(Trainer.__init__).parameters
        else "tokenizer"
    )


MEM_LIMIT_GB = 13.5


def _memory_probe(model: object, collator: object, train_ds: object, n: int = 8) -> float | None:
    """Expected peak VRAM (GiB) of training without gradient checkpointing.

    One forward + backward under fp16 autocast on the n longest training rows, plus 8 bytes per
    trainable parameter for the AdamW moments that the probe does not allocate. Runs before the
    Trainer is built, so the Trainer's own seeding is unaffected.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    with train_ds.formatted_as(None):
        lengths = [len(x) for x in train_ds["input_ids"]]
        idx = sorted(range(len(lengths)), key=lengths.__getitem__)[-n:]
        feats = [train_ds[i] for i in idx]
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in collator(feats).items()}
    was_training = model.training
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(**batch)
    out.loss.backward()
    peak = torch.cuda.max_memory_allocated()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.zero_grad(set_to_none=True)
    del out, batch
    model.train(was_training)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return round((peak + 8 * n_params) / 2**30, 2)


def _speed_settings(ctx: dict, cfg: dict, model: object, collator: object, train_ds: object) -> dict:
    """Rebuttal runs only (spec has "rebuttal"): gradient checkpointing off unless the memory probe
    says it will not fit, eval batch 64. The paper runs keep their original settings."""
    spec = ctx["spec"]
    if "rebuttal" not in spec:
        return {}
    gc = bool(spec.get("gradient_checkpointing", os.environ.get("CODRAFT_GC", "0") == "1"))
    probe = None
    if not gc:
        probe = _memory_probe(model, collator, train_ds)
        if probe is not None and probe > MEM_LIMIT_GB:
            gc = True
    cfg["training_args"]["gradient_checkpointing"] = gc
    cfg["training_args"]["per_device_eval_batch_size"] = 64
    print(
        f"Memory probe: {probe} GiB expected peak (limit {MEM_LIMIT_GB}) -> "
        f"gradient checkpointing {'ON' if gc else 'OFF'}"
    )
    return {"gradient_checkpointing": gc, "mem_probe_gb": probe}


def _run_multi(ctx: dict) -> tuple:
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    from config.config_model import CONFIG_MODEL
    from model.get_tokenizer import get_tokenizer
    from model.models.MultiTask import get_model_multi_task
    from utils.compute_weight import compute_class_weight
    from utils.evaluate import aux_accuracy, compute_metrics, get_preds_multi

    spec = ctx["spec"]
    tokenizer = get_tokenizer(spec["model"], add_class_tokens=True)
    dm = _data_manager(ctx, tokenizer, "multi_task")
    _check_class_tokens(tokenizer, list(dm.class_to_token.values()))
    train_ds, val_ds, test_ds = dm.datasets
    df_train, _, df_test = dm.get_data()
    assert len(test_ds) == 2 * len(df_test), "create_patterns must give two rows per pair"
    assert dm.NUM_PRODUCT_CLASSES == CONFIG_MODEL.NUM_PRODUCT_CLASSES

    cfg = CONFIG_MODEL.multi_task_args(seed=ctx["seed"], output_dir=ctx["scratch"])
    if ctx["smoke"]:
        cfg["training_args"]["num_train_epochs"] = 1
    loss = dict(cfg["loss_args"], loss_type=spec.get("loss_type", "rank_aware"))
    for k in ("aux_weight", "alpha"):
        if k in spec:
            loss[k] = spec[k]
    class_weights = (
        compute_class_weight(df_train["label_score"].values) if loss["loss_type"] == "ce" else None
    )
    model = get_model_multi_task(
        spec["model"],
        num_classes=ctx["n_classes"],
        num_product_classes=dm.NUM_PRODUCT_CLASSES,
        device=ctx["device"],
        class_weights=class_weights,
        tokenizer=tokenizer,
        **loss
    )
    assert model.get_input_embeddings().num_embeddings >= len(tokenizer)
    collator = DataCollatorWithPadding(tokenizer)
    speed = _speed_settings(ctx, cfg, model, collator, train_ds)
    args = TrainingArguments(**cfg["training_args"])
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        **{_trainer_tokenizer_kw(): tokenizer}
    )
    trainer.train()
    output = trainer.predict(test_ds)
    preds, true, logits = get_preds_multi(trainer, test_ds, df_test, output=output)
    aux_acc = aux_accuracy(output)
    print("Masked class head accuracy on test views:", aux_acc)

    def save() -> None:
        trainer.save_model(ctx["weights_dir"])
        tokenizer.save_pretrained(ctx["weights_dir"])

    extra = {
        "loss_args": loss,
        "class_weights": None if class_weights is None else list(map(float, class_weights)),
        "epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_batch": args.per_device_train_batch_size,
        "grad_accum": args.gradient_accumulation_steps,
        "effective_batch": args.per_device_train_batch_size
        * args.gradient_accumulation_steps
        * max(1, args.n_gpu),
        "fp16": args.fp16,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_val_f1_macro": trainer.state.best_metric,
        "vocab_size": len(tokenizer),
        "mask_hides_heading": bool(spec.get("mask_hides_heading", False)),
        "aux_acc_test": aux_acc,
        **speed
    }
    return preds, true, df_test, save, extra, logits


def _run_probe(ctx: dict) -> tuple:
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

    from model.models.MultiTask import JointClassSimBGE
    from preprocess.data_loader import create_patterns, to_dataset
    from preprocess.data_manager import class_maps
    from preprocess.preprocess_data import create_structured_text_enhanced, preprocess
    from utils.evaluate import build_result_df, get_preds_multi
    from utils.metrics import stats

    spec, results_dir, tag = ctx["spec"], ctx["results_dir"], ctx["tag"]
    ckpt = find_checkpoint(spec["source"])
    print("Checkpoint:", ckpt)
    tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=False)
    model = JointClassSimBGE.from_pretrained(ckpt).to(ctx["device"]).eval()
    class_to_token, class_to_id = class_maps()
    _check_class_tokens(tokenizer, list(class_to_token.values()))

    raw = pd.read_csv(os.path.join(ctx["data_root"], spec["variant"], "test.csv"), low_memory=False)
    base = preprocess(raw.copy())

    def frame(n1: list, p1: list, n2: list, p2: list, heading: bool = True) -> pd.DataFrame:
        d = base.copy()
        for side, n, p in (("1", n1, p1), ("2", n2, p2)):
            d[f"input_text_{side}"] = [
                create_structured_text_enhanced(t, a, b, c, heading=heading)
                for t, a, b, c in zip(d[f"Term {side}"], n, p, d[f"Class {side}"])
            ]
        return d

    N1, P1, N2, P2 = (
        raw[c].fillna("").astype(str).values
        for c in ("Nature 1", "Purpose 1", "Nature 2", "Purpose 2")
    )
    full = frame(N1, P1, N2, P2)
    rng = np.random.default_rng(SEED)
    pool = np.array(list(zip(np.r_[N1, N2], np.r_[P1, P2])), dtype=object)
    perm = pool[rng.permutation(len(pool))]
    n = len(raw)
    empty = [""] * n
    conditions = {
        "full": full,
        "no_nature": frame(empty, P1, empty, P2),
        "no_purpose": frame(N1, empty, N2, empty),
        "heading_only": frame(empty, empty, empty, empty),
        "no_heading": frame(N1, P1, N2, P2, heading=False),
        "bare_term": frame(empty, empty, empty, empty, heading=False),
        "shuffled_attributes": frame(perm[:n, 0], perm[:n, 1], perm[n:, 0], perm[n:, 1])
    }

    args = TrainingArguments(
        output_dir=ctx["scratch"],
        per_device_eval_batch_size=16,
        report_to="none",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=2
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=DataCollatorWithPadding(tokenizer),
        **{_trainer_tokenizer_kw(): tokenizer}
    )
    results = {}
    for name, d in conditions.items():
        ds = to_dataset(create_patterns(d, tokenizer, class_to_token, class_to_id), tokenizer)
        p, t, lg = get_preds_multi(trainer, ds, d)
        build_result_df(d, t, p).to_csv(
            os.path.join(results_dir, f"{tag}_{name}_preds.csv"), index=False
        )
        np.save(os.path.join(results_dir, f"{tag}_{name}_logits.npy"), lg)
        m = stats(t, p)
        results[name] = {
            "f1_macro": m["f1_macro"],
            "qwk": m["qwk"],
            "mae": m["mae"],
            "adjacent_acc": m["adjacent_acc"],
            "severe_rate": m["severe_rate"],
            "per_class_f1": [r["f1"] for r in m["per_class"]],
            "example_input": d["input_text_1"].iloc[0]
        }
        print(
            f"[probe] {name:20s} Macro-F1 {m['f1_macro'] * 100:.2f} | QWK {m['qwk']:.4f} | MAE {m['mae']:.4f}"
        )
        if name == "full":
            preds, true = p, t
    for r in results.values():
        r["delta_f1_vs_full"] = r["f1_macro"] - results["full"]["f1_macro"]
    return preds, true, full, None, {"checkpoint": ckpt, "conditions": results}, None


def _run_hybrid(ctx: dict) -> tuple:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import f1_score

    from utils.compute_weight import compute_class_weight

    device = ctx["device"]
    dm = _data_manager(ctx, None, "ml")
    df_train, df_val, df_test = dm.get_data()
    encoder_name = ctx["spec"]["model"]
    encoder = SentenceTransformer(encoder_name, device=str(device), trust_remote_code=True)
    texts = sorted(
        set(
            pd.concat(
                [
                    d[c]
                    for d in (df_train, df_val, df_test)
                    for c in ("input_text_1", "input_text_2")
                ]
            ).astype(str)
        )
    )
    emb = encoder.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    index = {t: i for i, t in enumerate(texts)}
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def feats(d: pd.DataFrame) -> np.ndarray:
        u = emb[[index[str(t)] for t in d["input_text_1"]]]
        v = emb[[index[str(t)] for t in d["input_text_2"]]]
        c1 = np.eye(45, dtype=np.float32)[d["Class 1"].values - 1]
        c2 = np.eye(45, dtype=np.float32)[d["Class 2"].values - 1]
        same = (d["Class 1"].values == d["Class 2"].values).astype(np.float32)[:, None]
        return np.hstack([u, v, np.abs(u - v), u * v, c1, c2, same]).astype(np.float32)

    Xtr, Xva, Xte = (torch.tensor(feats(d)) for d in (df_train, df_val, df_test))
    ytr = torch.tensor(df_train["label_score"].values)
    cw = compute_class_weight(df_train["label_score"].values)
    net = nn.Sequential(
        nn.Linear(Xtr.shape[1], 1024),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 5)
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device))
    max_epochs, patience = (2, 2) if ctx["smoke"] else (50, 8)
    best, best_state, best_epoch, wait = -1.0, None, 0, 0

    def predict(X: torch.Tensor) -> np.ndarray:
        net.eval()
        with torch.no_grad():
            chunks = [
                net(X[i : i + 1024].to(device)).argmax(1).cpu() for i in range(0, len(X), 1024)
            ]
        return torch.cat(chunks).numpy()

    for epoch in range(1, max_epochs + 1):
        net.train()
        order = torch.randperm(len(Xtr), generator=ctx["generator"])
        for i in range(0, len(order), 256):
            b = order[i : i + 256]
            opt.zero_grad()
            loss = crit(net(Xtr[b].to(device)), ytr[b].to(device))
            loss.backward()
            opt.step()
        val_f1 = f1_score(df_val["label_score"].values, predict(Xva), average="macro")
        print(f"Epoch {epoch} | loss {loss.item():.4f} | val macro-F1 {val_f1:.4f}")
        if val_f1 > best:
            best, best_epoch, wait = val_f1, epoch, 0
            best_state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)

    def save() -> None:
        torch.save(
            {
                "state_dict": best_state,
                "encoder": encoder_name,
                "in_dim": int(Xtr.shape[1]),
                "features": "[u, v, |u-v|, u*v, onehot(class1), onehot(class2), same_class]"
            },
            os.path.join(ctx["weights_dir"], "hybrid_mlp.pt")
        )

    extra = {
        "encoder": encoder_name,
        "feature_dim": int(Xtr.shape[1]),
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_val_f1_macro": float(best),
        "class_weights": list(map(float, cw)),
        "unique_texts": len(texts)
    }
    return predict(Xte), df_test["label_score"].values, df_test, save, extra, None


def _run_ce_recipe(ctx: dict) -> tuple:
    """A standard cross-encoder trained with the multi-task model's recipe (rebuttal E1).

    Same schedule, optimiser, batch and checkpoint selection (validation Macro-F1) as the full
    model, but its own head and loss: one 5-way head on [CLS], class-weighted cross-entropy, no
    class tokens, no masked views, no auxiliary head. It trains on the (A, B) order, as the
    original baselines did, and predicts both (A, B) and (B, A): preds.csv is (A, B); the
    average of the two orders (E1b) goes to {tag}_avg_preds.csv and settings["two_order"].
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments
    )

    from config.config_data import CONFIG_DATA
    from config.config_model import CONFIG_MODEL
    from model.get_tokenizer import get_tokenizer
    from utils.compute_weight import compute_class_weight
    from utils.evaluate import build_result_df, compute_metrics
    from utils.metrics import stats

    spec = ctx["spec"]
    tokenizer = get_tokenizer(spec["model"])
    dm = _data_manager(ctx, tokenizer, "pairs")
    df_train, df_val, df_test = dm.get_data()

    def dataset(d: object, swap: bool = False) -> object:
        a, b = ("input_text_2", "input_text_1") if swap else ("input_text_1", "input_text_2")
        ds = Dataset.from_dict(
            {
                "a": d[a].astype(str).tolist(),
                "b": d[b].astype(str).tolist(),
                "labels": d["label_score"].astype(int).tolist()
            }
        )
        ds = ds.map(
            lambda x: tokenizer(x["a"], x["b"], truncation=True, max_length=CONFIG_DATA.MAX_LEN),
            batched=True,
            remove_columns=["a", "b"]
        )
        return ds

    train_ds, val_ds = dataset(df_train), dataset(df_val)
    test_ab, test_ba = dataset(df_test), dataset(df_test, swap=True)

    class_weights = compute_class_weight(df_train["label_score"].values)
    weight = torch.tensor(class_weights, dtype=torch.float32)

    class WeightedTrainer(Trainer):
        def compute_loss(
            self, model: object, inputs: dict, return_outputs: bool = False, **kwargs: object
        ) -> object:
            labels = inputs.get("labels")
            out = model(**{k: v for k, v in inputs.items() if k != "labels"})
            loss = F.cross_entropy(
                out.logits.float(), labels, weight=weight.to(out.logits.device)
            )
            return (loss, out) if return_outputs else loss

    cfg = CONFIG_MODEL.multi_task_args(seed=ctx["seed"], output_dir=ctx["scratch"])
    if ctx["smoke"]:
        cfg["training_args"]["num_train_epochs"] = 1
    model = AutoModelForSequenceClassification.from_pretrained(
        spec["model"], num_labels=ctx["n_classes"], ignore_mismatched_sizes=True
    ).to(ctx["device"])
    collator = DataCollatorWithPadding(tokenizer)
    speed = _speed_settings(ctx, cfg, model, collator, train_ds)
    args = TrainingArguments(**cfg["training_args"])
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        **{_trainer_tokenizer_kw(): tokenizer}
    )
    trainer.train()

    def logits_of(ds: object) -> np.ndarray:
        p = trainer.predict(ds).predictions
        return p[0] if isinstance(p, tuple) else p

    ab, ba = logits_of(test_ab), logits_of(test_ba)
    if len(ab) != len(df_test) or len(ba) != len(df_test):
        raise ValueError(f"Got {len(ab)}/{len(ba)} prediction rows for {len(df_test)} test pairs")
    logits = np.stack([ab, ba], axis=1)
    true = df_test["label_score"].values
    preds = ab.argmax(-1)
    avg_preds = logits.mean(axis=1).argmax(-1)
    build_result_df(df_test, true, avg_preds).to_csv(
        os.path.join(ctx["results_dir"], f"{ctx['tag']}_avg_preds.csv"), index=False
    )
    m = stats(true, avg_preds, ctx["n_classes"])
    two_order = {
        k: m[k]
        for k in ("f1_macro", "qwk", "mae", "accuracy", "adjacent_acc", "severe_rate")
    }
    two_order["per_class_f1"] = [r["f1"] for r in m["per_class"]]
    print(
        f"(A,B)+(B,A) averaged: Macro-F1 {m['f1_macro'] * 100:.2f} | QWK {m['qwk']:.4f} | "
        f"MAE {m['mae']:.4f}"
    )

    def save() -> None:
        trainer.save_model(ctx["weights_dir"])
        tokenizer.save_pretrained(ctx["weights_dir"])

    extra = {
        "recipe": "multi-task (Table 7) with the baseline's own head and class-weighted CE",
        "class_weights": list(map(float, class_weights)),
        "epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_batch": args.per_device_train_batch_size,
        "grad_accum": args.gradient_accumulation_steps,
        "effective_batch": args.per_device_train_batch_size
        * args.gradient_accumulation_steps
        * max(1, args.n_gpu),
        "fp16": args.fp16,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_val_f1_macro": trainer.state.best_metric,
        "train_order": "(A, B)",
        "two_order": two_order,
        **speed
    }
    return preds, true, df_test, save, extra, logits


def _load_multi_checkpoint(ctx: dict) -> tuple:
    from transformers import AutoTokenizer

    from model.models.MultiTask import JointClassSimBGE
    from preprocess.data_manager import class_maps

    ckpt = find_checkpoint(ctx["spec"]["source"])
    print("Checkpoint:", ckpt)
    tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=False)
    model = JointClassSimBGE.from_pretrained(ckpt).to(ctx["device"]).eval()
    class_to_token, class_to_id = class_maps()
    _check_class_tokens(tokenizer, list(class_to_token.values()))
    return ckpt, tokenizer, model, class_to_token, class_to_id


def _predict_trainer(ctx: dict, model: object, tokenizer: object) -> object:
    """Prediction-only Trainer with the settings of the input probe (run 19), which reproduces run
    7's logits exactly: fp16, eval batch 16."""
    import torch
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    args = TrainingArguments(
        output_dir=ctx["scratch"],
        per_device_eval_batch_size=16,
        report_to="none",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=2
    )
    return Trainer(
        model=model,
        args=args,
        data_collator=DataCollatorWithPadding(tokenizer),
        **{_trainer_tokenizer_kw(): tokenizer}
    )


def _predict_views(trainer: object, frame: object, tokenizer: object, maps: tuple) -> tuple:
    from preprocess.data_loader import create_patterns, to_dataset
    from utils.evaluate import get_preds_multi

    ds = to_dataset(create_patterns(frame, tokenizer, *maps), tokenizer)
    out = trainer.predict(ds)
    preds, true, avg = get_preds_multi(trainer, ds, frame, output=out)
    return preds, true, avg, out


def _original_run(source: int) -> tuple:
    """preds.csv (and logits.npy if present) of a released run, for the reproduction checks."""
    import numpy as np
    import pandas as pd

    release = find_release_dir(source)
    preds = pd.read_csv(os.path.join(release, "preds.csv"))
    lg = os.path.join(release, "logits.npy")
    return preds, (np.load(lg) if os.path.isfile(lg) else None)


def _agreement(df_test: object, preds: object, logits: object, source: int) -> dict:
    import numpy as np
    import pandas as pd

    orig, orig_logits = _original_run(source)
    pos = pd.Series(np.arange(len(orig)), index=orig["Pair ID"].astype(str).values)
    idx = pos.reindex(df_test["Pair ID"].astype(str).values)
    keep = idx.notna().to_numpy()
    idx = idx[keep].astype(int).to_numpy()
    out = {
        "n_compared": int(keep.sum()),
        "pred_agreement": float((np.asarray(preds)[keep] == orig["pred"].values[idx]).mean())
    }
    if orig_logits is not None and logits is not None:
        diff = np.abs(np.asarray(logits)[keep] - orig_logits[idx])
        out["logits_max_abs_diff"] = float(diff.max())
    print(f"Reproduction of run {source}: {out}")
    return out


def _test_frame(ctx: dict, split: str = "test") -> object:
    import pandas as pd

    from preprocess.preprocess_data import preprocess

    raw = pd.read_csv(
        os.path.join(ctx["data_root"], ctx["spec"]["variant"], f"{split}.csv"), low_memory=False
    )
    return raw, preprocess(raw.copy())


def _run_aux_head(ctx: dict) -> tuple:
    """Rebuttal E9a: accuracy of run 7's masked class head, which training uses but the paper never
    measured. Inference only. Also checks that run 7 is reproduced (preds and logits)."""
    import numpy as np
    import pandas as pd

    spec = ctx["spec"]
    ckpt, tokenizer, model, c2t, c2i = _load_multi_checkpoint(ctx)
    id_to_class = {v: k for k, v in c2i.items()}
    trainer = _predict_trainer(ctx, model, tokenizer)
    extra = {"checkpoint": ckpt}
    for split in ("val", "test"):
        _, frame = _test_frame(ctx, split)
        preds, true, avg, out = _predict_views(trainer, frame, tokenizer, (c2t, c2i))
        aux_pred = np.argmax(out.predictions[1], axis=-1)
        aux_true = np.asarray(out.label_ids[1])
        hit = aux_pred == aux_true
        per_class = {
            int(id_to_class[c]): {"n": int((aux_true == c).sum()), "acc": float(hit[aux_true == c].mean())}
            for c in np.unique(aux_true)
        }
        extra[f"aux_acc_{split}"] = float(hit.mean())
        extra[f"aux_per_class_{split}"] = per_class
        extra[f"aux_worst_classes_{split}"] = sorted(per_class.items(), key=lambda kv: kv[1]["acc"])[:5]
        print(f"[E9a] {split}: masked class head accuracy {hit.mean() * 100:.2f}% on {len(hit)} views")
        if split == "test":
            test_frame, test = frame, (preds, true, avg)
            pd.DataFrame(
                {
                    "Pair ID": np.repeat(frame["Pair ID"].values, 2),
                    "masked_side": np.tile([1, 2], len(frame)),
                    "class_true": [int(id_to_class[c]) for c in aux_true],
                    "class_pred": [int(id_to_class[c]) for c in aux_pred]
                }
            ).to_csv(os.path.join(ctx["results_dir"], f"{ctx['tag']}_aux_preds.csv"), index=False)
    preds, true, avg = test
    extra["reproduction"] = _agreement(test_frame, preds, avg, spec["source"])
    return preds, true, test_frame, None, extra, avg


def _annotations(ctx: dict) -> tuple:
    """Majority vote of the three annotators behind the paper's Table 10 (TNH, VBD, NVTK) over the
    135 rated descriptions, keyed by (normalised description, NICE class), as in analysis/rebuttal.py
    (E5a). Returns (ratings, samples, annotate_dir, norm, counts)."""
    import re

    import pandas as pd

    def ok(d: str | None) -> bool:
        return bool(d) and os.path.isfile(os.path.join(d, "Samples.csv"))

    cands = [os.path.join(ctx["full_data_root"], "annotate"), os.path.join(REPO_DIR, "data", "annotate")]
    where = next((d for d in cands if ok(d)), None) or _search_kaggle(ok)
    if where is None:
        raise FileNotFoundError("data/annotate/Samples.csv not found (copy it from COLING/annotate/)")
    samples = pd.read_csv(os.path.join(where, "Samples.csv"))
    votes = {}
    for a in ("TNH", "VBD", "NVTK"):
        d = pd.read_csv(os.path.join(where, f"Annotation_{a}.csv"), encoding="utf-8-sig")
        assert (d["PairID"].values == samples["PairID"].values).all(), a
        votes[a] = d
    nat = sum(votes[a]["Nature"] for a in votes) >= 2
    pur = sum(votes[a]["Purpose"] for a in votes) >= 2

    def norm(x: object) -> str:
        return re.sub(r"\s+", " ", str(x)).strip().lower()

    ratings = {
        (norm(t), int(c)): (bool(n), bool(p))
        for t, c, n, p in zip(samples["Term"], samples["Class"], nat, pur)
    }
    counts = {
        "rows": len(samples),
        "distinct_descriptions": len(ratings),
        "nature_failed_rows": int((~nat).sum()),
        "purpose_failed_rows": int((~pur).sum()),
        "nature_failed_distinct": int(sum(not v[0] for v in ratings.values())),
        "purpose_failed_distinct": int(sum(not v[1] for v in ratings.values()))
    }
    return ratings, samples, where, norm, counts


def _run_counterfactual(ctx: dict) -> tuple:
    """Rebuttal E5 (counterfactual): re-predict with run 7 after replacing only the attributes that
    the annotators failed. Inference only, no retraining.

    Conditions (each on the whole test set; only the listed sides change):
      original             run 7's own inputs (must reproduce run 7)
      drop_failed          failed Nature/Purpose removed
      llm_<tag>            failed Nature/Purpose replaced by that open LLM's attribute for the
                           same pair and side (data/codraft_<tag>/, runs 21-24)
      control_drop_nature  on the *pass* pairs, the Nature of every rated side removed: how much
                           removing an attribute moves predictions by itself
      corrected            only if data/annotate/corrections.csv exists (PairID of Samples.csv,
                           Nature, Purpose; blank keeps the original)
    """
    import numpy as np
    import pandas as pd

    from preprocess.preprocess_data import create_structured_text_enhanced
    from utils.evaluate import build_result_df
    from utils.metrics import stats

    spec, tag = ctx["spec"], ctx["tag"]
    ratings, samples, annotate_dir, norm, counts = _annotations(ctx)
    raw, base = _test_frame(ctx)
    n = len(raw)
    attr = {
        (s, a): raw[f"{a} {s}"].fillna("").astype(str).to_numpy(dtype=object)
        for s in ("1", "2")
        for a in ("Nature", "Purpose")
    }
    status = {
        s: [ratings.get((norm(t), int(c))) for t, c in zip(raw[f"Term {s}"], raw[f"Class {s}"])]
        for s in ("1", "2")
    }
    failed = {
        (s, a): np.array([st is not None and not st[k] for st in status[s]])
        for s in ("1", "2")
        for k, a in enumerate(("Nature", "Purpose"))
    }
    rated = {s: np.array([st is not None for st in status[s]]) for s in ("1", "2")}
    fail_pair = np.zeros(n, bool)
    for m in failed.values():
        fail_pair |= m
    pass_pair = (rated["1"] | rated["2"]) & ~fail_pair

    def frame(values: dict) -> pd.DataFrame:
        d = base.copy()
        for s in ("1", "2"):
            d[f"input_text_{s}"] = [
                create_structured_text_enhanced(t, a, b, c)
                for t, a, b, c in zip(
                    d[f"Term {s}"], values[(s, "Nature")], values[(s, "Purpose")], d[f"Class {s}"]
                )
            ]
        return d

    def replaced(source: dict, where: dict) -> tuple:
        v = {k: x.copy() for k, x in attr.items()}
        for k, m in where.items():
            v[k][m] = source[k][m] if source is not None else ""
        return v

    conditions = {"original": (attr, None), "drop_failed": (replaced(None, failed), "fail")}
    llm_missing = {}
    for llm in ("qwen2.5-7b", "qwen3-8b", "llama3.1-8b", "nemotron-nano-8b"):
        try:
            root = find_variant_root(ctx["full_data_root"], f"codraft_{llm}")
        except FileNotFoundError:
            llm_missing[llm] = "variant not found"
            continue
        other = pd.read_csv(os.path.join(root, f"codraft_{llm}", "test.csv"), low_memory=False)
        other = other.set_index(other["Pair ID"].astype(str)).reindex(raw["Pair ID"].astype(str))
        src = {
            (s, a): other[f"{a} {s}"].fillna("").astype(str).to_numpy(dtype=object)
            for s in ("1", "2")
            for a in ("Nature", "Purpose")
        }
        conditions[f"llm_{llm}"] = (replaced(src, failed), "fail")
    nature_rated = {(s, "Nature"): rated[s] & pass_pair for s in ("1", "2")}
    conditions["control_drop_nature"] = (replaced(None, nature_rated), "pass")
    corr_path = os.path.join(annotate_dir, "corrections.csv")
    if os.path.isfile(corr_path):
        corr = pd.read_csv(corr_path).merge(samples[["PairID", "Term", "Class"]], on="PairID")
        fix = {}
        for _, r in corr.iterrows():
            fix[(norm(r["Term"]), int(r["Class"]))] = (r.get("Nature"), r.get("Purpose"))
        v = {k: x.copy() for k, x in attr.items()}
        for s in ("1", "2"):
            for i, (t, c) in enumerate(zip(raw[f"Term {s}"], raw[f"Class {s}"])):
                f = fix.get((norm(t), int(c)))
                for k, a in enumerate(("Nature", "Purpose")):
                    if f is not None and isinstance(f[k], str) and f[k].strip():
                        v[(s, a)][i] = f[k].strip()
        conditions["corrected"] = (v, "fail")

    ckpt, tokenizer, model, c2t, c2i = _load_multi_checkpoint(ctx)
    trainer = _predict_trainer(ctx, model, tokenizer)
    y = base["label_score"].to_numpy()
    subsets = {"fail": fail_pair, "pass": pass_pair}
    results, orig = {}, None
    for name, (values, subset) in conditions.items():
        d = frame(values)
        changed = np.zeros(n, bool)
        for s in ("1", "2"):
            changed |= d[f"input_text_{s}"].to_numpy() != base[f"input_text_{s}"].to_numpy()
        p, t, lg, _ = _predict_views(trainer, d, tokenizer, (c2t, c2i))
        build_result_df(d, t, p).to_csv(
            os.path.join(ctx["results_dir"], f"{tag}_{name}_preds.csv"), index=False
        )
        np.save(os.path.join(ctx["results_dir"], f"{tag}_{name}_logits.npy"), lg)
        if name == "original":
            orig = (p, lg)
            assert not changed.any(), "the original condition must rebuild run 7's inputs exactly"
        m = stats(t, p)
        r = {
            "pairs_with_changed_input": int(changed.sum()),
            "f1_macro_test": m["f1_macro"],
            "qwk_test": m["qwk"],
            "mae_test": m["mae"],
            "example": d.loc[changed, "input_text_1"].head(2).tolist() if changed.any() else []
        }
        if subset is not None:
            sel = subsets[subset]
            w0, w1 = orig[0][sel] != y[sel], p[sel] != y[sel]
            s0 = np.abs(orig[0][sel] - y[sel]) >= 2
            s1 = np.abs(p[sel] - y[sel]) >= 2
            r.update(
                subset=subset,
                n=int(sel.sum()),
                pred_changed=int((orig[0][sel] != p[sel]).sum()),
                wrong_to_right=int((w0 & ~w1).sum()),
                right_to_wrong=int((~w0 & w1).sum()),
                err_before=float(w0.mean()) if sel.any() else None,
                err_after=float(w1.mean()) if sel.any() else None,
                severe_before=float(s0.mean()) if sel.any() else None,
                severe_after=float(s1.mean()) if sel.any() else None
            )
        results[name] = r
        print(f"[E5] {name:22s} " + ", ".join(f"{k}={v}" for k, v in r.items() if k != "example"))
    extra = {
        "checkpoint": ckpt,
        "annotators": ["TNH", "VBD", "NVTK"],
        "failed_descriptions": counts,
        "pairs": {"fail": int(fail_pair.sum()), "pass": int(pass_pair.sum())},
        "failed_sides": {f"{a} {s}": int(m.sum()) for (s, a), m in failed.items()},
        "llm_missing": llm_missing,
        "corrections": corr_path if "corrected" in conditions else "none (no data/annotate/corrections.csv)",
        "conditions": results,
        "reproduction": _agreement(base, orig[0], orig[1], spec["source"])
    }
    return orig[0], y, base, None, extra, orig[1]


def _run_ce_two_order(ctx: dict) -> tuple:
    """Rebuttal E1b on the original cross-encoders (runs 3-6): predict (A, B), as the paper did,
    and (B, A), and average the two. Inference only, fp32 and batches of 32 like
    sentence-transformers' CrossEncoder.predict, so (A, B) reproduces the released preds.csv.
    preds.csv = the two orders averaged; conditions/ab_preds.csv = (A, B) alone."""
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from config.config_data import CONFIG_DATA
    from utils.evaluate import build_result_df
    from utils.metrics import stats

    spec = ctx["spec"]
    release = find_release_dir(spec["source"])
    if release is None:
        raise FileNotFoundError(
            f"weights/{RUNS[spec['source']]['path']}/model not found; attach codraft-weight"
        )
    ckpt = os.path.join(release, "model")
    print("Checkpoint:", ckpt)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(ctx["device"]).eval()
    dm = _data_manager(ctx, tokenizer, "pairs")
    df_test = dm.get_data()[2]
    a = [str(x) for x in df_test["input_text_1"]]
    b = [str(x) for x in df_test["input_text_2"]]

    def logits_of(first: list, second: list) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(first), 32):
                enc = tokenizer(
                    first[i : i + 32],
                    second[i : i + 32],
                    padding=True,
                    truncation="longest_first",
                    max_length=CONFIG_DATA.MAX_LEN,
                    return_tensors="pt"
                ).to(ctx["device"])
                out.append(model(**enc).logits.float().cpu().numpy())
        return np.concatenate(out)

    ab, ba = logits_of(a, b), logits_of(b, a)
    logits = np.stack([ab, ba], axis=1)
    true = df_test["label_score"].values
    ab_preds, preds = ab.argmax(-1), logits.mean(axis=1).argmax(-1)
    build_result_df(df_test, true, ab_preds).to_csv(
        os.path.join(ctx["results_dir"], f"{ctx['tag']}_ab_preds.csv"), index=False
    )
    m = stats(true, ab_preds, ctx["n_classes"])
    ab_metrics = {k: m[k] for k in ("f1_macro", "qwk", "mae", "accuracy", "adjacent_acc", "severe_rate")}
    ab_metrics["per_class_f1"] = [r["f1"] for r in m["per_class"]]
    print(
        f"[E1b] (A,B) only: Macro-F1 {m['f1_macro'] * 100:.2f} | QWK {m['qwk']:.4f} | MAE {m['mae']:.4f}"
    )
    extra = {
        "checkpoint": ckpt,
        "source_run": spec["source"],
        "inference": "fp32, batch 32, max_length 256, truncation longest_first",
        "ab_metrics": ab_metrics,
        "reproduction": _agreement(df_test, ab_preds, None, spec["source"])
    }
    return preds, true, df_test, None, extra, logits


FAMILIES = {
    "xgboost": _run_xgboost,
    "ce_recipe": _run_ce_recipe,
    "cross": _run_cross,
    "bi": _run_bi,
    "multi": _run_multi,
    "probe": _run_probe,
    "hybrid": _run_hybrid,
    "aux_head": _run_aux_head,
    "counterfactual": _run_counterfactual,
    "ce_two_order": _run_ce_two_order
}


def run_experiment(
    run_id: int, data_root: str | None = None, out_root: str = ".", smoke: bool = False
) -> dict:
    import numpy as np
    import torch

    from utils.evaluate import build_result_df
    from utils.metrics import binary_metrics, plot_confusion, stats

    spec = get_run(run_id)
    seed = run_seed(run_id)
    tag = run_tag(run_id, smoke)
    n_classes = 2 if spec.get("binary") else 5
    if n_classes == 2 and spec["family"] != "multi":
        raise ValueError("binary=True is implemented for the multi family only")
    data_root = find_variant_root(find_data_root(data_root), spec["variant"])
    results_dir = os.path.join(out_root, "results")
    weights_dir = os.path.join(out_root, "weights", tag)
    scratch = os.path.join("/tmp" if os.path.isdir("/tmp") else out_root, "codraft_ckpt", tag)
    for d in (results_dir, weights_dir, scratch):
        os.makedirs(d, exist_ok=True)

    print("=" * 78)
    print(f"Run {run_id}: {spec['name']}  [{spec['table']}] {spec['row']}")
    smoke_note = "  SMOKE TEST" if smoke else ""
    print(
        f"family={spec['family']} model={spec['model']} variant={spec['variant']} seed={seed}{smoke_note}"
    )
    print("=" * 78)
    full_data_root = data_root
    if smoke:
        data_root = _smoke_data(data_root, spec["variant"], os.path.join(scratch, "data"))
    print("Data:", data_root)

    ctx = {
        "spec": spec,
        "tag": tag,
        "smoke": smoke,
        "data_root": data_root,
        "full_data_root": full_data_root,
        "results_dir": results_dir,
        "weights_dir": weights_dir,
        "scratch": scratch,
        "n_classes": n_classes,
        "seed": seed,
        "generator": _setup_determinism(seed),
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    env = _environment()
    print("Environment:", json.dumps(env))
    t0 = time.time()
    try:
        preds, true, df_test, save, extra, logits = FAMILIES[spec["family"]](ctx)
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise

    true = np.asarray(true).astype(int)
    preds = np.asarray(preds).astype(int)
    if logits is not None:
        np.save(os.path.join(results_dir, f"{tag}_logits.npy"), logits)
    result_df = build_result_df(df_test, true, preds)
    if "label_5" in df_test.columns:
        result_df["label_5"] = df_test["label_5"].values
    result_df.to_csv(os.path.join(results_dir, f"{tag}_preds.csv"), index=False)
    m = stats(true, preds, n_classes)
    plot_confusion(m["confusion_matrix"], os.path.join(results_dir, f"cm_{tag}"))
    record = {
        "run_id": run_id,
        "tag": tag,
        "smoke": smoke,
        **spec,
        "seed": seed,
        "n_test": int(len(df_test)),
        **{
            k: m[k]
            for k in ("f1_macro", "qwk", "mae", "accuracy", "precision_macro", "recall_macro")
        },
        "adjacent_acc": m["adjacent_acc"],
        "severe_rate": m["severe_rate"],
        "per_class": m["per_class"],
        "n_classes": n_classes,
        "binary": binary_metrics(true, preds),
        "confusion_matrix": m["confusion_matrix"],
        "train_seconds": round(time.time() - t0, 1),
        "environment": env,
        "settings": extra
    }
    with open(os.path.join(results_dir, f"{tag}_metrics.json"), "w") as fh:
        json.dump(record, fh, indent=2, default=float)

    if save is not None and os.environ.get("CODRAFT_SAVE_WEIGHTS", "1") == "0":
        print("CODRAFT_SAVE_WEIGHTS=0: model weights not saved.")
    elif save is not None:
        try:
            save()
            print("Weights:", weights_dir)
        except Exception as e:
            print(f"Could not save the weights ({type(e).__name__}: {e}); the results are kept.")
    shutil.rmtree(scratch, ignore_errors=True)
    if os.path.isdir(weights_dir) and not os.listdir(weights_dir):
        os.rmdir(weights_dir)

    b = record["binary"]
    print("\n" + "=" * 78)
    print(
        f"DONE run {run_id} {spec['name']}: Macro-F1 {m['f1_macro'] * 100:.2f} | QWK {m['qwk']:.4f} | "
        f"MAE {m['mae']:.4f} | n={len(df_test)} | {record['train_seconds'] / 60:.1f} min"
    )
    print("Per-class F1: " + " / ".join(f"{r['f1'] * 100:.2f}" for r in m["per_class"]))
    balanced = (
        f", {b['balanced']['f1']:.4f} on {b['n_balanced']} balanced test sets"
        if "balanced" in b
        else ""
    )
    print(f"Binary (Le Nir): F1 {b['natural']['f1']:.4f} on the test set{balanced}")
    print("=" * 78)
    return record


def _gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        return len([line for line in out.stdout.splitlines() if line.startswith("GPU ")])
    except Exception:
        return 0


def _say(text: str) -> None:
    with PRINT_LOCK:
        print(text, flush=True)


def _stream(proc: subprocess.Popen, prefix: str, log_path: str, progress_every: int = 60) -> None:
    last, buf = 0.0, ""
    with open(log_path, "w", buffering=1) as log:
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            log.write(ch)
            if ch not in "\r\n":
                buf += ch
                continue
            line, buf = buf.strip(), ""
            if line and (ch == "\n" or time.time() - last > progress_every):
                _say(f"{prefix} {line}")
                if ch == "\r":
                    last = time.time()
        if buf.strip():
            _say(f"{prefix} {buf.strip()}")


def _fits(elapsed_min: float, est_min: float | None, budget_hours: float | None) -> bool:
    """True if a run expected to take est_min (with a 30% margin) ends inside the budget."""
    if budget_hours is None or est_min is None:
        return True
    return elapsed_min + 1.3 * est_min <= 60 * budget_hours


def launch(
    run_ids: int | list[int],
    data_root: str | None = None,
    out_root: str = ".",
    smoke: bool = False,
    budget_hours: float | None = None,
    package_zip: bool = True
) -> dict:
    """Run several experiments, one GPU each, from a shared queue.

    budget_hours: a run is started only if its EST_MIN (config/runs.py) fits in what is left, so a
    Kaggle session is never cut at its time limit. package_zip=False leaves the files loose for
    the caller to zip. Returns {"failed": [...], "skipped": [...]}; raises if a run failed.
    """
    from config.runs import EST_MIN

    t_start = time.time()
    ids = [run_ids] if isinstance(run_ids, int) else list(run_ids)
    for run_id in ids:
        get_run(run_id)
    data_root = find_data_root(data_root)
    n_gpu = _gpu_count()
    slots = max(1, n_gpu)
    log_dir = os.path.join(out_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Runs {ids} | GPUs found: {n_gpu} | data: {data_root} | out: {out_root}")
    if len(ids) > slots:
        print(f"More runs than GPUs: they will share {slots} card(s) in turn.")

    pending, running, failed, skipped = list(ids), {}, [], []
    while pending or running:
        for slot in range(slots):
            if slot in running or not pending:
                continue
            run_id = pending.pop(0)
            elapsed = (time.time() - t_start) / 60
            if not smoke and not _fits(elapsed, EST_MIN.get(run_id), budget_hours):
                skipped.append(run_id)
                print(
                    f"Skipped run {run_id}: ~{EST_MIN[run_id]} min would pass the "
                    f"{budget_hours} h budget ({elapsed:.0f} min used)"
                )
                continue
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            if n_gpu:
                env["CUDA_VISIBLE_DEVICES"] = str(slot)
            cmd = [
                sys.executable,
                "-u",
                os.path.join(REPO_DIR, "runner.py"),
                "--run-id",
                str(run_id),
                "--data-root",
                data_root,
                "--out-root",
                out_root
            ]
            if smoke:
                cmd.append("--smoke")
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            device = f"gpu{slot}" if n_gpu else "cpu"
            log_path = os.path.join(log_dir, f"{run_tag(run_id, smoke)}.log")
            thread = threading.Thread(
                target=_stream, daemon=True, args=(proc, f"[{run_id:02d}|{device}]", log_path)
            )
            thread.start()
            running[slot] = (run_id, proc, thread)
            print(f"Started run {run_id} on {device}")
        for slot, (run_id, proc, thread) in list(running.items()):
            if proc.poll() is None:
                continue
            thread.join()
            if proc.returncode != 0:
                failed.append(run_id)
                print(f"Run {run_id} FAILED (exit {proc.returncode}); see logs/")
            else:
                print(f"Run {run_id} finished")
            del running[slot]
        time.sleep(2)

    summary(out_root, smoke=smoke)
    done = [run_id for run_id in ids if run_id not in skipped]
    if package_zip and done:
        package([run_tag(run_id, smoke) for run_id in done], out_root)
    print(f"Wall time {(time.time() - t_start) / 60:.0f} min")
    if skipped:
        print(f"Skipped for time: {skipped}. Run them in another session.")
    if failed:
        raise RuntimeError(f"Runs failed: {failed}. The full output is in {log_dir}/")
    return {"failed": failed, "skipped": skipped}


def package(tags: list[str], out_root: str = ".") -> str | None:
    files = []
    for tag in tags:
        files += glob.glob(os.path.join(out_root, "results", f"{tag}_*"))
        files += glob.glob(os.path.join(out_root, "results", f"cm_{tag}.*"))
        files += glob.glob(os.path.join(out_root, "logs", f"{tag}.log"))
        for root, _, fs in os.walk(os.path.join(out_root, "weights", tag)):
            files += [os.path.join(root, f) for f in fs]
    if not files:
        print("Nothing to package.")
        return None
    ids = "-".join(t.split("_")[1 if t.startswith("SMOKE_") else 0] for t in tags)
    name = ("SMOKE_" if tags[0].startswith("SMOKE_") else "") + f"codraft_runs_{ids}.zip"
    path = os.path.join(out_root, name)
    with zipfile.ZipFile(path, "w", allowZip64=True) as zf:
        for f in sorted(set(files)):
            heavy = os.sep + "weights" + os.sep in f
            zf.write(
                f,
                os.path.relpath(f, out_root),
                compress_type=zipfile.ZIP_STORED if heavy else zipfile.ZIP_DEFLATED
            )
    with zipfile.ZipFile(path) as zf:
        bad, n = zf.testzip(), len(zf.namelist())
    if bad is not None:
        print(f"Zip check failed on {bad}; the loose files are kept.")
        return path
    for tag in tags:
        shutil.rmtree(os.path.join(out_root, "weights", tag), ignore_errors=True)
    print(f"\nPackaged {n} files -> {path} ({os.path.getsize(path) / 2**20:.1f} MB)")
    print("Download this one file from the Output panel.")
    return path


def release_dest(member: str, repo_dir: str = REPO_DIR) -> str | None:
    weights, data = os.path.join(repo_dir, "weights"), os.path.join(repo_dir, "data")
    top, base = member.split("/", 1)[0], os.path.basename(member)
    if top == "results" and base.startswith("cm_") and base.endswith((".png", ".pdf")):
        return None
    m = re.search(r"(?<![A-Za-z0-9])(\d\d)_([a-z0-9_.\-]+?)_seed(\d+)", member)
    if m and not member.startswith(("attributes/", "codraft_")):
        run_id, tag = int(m.group(1)), m.group(0)
        if run_id not in RUNS or RUNS[run_id]["name"] != m.group(2):
            raise ValueError(f"{member}: tag {tag} does not match run {run_id} in config/runs.py")
        root = os.path.join(weights, RUNS[run_id]["path"])
        if top == "weights":
            return os.path.join(root, "model", member.split(tag + "/", 1)[1])
        if base == f"{tag}.log":
            return os.path.join(root, "run.log")
        for suffix, name in RESULT_FILES.items():
            if base == tag + suffix:
                return os.path.join(root, name)
            if base.startswith(tag + "_") and base.endswith(suffix):
                return os.path.join(root, "conditions", base[len(tag) + 1 :])
        raise ValueError(f"{member}: no place for this file of {tag}")
    m = re.match(
        r"llm_(.+?)_(\d+shot)_([a-z0-9_.\-]+?)(_preds\.csv|_metrics\.json|_raw\.jsonl)$", base
    )
    if top == "results" and m:
        llm, shots, inp, kind = m.groups()
        return os.path.join(weights, "llm-classifiers", llm, f"{shots}_{inp}", RESULT_FILES[kind])
    if top.startswith("codraft_"):
        return os.path.join(data, member)
    if top == "attributes":
        if base.endswith("_raw.jsonl"):
            return os.path.join(
                weights, "llm-classifiers", base[: -len("_raw.jsonl")], "enrich_raw.jsonl"
            )
        return os.path.join(data, "attributes", base)
    if top == "logs":
        if base.startswith("vllm_"):
            return os.path.join(
                weights, "llm-classifiers", base[len("vllm_") : -len(".log")], "vllm.log"
            )
        m = re.match(r"(.+?)_((?:enrich|classify)(?:-(?:enrich|classify))?)\.log$", base)
        if m:
            return os.path.join(weights, "llm-classifiers", m.group(1), "run.log")
    raise ValueError(f"{member}: no place for it in weights/ or data/")


def install(zip_path: str, repo_dir: str = REPO_DIR, keep_zip: bool = False) -> list[str]:
    with zipfile.ZipFile(zip_path) as z:
        if z.testzip() is not None:
            raise ValueError(f"{zip_path} is damaged")
        plan = [(i, release_dest(i.filename, repo_dir)) for i in z.infolist() if not i.is_dir()]
        skipped = sum(d is None for _, d in plan)
        plan = [(i, d) for i, d in plan if d is not None]
        dsts = [d for _, d in plan]
        clash = sorted(
            {d for d in dsts if dsts.count(d) > 1} | {d for d in dsts if os.path.exists(d)}
        )
        if clash:
            raise FileExistsError(f"Would overwrite {len(clash)} files, e.g. {clash[:3]}")
        for info, dst in plan:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(info) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 24)
            if os.path.getsize(dst) != info.file_size:
                raise OSError(f"{dst}: size differs from the zip")
    if not keep_zip:
        os.remove(zip_path)
    where = sorted(
        {os.path.relpath(os.path.dirname(d), repo_dir).split(os.sep + "model")[0] for d in dsts}
    )
    skipped_note = f"; {skipped} figures skipped" if skipped else ""
    deleted_note = "" if keep_zip else "; zip deleted"
    print(
        f"{os.path.basename(zip_path)}: {len(plan)} files -> {', '.join(where)}{skipped_note}{deleted_note}"
    )
    return dsts


def summary(out_root: str = ".", smoke: bool = False) -> None:
    files = sorted(glob.glob(os.path.join(out_root, "results", "*_metrics.json")))
    files = [f for f in files if os.path.basename(f).startswith("SMOKE_") == smoke]
    if not files:
        print("No finished runs yet.")
        return
    print(f"\n{'run':<38}{'MacroF1':>8}{'QWK':>8}{'MAE':>8}{'adj':>8}{'n':>6}{'min':>7}")
    for f in files:
        with open(f) as fh:
            r = json.load(fh)
        print(
            f"{r['tag']:<38}{r['f1_macro'] * 100:8.2f}{r['qwk']:8.4f}{r['mae']:8.4f}"
            f"{r['adjacent_acc'] * 100:7.2f}%{r['n_test']:6d}{r['train_seconds'] / 60:7.1f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=int)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-root", default=".")
    ap.add_argument("--smoke", action="store_true", help="tiny subset, one epoch")
    ap.add_argument("--list", action="store_true")
    ap.add_argument(
        "--install", nargs="+", metavar="ZIP", help="unpack session zips into weights/ and data/"
    )
    ap.add_argument("--keep-zip", action="store_true", help="with --install: keep the zip")
    a = ap.parse_args()
    if a.install:
        for z in a.install:
            install(z, keep_zip=a.keep_zip)
    elif a.list or a.run_id is None:
        print(describe())
    else:
        run_experiment(a.run_id, data_root=a.data_root, out_root=a.out_root, smoke=a.smoke)


if __name__ == "__main__":
    main()
