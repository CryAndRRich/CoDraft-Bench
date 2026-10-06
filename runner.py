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
    "_raw.jsonl": "raw.jsonl",
}
PRINT_LOCK = threading.Lock()


def run_tag(run_id: int, smoke: bool = False) -> str:
    return ("SMOKE_" if smoke else "") + f"{run_id:02d}_{RUNS[run_id]['name']}_seed{SEED}"


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


def find_checkpoint(source: int) -> str:
    path = RUNS[source]["path"]
    tag = run_tag(source)

    def ok(d: str | None) -> bool:
        return bool(d) and all(
            os.path.isfile(os.path.join(d, f)) for f in ("model.safetensors", "config.json")
        )

    for d in (os.environ.get("CODRAFT_CKPT"), os.path.join(REPO_DIR, "weights", path, "model")):
        if ok(d):
            return d
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
        "n_gpu_visible": torch.cuda.device_count(),
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
    )
    for split, d in zip(("train", "val", "test"), dm.get_data()):
        for c in ("input_text_1", "input_text_2"):
            bad = d[c].astype(str).str.contains(r"(?:Nature|Use|Category): nan\b", regex=True)
            assert not bad.any(), (
                f'{split} {c}: {int(bad.sum())} inputs carry "nan", e.g. {d[c][bad].iloc[0]!r}'
            )
    print("Sample input:", dm.get_data()[2]["input_text_1"].iloc[0])
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
        device=ctx["device"],
    )
    model = get_model_cross_encoder(
        ctx["spec"]["model"],
        num_classes=5,
        max_len=CONFIG_DATA.MAX_LEN,
        weights_tensor=weights,
    )
    epochs = 1 if ctx["smoke"] else CONFIG_MODEL.MODEL_CONFIG["cross_encoder"]["epochs"]
    model = train_cross_encoder(
        model, train_dl, evaluator, output_path=ctx["weights_dir"], epochs=epochs
    )
    preds, true = get_preds_cross_encoder(model, df_test)
    extra = {
        "epochs": epochs,
        "class_weights": weights.tolist(),
        "batch_size": 32,
        "best_val_accuracy": float(getattr(model, "best_score", float("nan"))),
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
        "batch_size": CONFIG_MODEL.MODEL_CONFIG["siamese"]["physical_batch_size"],
    }
    return preds, true, df_test, None, extra, None


def _trainer_tokenizer_kw() -> str:
    import inspect

    from transformers import Trainer

    return (
        "processing_class"
        if "processing_class" in inspect.signature(Trainer.__init__).parameters
        else "tokenizer"
    )


def _run_multi(ctx: dict) -> tuple:
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    from config.config_model import CONFIG_MODEL
    from model.get_tokenizer import get_tokenizer
    from model.models.MultiTask import get_model_multi_task
    from utils.compute_weight import compute_class_weight
    from utils.evaluate import compute_metrics, get_preds_multi

    spec = ctx["spec"]
    tokenizer = get_tokenizer(spec["model"], add_class_tokens=True)
    dm = _data_manager(ctx, tokenizer, "multi_task")
    bad = [
        t
        for t in dm.class_to_token.values()
        if len(tokenizer.encode(t, add_special_tokens=False)) != 1
    ]
    assert not bad, f"Class markers split into subwords: {bad[:5]}"
    train_ds, val_ds, test_ds = dm.datasets
    df_train, _, df_test = dm.get_data()
    assert len(test_ds) == 2 * len(df_test), "create_patterns must give two rows per pair"
    assert dm.NUM_PRODUCT_CLASSES == CONFIG_MODEL.NUM_PRODUCT_CLASSES

    cfg = CONFIG_MODEL.multi_task_args(seed=SEED, output_dir=ctx["scratch"])
    if ctx["smoke"]:
        cfg["training_args"]["num_train_epochs"] = 1
    args = TrainingArguments(**cfg["training_args"])
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
        **loss,
    )
    assert model.get_input_embeddings().num_embeddings >= len(tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        **{_trainer_tokenizer_kw(): tokenizer},
    )
    trainer.train()
    preds, true, logits = get_preds_multi(trainer, test_ds, df_test)

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
    }
    return preds, true, df_test, save, extra, logits


def _run_probe(ctx: dict) -> tuple:
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

    from config.config_data import CONFIG_DATA
    from model.models.MultiTask import JointClassSimBGE
    from preprocess.data_loader import create_patterns, to_dataset
    from preprocess.preprocess_data import preprocess
    from utils.evaluate import build_result_df, get_preds_multi
    from utils.metrics import stats

    spec, results_dir, tag = ctx["spec"], ctx["results_dir"], ctx["tag"]
    ckpt = find_checkpoint(spec["source"])
    print("Checkpoint:", ckpt)
    tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=False)
    model = JointClassSimBGE.from_pretrained(ckpt).to(ctx["device"]).eval()
    nice = sorted(CONFIG_DATA.NICE_CLASS_MAP)
    class_to_token = {c: f"[CLASS_{c}]" for c in nice}
    class_to_id = {c: i for i, c in enumerate(nice)}
    bad = [
        t
        for t in class_to_token.values()
        if len(tokenizer.encode(t, add_special_tokens=False)) != 1
    ]
    assert not bad, f"Checkpoint tokenizer lacks the class tokens: {bad[:5]}"

    raw = pd.read_csv(os.path.join(ctx["data_root"], spec["variant"], "test.csv"), low_memory=False)
    base = preprocess(raw.copy())

    def build(term: str, nature: str, purpose: str, cls: int, use_c: bool = True) -> str:
        parts = []
        if str(nature).strip():
            parts.append(f"Nature: {str(nature).strip()}")
        if str(purpose).strip():
            parts.append(f"Use: {str(purpose).strip()}")
        if use_c and CONFIG_DATA.NICE_CLASS_MAP.get(int(cls), ""):
            parts.append(f"Category: {CONFIG_DATA.NICE_CLASS_MAP[int(cls)]}")
        text = str(term).strip()
        return f"{text} [ {' | '.join(parts)} ]" if parts else text

    def frame(n1: list, p1: list, n2: list, p2: list, use_c: bool = True) -> pd.DataFrame:
        d = base.copy()
        d["input_text_1"] = [build(*x, use_c=use_c) for x in zip(d["Term 1"], n1, p1, d["Class 1"])]
        d["input_text_2"] = [build(*x, use_c=use_c) for x in zip(d["Term 2"], n2, p2, d["Class 2"])]
        return d

    N1, P1, N2, P2 = (
        raw[c].fillna("").astype(str).values
        for c in ("Nature 1", "Purpose 1", "Nature 2", "Purpose 2")
    )
    full = frame(N1, P1, N2, P2)
    assert (full["input_text_1"] == base["input_text_1"]).all(), (
        "Probe input builder drifted from preprocess()"
    )
    assert (full["input_text_2"] == base["input_text_2"]).all(), (
        "Probe input builder drifted from preprocess()"
    )
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
        "no_heading": frame(N1, P1, N2, P2, use_c=False),
        "bare_term": frame(empty, empty, empty, empty, use_c=False),
        "shuffled_attributes": frame(perm[:n, 0], perm[:n, 1], perm[n:, 0], perm[n:, 1]),
    }

    args = TrainingArguments(
        output_dir=ctx["scratch"],
        per_device_eval_batch_size=16,
        report_to="none",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=DataCollatorWithPadding(tokenizer),
        **{_trainer_tokenizer_kw(): tokenizer},
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
            "example_input": d["input_text_1"].iloc[0],
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
        show_progress_bar=True,
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
        nn.Linear(256, 5),
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
                "features": "[u, v, |u-v|, u*v, onehot(class1), onehot(class2), same_class]",
            },
            os.path.join(ctx["weights_dir"], "hybrid_mlp.pt"),
        )

    extra = {
        "encoder": encoder_name,
        "feature_dim": int(Xtr.shape[1]),
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_val_f1_macro": float(best),
        "class_weights": list(map(float, cw)),
        "unique_texts": len(texts),
    }
    return predict(Xte), df_test["label_score"].values, df_test, save, extra, None


FAMILIES = {
    "xgboost": _run_xgboost,
    "cross": _run_cross,
    "bi": _run_bi,
    "multi": _run_multi,
    "probe": _run_probe,
    "hybrid": _run_hybrid,
}


def run_experiment(
    run_id: int, data_root: str | None = None, out_root: str = ".", smoke: bool = False
) -> dict:
    import numpy as np
    import torch

    from utils.evaluate import build_result_df
    from utils.metrics import binary_metrics, plot_confusion, stats

    spec = get_run(run_id)
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
        f"family={spec['family']} model={spec['model']} variant={spec['variant']} seed={SEED}{smoke_note}"
    )
    print("=" * 78)
    if smoke:
        data_root = _smoke_data(data_root, spec["variant"], os.path.join(scratch, "data"))
    print("Data:", data_root)

    ctx = {
        "spec": spec,
        "tag": tag,
        "smoke": smoke,
        "data_root": data_root,
        "results_dir": results_dir,
        "weights_dir": weights_dir,
        "scratch": scratch,
        "n_classes": n_classes,
        "generator": _setup_determinism(SEED),
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    env = _environment()
    print("Environment:", json.dumps(env))
    t0 = time.time()
    preds, true, df_test, save, extra, logits = FAMILIES[spec["family"]](ctx)

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
        "seed": SEED,
        **spec,
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
        "settings": extra,
    }
    with open(os.path.join(results_dir, f"{tag}_metrics.json"), "w") as fh:
        json.dump(record, fh, indent=2, default=float)

    if save is not None:
        try:
            save()
            print("Weights:", weights_dir)
        except Exception as e:
            print(f"Could not save the weights ({type(e).__name__}: {e}); the results are kept.")

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


def launch(
    run_ids: int | list[int], data_root: str | None = None, out_root: str = ".", smoke: bool = False
) -> None:
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

    pending, running, failed = list(ids), {}, []
    while pending or running:
        for slot in range(slots):
            if slot in running or not pending:
                continue
            run_id = pending.pop(0)
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
                out_root,
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
                bufsize=1,
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
    package([run_tag(run_id, smoke) for run_id in ids], out_root)
    if failed:
        raise RuntimeError(f"Runs failed: {failed}. The full output is in {log_dir}/")


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
                compress_type=zipfile.ZIP_STORED if heavy else zipfile.ZIP_DEFLATED,
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
