"""Run one experiment from config/runs.py, end to end.

    python runner.py --run-id 7 --data-root data --out-root .
    python runner.py --list

Every run uses one GPU, so all of them see the same hardware and the same optimisation
settings. The notebook can launch two runs side by side, one per GPU of a T4 x2 machine,
through launch(); each still runs alone on its own card.

Outputs, all named after the run tag  <ID>_<name>_seed<seed>:
    <out>/results/<tag>_preds.csv      per-example predictions, carrying Pair ID
    <out>/results/<tag>_metrics.json   every metric, the run spec and the environment
    <out>/results/<tag>_logits.npy     averaged logits (multi-task only)
    <out>/results/cm_<tag>.pdf/.png    confusion matrix
    <out>/weights/<tag>/               the trained model
    <out>/logs/<tag>.log               the full console output (written by launch)
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Lets cuBLAS run deterministically, which use_deterministic_algorithms asks for.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import glob
import json
import platform
import subprocess
import sys
import threading
import time

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from config.runs import RUNS, get_run, describe

LABEL_NAMES = ["Dissimilar", "Low similar", "Similar", "High similar", "Identical"]
DEFAULT_SEED = 42


# --------------------------------------------------------------------------- data

def find_data_root(hint=None):
    """The directory holding codraft/, plain/, category/ and expanded/."""
    def ok(d):
        return d and os.path.isfile(os.path.join(d, "codraft", "train.csv"))
    candidates = [hint, os.environ.get("CODRAFT_DATA"), os.path.join(REPO_DIR, "data")]
    for c in candidates:
        if ok(c):
            return c
    # Kaggle mounts datasets at different depths depending on how they were attached.
    for depth in ("*", "*/*", "*/*/*", "*/*/*/*"):
        for d in sorted(glob.glob(os.path.join("/kaggle/input", depth))):
            if ok(d):
                return d
    raise FileNotFoundError(
        f"no dataset found (tried {hint!r} and /kaggle/input). Expected a directory "
        f"containing codraft/train.csv."
    )


def find_variant_root(data_root, variant, extra=()):
    """The directory holding <variant>/, which is data_root for the four shipped variants.

    A variant built from another LLM's attributes (codraft_<llm>/, from scripts/llm.ipynb)
    may sit elsewhere: in one of the extra directories, or in any dataset under
    /kaggle/input (attach the LLM notebook's output as a dataset).
    """
    def ok(d):
        return d and os.path.isfile(os.path.join(d, variant, "test.csv"))
    for c in (data_root, *extra):
        if ok(c):
            return c
    for depth in ("*", "*/*", "*/*/*", "*/*/*/*"):
        for d in sorted(glob.glob(os.path.join("/kaggle/input", depth))):
            if ok(d):
                return d
    raise FileNotFoundError(f"variant {variant!r} not found in {data_root} or under /kaggle/input; "
                            f"expected a directory containing {variant}/test.csv")


def find_checkpoint(source, hint=None):
    """A trained multi-task checkpoint: the folder holding model.safetensors and config.json.

    Looks in weights/<source>/model locally, then anywhere under /kaggle/input (attach the
    run's model/ folder as a Kaggle dataset). Kaggle may flatten the folder name, so a
    multi-task checkpoint is recognised by its config rather than its path.
    """
    def ok(d):
        return d and os.path.isfile(os.path.join(d, "model.safetensors")) \
            and os.path.isfile(os.path.join(d, "config.json"))
    for c in (hint, os.environ.get("CODRAFT_CKPT"),
              os.path.join(REPO_DIR, "weights", source, "model"),
              os.path.join(REPO_DIR, "weights", source)):
        if ok(c):
            return c
    found = []
    if os.path.isdir("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if "model.safetensors" in files and "added_tokens.json" in files and "config.json" in files:
                try:
                    if json.load(open(os.path.join(root, "config.json"))).get("num_product_classes"):
                        found.append(root)
                except Exception:
                    pass
    named = [f for f in found if source in f]
    if len(named) == 1:
        return named[0]
    if len(found) == 1:
        return found[0]
    raise FileNotFoundError(
        f"checkpoint for {source} not found" + (f"; candidates: {found}" if found else "")
        + f". Attach weights/{source}/model as a Kaggle dataset, or set CODRAFT_CKPT.")


def _smoke_data(data_root, variant, tmp_root):
    """A tiny copy of one variant that still holds all five labels, for a quick check."""
    import pandas as pd
    dst = os.path.join(tmp_root, variant)
    os.makedirs(dst, exist_ok=True)
    per_class = {"train": 40, "val": 10, "test": 10}
    for split, k in per_class.items():
        d = pd.read_csv(os.path.join(data_root, variant, f"{split}.csv"), low_memory=False)
        d = d.groupby("Similarity", sort=False, group_keys=False).head(k)
        d.sort_index().to_csv(os.path.join(dst, f"{split}.csv"), index=False)
    return tmp_root


# ---------------------------------------------------------------------- one run

def _setup_determinism(seed):
    import torch
    from utils import set_seed
    set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _environment():
    import torch
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "n_gpu_visible": torch.cuda.device_count()}
    for pkg in ("transformers", "sentence_transformers", "xgboost", "sklearn", "datasets"):
        try:
            env[pkg] = __import__(pkg).__version__
        except Exception:
            env[pkg] = None
    return env


def run_experiment(run_id, data_root=None, out_root=".", seed=DEFAULT_SEED, smoke=False,
                   save_weights=True):
    import numpy as np
    import pandas as pd
    import torch

    from config import CONFIG_DATA, CONFIG_MODEL
    from preprocess import DataManager
    from utils import seed_worker, compute_class_weight, get_stats, build_result_df

    spec = get_run(run_id)
    tag = f"{run_id:02d}_{spec['name']}_seed{seed}"
    if smoke:
        tag = "SMOKE_" + tag
    data_root = find_variant_root(find_data_root(data_root), spec["variant"])
    results_dir = os.path.join(out_root, "results")
    weights_dir = os.path.join(out_root, "weights", tag)
    scratch = os.path.join("/tmp" if os.path.isdir("/tmp") else out_root, "codraft_ckpt", tag)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(scratch, exist_ok=True)
    if save_weights:
        os.makedirs(weights_dir, exist_ok=True)

    print("=" * 78)
    print(f"run {run_id}: {spec['name']}  [{spec['table']}] {spec['row']}")
    print(f"family={spec['family']} model={spec['model']} variant={spec['variant']} seed={seed}"
          + ("  SMOKE TEST" if smoke else ""))
    print("=" * 78)

    if smoke:
        data_root = _smoke_data(data_root, spec["variant"], os.path.join(scratch, "data"))
    print("data:", data_root)

    generator = _setup_determinism(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = _environment()
    print("env:", json.dumps(env))
    t0 = time.time()

    def data_manager(tokenizer, build_for):
        return DataManager(input_root=data_root, work_dir=scratch, config_data=CONFIG_DATA,
                           tokenizer=tokenizer, seed_worker=seed_worker,
                           data_generator=generator, random_seed=seed, rebalance=False,
                           variant=spec["variant"], build_for=build_for)

    extra = {}
    logits = None
    family = spec["family"]

    if family == "xgboost":
        from model import train_xgboost
        from utils import get_preds_ml
        dm = data_manager(None, "ml")
        X_train, y_train, X_val, y_val, X_test, y_test = dm.get_ml_data()
        _, _, df_test = dm.get_data()
        model, _ = train_xgboost(X_train, y_train, X_val, y_val, compute_class_weight(y_train))
        preds, true = get_preds_ml(model, X_test, y_test)
        # Through the booster: XGBClassifier.save_model reads _estimator_type, which
        # scikit-learn >= 1.6 no longer defines, and fails with older xgboost.
        save_fn = lambda: model.get_booster().save_model(os.path.join(weights_dir, "xgboost.json"))
        bi = getattr(model, "best_iteration", None)
        extra["best_iteration"] = None if bi is None else int(bi)

    elif family == "cross":
        from model import train_cross_encoder
        from model.models import get_model
        from utils import get_preds_cross_encoder
        dm = data_manager(None, "cross_encoder")
        train_dl, evaluator = dm.get_dataloaders(model_type="cross_encoder")
        df_train, _, df_test = dm.get_data()
        weights = torch.tensor(compute_class_weight(df_train["label_score"].values),
                               dtype=torch.float32, device=device)
        model, _ = get_model(model_type="cross_encoder", model_name=spec["model"],
                             num_classes=5, max_len=CONFIG_MODEL.MAX_LEN, weights_tensor=weights)
        epochs = 1 if smoke else CONFIG_MODEL.MODEL_CONFIG["cross_encoder"]["epochs"]
        model = train_cross_encoder(model, train_dl, evaluator,
                                    output_path=weights_dir if save_weights else scratch,
                                    epochs=epochs)
        preds, true = get_preds_cross_encoder(model=model, df_test=df_test)
        save_fn = None   # fit() already saved the best checkpoint into weights_dir
        extra.update(epochs=epochs, class_weights=weights.tolist(), batch_size=32,
                     best_val_accuracy=float(getattr(model, "best_score", float("nan"))))

    elif family == "bi":
        from transformers import AutoTokenizer
        from model import train_bi_encoder_baseline
        from utils import get_preds_siamese
        # The same (fast) tokenizer SentenceTransformer loads for prediction.
        tok = AutoTokenizer.from_pretrained(spec["model"])
        dm = data_manager(tok, "siamese")
        train_loader, val_loader = dm.get_dataloaders(model_type="siamese")
        df_train, _, df_test = dm.get_data()
        cw = compute_class_weight(df_train["label_score"].values)
        epochs = 1 if smoke else CONFIG_MODEL.MODEL_CONFIG["siamese"]["num_epochs_cls"]
        out = weights_dir if save_weights else scratch
        train_bi_encoder_baseline(spec["model"], train_loader, val_loader, cw, out, device,
                                  epochs=epochs)
        preds, true = get_preds_siamese(df_test, out, spec["model"], device)
        save_fn = None   # training saved the best epoch into weights_dir
        extra.update(epochs=epochs, class_weights=list(map(float, cw)),
                     batch_size=CONFIG_MODEL.MODEL_CONFIG["siamese"]["physical_batch_size"])

    elif family == "multi":
        import inspect
        from transformers import Trainer, DataCollatorWithPadding
        from model import get_tokenizer
        from model.models import get_model, get_training_args
        from utils import compute_metrics, get_preds_multi

        tokenizer = get_tokenizer(spec["model"], add_class_tokens=True)
        dm = data_manager(tokenizer, "multi_task")
        bad = [t for t in dm.class_to_token.values()
               if len(tokenizer.encode(t, add_special_tokens=False)) != 1]
        assert not bad, f"class markers split into subwords: {bad[:5]}"
        train_ds, val_ds, test_ds = dm.get_dataset()
        df_train, _, df_test = dm.get_data()
        assert len(test_ds) == 2 * len(df_test), "create_patterns must give two rows per pair"
        assert dm.NUM_PRODUCT_CLASSES == CONFIG_MODEL.NUM_PRODUCT_CLASSES

        cfg = CONFIG_MODEL.multi_task_args(seed=seed, output_dir=scratch, n_gpu=1)
        if smoke:
            cfg["training_args"]["num_train_epochs"] = 1
        training_args = get_training_args(**cfg["training_args"])

        loss = dict(cfg["loss_args"])
        loss["loss_type"] = spec.get("loss_type", "rank_aware")
        for k in ("aux_weight", "alpha"):
            if k in spec:
                loss[k] = spec[k]
        class_weights = (compute_class_weight(df_train["label_score"].values)
                         if loss["loss_type"] == "ce" else None)

        model, _ = get_model(model_type="multi_task", model_name=spec["model"],
                             num_classes=5, num_product_classes=dm.NUM_PRODUCT_CLASSES,
                             device=device, class_weights=class_weights, tokenizer=tokenizer,
                             **loss)
        assert model.get_input_embeddings().num_embeddings >= len(tokenizer)

        # transformers renamed Trainer(tokenizer=) to processing_class= in 4.46; the pinned
        # 4.45.2 only knows the old name.
        tok_kw = ("processing_class" if "processing_class" in
                  inspect.signature(Trainer.__init__).parameters else "tokenizer")
        trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                          eval_dataset=val_ds, data_collator=DataCollatorWithPadding(tokenizer),
                          compute_metrics=compute_metrics, **{tok_kw: tokenizer})
        trainer.train()
        preds, true, logits = get_preds_multi(trainer, test_ds, df_test, return_logits=True)
        np.save(os.path.join(results_dir, f"{tag}_logits.npy"), logits)
        def save_fn():
            trainer.save_model(weights_dir)
            tokenizer.save_pretrained(weights_dir)
        a = training_args
        extra.update(loss_args=loss, class_weights=None if class_weights is None
                     else list(map(float, class_weights)),
                     epochs=a.num_train_epochs, learning_rate=a.learning_rate,
                     per_device_batch=a.per_device_train_batch_size,
                     grad_accum=a.gradient_accumulation_steps,
                     effective_batch=a.per_device_train_batch_size * a.gradient_accumulation_steps
                     * max(1, a.n_gpu), fp16=a.fp16,
                     best_checkpoint=trainer.state.best_model_checkpoint,
                     best_val_f1_macro=trainer.state.best_metric,
                     vocab_size=len(tokenizer))
    elif family == "probe":
        import inspect
        from datasets import Dataset
        from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
        from model.models.MultiTask import JointClassSimBGE
        from preprocess.preprocess_data import preprocess
        from preprocess.data_loader import create_patterns, preprocess_dataset
        from utils import get_preds_multi

        ckpt = find_checkpoint(spec["source"])
        print("checkpoint:", ckpt)
        tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=False)
        model = JointClassSimBGE.from_pretrained(ckpt).to(device).eval()
        nice = sorted(CONFIG_DATA.NICE_CLASS_MAP)
        class_to_token = {c: f"[CLASS_{c}]" for c in nice}
        class_to_id = {c: i for i, c in enumerate(nice)}
        bad = [t for t in class_to_token.values() if len(tokenizer.encode(t, add_special_tokens=False)) != 1]
        assert not bad, f"checkpoint tokenizer lacks the class tokens: {bad[:5]}"

        raw = pd.read_csv(os.path.join(data_root, spec["variant"], "test.csv"), low_memory=False)
        base = preprocess(raw.copy())

        def build(term, nature, purpose, cls, use_n=True, use_p=True, use_c=True):
            # Same format as preprocess_data.create_structured_text_enhanced, with switches.
            parts = []
            if use_n and str(nature).strip():
                parts.append(f"Nature: {str(nature).strip()}")
            if use_p and str(purpose).strip():
                parts.append(f"Use: {str(purpose).strip()}")
            if use_c and CONFIG_DATA.NICE_CLASS_MAP.get(int(cls), ""):
                parts.append(f"Category: {CONFIG_DATA.NICE_CLASS_MAP[int(cls)]}")
            t = str(term).strip()
            return f"{t} [ {' | '.join(parts)} ]" if parts else t

        def frame(n1, p1, n2, p2, **kw):
            d = base.copy()
            d["input_text_1"] = [build(t, n, p, c, **kw) for t, n, p, c in zip(d["Term 1"], n1, p1, d["Class 1"])]
            d["input_text_2"] = [build(t, n, p, c, **kw) for t, n, p, c in zip(d["Term 2"], n2, p2, d["Class 2"])]
            return d

        N1, P1, N2, P2 = (raw[c].fillna("").astype(str).values for c in ("Nature 1", "Purpose 1", "Nature 2", "Purpose 2"))
        full = frame(N1, P1, N2, P2)
        assert (full["input_text_1"] == base["input_text_1"]).all() and \
            (full["input_text_2"] == base["input_text_2"]).all(), "probe input builder drifted from preprocess()"
        # shuffled: every side gets the (Nature, Purpose) of a random other side
        rng = np.random.default_rng(seed)
        pool = np.array(list(zip(np.r_[N1, N2], np.r_[P1, P2])), dtype=object)
        perm = pool[rng.permutation(len(pool))]
        n = len(raw)
        conds = {
            "full": full,
            "no_nature": frame([""] * n, P1, [""] * n, P2),
            "no_purpose": frame(N1, [""] * n, N2, [""] * n),
            "heading_only": frame([""] * n, [""] * n, [""] * n, [""] * n),
            "no_heading": frame(N1, P1, N2, P2, use_c=False),
            "bare_term": frame([""] * n, [""] * n, [""] * n, [""] * n, use_c=False),
            "shuffled_attributes": frame(perm[:n, 0], perm[:n, 1], perm[n:, 0], perm[n:, 1]),
        }

        tok_kw = ("processing_class" if "processing_class" in
                  inspect.signature(Trainer.__init__).parameters else "tokenizer")
        args = TrainingArguments(output_dir=scratch, per_device_eval_batch_size=16, report_to="none",
                                 fp16=torch.cuda.is_available(), remove_unused_columns=False,
                                 dataloader_num_workers=2)
        trainer = Trainer(model=model, args=args, data_collator=DataCollatorWithPadding(tokenizer),
                          **{tok_kw: tokenizer})
        cols = ["input_ids", "attention_mask", "labels", "aux_labels"]
        results = {}
        for name, d in conds.items():
            aug = create_patterns(d, tokenizer, class_to_token, class_to_id)
            ds = Dataset.from_pandas(aug).map(preprocess_dataset, batched=True,
                                              fn_kwargs={"tokenizer": tokenizer},
                                              remove_columns=aug.columns.tolist())
            ds.set_format(type="torch", columns=cols)
            p, t, lg = get_preds_multi(trainer, ds, d, return_logits=True)
            build_result_df(d, t, p).to_csv(os.path.join(results_dir, f"{tag}_{name}_preds.csv"), index=False)
            np.save(os.path.join(results_dir, f"{tag}_{name}_logits.npy"), lg)
            mc = get_stats(build_result_df(d, t, p), fig_prefix=os.path.join(results_dir, f"cm_{tag}_{name}"),
                           return_metrics=True)
            results[name] = {"f1_macro": mc["f1_macro"], "qwk": float(mc["qwk"]), "mae": float(mc["mae"]),
                             "adjacent_acc": mc["adjacent_acc"], "severe_rate": mc["severe_rate"],
                             "per_class_f1": [r["f1"] for r in mc["per_class"]],
                             "example_input": d["input_text_1"].iloc[0]}
            print(f"[probe] {name:20s} Macro-F1 {mc['f1_macro']*100:.2f} | QWK {mc['qwk']:.4f} | MAE {mc['mae']:.4f}")
            if name == "full":
                preds, true, df_test = p, t, d
        for name in results:
            results[name]["delta_f1_vs_full"] = results[name]["f1_macro"] - results["full"]["f1_macro"]
        save_fn = None   # inference only; no new weights
        extra.update(checkpoint=ckpt, conditions=results)

    elif family == "hybrid":
        import torch.nn as nn
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics import f1_score as _f1

        dm = data_manager(None, "ml")
        df_train, df_val, df_test = dm.get_data()
        # CODRAFT_HYBRID_ENCODER swaps in a small encoder for local testing only.
        enc_name = os.environ.get("CODRAFT_HYBRID_ENCODER", spec["model"])
        enc = SentenceTransformer(enc_name, device=str(device), trust_remote_code=True)
        texts = sorted(set(pd.concat([d[c] for d in (df_train, df_val, df_test)
                                      for c in ("input_text_1", "input_text_2")]).astype(str)))
        emb = enc.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=True)
        row = {t: i for i, t in enumerate(texts)}
        del enc
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        def feats(d):
            u = emb[[row[str(t)] for t in d["input_text_1"]]]
            v = emb[[row[str(t)] for t in d["input_text_2"]]]
            c1 = np.eye(45, dtype=np.float32)[d["Class 1"].values - 1]
            c2 = np.eye(45, dtype=np.float32)[d["Class 2"].values - 1]
            same = (d["Class 1"].values == d["Class 2"].values).astype(np.float32)[:, None]
            return np.hstack([u, v, np.abs(u - v), u * v, c1, c2, same]).astype(np.float32)

        Xtr, Xva, Xte = (torch.tensor(feats(d)) for d in (df_train, df_val, df_test))
        ytr = torch.tensor(df_train["label_score"].values)
        cw = compute_class_weight(df_train["label_score"].values)
        net = nn.Sequential(nn.Linear(Xtr.shape[1], 1024), nn.ReLU(), nn.Dropout(0.1),
                            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.1),
                            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
                            nn.Linear(256, 5)).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device))
        max_epochs, patience = (2, 2) if smoke else (50, 8)
        best, best_state, best_epoch, wait = -1.0, None, 0, 0

        def predict(X):
            net.eval()
            with torch.no_grad():
                return torch.cat([net(X[i:i + 1024].to(device)).argmax(1).cpu()
                                  for i in range(0, len(X), 1024)]).numpy()
        for epoch in range(1, max_epochs + 1):
            net.train()
            order = torch.randperm(len(Xtr), generator=generator)
            for i in range(0, len(order), 256):
                b = order[i:i + 256]
                opt.zero_grad()
                loss = crit(net(Xtr[b].to(device)), ytr[b].to(device))
                loss.backward()
                opt.step()
            vf = _f1(df_val["label_score"].values, predict(Xva), average="macro")
            print(f"epoch {epoch} | loss {loss.item():.4f} | val macro-F1 {vf:.4f}")
            if vf > best:
                best, best_epoch, wait = vf, epoch, 0
                best_state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        net.load_state_dict(best_state)
        preds, true = predict(Xte), df_test["label_score"].values
        def save_fn():
            torch.save({"state_dict": best_state, "encoder": enc_name, "in_dim": int(Xtr.shape[1]),
                        "features": "[u, v, |u-v|, u*v, onehot(class1), onehot(class2), same_class]"},
                       os.path.join(weights_dir, "hybrid_mlp.pt"))
        extra.update(encoder=enc_name, feature_dim=int(Xtr.shape[1]), best_epoch=best_epoch,
                     epochs_run=epoch, best_val_f1_macro=float(best), class_weights=list(map(float, cw)),
                     unique_texts=len(texts))
    else:
        raise ValueError(f"unknown family {family!r}")

    # ------------------------------------------------------------- evaluation
    true = np.asarray(true).astype(int)
    preds = np.asarray(preds).astype(int)
    result_df = build_result_df(df_test, true, preds)
    result_df.to_csv(os.path.join(results_dir, f"{tag}_preds.csv"), index=False)
    m = get_stats(result_df, fig_prefix=os.path.join(results_dir, f"cm_{tag}"),
                  return_metrics=True)

    record = {
        "run_id": run_id, "tag": tag, "smoke": smoke, "seed": seed, **spec,
        "n_test": int(len(df_test)),
        "f1_macro": m["f1_macro"], "qwk": float(m["qwk"]), "mae": float(m["mae"]),
        "accuracy": m["accuracy"], "precision_macro": m["precision_macro"],
        "recall_macro": m["recall_macro"], "adjacent_acc": m["adjacent_acc"],
        "severe_rate": m["severe_rate"],
        "per_class": [{**r, "name": LABEL_NAMES[r["class"]]} for r in m["per_class"]],
        "confusion_matrix": m["confusion_matrix"].tolist(),
        "train_seconds": round(time.time() - t0, 1),
        "environment": env, "settings": extra,
    }
    with open(os.path.join(results_dir, f"{tag}_metrics.json"), "w") as fh:
        json.dump(record, fh, indent=2, default=float)

    # Weights last, once the results are safely written: a failure here must not cost
    # the numbers of a run that has already finished.
    if save_weights and save_fn is not None:
        try:
            save_fn()
            print("weights:", weights_dir)
        except Exception as e:
            print(f"!! could not save weights ({type(e).__name__}: {e}); results are kept")

    print("\n" + "=" * 78)
    print(f"DONE run {run_id} {spec['name']}: Macro-F1 {m['f1_macro']*100:.2f} | "
          f"QWK {m['qwk']:.4f} | MAE {m['mae']:.4f} | n={len(df_test)} | "
          f"{record['train_seconds']/60:.1f} min")
    print("per-class F1: " + " / ".join(f"{r['f1']*100:.2f}" for r in m["per_class"]))
    print("=" * 78)
    return record


# ------------------------------------------------------------------- launching

def _gpu_count():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        return len([l for l in out.stdout.splitlines() if l.startswith("GPU ")])
    except Exception:
        return 0


_PRINT_LOCK = threading.Lock()


def _say(text):
    # Two runs stream at once; without the lock their lines interleave mid-line.
    with _PRINT_LOCK:
        print(text, flush=True)


def _stream(proc, prefix, log_path, progress_every=60):
    """Echo a child's output with a prefix. Progress-bar redraws (\\r) are throttled so the
    notebook output stays readable; the log file keeps everything."""
    last = 0.0
    buf = ""
    with open(log_path, "w", buffering=1) as log:
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            log.write(ch)
            if ch in "\r\n":
                line, buf = buf.strip(), ""
                if not line:
                    continue
                if ch == "\n" or time.time() - last > progress_every:
                    _say(f"{prefix} {line}")
                    if ch == "\r":
                        last = time.time()
            else:
                buf += ch
        if buf.strip():
            _say(f"{prefix} {buf.strip()}")


def launch(run_ids, data_root=None, out_root=".", seed=DEFAULT_SEED, smoke=False,
           save_weights=True):
    """Run one ID, or several side by side with one GPU each.

    With more runs than GPUs the extra ones wait for a free card. Raises at the end if any
    run failed, so a Kaggle "Save & Run All" shows the failure.
    """
    ids = [run_ids] if isinstance(run_ids, int) else list(run_ids)
    for i in ids:
        get_run(i)
    data_root = find_data_root(data_root)
    n_gpu = _gpu_count()
    slots = max(1, n_gpu)
    log_dir = os.path.join(out_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"runs {ids} | GPUs found: {n_gpu} | data: {data_root} | out: {out_root}")
    if len(ids) > slots:
        print(f"more runs than GPUs: they will share {slots} card(s) in turn")

    pending = list(ids)
    running = {}          # gpu slot -> (run id, proc, thread)
    failed = []
    while pending or running:
        for slot in range(slots):
            if slot not in running and pending:
                rid = pending.pop(0)
                env = dict(os.environ)
                if n_gpu:
                    env["CUDA_VISIBLE_DEVICES"] = str(slot)
                env["PYTHONUNBUFFERED"] = "1"
                cmd = [sys.executable, "-u", os.path.join(REPO_DIR, "runner.py"),
                       "--run-id", str(rid), "--data-root", data_root,
                       "--out-root", out_root, "--seed", str(seed)]
                if smoke:
                    cmd.append("--smoke")
                if not save_weights:
                    cmd.append("--no-weights")
                tag = ("SMOKE_" if smoke else "") + f"{rid:02d}_{RUNS[rid]['name']}_seed{seed}"
                proc = subprocess.Popen(cmd, cwd=REPO_DIR, env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
                th = threading.Thread(target=_stream, daemon=True,
                                      args=(proc, f"[{rid:02d}|{'gpu' + str(slot) if n_gpu else 'cpu'}]",
                                            os.path.join(log_dir, f"{tag}.log")))
                th.start()
                running[slot] = (rid, proc, th)
                print(f"started run {rid} on gpu{slot}" if n_gpu else f"started run {rid} on CPU")
        for slot, (rid, proc, th) in list(running.items()):
            if proc.poll() is not None:
                th.join()
                if proc.returncode != 0:
                    failed.append(rid)
                    print(f"!! run {rid} FAILED (exit {proc.returncode}); see logs/")
                else:
                    print(f"run {rid} finished")
                del running[slot]
        time.sleep(2)

    summary(out_root, smoke=smoke)
    tags = [("SMOKE_" if smoke else "") + f"{rid:02d}_{RUNS[rid]['name']}_seed{seed}" for rid in ids]
    package(tags, out_root, include_weights=save_weights)
    if failed:
        raise RuntimeError(f"runs failed: {failed}. Full output is in {log_dir}/")


def package(tags, out_root=".", include_weights=True):
    """Put everything a session produced into one zip, so a Kaggle run is one download.

    Holds results/, logs/ and weights/ for the given tags. Weights are stored rather than
    compressed (safetensors do not shrink and deflating 2 GB is slow), and once the zip is
    verified the loose weight folders are removed so /kaggle/working is not holding every
    checkpoint twice. Runs that did not finish are skipped; their logs are still included.
    """
    import zipfile
    files = []
    for tag in tags:
        files += glob.glob(os.path.join(out_root, "results", f"{tag}_*"))
        files += glob.glob(os.path.join(out_root, "results", f"cm_{tag}.*"))
        files += glob.glob(os.path.join(out_root, "logs", f"{tag}.log"))
        if include_weights:
            for root, _, fs in os.walk(os.path.join(out_root, "weights", tag)):
                files += [os.path.join(root, f) for f in fs]
    if not files:
        print("nothing to package")
        return None

    ids = "-".join(t.split("_")[1 if t.startswith("SMOKE_") else 0] for t in tags)
    name = ("SMOKE_" if tags[0].startswith("SMOKE_") else "") + f"codraft_runs_{ids}.zip"
    path = os.path.join(out_root, name)
    with zipfile.ZipFile(path, "w", allowZip64=True) as zf:
        for f in sorted(set(files)):
            heavy = os.sep + "weights" + os.sep in f
            zf.write(f, os.path.relpath(f, out_root),
                     compress_type=zipfile.ZIP_STORED if heavy else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        n = len(zf.namelist())
    if bad is not None:
        print(f"!! zip check failed on {bad}; loose files kept")
        return path

    if include_weights:
        import shutil
        for tag in tags:
            shutil.rmtree(os.path.join(out_root, "weights", tag), ignore_errors=True)
    print(f"\npackaged {n} files -> {path} ({os.path.getsize(path) / 2**20:.1f} MB)")
    print("download this one file from the Output panel")
    return path


def summary(out_root=".", smoke=False):
    """One line per finished run, read back from the metrics files."""
    files = sorted(glob.glob(os.path.join(out_root, "results", "*_metrics.json")))
    files = [f for f in files if os.path.basename(f).startswith("SMOKE_") == smoke]
    if not files:
        print("no finished runs yet")
        return
    print(f"\n{'run':<38}{'MacroF1':>8}{'QWK':>8}{'MAE':>8}{'adj':>8}{'n':>6}{'min':>7}")
    for f in files:
        r = json.load(open(f))
        print(f"{r['tag']:<38}{r['f1_macro']*100:8.2f}{r['qwk']:8.4f}{r['mae']:8.4f}"
              f"{r['adjacent_acc']*100:7.2f}%{r['n_test']:6d}{r['train_seconds']/60:7.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", type=int)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-root", default=".")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subset, one epoch: checks the pipeline in minutes")
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or a.run_id is None:
        print(describe())
        return
    run_experiment(a.run_id, data_root=a.data_root, out_root=a.out_root, seed=a.seed,
                   smoke=a.smoke, save_weights=not a.no_weights)


if __name__ == "__main__":
    main()
