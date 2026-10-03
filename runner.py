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
    data_root = find_data_root(data_root)
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
