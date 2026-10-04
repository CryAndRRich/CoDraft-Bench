"""Run an LLM from config/llms.py on CoDraft: enrichment, classification, or both.

    python llm_runner.py --llm qwen2.5-7b --task enrich              # all 7,724 terms
    python llm_runner.py --llm qwen2.5-7b --task enrich --scope test # the 2,928 test terms
    python llm_runner.py --llm qwen2.5-7b --task classify            # 0- and 10-shot, plain and codraft
    python llm_runner.py --llm qwen2.5-7b --task enrich classify --smoke
    python llm_runner.py --check      # the variant builder and the metrics, against the shipped data
    python llm_runner.py --list

A vLLM model is served once per call, on both GPUs, and stopped at the end. This runs in
its own environment (scripts/llm.ipynb installs vLLM); it never imports the training code.

Outputs under <out>/, all named after the LLM tag:
    attributes/<tag>_attributes.csv    one row per (term, class), term_attributes.csv columns + Class
    attributes/<tag>_raw.jsonl         every answer as it arrived (resumes an interrupted run)
    attributes/<tag>_enrich.json       model revision, versions, prompt, coverage, tokens, time
    codraft_<tag>/{train,val,test}.csv the data variant, row-aligned with data/codraft/
    results/llm_<tag>_<k>shot_<variant>_preds.csv / _metrics.json / cm_*.pdf/.png
    logs/<tag>_<task>.log, logs/vllm_<tag>.log
and one zip of all of it, codraft_llm_<tag>_<tasks>.zip.
"""
import argparse
import glob
import json
import os
import sys
import time
import zipfile

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from config.llms import get_llm, describe
from runner import find_data_root, find_variant_root


class _Tee:
    """Copy everything printed to a log file as well."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fh = open(self.path, "a", buffering=1)
        self.stdout = sys.stdout
        sys.stdout = self
        return self

    def write(self, s):
        self.stdout.write(s)
        self.fh.write(s)

    def flush(self):
        self.stdout.flush()
        self.fh.flush()

    def __exit__(self, *exc):
        sys.stdout = self.stdout
        self.fh.close()


def _server(spec, out_root, port):
    """A context that serves the model (vLLM), or does nothing (hosted API)."""
    import contextlib
    from codraft_enrichment.llm import VLLMServer
    if spec["backend"] != "vllm":
        return contextlib.nullcontext(None)
    return VLLMServer(spec, port=port, log_path=os.path.join(out_root, "logs", f"vllm_{spec['tag']}.log"))


def enrich_task(client, spec, meta, data_root, out_root, scope, smoke, batch_size, workers):
    from codraft_enrichment.enrich import term_table, enrich
    from codraft_enrichment.variant import build_variant
    tag = ("SMOKE_" if smoke else "") + spec["tag"]
    table = term_table(data_root, scope, gemini_cache=os.path.join(data_root, "term_attributes.csv"))
    if smoke:
        table = table.groupby("Class", sort=False).head(1).reset_index(drop=True)   # one term per class
    print(f"enrichment: {len(table)} (term, class) pairs, scope={scope}")
    attr_dir = os.path.join(out_root, "attributes")
    attrs, st = enrich(client, table, attr_dir, tag, batch_size=batch_size, workers=workers)
    path = os.path.join(attr_dir, f"{tag}_attributes.csv")
    attrs.drop(columns=["key", "returned_term"]).to_csv(path, index=False)
    print(f"attributes: {len(attrs)}/{len(table)} terms -> {path}; failed: {st['n_failed']}; "
          f"by mode: {st['by_mode']}; generic natures: {st['generic_nature']}")
    written = {}
    # next to the shipped variants locally; on Kaggle the dataset is read-only, so in the output
    variant_root = out_root if os.path.abspath(data_root).startswith("/kaggle/input") else data_root
    if not smoke:
        try:
            written = build_variant(data_root, path, f"codraft_{spec['tag']}", out_root=variant_root)
        except ValueError as e:
            print(f"!! no variant written: {e}")
    record = {**meta, "task": "enrich", "tag": tag, "scope": scope, "smoke": smoke,
              "attributes": os.path.relpath(path, out_root), "variant_splits": written, **st}
    with open(os.path.join(attr_dir, f"{tag}_enrich.json"), "w") as fh:
        json.dump(record, fh, indent=2, default=str)
    files = [path, os.path.join(attr_dir, f"{tag}_raw.jsonl"), os.path.join(attr_dir, f"{tag}_enrich.json")]
    files += glob.glob(os.path.join(variant_root, f"codraft_{spec['tag']}", "*.csv")) if written else []
    return files


def classify_task(client, spec, meta, data_root, out_root, variants, shots, smoke, workers):
    from codraft_enrichment.classify import load_pairs, demonstrations, classify
    from codraft_enrichment.metrics import stats, plot_confusion
    import pandas as pd
    res = os.path.join(out_root, "results")
    os.makedirs(res, exist_ok=True)
    files = []
    for variant in variants:
        root = find_variant_root(data_root, variant, extra=(out_root,))
        test = load_pairs(root, variant, "test")
        if smoke:
            test = test.groupby("label_score", sort=False).head(10).reset_index(drop=True)
        has_train = os.path.isfile(os.path.join(root, variant, "train.csv"))
        train = load_pairs(root, variant, "train") if has_train and any(shots) else None
        for k in shots:
            if k and train is None:
                # a test-only variant (enrich --scope test) has no demonstrations to draw
                print(f"!! skipping {k}-shot on {variant}: it has no train split "
                      f"(enrich with --scope all to get one)")
                continue
            tag = ("SMOKE_" if smoke else "") + f"llm_{spec['tag']}_{k}shot_{variant}"
            print("=" * 78 + f"\n{tag}: {len(test)} pairs, {k} demonstrations\n" + "=" * 78)
            demos = demonstrations(train, k=k) if k else None
            preds, st = classify(client, test, os.path.join(res, f"{tag}_raw.jsonl"),
                                 demos=demos, workers=workers)
            ok = [p is not None for p in preds]
            if not all(ok):
                print(f"!! {ok.count(False)} pairs got no answer; scored as Dissimilar (label 0)")
            y = test["label_score"].astype(int).tolist()
            p = [q if q is not None else 0 for q in preds]
            m = stats(y, p)
            pd.DataFrame({"Pair ID": test["Pair ID"], "label": y, "pred": p,
                          "Class 1": test["Class 1"], "Class 2": test["Class 2"]}) \
                .to_csv(os.path.join(res, f"{tag}_preds.csv"), index=False)
            plot_confusion(m["confusion_matrix"], os.path.join(res, f"cm_{tag}"))
            record = {**meta, "task": "classify", "tag": tag, "smoke": smoke, "family": "llm",
                      "variant": variant, "shots": k, "n_test": len(test),
                      **{key: m[key] for key in ("f1_macro", "qwk", "mae", "accuracy",
                                                 "precision_macro", "recall_macro",
                                                 "adjacent_acc", "severe_rate", "per_class",
                                                 "confusion_matrix")},
                      "settings": st}
            with open(os.path.join(res, f"{tag}_metrics.json"), "w") as fh:
                json.dump(record, fh, indent=2, default=str)
            print(f"DONE {tag}: Macro-F1 {m['f1_macro']*100:.2f} | QWK {m['qwk']:.4f} | "
                  f"MAE {m['mae']:.4f} | n={len(test)} | {st['seconds']/60:.1f} min")
            print("per-class F1: " + " / ".join(f"{r['f1']*100:.2f}" for r in m["per_class"]))
            files += glob.glob(os.path.join(res, f"{tag}_*")) + glob.glob(os.path.join(res, f"cm_{tag}.*"))
    return files


def run_llm(llm, tasks=("enrich",), data_root=None, out_root=".", scope="all",
            variants=("plain", "codraft"), shots=(0, 10), smoke=False, batch_size=None,
            workers=None, port=8000, seed=0):
    from codraft_enrichment.llm import make_client, served_revision, package_versions
    tasks = [tasks] if isinstance(tasks, str) else list(tasks)
    bad = set(tasks) - {"enrich", "classify"}
    if bad:
        raise ValueError(f"unknown task(s) {sorted(bad)}; use enrich and/or classify")
    spec = get_llm(llm)
    data_root = find_data_root(data_root)
    batch_size = batch_size or spec["batch_size"]
    workers = workers or spec["workers"]
    name = ("SMOKE_" if smoke else "") + spec["tag"]
    log = os.path.join(out_root, "logs", f"{name}_{'-'.join(tasks)}.log")
    files = [log]
    with _Tee(log):
        print(f"LLM {spec['tag']} = {spec['model']} ({spec['backend']}) | tasks {tasks} | "
              f"data {data_root} | out {out_root}" + ("  SMOKE TEST" if smoke else ""))
        meta = {"llm": spec["tag"], "model": spec["model"], "serving": spec,
                "revision": served_revision(spec), "environment": package_versions(),
                "seed": seed, "temperature": 0, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        print("revision:", meta["revision"], "| env:", json.dumps(meta["environment"]))
        with _server(spec, out_root, port) as server:
            client = make_client(spec, server.base_url if server else None, seed=seed)
            if "enrich" in tasks:
                files += enrich_task(client, spec, meta, data_root, out_root, scope, smoke,
                                     batch_size, workers)
            if "classify" in tasks:
                # short answers: a local server takes many at once, a hosted API its own limit
                files += classify_task(client, spec, meta, data_root, out_root, variants, shots,
                                       smoke, max(workers, 32) if spec["backend"] == "vllm" else workers)
        if spec["backend"] == "vllm":
            files.append(os.path.join(out_root, "logs", f"vllm_{spec['tag']}.log"))
    return package(files, out_root, f"{'SMOKE_' if smoke else ''}codraft_llm_{spec['tag']}_{'-'.join(tasks)}.zip")


def package(files, out_root, name):
    path = os.path.join(out_root, name)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for f in sorted(set(files)):
            if os.path.isfile(f):
                zf.write(f, os.path.relpath(f, out_root))
    with zipfile.ZipFile(path) as zf:
        bad, n = zf.testzip(), len(zf.namelist())
    if bad is not None:
        print(f"!! zip check failed on {bad}")
    print(f"\npackaged {n} files -> {path} ({os.path.getsize(path) / 2**20:.1f} MB)")
    print("download this one file from the Output panel")
    return path


def check(data_root=None, tmp_dir=None):
    """Offline checks, no LLM: (1) the variant builder rebuilds data/codraft/ from the Gemini
    cache; (2) metrics.stats reproduces every saved metrics file in weights/."""
    import tempfile
    import pandas as pd
    from codraft_enrichment.variant import check_rule
    from codraft_enrichment.metrics import stats
    data_root = find_data_root(data_root)
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
        check_rule(data_root, tmp)
    n = 0
    for f in sorted(glob.glob(os.path.join(REPO_DIR, "weights", "*", "*_metrics.json"))):
        r = json.load(open(f))
        p = pd.read_csv(f.replace("_metrics.json", "_preds.csv"))
        m = stats(p["label"], p["pred"])
        for k in ("f1_macro", "qwk", "mae", "accuracy", "precision_macro", "recall_macro",
                  "adjacent_acc", "severe_rate"):
            assert abs(m[k] - r[k]) < 1e-9, (r["tag"], k, m[k], r[k])
        assert m["confusion_matrix"] == r["confusion_matrix"], r["tag"]
        assert all(abs(a["f1"] - b["f1"]) < 1e-9 for a, b in zip(m["per_class"], r["per_class"]))
        n += 1
    print(f"check: metrics.stats matches the saved metrics of {n} runs in weights/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm")
    ap.add_argument("--task", nargs="+", default=["enrich"], choices=["enrich", "classify"])
    ap.add_argument("--scope", default="all", choices=["all", "test"])
    ap.add_argument("--variants", nargs="+", default=["plain", "codraft"])
    ap.add_argument("--shots", nargs="+", type=int, default=[0, 10])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-root", default=".")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--smoke", action="store_true", help="45 terms / 50 pairs: checks the pipeline")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check(a.data_root)
    if a.list or not a.llm:
        print(describe())
        return
    run_llm(a.llm, a.task, data_root=a.data_root, out_root=a.out_root, scope=a.scope,
            variants=a.variants, shots=a.shots, smoke=a.smoke, batch_size=a.batch_size,
            workers=a.workers, port=a.port)


if __name__ == "__main__":
    main()
