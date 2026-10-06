import argparse
import contextlib
import glob
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import zipfile

import pandas as pd

from config.llms import describe, get_llm
from runner import REPO_DIR, find_data_root, find_variant_root

SMOKE_PER_LABEL = 10


def _smoke_pairs(test: pd.DataFrame, label_col: str) -> pd.DataFrame:
    return test.groupby(label_col, sort=False).head(SMOKE_PER_LABEL).reset_index(drop=True)


class _Tee:
    def __init__(self, path: str) -> None:
        self.path = path
        self.fh = None
        self.stdout = None

    def __enter__(self) -> "_Tee":
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fh = open(self.path, "a", buffering=1)
        self.stdout = sys.stdout
        sys.stdout = self
        return self

    def write(self, s: str) -> None:
        self.stdout.write(s)
        self.fh.write(s)

    def flush(self) -> None:
        self.stdout.flush()
        self.fh.flush()

    def __exit__(self, *exc: object) -> None:
        sys.stdout = self.stdout
        self.fh.close()


def _rescue(spec: dict) -> dict | None:
    if spec["backend"] == "vllm" and spec["rescue_penalty"]:
        return {"repetition_penalty": spec["rescue_penalty"]}
    return None


def _resume_from(path: str, spec: dict, smoke: bool, out_root: str) -> None:
    prefix = "SMOKE_" if smoke else ""
    enrich_file = f"{prefix}{spec['tag']}_raw.jsonl"
    found = {}
    for root, _, files in os.walk(path):
        for f in files:
            if f == enrich_file or (
                f.startswith(f"{prefix}llm_{spec['tag']}_") and f.endswith("_raw.jsonl")
            ):
                found.setdefault(f, []).append(os.path.join(root, f))
    if not found:
        raise FileNotFoundError(f"--resume-from {path}: no *_raw.jsonl of {spec['tag']} in it")
    for f, srcs in sorted(found.items()):
        if len(srcs) > 1:
            raise ValueError(f"--resume-from {path}: {f} found {len(srcs)} times: {srcs}")
        dst = os.path.join(out_root, "attributes" if f == enrich_file else "results", f)
        if os.path.exists(dst):
            print(f"Resume: keeping {dst} (already here)")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(srcs[0], dst)
        with open(dst) as fh:
            print(f"Resume: {sum(1 for _ in fh)} answers from {srcs[0]}")


def _server(spec: dict, out_root: str, port: int) -> contextlib.AbstractContextManager:
    from codraft_enrichment.llm import VLLMServer

    if spec["backend"] != "vllm":
        return contextlib.nullcontext(None)
    return VLLMServer(
        spec, port=port, log_path=os.path.join(out_root, "logs", f"vllm_{spec['tag']}.log")
    )


def enrich_task(
    client: object,
    spec: dict,
    meta: dict,
    data_root: str,
    out_root: str,
    scope: str,
    smoke: bool,
    batch_size: int,
    workers: int,
) -> list[str]:
    from codraft_enrichment.enrich import enrich, norm, term_table
    from codraft_enrichment.variant import build_variant

    tag = ("SMOKE_" if smoke else "") + spec["tag"]
    table = term_table(
        data_root, scope, gemini_cache=os.path.join(data_root, "term_attributes.csv")
    )
    smoke_ids = None
    if smoke:
        test = pd.read_csv(os.path.join(data_root, "codraft", "test.csv"), low_memory=False)
        pick = _smoke_pairs(test, "Similarity")
        smoke_ids = list(pick["Pair ID"])
        keys = {
            (norm(t), int(c))
            for side in "12"
            for t, c in zip(pick[f"Term {side}"], pick[f"Class {side}"])
        }
        table = table[[(k, c) in keys for k, c in zip(table["key"], table["Class"])]].reset_index(
            drop=True
        )
    print(f"Enrichment: {len(table)} (term, class) pairs, scope={scope}")
    in_data = not smoke and not os.path.abspath(data_root).startswith("/kaggle/input")
    work_dir = os.path.join(out_root, "attributes")
    attr_dir = os.path.join(data_root, "attributes") if in_data else work_dir
    variant_root = data_root if in_data else out_root
    attrs, st = enrich(
        client,
        table,
        work_dir,
        tag,
        batch_size=batch_size,
        workers=workers,
        rescue_sampling=_rescue(spec),
    )
    os.makedirs(attr_dir, exist_ok=True)
    path = os.path.join(attr_dir, f"{tag}_attributes.csv")
    attrs.drop(columns=["key", "returned_term"]).to_csv(path, index=False)
    print(
        f"Attributes: {len(attrs)}/{len(table)} terms -> {path}; failed: {st['n_failed']}; "
        f"by mode: {st['by_mode']}; generic natures: {st['generic_nature']}"
    )
    written = {}
    vname = ("SMOKE_" if smoke else "") + f"codraft_{spec['tag']}"
    try:
        written = build_variant(
            data_root,
            path,
            vname,
            out_root=variant_root,
            splits=("test",) if smoke else None,
            pair_ids=smoke_ids,
        )
    except ValueError as e:
        print(f"No variant written: {e}")
    record = {
        **meta,
        "task": "enrich",
        "tag": tag,
        "scope": scope,
        "smoke": smoke,
        "attributes": os.path.relpath(path, out_root),
        "variant_splits": written,
        **st,
    }
    meta_path = os.path.join(attr_dir, f"{tag}_enrich.json")
    with open(meta_path, "w") as fh:
        json.dump(record, fh, indent=2, default=str)
    files = [path, os.path.join(work_dir, f"{tag}_raw.jsonl"), meta_path]
    if written:
        files += glob.glob(os.path.join(variant_root, vname, "*.csv"))
    return files


def classify_task(
    client: object,
    spec: dict,
    meta: dict,
    data_root: str,
    out_root: str,
    variants: list[str],
    shots: list[int],
    smoke: bool,
    workers: int,
) -> list[str]:
    from codraft_enrichment.classify import classify, demonstrations, load_pairs
    from utils.metrics import plot_confusion, stats

    res = os.path.join(out_root, "results")
    os.makedirs(res, exist_ok=True)
    prefix = "SMOKE_" if smoke else ""
    files = []
    for name in variants:
        variant = f"{prefix}codraft_{spec['tag']}" if name == "own" else name
        try:
            root = find_variant_root(data_root, variant, extra=(out_root,))
        except FileNotFoundError as e:
            if name != "own":
                raise
            raise FileNotFoundError(
                f'{variant}/ not found. Run TASK = ["enrich", "classify"] so the attributes are '
                f"made first, or attach a run that made them."
            ) from e
        test = load_pairs(root, variant, "test")
        if smoke:
            test = _smoke_pairs(test, "label_score")
        has_train = os.path.isfile(os.path.join(root, variant, "train.csv"))
        train = load_pairs(root, variant, "train") if has_train and any(shots) else None
        for k in shots:
            if k and train is None:
                why = (
                    "the smoke test enriches its 50 test pairs only"
                    if smoke
                    else "enrich with --scope all"
                )
                print(f"Skipping {k}-shot on {variant}: it has no train split ({why})")
                continue
            tag = f"{prefix}llm_{spec['tag']}_{k}shot_{name}"
            print("=" * 78 + f"\n{tag}: {len(test)} pairs, {k} demonstrations\n" + "=" * 78)
            preds, st = classify(
                client,
                test,
                os.path.join(res, f"{tag}_raw.jsonl"),
                demos=demonstrations(train, k=k) if k else None,
                workers=workers,
                rescue_sampling=_rescue(spec),
            )
            unanswered = sum(p is None for p in preds)
            if unanswered:
                print(f"{unanswered} pairs got no answer; scored as Dissimilar (label 0)")
            y = test["label_score"].astype(int).tolist()
            p = [q if q is not None else 0 for q in preds]
            m = stats(y, p)
            pd.DataFrame(
                {
                    "Pair ID": test["Pair ID"],
                    "label": y,
                    "pred": p,
                    "Class 1": test["Class 1"],
                    "Class 2": test["Class 2"],
                }
            ).to_csv(os.path.join(res, f"{tag}_preds.csv"), index=False)
            plot_confusion(m["confusion_matrix"], os.path.join(res, f"cm_{tag}"))
            keys = (
                "f1_macro",
                "qwk",
                "mae",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "adjacent_acc",
                "severe_rate",
                "per_class",
                "confusion_matrix",
            )
            record = {
                **meta,
                "task": "classify",
                "tag": tag,
                "smoke": smoke,
                "family": "llm",
                "variant": variant,
                "shots": k,
                "n_test": len(test),
                **{key: m[key] for key in keys},
                "settings": st,
            }
            with open(os.path.join(res, f"{tag}_metrics.json"), "w") as fh:
                json.dump(record, fh, indent=2, default=str)
            print(
                f"DONE {tag}: Macro-F1 {m['f1_macro'] * 100:.2f} | QWK {m['qwk']:.4f} | "
                f"MAE {m['mae']:.4f} | n={len(test)} | {st['seconds'] / 60:.1f} min"
            )
            print("Per-class F1: " + " / ".join(f"{r['f1'] * 100:.2f}" for r in m["per_class"]))
            files += glob.glob(os.path.join(res, f"{tag}_*")) + glob.glob(
                os.path.join(res, f"cm_{tag}.*")
            )
    return files


def _tasks(
    spec: dict,
    tasks: list[str],
    files: list[str],
    data_root: str,
    out_root: str,
    scope: str,
    variants: list[str],
    shots: list[int],
    smoke: bool,
    batch_size: int,
    workers: int,
    port: int,
    seed: int,
) -> None:
    from codraft_enrichment.llm import make_client, package_versions, served_revision

    smoke_note = "  SMOKE TEST" if smoke else ""
    print(
        f"LLM {spec['tag']} = {spec['model']} ({spec['backend']}) | tasks {tasks} | "
        f"data {data_root} | out {out_root}{smoke_note}"
    )
    meta = {
        "llm": spec["tag"],
        "model": spec["model"],
        "serving": spec,
        "revision": served_revision(spec),
        "environment": package_versions(),
        "seed": seed,
        "temperature": 0,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("Revision:", meta["revision"], "| environment:", json.dumps(meta["environment"]))
    with _server(spec, out_root, port) as server:
        client = make_client(spec, server.base_url if server else None, seed=seed)
        if "enrich" in tasks:
            files += enrich_task(
                client, spec, meta, data_root, out_root, scope, smoke, batch_size, workers
            )
        if "classify" in tasks:
            classify_workers = max(workers, 32) if spec["backend"] == "vllm" else workers
            files += classify_task(
                client,
                spec,
                meta,
                data_root,
                out_root,
                variants,
                shots,
                smoke,
                classify_workers,
            )


def run_llm(
    llm: str,
    tasks: list[str],
    data_root: str | None = None,
    out_root: str = ".",
    scope: str = "all",
    variants: tuple[str, ...] | list[str] = ("plain", "codraft", "own"),
    shots: tuple[int, ...] | list[int] = (0, 10),
    smoke: bool = False,
    batch_size: int | None = None,
    workers: int | None = None,
    port: int = 8000,
    seed: int = 0,
    resume_from: str | None = None,
) -> str:
    spec = get_llm(llm)
    data_root = find_data_root(data_root)
    prefix = "SMOKE_" if smoke else ""
    name = prefix + spec["tag"]
    log = os.path.join(out_root, "logs", f"{name}_{'-'.join(tasks)}.log")
    files = [log]
    finished = False
    try:
        with _Tee(log):
            try:
                if resume_from:
                    _resume_from(resume_from, spec, smoke, out_root)
                _tasks(
                    spec,
                    tasks,
                    files,
                    data_root,
                    out_root,
                    scope,
                    variants,
                    shots,
                    smoke,
                    batch_size or spec["batch_size"],
                    workers or spec["workers"],
                    port,
                    seed,
                )
            except BaseException:
                traceback.print_exc(file=sys.stdout)
                raise
        finished = True
    finally:
        for pattern in (
            f"attributes/{name}_*",
            f"results/{prefix}llm_{spec['tag']}_*",
            f"results/cm_{prefix}llm_{spec['tag']}_*",
            f"logs/vllm_{spec['tag']}.log",
        ):
            files += glob.glob(os.path.join(out_root, pattern))
        zip_name = f"{prefix}codraft_llm_{spec['tag']}_{'-'.join(tasks)}" + (
            "" if finished else "_FAILED"
        )
        path = package(files, out_root, data_root, zip_name + ".zip")
    return path


def package(files: list[str], out_root: str, data_root: str, name: str) -> str:
    path = os.path.join(out_root, name)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for f in sorted(set(files)):
            if os.path.isfile(f):
                inside = os.path.abspath(f).startswith(os.path.abspath(out_root) + os.sep)
                zf.write(f, os.path.relpath(f, out_root if inside else data_root))
    with zipfile.ZipFile(path) as zf:
        bad, n = zf.testzip(), len(zf.namelist())
    if bad is not None:
        print(f"Zip check failed on {bad}")
    print(f"\nPackaged {n} files -> {path} ({os.path.getsize(path) / 2**20:.1f} MB)")
    print("Download this one file from the Output panel.")
    return path


def check(data_root: str | None = None) -> None:
    from codraft_enrichment.variant import check_rule
    from utils.metrics import stats

    data_root = find_data_root(data_root)
    with tempfile.TemporaryDirectory() as tmp:
        check_rule(data_root, tmp)
    n = 0
    for f in sorted(
        glob.glob(os.path.join(REPO_DIR, "weights", "**", "metrics.json"), recursive=True)
    ):
        with open(f) as fh:
            r = json.load(fh)
        p = pd.read_csv(os.path.join(os.path.dirname(f), "preds.csv"))
        m = stats(p["label"], p["pred"], r.get("n_classes", 5))
        for k in (
            "f1_macro",
            "qwk",
            "mae",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "adjacent_acc",
            "severe_rate",
        ):
            assert abs(m[k] - r[k]) < 1e-9, (r["tag"], k, m[k], r[k])
        assert m["confusion_matrix"] == r["confusion_matrix"], r["tag"]
        assert all(abs(a["f1"] - b["f1"]) < 1e-9 for a, b in zip(m["per_class"], r["per_class"])), (
            r["tag"]
        )
        n += 1
    print(f"Check: utils.metrics.stats matches the saved metrics of {n} runs in weights/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm")
    ap.add_argument("--task", nargs="+", default=["enrich"], choices=["enrich", "classify"])
    ap.add_argument("--scope", default="all", choices=["all", "test"])
    ap.add_argument("--variants", nargs="+", default=["plain", "codraft", "own"])
    ap.add_argument("--shots", nargs="+", type=int, default=[0, 10])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-root", default=".")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.check:
        check(a.data_root)
    elif a.list or not a.llm:
        print(describe())
    else:
        run_llm(
            a.llm,
            a.task,
            data_root=a.data_root,
            out_root=a.out_root,
            scope=a.scope,
            variants=a.variants,
            shots=a.shots,
            smoke=a.smoke,
            batch_size=a.batch_size,
            workers=a.workers,
            port=a.port,
            resume_from=a.resume_from,
        )


if __name__ == "__main__":
    main()
