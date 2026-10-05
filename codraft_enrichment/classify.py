"""An LLM as the classifier: it reads a pair and answers with one of the five labels.

The two sides are the same strings the trained models read, built by
preprocess_data.preprocess() from data/<variant>/: the bare name for "plain", the
"Term [ Nature: .. | Use: .. | Category: .. ]" string for "codraft". The answer is
constrained to the five label names, so every pair gets a label.

Few-shot demonstrations come from train, two per label, drawn once with a fixed seed and
shown in the same order for every test pair.
"""
import importlib.util
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import pandas as pd
from pydantic import BaseModel

LABELS = ["Dissimilar", "Low similar", "Similar", "High similar", "Identical"]
PROMPT_VERSION = "cls-v1"

SYSTEM = ("You are an examiner at the European Union Intellectual Property Office (EUIPO) "
          "comparing goods and services in trade mark opposition proceedings.")

INSTRUCTIONS = """Rate how similar the two goods or services below are, on the five-level scale used in EUIPO decisions.

Weigh the factors of EUIPO practice: their nature, intended purpose and method of use; whether they are in competition with each other or complementary; their distribution channels and sales outlets; their relevant public; and their usual commercial origin.

Scale:
- Dissimilar: no relevant factor connects them.
- Low similar: similar to a low degree; they share only a minor factor.
- Similar: similar to an average degree.
- High similar: similar to a high degree; they share most factors.
- Identical: the same goods or services, including when one term is included in, or includes, the other.

Each item may be followed by a description in square brackets."""


class Verdict(BaseModel):
    label: Literal["Dissimilar", "Low similar", "Similar", "High similar", "Identical"]


def _text_builder():
    """preprocess() from preprocess/preprocess_data.py, loaded without the preprocess
    package, whose __init__ needs the training stack (sentence-transformers, datasets)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "codraft_preprocess_data", os.path.join(here, "preprocess", "preprocess_data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.preprocess


def load_pairs(variant_root, variant, split):
    d = pd.read_csv(os.path.join(variant_root, variant, f"{split}.csv"), low_memory=False)
    return _text_builder()(d).reset_index(drop=True)


def demonstrations(train, k=10, seed=0):
    per = k // len(LABELS)
    picked = [train[train["label_score"] == y].sample(per, random_state=seed + y)
              for y in range(len(LABELS))]
    return pd.concat(picked).sample(frac=1, random_state=seed).reset_index(drop=True)


def user_prompt(a, b, demos=None):
    parts = [INSTRUCTIONS]
    if demos is not None and len(demos):
        parts.append("Examples:")
        for r in demos.itertuples(index=False):
            parts.append(f"Item A: {r.input_text_1}\nItem B: {r.input_text_2}\n"
                         f"Answer: {json.dumps({'label': LABELS[r.label_score]})}")
        parts.append("Now the pair to rate.")
    parts.append(f"Item A: {a}\nItem B: {b}\nAnswer with JSON: {{\"label\": <one of the five labels>}}")
    return "\n\n".join(parts)


def classify(client, pairs, out_path, demos=None, workers=32, rescue_sampling=None):
    """Label every row of pairs; returns (predictions, stats). Resumes from out_path (JSONL).

    A pair left without an answer gets one more try with rescue_sampling (a repetition
    penalty), marked "rescued" in the JSONL and counted in the stats.
    """
    done = {}
    if os.path.isfile(out_path):
        for line in open(out_path):
            r = json.loads(line)
            done[r["Pair ID"]] = r
        print(f"resuming: {len(done)} pairs already in {out_path}")
    lock = threading.Lock()
    fh = open(out_path, "a", buffering=1)
    stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "errors": []}

    def one(r, sampling=None):
        out, info = client.json(SYSTEM, user_prompt(r["input_text_1"], r["input_text_2"], demos),
                                Verdict, max_tokens=32, sampling=sampling)
        rec = {"Pair ID": r["Pair ID"], "label": int(r["label_score"]),
               "pred": LABELS.index(out.label) if out else None, "error": info["error"]}
        if sampling:
            rec["rescued"] = True
        with lock:
            stats["calls"] += info["calls"]
            stats["prompt_tokens"] += info["prompt_tokens"]
            stats["completion_tokens"] += info["completion_tokens"]
            if info["error"]:
                stats["errors"].append(info["error"])
            if rec["pred"] is not None:
                fh.write(json.dumps(rec) + "\n")
                done[rec["Pair ID"]] = rec
        return rec

    def run(rows, sampling=None):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(one, r, sampling) for r in rows]
            for i, f in enumerate(as_completed(futs), 1):
                f.result()
                if i % max(1, len(rows) // 20) == 0 or i == len(rows):
                    print(f"[classify] {i}/{len(rows)} | {time.time() - t0:.0f}s", flush=True)

    rows = pairs.to_dict("records")
    t0 = time.time()
    run([r for r in rows if r["Pair ID"] not in done])
    left = [r for r in rows if r["Pair ID"] not in done]
    if left and rescue_sampling:
        print(f"{len(left)} pairs got no answer; one more try with {rescue_sampling}")
        run(left, rescue_sampling)
    fh.close()
    preds = [done.get(pid, {}).get("pred") for pid in pairs["Pair ID"]]
    stats.update(n_pairs=len(pairs), n_failed=sum(p is None for p in preds),
                 n_rescued=sum(bool(done.get(pid, {}).get("rescued")) for pid in pairs["Pair ID"]),
                 rescue_sampling=rescue_sampling,
                 seconds=round(time.time() - t0, 1), prompt_version=PROMPT_VERSION,
                 n_demonstrations=0 if demos is None else len(demos),
                 demonstration_ids=[] if demos is None else list(demos["Pair ID"]))
    stats["n_errors"] = len(stats["errors"])
    stats["errors"] = stats["errors"][:50]
    return preds, stats
