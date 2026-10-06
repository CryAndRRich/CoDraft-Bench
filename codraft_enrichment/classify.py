import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from codraft_enrichment.llm import LLMClient
from preprocess.preprocess_data import preprocess

LABELS = ["Dissimilar", "Low similar", "Similar", "High similar", "Identical"]
PROMPT_VERSION = "cls-v1"

SYSTEM = (
    "You are an examiner at the European Union Intellectual Property Office (EUIPO) "
    "comparing goods and services in trade mark opposition proceedings."
)

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


def load_pairs(variant_root: str, variant: str, split: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(variant_root, variant, f"{split}.csv"), low_memory=False)
    return preprocess(d).reset_index(drop=True)


def demonstrations(train: pd.DataFrame, k: int = 10, seed: int = 0) -> pd.DataFrame:
    per = k // len(LABELS)
    picked = [
        train[train["label_score"] == y].sample(per, random_state=seed + y)
        for y in range(len(LABELS))
    ]
    return pd.concat(picked).sample(frac=1, random_state=seed).reset_index(drop=True)


def user_prompt(a: str, b: str, demos: pd.DataFrame | None = None) -> str:
    parts = [INSTRUCTIONS]
    if demos is not None and len(demos):
        parts.append("Examples:")
        for r in demos.itertuples(index=False):
            answer = json.dumps({"label": LABELS[r.label_score]})
            parts.append(f"Item A: {r.input_text_1}\nItem B: {r.input_text_2}\nAnswer: {answer}")
        parts.append("Now the pair to rate.")
    parts.append(
        f'Item A: {a}\nItem B: {b}\nAnswer with JSON: {{"label": <one of the five labels>}}'
    )
    return "\n\n".join(parts)


def classify(
    client: LLMClient,
    pairs: pd.DataFrame,
    out_path: str,
    demos: pd.DataFrame | None = None,
    workers: int = 32,
    rescue_sampling: dict | None = None,
) -> tuple[list[int | None], dict]:
    done = {}
    if os.path.isfile(out_path):
        with open(out_path) as fh:
            for line in fh:
                r = json.loads(line)
                done[r["Pair ID"]] = r
        print(f"Resuming: {len(done)} pairs already in {out_path}")
    lock = threading.Lock()
    fh = open(out_path, "a", buffering=1)
    stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "errors": []}

    def one(r: dict, sampling: dict | None = None) -> None:
        out, info = client.json(
            SYSTEM,
            user_prompt(r["input_text_1"], r["input_text_2"], demos),
            Verdict,
            max_tokens=32,
            sampling=sampling,
        )
        rec = {
            "Pair ID": r["Pair ID"],
            "label": int(r["label_score"]),
            "pred": LABELS.index(out.label) if out else None,
            "error": info["error"],
        }
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

    def run(rows: list[dict], sampling: dict | None = None) -> None:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(one, r, sampling) for r in rows]
            for i, f in enumerate(as_completed(futures), 1):
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
    stats.update(
        n_pairs=len(pairs),
        n_failed=sum(p is None for p in preds),
        n_rescued=sum(bool(done.get(pid, {}).get("rescued")) for pid in pairs["Pair ID"]),
        rescue_sampling=rescue_sampling,
        seconds=round(time.time() - t0, 1),
        prompt_version=PROMPT_VERSION,
        n_demonstrations=0 if demos is None else len(demos),
        demonstration_ids=[] if demos is None else list(demos["Pair ID"]),
    )
    stats["n_errors"] = len(stats["errors"])
    stats["errors"] = stats["errors"][:50]
    return preds, stats
