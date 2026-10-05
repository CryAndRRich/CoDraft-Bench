"""CoDraft enrichment of every term the splits use, by any LLM in config/llms.py.

Same prompt (v1) and schema as the Gemini run that produced data/term_attributes.csv; only
the model and the batch size differ. Each term goes in with the casing Gemini saw and the
official heading of its NICE class.

Progress is appended to a JSONL file after every batch, so a run that stops halfway
continues where it left off when started again on the same output folder.
"""
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from config.config_data import CONFIG_DATA
from .llm import strict_schema
from .prompt import ENRICHMENT_PROMPTS
from .schemas import SCHEMA_MAP

GENERIC = {"product", "products", "item", "items", "goods", "service", "services", "thing"}
ATTRIBUTE_COLUMNS = ["Original_Term", "Class", "Nature", "Purpose", "Expanded_Name",
                     "Logic_Trace", "Brainstorming", "mode"]


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().strip("\"'").strip().lower()


def term_table(data_root, scope="all", gemini_cache=None):
    """Every (term, NICE class) in the chosen splits, one row each, in a fixed order.

    scope: "all" (train + val + test) or "test".
    gemini_cache: term_attributes.csv, used only to recover the casing Gemini was given,
        since the split files hold lower-cased terms.
    """
    splits = {"all": ("train", "val", "test"), "test": ("test",)}[scope]
    seen = {}
    for split in splits:
        d = pd.read_csv(os.path.join(data_root, "codraft", f"{split}.csv"), low_memory=False)
        for side in ("1", "2"):
            for t, c in zip(d[f"Term {side}"], d[f"Class {side}"]):
                seen.setdefault((norm(t), int(c)), str(t).strip())
    casing = {}
    if gemini_cache and os.path.isfile(gemini_cache):
        g = pd.read_csv(gemini_cache, low_memory=False)
        for t in g["Original_Term"].astype(str):
            casing.setdefault(norm(t), t.strip())
    rows = [{"key": k, "Class": c, "Term": casing.get(k, t),
             "Description": CONFIG_DATA.NICE_CLASS_HEADINGS[c]}
            for (k, c), t in seen.items()]
    return pd.DataFrame(rows).sort_values(["Class", "key"], kind="stable").reset_index(drop=True)


def _valid(item):
    return bool(str(item.nature).strip()) and bool(str(item.purpose).strip())


def _record(row, item, mode):
    return {"key": row["key"], "Class": int(row["Class"]), "Original_Term": row["Term"],
            "Nature": str(item.nature).strip(), "Purpose": str(item.purpose).strip(),
            "Expanded_Name": str(item.expanded_name).strip(),
            "Logic_Trace": str(item.step2_context_verification).strip(),
            "Brainstorming": json.dumps(list(item.step1_brainstorming), ensure_ascii=False),
            "returned_term": str(item.original_term), "mode": mode}


def enrich(client, table, out_dir, name, batch_size=10, workers=16, version="v1",
           tokens_per_term=400, rescue_sampling=None):
    """Enrich every row of term_table(); returns (attributes frame, stats).

    A batch answer is matched back to its inputs by term. Terms a batch answer leaves out
    (or garbles) are sent again one at a time. A term that fails alone too gets one last
    try with rescue_sampling (a repetition penalty: greedy decoding can loop on a phrase
    until the token budget runs out), marked mode "single_penalized"; if that fails as
    well, it is reported as failed rather than filled in.
    """
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"{name}_raw.jsonl")
    done = {}
    if os.path.isfile(raw_path):
        for line in open(raw_path):
            r = json.loads(line)
            done[(r["key"], r["Class"])] = r
        print(f"resuming: {len(done)} terms already in {raw_path}")
    lock = threading.Lock()
    raw = open(raw_path, "a", buffering=1)
    system = ENRICHMENT_PROMPTS[version]["system"]
    template = ENRICHMENT_PROMPTS[version]["user"]
    Response = SCHEMA_MAP[version]
    stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "errors": []}

    def call(rows, mode):
        sampling = rescue_sampling if mode == "single_penalized" else None
        batch = [{"Term": r["Term"], "Class": int(r["Class"]), "Description": r["Description"]}
                 for r in rows]
        # exactly one item per input term, and a bounded brainstorm (the prompt asks for 3)
        schema = strict_schema(Response, max_items={"items": len(rows), "step1_brainstorming": 5},
                               min_items={"items": len(rows)})
        user = template.format(batch_json=json.dumps(batch, indent=2, ensure_ascii=False))
        out, info = client.json(system, user, Response, schema=schema,
                                max_tokens=tokens_per_term * len(rows) + 256, sampling=sampling)
        items = list(out.items) if out else []
        matched, missing = [], []
        if mode != "batch":
            # one input, so the one answer is for it whatever name it echoes back
            if items and _valid(items[0]):
                matched.append(_record(rows[0], items[0], mode))
            else:
                missing = list(rows)
        else:
            pool = {}
            for it in items:
                pool.setdefault(norm(it.original_term), []).append(it)
            for r in rows:
                cand = pool.get(norm(r["Term"]))
                if cand and _valid(cand[0]):
                    matched.append(_record(r, cand.pop(0), mode))
                else:
                    missing.append(r)
        with lock:
            stats["calls"] += info["calls"]
            stats["prompt_tokens"] += info["prompt_tokens"]
            stats["completion_tokens"] += info["completion_tokens"]
            if info["error"]:
                stats["errors"].append(info["error"])
            for rec in matched:
                raw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done[(rec["key"], rec["Class"])] = rec
        return missing

    def run(groups, mode, desc):
        if not groups:
            return []
        t0, left, n = time.time(), [], 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(call, g, mode) for g in groups]
            for f in as_completed(futs):
                left += f.result()
                n += 1
                if n % max(1, len(groups) // 20) == 0 or n == len(groups):
                    print(f"[{desc}] {n}/{len(groups)} calls | {len(done)} terms done | "
                          f"{time.time() - t0:.0f}s", flush=True)
        return left

    t0 = time.time()
    todo = [r for r in table.to_dict("records") if (r["key"], r["Class"]) not in done]
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    print(f"{len(table)} terms, {len(todo)} to do, in {len(batches)} batches of {batch_size}")
    missing = run(batches, "batch" if batch_size > 1 else "single", "batches")
    if missing and batch_size > 1:
        print(f"{len(missing)} terms missing from batch answers; asking for them one at a time")
        missing = run([[r] for r in missing], "single", "singles")
    if missing and rescue_sampling:
        print(f"{len(missing)} terms failed alone too; one last try with {rescue_sampling}")
        missing = run([[r] for r in missing], "single_penalized", "penalized")
    raw.close()

    recs = [done[(r["key"], r["Class"])] for r in table.to_dict("records")
            if (r["key"], r["Class"]) in done]
    attrs = pd.DataFrame(recs)
    attrs = attrs[ATTRIBUTE_COLUMNS + ["key", "returned_term"]] if len(attrs) else \
        pd.DataFrame(columns=ATTRIBUTE_COLUMNS + ["key", "returned_term"])
    failed = [{"Term": r["Term"], "Class": int(r["Class"])} for r in missing]
    stats.update(
        n_terms=len(table), n_done=len(attrs), n_failed=len(failed), failed=failed[:200],
        by_mode=attrs["mode"].value_counts().to_dict() if len(attrs) else {},
        generic_nature=int(attrs["Nature"].str.lower().str.strip().isin(GENERIC).sum()) if len(attrs) else 0,
        echoed_other_name=int((attrs["returned_term"].map(norm) != attrs["key"]).sum()) if len(attrs) else 0,
        seconds=round(time.time() - t0, 1), batch_size=batch_size, workers=workers,
        prompt_version=version, rescue_sampling=rescue_sampling)
    stats["n_errors"] = len(stats["errors"])
    stats["errors"] = stats["errors"][:50]
    return attrs, stats
