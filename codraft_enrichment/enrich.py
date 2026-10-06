import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from codraft_enrichment.llm import LLMClient, strict_schema
from codraft_enrichment.prompt import PROMPT_VERSION, SYSTEM_PROMPT, USER_PROMPT
from codraft_enrichment.schemas import BatchAnalysis, NiceCoDraft
from config.config_data import CONFIG_DATA

GENERIC = {"product", "products", "item", "items", "goods", "service", "services", "thing"}
ATTRIBUTE_COLUMNS = [
    "Original_Term",
    "Class",
    "Nature",
    "Purpose",
    "Expanded_Name",
    "Logic_Trace",
    "Brainstorming",
    "mode",
]


def norm(s: object) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().strip("\"'").strip().lower()


def term_table(data_root: str, scope: str = "all", gemini_cache: str | None = None) -> pd.DataFrame:
    splits = {"all": ("train", "val", "test"), "test": ("test",)}[scope]
    seen = {}
    for split in splits:
        d = pd.read_csv(os.path.join(data_root, "codraft", f"{split}.csv"), low_memory=False)
        for side in ("1", "2"):
            for t, c in zip(d[f"Term {side}"], d[f"Class {side}"]):
                seen.setdefault((norm(t), int(c)), str(t).strip())
    casing = {}
    if gemini_cache and os.path.isfile(gemini_cache):
        for t in pd.read_csv(gemini_cache, low_memory=False)["Original_Term"].astype(str):
            casing.setdefault(norm(t), t.strip())
    rows = [
        {
            "key": k,
            "Class": c,
            "Term": casing.get(k, t),
            "Description": CONFIG_DATA.NICE_CLASS_HEADINGS[c],
        }
        for (k, c), t in seen.items()
    ]
    return pd.DataFrame(rows).sort_values(["Class", "key"], kind="stable").reset_index(drop=True)


def _valid(item: NiceCoDraft) -> bool:
    return bool(str(item.nature).strip()) and bool(str(item.purpose).strip())


def _record(row: dict, item: NiceCoDraft, mode: str) -> dict:
    return {
        "key": row["key"],
        "Class": int(row["Class"]),
        "Original_Term": row["Term"],
        "Nature": str(item.nature).strip(),
        "Purpose": str(item.purpose).strip(),
        "Expanded_Name": str(item.expanded_name).strip(),
        "Logic_Trace": str(item.step2_context_verification).strip(),
        "Brainstorming": json.dumps(list(item.step1_brainstorming), ensure_ascii=False),
        "returned_term": str(item.original_term),
        "mode": mode,
    }


def enrich(
    client: LLMClient,
    table: pd.DataFrame,
    out_dir: str,
    name: str,
    batch_size: int = 10,
    workers: int = 16,
    tokens_per_term: int = 400,
    rescue_sampling: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"{name}_raw.jsonl")
    done = {}
    if os.path.isfile(raw_path):
        with open(raw_path) as fh:
            for line in fh:
                r = json.loads(line)
                done[(r["key"], r["Class"])] = r
        print(f"Resuming: {len(done)} terms already in {raw_path}")
    lock = threading.Lock()
    raw = open(raw_path, "a", buffering=1)
    stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "errors": []}

    def call(rows: list[dict], mode: str) -> list[dict]:
        sampling = rescue_sampling if mode == "single_penalized" else None
        batch = [
            {"Term": r["Term"], "Class": int(r["Class"]), "Description": r["Description"]}
            for r in rows
        ]
        schema = strict_schema(
            BatchAnalysis,
            max_items={"items": len(rows), "step1_brainstorming": 5},
            min_items={"items": len(rows)},
        )
        user = USER_PROMPT.format(batch_json=json.dumps(batch, indent=2, ensure_ascii=False))
        out, info = client.json(
            SYSTEM_PROMPT,
            user,
            BatchAnalysis,
            schema=schema,
            max_tokens=tokens_per_term * len(rows) + 256,
            sampling=sampling,
        )
        items = list(out.items) if out else []
        matched, missing = [], []
        if mode != "batch":
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

    def run(groups: list[list[dict]], mode: str, desc: str) -> list[dict]:
        if not groups:
            return []
        t0, left, n = time.time(), [], 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for f in as_completed([ex.submit(call, g, mode) for g in groups]):
                left += f.result()
                n += 1
                if n % max(1, len(groups) // 20) == 0 or n == len(groups):
                    print(
                        f"[{desc}] {n}/{len(groups)} calls | {len(done)} terms done | {time.time() - t0:.0f}s",
                        flush=True,
                    )
        return left

    t0 = time.time()
    todo = [r for r in table.to_dict("records") if (r["key"], r["Class"]) not in done]
    batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    print(f"{len(table)} terms, {len(todo)} to do, in {len(batches)} batches of {batch_size}")
    missing = run(batches, "batch" if batch_size > 1 else "single", "batches")
    if missing and batch_size > 1:
        print(f"{len(missing)} terms missing from batch answers; asking for them one at a time")
        missing = run([[r] for r in missing], "single", "singles")
    if missing and rescue_sampling:
        print(f"{len(missing)} terms failed alone too; one last try with {rescue_sampling}")
        missing = run([[r] for r in missing], "single_penalized", "penalized")
    raw.close()

    keys = [(r["key"], r["Class"]) for r in table.to_dict("records")]
    columns = ATTRIBUTE_COLUMNS + ["key", "returned_term"]
    attrs = pd.DataFrame([done[k] for k in keys if k in done], columns=columns)
    failed = [{"Term": r["Term"], "Class": int(r["Class"])} for r in missing]
    stats.update(
        n_terms=len(table),
        n_done=len(attrs),
        n_failed=len(failed),
        failed=failed[:200],
        by_mode=attrs["mode"].value_counts().to_dict(),
        generic_nature=int(attrs["Nature"].str.lower().str.strip().isin(GENERIC).sum()),
        echoed_other_name=int((attrs["returned_term"].map(norm) != attrs["key"]).sum()),
        seconds=round(time.time() - t0, 1),
        batch_size=batch_size,
        workers=workers,
        prompt_version=PROMPT_VERSION,
        rescue_sampling=rescue_sampling,
    )
    stats["n_errors"] = len(stats["errors"])
    stats["errors"] = stats["errors"][:50]
    return attrs, stats
