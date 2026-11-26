"""Rebuttal analyses for ARR October 2026 (Submission 13566). CPU only, light on memory.

    python analysis/rebuttal.py        # writes analysis/out/rebuttal_summary.md and rebuttal_*.csv

Reads the splits in data/, the per-pair predictions in weights/<path>/preds.csv (paths from
config/runs.py) and the human evaluation in ../annotate/. It trains nothing and loads no model.

Sections (IDs from docs/arr-2026-10_sub13566/rebuttal-plan.md):
  checks  run 7 and the 7 - 12 interval reproduce the paper
  E3      train/test description overlap; scores on seen / unseen subsets
  E4      duplicates and label consistency over the 42,680 description instances
  E5a     error rate of run 7 on test pairs whose descriptions were rated by the annotators
  E13     binary rows, unrounded Table 4 deltas, Holm correction, per-class support
  new     every rebuttal run found in weights/ (E1, E1b, E2, E6, E7, E9b), compared with run 7

Paired comparisons use one bootstrap over test pairs, B = 2000, numpy seed 0, the same
resampled indices for every comparison on the full test set (as in the paper).
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from config.runs import RUNS  # noqa: E402
from utils.metrics import stats  # noqa: E402

DATA = os.path.join(REPO, "data")
WEIGHTS = os.path.join(REPO, "weights")
ANNOTATE = os.path.join(os.path.dirname(REPO), "annotate")
OUT = os.path.join(REPO, "analysis", "out")
LABELS = {"Dissimilar": 0, "Low similar": 1, "Similar": 2, "High similar": 3, "Identical": 4}
B, SEED = 2000, 0
PAPER_ANNOTATORS = ("TNH", "VBD", "NVTK")


# ------------------------------------------------------------------ metrics and bootstrap


def macro_f1(y: np.ndarray, p: np.ndarray, k: int = 5) -> float:
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    return float(f1.mean() * 100)


def qwk(y: np.ndarray, p: np.ndarray, k: int = 5) -> float:
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k).astype(float)
    w = np.subtract.outer(np.arange(k), np.arange(k)) ** 2 / (k - 1) ** 2
    exp = np.outer(cm.sum(1), cm.sum(0)) / cm.sum()
    return float(1 - (w * cm).sum() / (w * exp).sum())


def mae(y: np.ndarray, p: np.ndarray, k: int = 5) -> float:
    return float(np.abs(y - p).mean())


def sev(y: np.ndarray, p: np.ndarray, k: int = 5) -> float:
    return float((np.abs(y - p) >= 2).mean() * 100)


METRICS = {"f1": macro_f1, "qwk": qwk, "mae": mae, "sev": sev}


def boot_indices(n: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    for _ in range(B):
        yield rng.integers(0, n, n)


def paired(y: np.ndarray, a: np.ndarray, b: np.ndarray, metric: str = "f1", k: int = 5) -> dict:
    """Δ = metric(a) − metric(b), 95% percentile interval and a two-sided bootstrap p-value."""
    fn = METRICS[metric]
    d = np.array([fn(y[i], a[i], k) - fn(y[i], b[i], k) for i in boot_indices(len(y))])
    delta = fn(y, a, k) - fn(y, b, k)
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {
        "delta": delta,
        "lo": float(np.percentile(d, 2.5)),
        "hi": float(np.percentile(d, 97.5)),
        "p": float(p),
        "sig": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)
    }


def fmt(r: dict, nd: int = 2) -> str:
    star = " †" if r["sig"] else ""
    return f"{r['delta']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]{star}"


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m, adj, running = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


# ------------------------------------------------------------------ loading


def norm(s: object) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def split(name: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(DATA, "codraft", f"{name}.csv"), low_memory=False)
    d["y"] = d["Similarity"].map(LABELS).astype(int)
    return d


def run_dir(run_id: int) -> str:
    return os.path.join(WEIGHTS, RUNS[run_id]["path"])


def load_preds(run_id: int, ref_ids: np.ndarray, condition: str | None = None) -> np.ndarray | None:
    d = run_dir(run_id)
    f = os.path.join(d, "conditions", f"{condition}_preds.csv") if condition else os.path.join(
        d, "preds.csv"
    )
    if not os.path.isfile(f):
        return None
    p = pd.read_csv(f)
    if not (p["Pair ID"].astype(str).values == ref_ids).all():
        raise ValueError(f"{f}: Pair ID order differs from the test split")
    return p["pred"].to_numpy().astype(int)


def load_metrics(run_id: int) -> dict | None:
    f = os.path.join(run_dir(run_id), "metrics.json")
    return json.load(open(f)) if os.path.isfile(f) else None


# ------------------------------------------------------------------ sections


def section_checks(y: np.ndarray, P: dict) -> list[str]:
    m = stats(y, P[7])
    ok = abs(m["f1_macro"] * 100 - 81.83) < 0.005 and round(m["qwk"], 3) == 0.957 and round(
        m["mae"], 3
    ) == 0.125
    assert abs(macro_f1(y, P[7]) - m["f1_macro"] * 100) < 1e-9, "fast macro-F1 disagrees with stats()"
    assert abs(qwk(y, P[7]) - m["qwk"]) < 1e-9, "fast QWK disagrees with stats()"
    r = paired(y, P[7], P[12])
    ok2 = f"{r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]" == "+2.30 [+0.24, +4.51]"
    assert ok and ok2, (m["f1_macro"], m["qwk"], m["mae"], r)
    return [
        "## Checks",
        "",
        f"- Run 7 recomputed: Macro-F1 {m['f1_macro'] * 100:.2f}, QWK {m['qwk']:.3f}, "
        f"MAE {m['mae']:.3f} (paper: 81.83 / 0.957 / 0.125).",
        f"- 7 − 12 bootstrap: {fmt(r)} (paper: +2.30 [+0.24, +4.51]).",
        "",
    ]


def section_e3(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame, P: dict) -> tuple:
    y = te["y"].to_numpy()
    lines, rows = ["## E3. Train/test overlap and scores on unseen descriptions", ""], []
    for label, key in (("normalised", norm), ("exact", lambda s: str(s))):
        seen = set(tr["Term 1"].map(key)) | set(tr["Term 2"].map(key))
        s1 = te["Term 1"].map(key).isin(seen).to_numpy()
        s2 = te["Term 2"].map(key).isin(seen).to_numpy()
        test_desc = set(te["Term 1"].map(key)) | set(te["Term 2"].map(key))
        lines.append(
            f"- **{label} strings:** {len(test_desc):,} distinct test descriptions, "
            f"{len(test_desc & seen):,} ({len(test_desc & seen) / len(test_desc) * 100:.1f}%) also "
            f"appear in train. Test pairs: both seen {(s1 & s2).mean() * 100:.1f}%, one unseen "
            f"{(s1 ^ s2).mean() * 100:.1f}%, both unseen {(~s1 & ~s2).mean() * 100:.1f}%."
        )
        if label == "normalised":
            groups = {"both seen": s1 & s2, "one unseen": s1 ^ s2, "both unseen": ~s1 & ~s2}

    def pair_key(d: pd.DataFrame, with_class: bool) -> list:
        if with_class:
            return [
                frozenset({(norm(a), c), (norm(b), e)})
                for a, c, b, e in zip(d["Term 1"], d["Class 1"], d["Term 2"], d["Class 2"])
            ]
        return [frozenset({norm(a), norm(b)}) for a, b in zip(d["Term 1"], d["Term 2"])]

    for with_class in (True, False):
        k_tr, k_va, k_te = (set(pair_key(d, with_class)) for d in (tr, va, te))
        lines.append(
            f"- Unordered pairs shared between splits ({'with' if with_class else 'without'} NICE "
            f"classes): train∩test {len(k_tr & k_te)}, train∩val {len(k_tr & k_va)}, "
            f"val∩test {len(k_va & k_te)}."
        )
    lines += ["", "| test subset | n | " + " | ".join(f"run {r} F1" for r in P) + " | Δ 7−12 | Δ 7−5 |"]
    lines.append("|---|---|" + "---|" * (len(P) + 2))
    for g, m in groups.items():
        n = int(m.sum())
        cells = [f"{macro_f1(y[m], P[r][m]):.2f}" for r in P]
        ds = [
            fmt(paired(y[m], P[a][m], P[b][m])) if n >= 30 else "n < 30" for a, b in ((7, 12), (7, 5))
        ]
        lines.append(f"| {g} | {n} | " + " | ".join(cells) + " | " + " | ".join(ds) + " |")
        rows.append({"section": "E3", "group": g, "n": n, **{f"run{r}_f1": macro_f1(y[m], P[r][m]) for r in P}})
    lines += [
        "",
        "Within a subset the bootstrap resamples that subset (seed 0). Split: by pair, stratified by "
        "label; there is no split-creation code in the repo.",
        "",
    ]
    return lines, rows


def section_e4(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame) -> list[str]:
    allp = pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")])
    terms = pd.concat([allp["Term 1"], allp["Term 2"]]).astype(str)
    classes = pd.concat([allp["Class 1"], allp["Class 2"]])
    lines = [
        "## E4. Duplicates and label consistency",
        "",
        f"- Pairs: train {len(tr):,}, val {len(va):,}, test {len(te):,}, total {len(allp):,}. "
        f"Distinct Pair IDs: {allp['Pair ID'].nunique():,}.",
        f"- Description instances: {len(terms):,}. Distinct descriptions: {terms.nunique():,} exact, "
        f"{terms.map(norm).nunique():,} normalised, "
        f"{len(set(zip(terms.map(norm), classes))):,} normalised (description, class).",
    ]
    for with_class in (True, False):
        if with_class:
            key = [
                tuple(sorted([(norm(a), c), (norm(b), e)]))
                for a, c, b, e in zip(allp["Term 1"], allp["Class 1"], allp["Term 2"], allp["Class 2"])
            ]
        else:
            key = [tuple(sorted([norm(a), norm(b)])) for a, b in zip(allp["Term 1"], allp["Term 2"])]
        g = pd.DataFrame({"k": key, "y": allp["y"].to_numpy()}).groupby("k")["y"]
        size, nlab, spread = g.size(), g.nunique(), g.max() - g.min()
        dup = size[size > 1]
        conflict = spread[nlab > 1]
        lines.append(
            f"- Unordered pairs {'with' if with_class else 'without'} classes: {len(size):,} distinct; "
            f"{len(dup):,} occur more than once ({int(dup.sum()):,} rows); "
            f"{len(conflict):,} carry conflicting labels"
            + (
                f" (label spread: " + ", ".join(f"{k}: {v}" for k, v in conflict.value_counts().sort_index().items()) + ")"
                if len(conflict)
                else ""
            )
            + "."
        )
    sid = allp.groupby("Pair ID")["split"].nunique()
    lines += [f"- Pair IDs in more than one split: {int((sid > 1).sum())}.", ""]
    return lines


def section_e5(te: pd.DataFrame, P: dict) -> tuple:
    samples = pd.read_csv(os.path.join(ANNOTATE, "Samples.csv"))
    votes = {}
    for a in PAPER_ANNOTATORS:
        d = pd.read_csv(os.path.join(ANNOTATE, f"Annotation_{a}.csv"), encoding="utf-8-sig")
        assert (d["PairID"].values == samples["PairID"].values).all(), a
        votes[a] = d
    nat = (sum(votes[a]["Nature"] for a in PAPER_ANNOTATORS) >= 2).to_numpy()
    pur = (sum(votes[a]["Purpose"] for a in PAPER_ANNOTATORS) >= 2).to_numpy()
    rated = {}
    for t, c, n_ok, p_ok in zip(samples["Term"], samples["Class"], nat, pur):
        rated[(norm(t), int(c))] = (bool(n_ok), bool(p_ok))
    y = te["y"].to_numpy()
    side_status = []
    for t1, c1, t2, c2 in zip(te["Term 1"], te["Class 1"], te["Term 2"], te["Class 2"]):
        st = [rated.get((norm(t1), int(c1))), rated.get((norm(t2), int(c2)))]
        st = [s for s in st if s is not None]
        side_status.append("none" if not st else ("fail" if any(not (n and p) for n, p in st) else "pass"))
    side_status = np.array(side_status)
    lines = [
        "## E5a. Do failed attributes go with wrong predictions? (run 7, no retraining)",
        "",
        f"Majority of {', '.join(PAPER_ANNOTATORS)} (the three annotators behind Table 10) over the "
        f"135 rated descriptions: Nature fails {int((~nat).sum())}, Purpose fails {int((~pur).sum())}. "
        f"A test pair is *fail* if any of its rated sides has a failed Nature or Purpose, *pass* if "
        f"all its rated sides pass.",
        "",
        "| test pairs | n | run 7 error % | run 7 severe % | run 12 error % (no attributes) |",
        "|---|---|---|---|---|",
    ]
    rows = []
    for g in ("fail", "pass", "none"):
        m = side_status == g
        n = int(m.sum())
        e7 = (P[7][m] != y[m]).mean() * 100 if n else float("nan")
        s7 = (np.abs(P[7][m] - y[m]) >= 2).mean() * 100 if n else float("nan")
        e12 = (P[12][m] != y[m]).mean() * 100 if n else float("nan")
        lines.append(f"| {g} | {n} | {e7:.1f} | {s7:.1f} | {e12:.1f} |")
        rows.append({"section": "E5a", "group": g, "n": n, "err7": e7, "sev7": s7, "err12": e12})
    f, p_ = side_status == "fail", side_status == "pass"
    if f.sum() and p_.sum():
        from scipy.stats import fisher_exact

        w7 = P[7] != y
        table = [[int(w7[f].sum()), int((~w7[f]).sum())], [int(w7[p_].sum()), int((~w7[p_]).sum())]]
        odds, pv = fisher_exact(table)
        rng = np.random.default_rng(SEED)
        fi, pi = np.where(f)[0], np.where(p_)[0]
        diffs = [
            w7[rng.choice(fi, len(fi))].mean() - w7[rng.choice(pi, len(pi))].mean() for _ in range(B)
        ]
        lines += [
            "",
            f"Run 7 error rate, fail − pass: {(w7[f].mean() - w7[p_].mean()) * 100:+.1f} points, "
            f"95% bootstrap CI [{np.percentile(diffs, 2.5) * 100:+.1f}, "
            f"{np.percentile(diffs, 97.5) * 100:+.1f}]; Fisher exact p = {pv:.3g} (odds ratio {odds:.2f}).",
        ]
    lines += [
        "",
        "Small n: indicative only. Matching is on (normalised description, class). The run 12 column "
        "is a control: if it is just as high on *fail* pairs, those pairs are hard regardless of the "
        "attributes. The counterfactual (replace the failed attribute, re-predict with run 7) is run 43.",
        "",
    ]
    return lines, rows


def section_e13(y: np.ndarray, te_ids: np.ndarray, P: dict) -> tuple:
    lines, rows = ["## E13. Statistics", ""], []
    # (a) binary rows
    p25 = pd.read_csv(os.path.join(run_dir(25), "preds.csv"))
    assert (p25["Pair ID"].astype(str).values == te_ids).all()
    yb = (y >= 1).astype(int)
    assert (p25["label"].to_numpy() == yb).all()
    r = paired(yb, (P[7] >= 1).astype(int), p25["pred"].to_numpy().astype(int), k=2)
    lines += [
        "**(a) Binary.** Macro-F1 of run 7 collapsed to binary − run 25 trained on binary: "
        f"{fmt(r)} (collapsed {macro_f1(yb, (P[7] >= 1).astype(int), 2):.2f}, trained "
        f"{macro_f1(yb, p25['pred'].to_numpy().astype(int), 2):.2f}).",
        "",
    ]
    # (b, c) Table 4 unrounded, Holm over the retrained family
    names = {
        12: "w/o CoDraft Enrichment",
        13: "w/o Nature & Purpose",
        14: "w/ Free-text Expansion",
        15: "w/o Multi-Task Head",
        16: "w/o Rank-Aware Loss",
        17: "w/o Rank Penalty (α = 0)",
        18: "w/o Head & Rank-Aware Loss",
    }
    res = {rid: {m: paired(y, P[rid], P[7], m) for m in METRICS} for rid in names}
    adj = {
        m: dict(zip(names, holm([res[rid][m]["p"] for rid in names]))) for m in ("f1", "qwk", "mae")
    }
    lines += [
        "**(b, c) Table 4, retrained rows, Δ = ablation − full model from unrounded values.** p: "
        "two-sided bootstrap; Holm: Holm–Bonferroni over these 7 rows, per metric.",
        "",
        "| row | run | Δ Macro-F1 [95% CI] | p | Holm p | Δ QWK [95% CI] | Holm p | Δ MAE [95% CI] | Holm p | Δ Sev. pts |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rid, nm in names.items():
        R = res[rid]
        lines.append(
            f"| {nm} | {rid} | {fmt(R['f1'])} | {R['f1']['p']:.3f} | {adj['f1'][rid]:.3f} | "
            f"{fmt(R['qwk'], 3)} | {adj['qwk'][rid]:.3f} | {fmt(R['mae'], 3)} | {adj['mae'][rid]:.3f} | "
            f"{R['sev']['delta']:+.2f} |"
        )
        rows.append({"section": "E13c", "run": rid, **{f"{m}_{k}": R[m][k] for m in R for k in ("delta", "lo", "hi", "p")}, **{f"{m}_holm": adj[m][rid] for m in adj}})
    probe = ["no_nature", "no_purpose", "no_heading", "heading_only", "bare_term", "shuffled_attributes"]
    cells = []
    for c in probe:
        pc = load_preds(19, te_ids, c)
        if pc is not None:
            cells.append(f"{c} {macro_f1(y, pc) - macro_f1(y, P[7]):+.2f}")
    lines += ["", "Probe rows, unrounded Δ Macro-F1 vs full: " + "; ".join(cells) + ".", ""]
    # (d) support
    sup = np.bincount(y, minlength=5)
    lines += [f"**(d) Test support per level:** {' / '.join(f'{v:,}' for v in sup)} (n = {sup.sum():,}).", ""]
    return lines, rows


def section_new(y: np.ndarray, te_ids: np.ndarray, P: dict) -> tuple:
    lines, rows = ["## Rebuttal runs found in weights/", ""], []
    new = [r for r in sorted(RUNS) if r >= 26]
    found = {r: load_preds(r, te_ids) for r in new}
    found = {r: p for r, p in found.items() if p is not None}
    if not found:
        return lines + ["None yet. Install the Kaggle zips with `python runner.py --install <zip>`.", ""], rows
    lines += ["| run | what | Macro-F1 | QWK | MAE | Sev. % | Δ F1 vs run 7 | extra |", "|---|---|---|---|---|---|---|---|"]
    for r, p in found.items():
        met = load_metrics(r) or {}
        st = met.get("settings", {})
        extra = []
        if "two_order" in st:
            t = st["two_order"]
            extra.append(f"(A,B)+(B,A): {t['f1_macro'] * 100:.2f} / {t['qwk']:.3f} / {t['mae']:.3f}")
        if st.get("aux_acc_test") is not None:
            extra.append(f"class head acc {st['aux_acc_test'] * 100:.1f}%")
        lines.append(
            f"| {r} | {RUNS[r]['row']} | {macro_f1(y, p):.2f} | {qwk(y, p):.3f} | {mae(y, p):.3f} | "
            f"{sev(y, p):.2f} | {fmt(paired(y, P[7], p))} | {'; '.join(extra)} |"
        )
        rows.append({"section": "new", "run": r, "f1": macro_f1(y, p), "qwk": qwk(y, p), "mae": mae(y, p), "sev": sev(y, p)})
    P = {**P, **found}
    comps = [
        ((27, 26), "E1 CoDraft − vanilla, CE bge, same recipe"),
        ((29, 28), "E1 CoDraft − vanilla, CE deberta, same recipe"),
        ((7, 27), "E1 full model − CE bge CoDraft, same recipe"),
        ((27, 5), "E1 recipe effect on CE bge CoDraft (same-recipe − original)"),
        ((26, 6), "E1 recipe effect on CE bge vanilla"),
        ((34, 20), "E6 Hybrid MLP, CoDraft − vanilla"),
        ((35, 7), "E9b masked side without heading − full model"),
        ((36, 18), "E7 CORAL − run 18 (same input, no head, no rank-aware loss)"),
        ((37, 18), "E7 CORN − run 18"),
        ((30, 32), "E2 seed 13: CoDraft − vanilla"),
        ((31, 33), "E2 seed 2026: CoDraft − vanilla"),
    ]
    lines += ["", "| comparison | Δ Macro-F1 | Δ QWK | Δ MAE |", "|---|---|---|---|"]
    for (a, b), what in comps:
        if a in P and b in P:
            lines.append(
                f"| {what} ({a} − {b}) | {fmt(paired(y, P[a], P[b]))} | "
                f"{fmt(paired(y, P[a], P[b], 'qwk'), 3)} | {fmt(paired(y, P[a], P[b], 'mae'), 3)} |"
            )
    for avg in (26, 27, 28, 29):
        pa = load_preds(avg, te_ids, "avg") if avg in found else None
        if pa is not None:
            lines.append(
                f"| E1b run {avg}: two orders averaged − (A, B) only | {fmt(paired(y, pa, P[avg]))} | "
                f"{fmt(paired(y, pa, P[avg], 'qwk'), 3)} | {fmt(paired(y, pa, P[avg], 'mae'), 3)} |"
            )
    for old in (44, 45, 46, 47):
        pa = P.get(old)
        pab = load_preds(old, te_ids, "ab") if old in found else None
        if pa is not None and pab is not None:
            src = RUNS[old]["source"]
            lines.append(
                f"| E1b original CE run {src}: two orders averaged − (A, B) only | {fmt(paired(y, pa, pab))} | "
                f"{fmt(paired(y, pa, pab, 'qwk'), 3)} | {fmt(paired(y, pa, pab, 'mae'), 3)} |"
            )
    rep = []
    for r in (42, 43, 44, 45, 46, 47):
        st = (load_metrics(r) or {}).get("settings", {}) if r in found else {}
        if "reproduction" in st:
            rr = st["reproduction"]
            diff = f", max |Δ logit| {rr['logits_max_abs_diff']:.2g}" if "logits_max_abs_diff" in rr else ""
            rep.append(f"run {r}: {rr['pred_agreement'] * 100:.2f}% of {rr['n_compared']}{diff}")
    if rep:
        lines += ["", "Reproduction of the released predictions by the inference runs: " + "; ".join(rep) + "."]
    st42 = (load_metrics(42) or {}).get("settings", {}) if 42 in found else {}
    if "aux_acc_test" in st42:
        worst = ", ".join(
            f"class {c} {v['acc'] * 100:.1f}% (n={v['n']})" for c, v in st42.get("aux_worst_classes_test", [])
        )
        lines += [
            "",
            f"- E9a run 7 masked class head: val {st42['aux_acc_val'] * 100:.2f}%, test "
            f"{st42['aux_acc_test'] * 100:.2f}% of masked views; lowest classes on test: {worst}.",
        ]
    seed_aux = [
        (r, (load_metrics(r) or {}).get("settings", {}).get("aux_acc_test")) for r in (30, 31, 35) if r in found
    ]
    seed_aux = [(r, a) for r, a in seed_aux if a is not None]
    if seed_aux:
        lines.append("- Class head accuracy on test, new runs: " + ", ".join(f"run {r} {a * 100:.2f}%" for r, a in seed_aux))
    st43 = (load_metrics(43) or {}).get("settings", {}) if 43 in found else {}
    if "conditions" in st43:
        lines += [
            "",
            f"**E5 counterfactual (run 7, inference only).** Pairs with a failed attribute: "
            f"{st43['pairs']['fail']}; pairs whose rated attributes all pass: {st43['pairs']['pass']}. "
            f"Corrections: {st43['corrections']}.",
            "",
            "| condition | pairs | inputs changed | predictions changed | wrong → right | right → wrong | error % before → after | severe % before → after | test Macro-F1 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for name, c in st43["conditions"].items():
            if "n" not in c:
                continue
            lines.append(
                f"| {name} | {c['subset']} ({c['n']}) | {c['pairs_with_changed_input']} | {c['pred_changed']} | "
                f"{c['wrong_to_right']} | {c['right_to_wrong']} | {c['err_before'] * 100:.1f} → {c['err_after'] * 100:.1f} | "
                f"{c['severe_before'] * 100:.1f} → {c['severe_after'] * 100:.1f} | {c['f1_macro_test'] * 100:.2f} |"
            )
        rows += [{"section": "E5cf", "condition": k, **{kk: vv for kk, vv in c.items() if kk != "example"}} for k, c in st43["conditions"].items()]
    seeds = {
        "full model (7, 30, 31)": (7, 30, 31),
        "w/o CoDraft (12, 32, 33)": (12, 32, 33),
        "same-recipe CE bge, vanilla (26, 38, 40)": (26, 38, 40),
        "same-recipe CE bge, CoDraft (27, 39, 41)": (27, 39, 41),
    }
    lines += [""]
    for what, ids in seeds.items():
        have = [i for i in ids if i in P]
        if len(have) > 1:
            vals = {m: [METRICS[m](y, P[i]) for i in have] for m in ("f1", "qwk", "mae", "sev")}
            lines.append(
                f"- E2 {what}, {len(have)} seeds: "
                + ", ".join(f"{m} {np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}" for m, v in vals.items())
            )
    for what, pairs_ in (
        ("full model − same-recipe CE bge CoDraft per seed", ((7, 27), (30, 39), (31, 41))),
        ("same-recipe CE bge CoDraft − vanilla per seed", ((27, 26), (39, 38), (41, 40))),
    ):
        g = [macro_f1(y, P[a]) - macro_f1(y, P[b]) for a, b in pairs_ if a in P and b in P]
        if len(g) > 1:
            lines.append(f"- E2 {what}: {', '.join(f'{x:+.2f}' for x in g)}; mean {np.mean(g):+.2f} ± {np.std(g, ddof=1):.2f}.")
    gains = [
        macro_f1(y, P[a]) - macro_f1(y, P[b]) for a, b in ((7, 12), (30, 32), (31, 33)) if a in P and b in P
    ]
    if len(gains) > 1:
        lines.append(
            f"- E2 CoDraft gain per seed: {', '.join(f'{g:+.2f}' for g in gains)}; mean "
            f"{np.mean(gains):+.2f} ± {np.std(gains, ddof=1):.2f}; positive in {sum(g > 0 for g in gains)}/{len(gains)}."
        )
    return lines + [""], rows


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    tr, va, te = split("train"), split("val"), split("test")
    te_ids = te["Pair ID"].astype(str).to_numpy()
    y = te["y"].to_numpy()
    P = {r: load_preds(r, te_ids) for r in (5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 20)}
    missing = [r for r, p in P.items() if p is None]
    assert not missing, f"missing preds for runs {missing}"
    out, rows = ["# Rebuttal analyses (generated by analysis/rebuttal.py)", ""], []
    out += section_checks(y, P)
    for fn in (
        lambda: section_e3(tr, va, te, {r: P[r] for r in (7, 12, 5)}),
        lambda: (section_e4(tr, va, te), []),
        lambda: section_e5(te, P),
        lambda: section_e13(y, te_ids, P),
        lambda: section_new(y, te_ids, P),
    ):
        lines, rs = fn()
        out += lines
        rows += rs
        print("\n".join(lines), flush=True)
    with open(os.path.join(OUT, "rebuttal_summary.md"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "rebuttal_tables.csv"), index=False)
    print(f"\nWrote {OUT}/rebuttal_summary.md and rebuttal_tables.csv")


if __name__ == "__main__":
    main()
