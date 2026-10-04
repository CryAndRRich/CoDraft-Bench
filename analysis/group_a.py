"""Analyses that need no GPU: they read the trained runs in weights/ and the splits in data/.

    python analysis/group_a.py            # writes analysis/out/group_a.md and CSVs

1. class-pair-only baselines (no text)
2. test pairs split by whether their terms were seen in training
3. same-class vs cross-class pairs
4. lexical-overlap buckets, including the "deceptive" pairs
5. floor baselines: random, majority, exact match, TF-IDF cosine with tuned thresholds
6. calibration (ECE) and risk-coverage, from the saved multi-task logits
7. cost of the enrichment and inference speed

Paired comparisons use one bootstrap over test pairs (B=2000, seed 0) shared by every
comparison, and the runs are matched on Pair ID, which every prediction file carries.
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.neural_network import MLPClassifier

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
WEIGHTS = os.path.join(REPO, "weights")
OUT = os.path.join(REPO, "analysis", "out")
LABELS = {"Dissimilar": 0, "Low similar": 1, "Similar": 2, "High similar": 3, "Identical": 4}
B, SEED = 2000, 0


# ------------------------------------------------------------------ loading

def split(variant, name):
    d = pd.read_csv(os.path.join(DATA, variant, f"{name}.csv"), low_memory=False)
    d["y"] = d["Similarity"].map(LABELS).astype(int)
    return d


def load_runs():
    runs = {}
    for f in sorted(glob.glob(os.path.join(WEIGHTS, "[0-9][0-9]_*", "*_metrics.json"))):
        r = json.load(open(f))
        tag = r["tag"]
        d = os.path.dirname(f)
        p = pd.read_csv(os.path.join(d, f"{tag}_preds.csv"))
        lg = os.path.join(d, f"{tag}_logits.npy")
        runs[r["run_id"]] = {"meta": r, "pred": p["pred"].to_numpy(),
                             "logits": np.load(lg) if os.path.isfile(lg) else None,
                             "pair_id": p["Pair ID"].to_numpy()}
    return runs


# ------------------------------------------------------------------ metrics

def f1(y, p):
    return f1_score(y, p, average="macro", labels=range(5), zero_division=0) * 100


def metrics(y, p):
    return {"n": len(y), "f1": f1(y, p),
            "qwk": cohen_kappa_score(y, p, weights="quadratic") if len(set(y)) > 1 else float("nan"),
            "mae": float(np.abs(y - p).mean()),
            "acc": float((y == p).mean() * 100),
            "bin_f1": f1_score(y >= 1, p >= 1, zero_division=0)}


IDX = None


def paired(y, a, b, mask=None):
    """Δ macro-F1 of a over b with a 95% bootstrap interval, optionally on a subset."""
    global IDX
    if IDX is None:
        rng = np.random.default_rng(SEED)
        IDX = [rng.integers(0, len(y), len(y)) for _ in range(B)]
    if mask is not None:
        y, a, b = y[mask], a[mask], b[mask]
        rng = np.random.default_rng(SEED)
        idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]
    else:
        idx = IDX
    d = np.array([f1(y[i], a[i]) - f1(y[i], b[i]) for i in idx])
    return f1(y, a) - f1(y, b), np.percentile(d, 2.5), np.percentile(d, 97.5)


def fmt_delta(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"


def norm(s):
    return str(s).strip().lower()


def tokens(s):
    return set(re.findall(r"[a-z0-9]+", norm(s)))


# ------------------------------------------------------------------ sections

def class_pair_baselines(tr, va, te):
    """1. Predict from the two NICE classes alone, no text."""
    def key(d):
        return [tuple(sorted(x)) for x in zip(d["Class 1"], d["Class 2"])]
    rows = []

    # majority label per unordered class pair, backing off to the global majority
    maj = pd.Series(tr["y"].values, index=key(tr)).groupby(level=0).agg(lambda s: s.value_counts().idxmax())
    glob_maj = int(tr["y"].value_counts().idxmax())
    kt = key(te)
    pred = np.array([maj.get(k, glob_maj) for k in kt])
    seen = np.mean([k in maj.index for k in kt]) * 100
    rows.append(("majority label per class pair", metrics(te["y"].values, pred), f"{seen:.1f}% of test pairs have a class pair seen in train"))

    # linear and MLP models on one-hot classes
    def onehot(d):
        x = np.zeros((len(d), 45 * 2 + 1))
        x[np.arange(len(d)), d["Class 1"].values - 1] = 1
        x[np.arange(len(d)), 45 + d["Class 2"].values - 1] = 1
        x[:, -1] = (d["Class 1"].values == d["Class 2"].values)
        return x
    Xtr, Xte = onehot(tr), onehot(te)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, tr["y"])
    rows.append(("logistic regression, one-hot classes, balanced", metrics(te["y"].values, lr.predict(Xte)), ""))
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True, random_state=SEED,
                        max_iter=300).fit(Xtr, tr["y"])
    rows.append(("MLP 256-128, one-hot classes (as P1's \"No Emb\")", metrics(te["y"].values, mlp.predict(Xte)), ""))
    return rows


def floor_baselines(tr, va, te):
    """5. Random, majority, exact match and TF-IDF cosine with tuned thresholds."""
    y = te["y"].values
    rng = np.random.default_rng(SEED)
    prior = tr["y"].value_counts(normalize=True).sort_index().values
    rows = [("random, uniform", metrics(y, rng.integers(0, 5, len(y))), ""),
            ("random, train prior", metrics(y, rng.choice(5, len(y), p=prior)), ""),
            ("majority (always Dissimilar)", metrics(y, np.zeros(len(y), int)), "")]
    exact = np.where([norm(a) == norm(b) for a, b in zip(te["Term 1"], te["Term 2"])], 4, 0)
    rows.append(("exact string match -> Identical, else Dissimilar", metrics(y, exact), f"{(exact == 4).sum()} test pairs match exactly"))

    vec = TfidfVectorizer(lowercase=True).fit(pd.concat([tr["Term 1"], tr["Term 2"]]).astype(str))
    def cos(d):
        a = vec.transform(d["Term 1"].astype(str)); b = vec.transform(d["Term 2"].astype(str))
        num = np.asarray(a.multiply(b).sum(1)).ravel()
        den = np.sqrt(np.asarray(a.multiply(a).sum(1)).ravel() * np.asarray(b.multiply(b).sum(1)).ravel())
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    cv, ct = cos(va), cos(te)
    # four ordered cut-points, tuned on validation by coordinate ascent over a grid
    grid = np.unique(np.quantile(cv, np.linspace(0, 1, 101)))
    cuts = list(np.quantile(cv, [0.55, 0.59, 0.72, 0.74]))
    def apply(c, cuts):
        return np.searchsorted(np.sort(cuts), c, side="right")
    best = f1(va["y"].values, apply(cv, cuts))
    for _ in range(4):
        for i in range(4):
            for g in grid:
                trial = cuts.copy(); trial[i] = g
                s = f1(va["y"].values, apply(cv, trial))
                if s > best:
                    best, cuts = s, trial
    rows.append(("TF-IDF cosine, 4 thresholds tuned on val", metrics(y, apply(ct, cuts)), f"val macro-F1 {best:.2f}"))
    return rows


def subgroup_table(name, masks, runs, y, compare):
    """Per-group macro-F1 for key runs, and paired gains within each group."""
    lines = [f"| group | n | " + " | ".join(f"run {r}" for r in compare["show"]) + " | "
             + " | ".join(f"Δ {a}−{b}" for a, b in compare["pairs"]) + " |",
             "|---|---|" + "---|" * (len(compare["show"]) + len(compare["pairs"]))]
    csv = []
    for g, m in masks.items():
        n = int(m.sum())
        cells = [f"{f1(y[m], runs[r]['pred'][m]):.2f}" for r in compare["show"]]
        ds = [fmt_delta(paired(y, runs[a]["pred"], runs[b]["pred"], m)) if n >= 30 else "n too small"
              for a, b in compare["pairs"]]
        lines.append(f"| {g} | {n} | " + " | ".join(cells) + " | " + " | ".join(ds) + " |")
        csv.append({"section": name, "group": g, "n": n,
                    **{f"run{r}_f1": f1(y[m], runs[r]["pred"][m]) for r in compare["show"]}})
    return lines, csv


def ece(prob, y, bins=15):
    conf = prob.max(1); pred = prob.argmax(1)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def calibration(runs, y):
    """6. ECE and risk-coverage for every run that saved logits."""
    rows = []
    for rid, r in sorted(runs.items()):
        if r["logits"] is None:
            continue
        z = r["logits"] - r["logits"].max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        order = np.argsort(-p.max(1)); pred = p.argmax(1)
        cov = {}
        for c in (0.5, 0.7, 0.9, 1.0):
            k = order[: int(round(c * len(y)))]
            cov[c] = (f1(y[k], pred[k]), (y[k] == pred[k]).mean() * 100)
        risk = 1 - np.cumsum(y[order] == pred[order]) / np.arange(1, len(y) + 1)
        rows.append((rid, r["meta"]["name"], ece(p, y), cov, float(risk.mean())))
    return rows


def cost():
    """7. Enrichment cost estimate and logged inference speed."""
    attr = pd.read_csv(os.path.join(DATA, "term_attributes.csv"), low_memory=False)
    out_chars = sum(attr[c].astype(str).str.len().sum() for c in ["Nature", "Purpose", "Expanded_Name", "Logic_Trace"])
    n_terms = len(attr)
    import importlib.util
    spec = importlib.util.spec_from_file_location("p", os.path.join(REPO, "codraft_enrichment", "prompt.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    tmpl = len(m.ENRICHMENT_PROMPTS["v1"]["system"]) + len(m.ENRICHMENT_PROMPTS["v1"]["user"])
    batch = 30
    calls = int(np.ceil(n_terms / batch))
    speed = {}
    for rid, pat in [(7, "07_*"), (5, "05_*"), (8, "08_*")]:
        for lg in glob.glob(os.path.join(WEIGHTS, pat, "*.log")):
            txt = open(lg, errors="ignore").read()
            v = re.findall(r"'eval_samples_per_second': ([0-9.]+)", txt)
            if v:
                speed[rid] = f"{float(v[-1]) / 2:.0f} pairs/s (validation pass, two masked rows per pair)"
            b = re.findall(r"Predicting Baseline: 100%.*?\[(\d+):(\d+)<", txt)
            if b:
                s = int(b[-1][0]) * 60 + int(b[-1][1])
                speed[rid] = f"{4266 / max(s, 1):.0f} pairs/s (test prediction)"
    return {"terms": n_terms, "calls": calls, "out_chars": int(out_chars),
            "template_chars": tmpl, "speed": speed}


# ------------------------------------------------------------------ report

def table(rows, title):
    lines = [f"| {title} | Macro-F1 | QWK | MAE | binary F1 | note |", "|---|---|---|---|---|---|"]
    for name, m, note in rows:
        lines.append(f"| {name} | {m['f1']:.2f} | {m['qwk']:.3f} | {m['mae']:.3f} | {m['bin_f1']:.3f} | {note} |")
    return lines


def main():
    os.makedirs(OUT, exist_ok=True)
    tr, va, te = (split("plain", s) for s in ("train", "val", "test"))
    runs = load_runs()
    y = te["y"].values
    for rid, r in runs.items():
        assert (r["pair_id"] == te["Pair ID"].values).all(), f"run {rid} is not aligned to the test split"
    md = ["# Group A analyses", "",
          f"Generated by `analysis/group_a.py` from {len(runs)} runs in `weights/`. Paired intervals: "
          f"bootstrap over test pairs, B = {B}, 95%.", ""]
    csv = []

    md += ["## 1. Class-pair-only baselines (no text)", ""]
    rows = class_pair_baselines(tr, va, te)
    md += table(rows, "model") + [""]
    md += [f"For reference, run 7 (full model): Macro-F1 {f1(y, runs[7]['pred']):.2f}, binary F1 "
           f"{f1_score(y >= 1, runs[7]['pred'] >= 1):.3f}. P1 reports 0.807 binary F1 for its no-text model "
           f"on its own data.", ""]

    md += ["## 5. Floor baselines", ""]
    md += table(floor_baselines(tr, va, te), "baseline") + [""]

    seen = set(map(norm, tr["Term 1"])) | set(map(norm, tr["Term 2"]))
    s1 = te["Term 1"].map(norm).isin(seen).values; s2 = te["Term 2"].map(norm).isin(seen).values
    groups = {"both terms seen in train": s1 & s2, "one term seen": s1 ^ s2, "neither term seen": ~s1 & ~s2}
    cmp_ = {"show": [7, 12, 5, 4, 3], "pairs": [(7, 12), (5, 6), (4, 3), (2, 1)]}
    md += ["## 2. Seen vs unseen terms", ""]
    l, c = subgroup_table("seen", groups, runs, y, cmp_); md += l + [""]; csv += c

    same = (te["Class 1"] == te["Class 2"]).values
    md += ["## 3. Same-class vs cross-class pairs", ""]
    l, c = subgroup_table("class", {"same NICE class": same, "different NICE classes": ~same}, runs, y, cmp_)
    md += l + [""]; csv += c

    jac = np.array([len(tokens(a) & tokens(b)) / max(1, len(tokens(a) | tokens(b)))
                    for a, b in zip(te["Term 1"], te["Term 2"])])
    buckets = {"no shared word (J = 0)": jac == 0, "some overlap (0 < J < 0.5)": (jac > 0) & (jac < 0.5),
               "high overlap (J >= 0.5)": jac >= 0.5}
    md += ["## 4. Lexical overlap (token Jaccard of the two names)", ""]
    l, c = subgroup_table("overlap", buckets, runs, y, cmp_); md += l + [""]; csv += c
    dec = (jac >= 0.5) & (y <= 1)
    hid = (jac == 0) & (y >= 3)
    md += [f"**Deceptive pairs** (J >= 0.5 but labelled Dissimilar or Low similar): n = {int(dec.sum())}. "
           f"**Hidden-similarity pairs** (no shared word but High similar or Identical): n = {int(hid.sum())}.", "",
           "| run | deceptive: accuracy | deceptive: predicted Similar or higher | hidden: accuracy | hidden: predicted Low or lower |",
           "|---|---|---|---|---|"]
    for rid in [7, 12, 5, 4, 3, 2, 1]:
        p = runs[rid]["pred"]
        md.append(f"| {rid} {runs[rid]['meta']['name']} | {(p[dec] == y[dec]).mean()*100:.1f}% | {(p[dec] >= 2).mean()*100:.1f}% "
                  f"| {(p[hid] == y[hid]).mean()*100:.1f}% | {(p[hid] <= 1).mean()*100:.1f}% |")
    md += [""]

    md += ["## 6. Calibration and risk-coverage (multi-task runs, from saved logits)", "",
           "| run | ECE (15 bins) | Macro-F1 / acc at 50% coverage | at 70% | at 90% | at 100% | AURC |",
           "|---|---|---|---|---|---|---|"]
    for rid, name, e, cov, aurc in calibration(runs, y):
        cells = " | ".join(f"{cov[c][0]:.1f} / {cov[c][1]:.1f}" for c in (0.5, 0.7, 0.9, 1.0))
        md.append(f"| {rid} {name} | {e:.4f} | {cells} | {aurc:.4f} |")
    md += ["", "Coverage keeps the most confident pairs; the rest would go to a human examiner.", ""]

    k = cost()
    out_tok = k["out_chars"] / 4
    md += ["## 7. Cost", "",
           f"- Enrichment runs once per unique term: **{k['terms']:,} terms** in **{k['calls']} calls** of 30 terms.",
           f"- Generated text (Nature, Purpose, expanded name, reasoning trace): {k['out_chars']:,} characters, "
           f"about **{out_tok/1e6:.2f} M output tokens** at 4 characters per token.",
           f"- Prompt template: {k['template_chars']:,} characters per call, plus each term and its official class "
           f"heading. The inputs were not logged, so input tokens are not reported here.",
           "- The attribute cache (`data/term_attributes.csv`) is released, so reproducing the experiments needs no API call.",
           "- Inference speed on one T4, from the run logs: "
           + "; ".join(f"run {r}: {s}" for r, s in sorted(k["speed"].items())), ""]

    open(os.path.join(OUT, "group_a.md"), "w").write("\n".join(md) + "\n")
    pd.DataFrame(csv).to_csv(os.path.join(OUT, "group_a_subgroups.csv"), index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
