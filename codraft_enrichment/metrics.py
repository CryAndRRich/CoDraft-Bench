"""The metrics of utils/evaluate.get_stats, without its imports (torch, sentence-transformers),
so LLM rows are scored exactly as runs 1-20 are. llm_runner.py --check verifies the two
agree on every saved run in weights/.
"""
import numpy as np
from sklearn.metrics import cohen_kappa_score

LABEL_NAMES = ["Dissimilar", "Low similar", "Similar", "High similar", "Identical"]


def _div(a, b):
    return float(a) / float(b) if b else 0.0


def stats(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp = np.diag(cm)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    tn = cm.sum() - (tp + fp + fn)
    per_class = []
    for c in range(5):
        p, r = _div(tp[c], tp[c] + fp[c]), _div(tp[c], tp[c] + fn[c])
        per_class.append({"class": c, "support": int(cm[c].sum()),
                          "accuracy": _div(tp[c] + tn[c], cm.sum()),
                          "precision": p, "recall": r, "f1": _div(2 * p * r, p + r),
                          "name": LABEL_NAMES[c]})
    err = np.abs(y_true - y_pred)
    return {"n": int(len(y_true)), "accuracy": _div((y_true == y_pred).sum(), len(y_true)),
            "mae": float(err.mean()), "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
            "f1_macro": float(np.mean([r["f1"] for r in per_class])),
            "precision_macro": float(np.mean([r["precision"] for r in per_class])),
            "recall_macro": float(np.mean([r["recall"] for r in per_class])),
            "adjacent_acc": float((err <= 1).mean()), "severe_rate": float((err >= 2).mean()),
            "per_class": per_class, "confusion_matrix": cm.tolist()}


def plot_confusion(cm, prefix):
    """cm_<tag>.pdf / .png like the other runs; skipped if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    cm = np.asarray(cm)
    names = ['Dissimilar (0)', 'Low (1)', 'Similar (2)', 'High (3)', 'Identical (4)']
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm, cmap="Blues")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(5), names, rotation=45, ha="right")
    ax.set_yticks(range(5), names)
    fig.savefig(f"{prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
