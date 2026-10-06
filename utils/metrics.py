import numpy as np
from sklearn.metrics import cohen_kappa_score

LABEL_NAMES = ["Dissimilar", "Low similar", "Similar", "High similar", "Identical"]
BINARY_NAMES = ["Dissimilar", "Similar"]


def _div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def stats(y_true: list | np.ndarray, y_pred: list | np.ndarray, num_classes: int = 5) -> dict:
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    names = LABEL_NAMES if num_classes == 5 else BINARY_NAMES
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp = np.diag(cm)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    tn = cm.sum() - (tp + fp + fn)
    per_class = []
    for c in range(num_classes):
        precision, recall = _div(tp[c], tp[c] + fp[c]), _div(tp[c], tp[c] + fn[c])
        per_class.append(
            {
                "class": c,
                "support": int(cm[c].sum()),
                "accuracy": _div(tp[c] + tn[c], cm.sum()),
                "precision": precision,
                "recall": recall,
                "f1": _div(2 * precision * recall, precision + recall),
                "name": names[c],
            }
        )
    err = np.abs(y_true - y_pred)
    return {
        "n": int(len(y_true)),
        "accuracy": _div((y_true == y_pred).sum(), len(y_true)),
        "mae": float(err.mean()),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "f1_macro": float(np.mean([r["f1"] for r in per_class])),
        "precision_macro": float(np.mean([r["precision"] for r in per_class])),
        "recall_macro": float(np.mean([r["recall"] for r in per_class])),
        "adjacent_acc": float((err <= 1).mean()),
        "severe_rate": float((err >= 2).mean()),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def binary_metrics(
    true: list | np.ndarray,
    pred: list | np.ndarray,
    n_balanced: int = 30,
    seed: int = 0,
) -> dict:
    y = (np.asarray(true) >= 1).astype(int)
    p = (np.asarray(pred) >= 1).astype(int)

    def score(y: np.ndarray, p: np.ndarray) -> dict:
        tp, fp = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum())
        fn, tn = int(((y == 1) & (p == 0)).sum()), int(((y == 0) & (p == 0)).sum())
        rec, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1)
        return {
            "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "recall": rec,
            "specificity": spec,
            "precision": prec,
        }

    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = {"natural": score(y, p), "n_similar": len(pos), "n_dissimilar": len(neg)}
    if len(pos) and len(neg) >= len(pos):
        rng = np.random.default_rng(seed)
        sets = [np.r_[pos, rng.choice(neg, len(pos), replace=False)] for _ in range(n_balanced)]
        runs = [score(y[i], p[i]) for i in sets]
        out["balanced"] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        out["balanced_std"] = {k: float(np.std([r[k] for r in runs])) for k in runs[0]}
        out["n_balanced"] = n_balanced
    return out


def plot_confusion(cm: list | np.ndarray, prefix: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    k = len(cm)
    if k == 5:
        names = ["Dissimilar (0)", "Low (1)", "Similar (2)", "High (3)", "Identical (4)"]
    else:
        names = ["Dissimilar (0)", "Similar (1)"]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm, cmap="Blues")
    for i in range(k):
        for j in range(k):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    ax.set_xticks(range(k), names, rotation=45, ha="right")
    ax.set_yticks(range(k), names)
    fig.savefig(f"{prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
