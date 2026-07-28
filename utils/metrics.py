import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


def compute_binary_metrics(preds, labels) -> dict:
    preds = np.asarray(preds, dtype=int)
    labels = np.asarray(labels, dtype=int)
    return {
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def compute_ranking_metrics(scores, labels) -> dict:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return {"auroc": 0.0, "auprc": 0.0}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def find_best_threshold(scores, labels, n_steps: int = 1000) -> dict:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for t in np.linspace(0.0, 1.0, n_steps + 1):
        preds = (scores >= t).astype(int)
        m = compute_binary_metrics(preds, labels)
        if m["f1"] > best["f1"]:
            best = {"threshold": float(t), **m}
    return best
