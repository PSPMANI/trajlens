"""Agreement statistics for meta-evaluation: how far does a grader sit from the humans?

Pure Python on purpose, so the core package installs with no dependencies.
"""
from __future__ import annotations

import random

POSITIVE = "FAILED"  # a grader's job is to catch failures, so FAILED is the positive class


def cohens_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two PASSED/FAILED label lists."""
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(x == y for x, y in zip(a, b, strict=False)) / n
    pa = sum(x == "PASSED" for x in a) / n
    pb = sum(y == "PASSED" for y in b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe >= 1.0 else (po - pe) / (1 - pe)


def confusion(truth: list[str], pred: list[str]) -> dict:
    tp = sum(t == POSITIVE and p == POSITIVE for t, p in zip(truth, pred, strict=False))
    fn = sum(t == POSITIVE and p != POSITIVE for t, p in zip(truth, pred, strict=False))
    fp = sum(t != POSITIVE and p == POSITIVE for t, p in zip(truth, pred, strict=False))
    tn = sum(t != POSITIVE and p != POSITIVE for t, p in zip(truth, pred, strict=False))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / len(truth) if truth else 0.0}


def kappa_ci(a: list[str], b: list[str], n_boot: int = 2000, seed: int = 0,
             alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for Cohen's kappa.

    With a small corpus the interval is wide, and reporting it is the honest thing to do:
    a point estimate from 14 items should not be read to two decimals.
    """
    rng = random.Random(seed)
    n = len(a)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(cohens_kappa([a[i] for i in idx], [b[i] for i in idx]))
    stats.sort()
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def compare(truth: list[str], pred: list[str]) -> dict:
    """Everything the dashboard shows for one grader against the human labels."""
    lo, hi = kappa_ci(truth, pred)
    return {"agree": sum(t == p for t, p in zip(truth, pred, strict=False)), "n": len(truth),
            "kappa": cohens_kappa(truth, pred), "kappa_ci": [lo, hi], **confusion(truth, pred)}
