"""Monotonic FPR calibration and rank-preserving batch positive budget."""
import math
import numpy as np


class FPRCalibrator:
    """Isotonic calibration + piecewise-linear remap so the operating point
    with human FPR == target lands exactly on score 0.5. Monotonic => AP and
    recall@FPR are unchanged; only the 0.5-threshold sanity improves."""

    def __init__(self, target_fpr=0.04):
        self.target_fpr = target_fpr

    def fit(self, raw_scores, y):
        from sklearn.isotonic import IsotonicRegression
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso.fit(raw_scores, y)
        cal = self.iso.transform(raw_scores)
        humans = cal[np.asarray(y) == 0]
        self.t = float(np.quantile(humans, 1.0 - self.target_fpr)) if len(humans) else 0.5
        self.t = min(max(self.t, 1e-6), 1 - 1e-6)
        return self

    def transform(self, raw_scores):
        cal = self.iso.transform(np.asarray(raw_scores, dtype=float))
        out = np.where(
            cal <= self.t,
            0.5 * cal / self.t,
            0.5 + 0.5 * (cal - self.t) / (1.0 - self.t),
        )
        return np.clip(out, 0.0, 1.0)


def batch_guard(scores, min_pos=1, max_frac=0.25):
    """Rank-preserving positive budget: >=min_pos and <=max_frac*n scores >= 0.5."""
    s = np.asarray(scores, dtype=float).copy()
    order = np.argsort(-s)
    max_pos = max(min_pos, int(math.floor(max_frac * len(s))))
    pos = np.flatnonzero(s >= 0.5)
    if len(pos) > max_pos:
        keep = set(order[:max_pos])
        excess = sorted((i for i in pos if i not in keep), key=lambda i: -s[i])
        lo, hi = 0.470, 0.499
        for rank, i in enumerate(excess):
            s[i] = hi - (hi - lo) * rank / max(len(excess), 1)
    elif len(pos) < min_pos:
        for i in order[:min_pos]:
            s[i] = max(s[i], 0.55)
    return np.clip(s, 0.0, 1.0)


def rank01(p):
    from scipy.stats import rankdata
    return (rankdata(p) - 0.5) / len(p)
