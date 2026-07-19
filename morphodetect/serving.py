"""Production scorer: heterogeneous tree-stack on within-batch rank-normalized,
sanitize-invariant features, with an exact rank-budget output.

No neural net: the 6-max-trained transformer helped benchmark AP but hurt live
transfer (a "benchmark mirage"), so the model is a tree ensemble like the top
miners. CPU-only, no GPU needed to serve.
"""
import os
import pathlib

import joblib
import numpy as np

from .calibration import batch_rank, rank_budget
from .features import chunk_features_v2

DEFAULT_ARTIFACTS = pathlib.Path(__file__).resolve().parent.parent / "artifacts"


class Detector:
    def __init__(self, artifacts_dir=DEFAULT_ARTIFACTS, device=None):
        artifacts_dir = pathlib.Path(artifacts_dir)
        self.device = "cpu"
        bundle = joblib.load(artifacts_dir / "bundle.joblib")
        # "model" is the tree-stack; fall back to legacy "gbm" for old bundles.
        self.model = bundle.get("model", bundle.get("gbm"))
        self.feature_names = bundle["feature_names"]
        self.batch_rank_features = bundle.get("batch_rank_gbm", True)
        self.pos_frac = float(os.getenv("P44_POS_FRAC", str(bundle.get("pos_frac", 0.10))))

    def _probs(self, chunks):
        rows = [chunk_features_v2(c) for c in chunks]
        X = np.array([[r.get(k, 0.0) for k in self.feature_names] for r in rows],
                     dtype=np.float32)
        if self.batch_rank_features and len(chunks) > 1:
            X = batch_rank(X)
        return self.model.predict_proba(X)[:, 1]

    def score_chunks(self, chunks):
        """One bot-risk score in [0,1] per chunk."""
        chunks = [c if c else [{}] for c in chunks]
        if len(chunks) == 1:
            # no batch to rank against; return a neutral-ish single prob
            return [float(np.clip(self._probs(chunks)[0], 0.0, 1.0))]
        # Exact rank budget: exactly ~pos_frac of the batch crosses 0.5,
        # rank-preserving. Immune to the live-feed score compression that a
        # benchmark-fit calibrator mishandles.
        scores = rank_budget(self._probs(chunks), frac=self.pos_frac)
        return [float(round(s, 6)) for s in scores]
