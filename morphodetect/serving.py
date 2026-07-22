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
        """One OPERATIONAL bot-risk probability in [0,1] per chunk.

        The model (batch-rank-normalized tree-stack) already emits well-spread,
        operational probabilities on the live feed (min~0.07, max~0.95, ~35%
        cross 0.5). We return them directly. We deliberately do NOT compress
        them into a narrow rank-budget band around 0.5: the subnet's runtime-431
        validators (deploy 0.1.35) apply threshold-sanity/calibration checks that
        penalize "a rank ordering compressed below the 0.5 threshold" (see
        docs/miner.md), which is exactly what rank_budget did — it silently
        collapsed our live composite from ~0.53 to ~0.10.
        """
        chunks = [c if c else [{}] for c in chunks]
        probs = self._probs(chunks)
        return [float(np.clip(round(float(p), 6), 0.0, 1.0)) for p in probs]
