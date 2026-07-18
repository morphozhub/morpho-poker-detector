"""Production scorer: rank-blend of the chunk transformer and the GBM
on sanitize-invariant features, with FPR calibration and batch guard."""
import pathlib

import joblib
import numpy as np
import torch

import os

from .calibration import batch_guard, batch_rank, rank01, rank_budget
from .features import chunk_features_v2
from .net import ChunkNet, predict, tokenize_hand

DEFAULT_ARTIFACTS = pathlib.Path(__file__).resolve().parent.parent / "artifacts"


class Detector:
    def __init__(self, artifacts_dir=DEFAULT_ARTIFACTS, device=None):
        artifacts_dir = pathlib.Path(artifacts_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        bundle = joblib.load(artifacts_dir / "bundle.joblib")
        self.gbm = bundle["gbm"]
        self.feature_names = bundle["feature_names"]
        self.calibrator = bundle["calibrator"]
        self.max_pos_frac = bundle.get("max_pos_frac", 0.25)
        self.batch_rank_gbm = bundle.get("batch_rank_gbm", False)
        self.nets = []
        for f in sorted(artifacts_dir.glob("transformer_s*.pt")):
            net = ChunkNet()
            net.load_state_dict(torch.load(f, map_location=self.device))
            net.to(self.device).eval()
            self.nets.append(net)
        if not self.nets:
            raise FileNotFoundError(f"no transformer_s*.pt in {artifacts_dir}")

    def _gbm_probs(self, chunks):
        rows = [chunk_features_v2(c) for c in chunks]
        X = np.array([[r.get(k, 0.0) for k in self.feature_names] for r in rows],
                     dtype=np.float32)
        if self.batch_rank_gbm and len(chunks) > 1:
            X = batch_rank(X)
        return self.gbm.predict_proba(X)[:, 1]

    def _net_probs(self, chunks):
        toks = [[tokenize_hand(h) for h in c] for c in chunks]
        ps = [predict(net, toks, self.device) for net in self.nets]
        return np.mean(ps, axis=0)

    def score_chunks(self, chunks):
        """One bot-risk score in [0,1] per chunk."""
        chunks = [c if c else [{}] for c in chunks]
        if len(chunks) == 1:
            # rank blending needs a batch; fall back to calibrated mean prob
            raw = (self._net_probs(chunks) + self._gbm_probs(chunks)) / 2
            return self.calibrator.transform(raw).tolist()
        # Rank-fusion of the two decorrelated members, then an exact rank budget:
        # exactly ~frac of the batch crosses 0.5 (rank-preserving). Immune to the
        # live-feed score compression that a benchmark-fit calibrator mishandles.
        blend = (rank01(self._net_probs(chunks)) + rank01(self._gbm_probs(chunks))) / 2
        frac = float(os.getenv("P44_POS_FRAC", "0.10"))
        scores = rank_budget(blend, frac=frac)
        return [float(round(s, 6)) for s in scores]
