"""Production Poker44 miner serving the morpho-poker-detector ensemble."""
import hashlib
import pathlib
import sys
import time
from typing import Tuple

import bittensor as bt

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

from morphodetect.serving import Detector

MODEL_NAME = "morpho-poker-detector"
MODEL_VERSION = "1.0"
IMPLEMENTATION_FILES = [
    REPO_ROOT / "morphodetect" / "features.py",
    REPO_ROOT / "morphodetect" / "net.py",
    REPO_ROOT / "morphodetect" / "calibration.py",
    REPO_ROOT / "morphodetect" / "serving.py",
    pathlib.Path(__file__).resolve(),
]


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.detector = Detector()
        bt.logging.info(f"Detector loaded on {self.detector.device} "
                        f"({len(self.detector.nets)} transformer seeds + GBM)")
        self.model_manifest = build_local_model_manifest(
            repo_root=REPO_ROOT,
            implementation_files=IMPLEMENTATION_FILES,
            defaults={
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "framework": "pytorch+sklearn",
                "license": "MIT",
                "repo_url": "https://github.com/morphozhub/morpho-poker-detector",
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained exclusively on the public Poker44 benchmark API "
                    "(api.poker44.net/api/v1/benchmark); release hashes in "
                    "data_attestation.json. Hands are projected through the "
                    "validator's own sanitizer before feature extraction (train==serve)."
                ),
                "training_data_sources": ["poker44-public-benchmark"],
                "private_data_attestation": (
                    "No validator-only evaluation data, no live labels and no other "
                    "miners' artifacts were used for training or calibration."
                ),
                "data_attestation": (
                    "data_attestation.json (sha256 per benchmark release) in the repo root."
                ),
                "notes": (
                    "Rank-blend of a hierarchical action-sequence transformer (2 seeds) "
                    "and a HistGradientBoosting model on 206 sanitize-invariant chunk "
                    "features, with monotonic FPR calibration and a rank-preserving "
                    "batch positive budget. Full training code in train/."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        bt.logging.info(
            f"Manifest status: {self.manifest_compliance['status']} "
            f"(missing={self.manifest_compliance['missing_fields']}) "
            f"digest={manifest_digest(self.model_manifest)}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        started = time.monotonic()
        try:
            scores = self.detector.score_chunks(chunks)
        except Exception as exc:  # never return a length-mismatched response
            bt.logging.error(f"Scoring failed ({exc}); falling back to neutral scores.")
            scores = [0.5] * len(chunks)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(
            f"Scored {len(chunks)} chunks "
            f"({sum(len(c) for c in chunks)} hands) in {time.monotonic() - started:.2f}s"
        )
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("morpho-poker-detector miner running")
        while True:
            bt.logging.info(
                f"UID {miner.uid} | incentive {miner.metagraph.I[miner.uid]:.6f}"
            )
            time.sleep(300)
