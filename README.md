# morpho-poker-detector

Poker bot detection miner for Bittensor subnet 126 ([Poker44](https://poker44.net)).

One bot-risk score per chunk of poker hands, produced by a rank-blend of two
decorrelated models plus a calibration layer:

1. **Hierarchical action-sequence transformer** (`morphodetect/net.py`, 2 seeds averaged):
   action tokens (street, action type, hero flag, validator-grid size bucket, pot bucket,
   seat) → per-hand encoder → permutation-invariant chunk pooling → logit.
2. **HistGradientBoosting** on **206 sanitize-invariant chunk features**
   (`morphodetect/features.py`): order-statistic aggregation of per-hand behavioral
   scalars, cross-hand repeat signatures, compression diversity (gzip ratio, LZ76,
   action transition entropy, bigram Jaccard). Hero appears only as a share.
3. **Calibration** (`morphodetect/calibration.py`): isotonic regression + monotonic
   piecewise-linear remap that puts the target-FPR operating point exactly at 0.5,
   and a rank-preserving batch positive budget (≥1, ≤25% of scores ≥ 0.5).

Key design decision: **train == serve** — every hand is projected through the
validator's own sanitizer (`poker44.validator.payload_view.prepare_hand_for_miner`)
before feature extraction and tokenization, and training chunks are augmented by
same-date/label pooling to live sizes (25–105 hands).

## Reproduce

```bash
pip install -r requirements.txt
python train/fetch_data.py     # public benchmark API, writes data_attestation.json
python train/train_final.py    # -> artifacts/ (last 2 release dates held out for calibration)
```

Training data: exclusively the public Poker44 benchmark
(`https://api.poker44.net/api/v1/benchmark`). Per-release SHA-256 hashes are published
in [`data_attestation.json`](data_attestation.json). No validator-private data, no live
labels, no third-party miner artifacts.

Walk-forward validation (train < calibration dates < test date, native subnet
`reward()`): mean reward **0.952** over the last 5 release dates
(AP 0.959–0.983, hard FPR@0.5 ≤ 0.028).

## Run the miner

```bash
WALLET_NAME=my_cold HOTKEY=my_hot AXON_PORT=8091 ./scripts/run_miner.sh
```

## Model artifacts

Weights are reproducible from the public data via `train/train_final.py`
(fixed seeds 42/7). SHA-256 of the served artifacts is published in the
`model_manifest` returned with every response.

## License

MIT
