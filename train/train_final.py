"""Train production artifacts.

Split: all release dates except the last two -> training; the last two dates
are held out for early stopping and FPR calibration of the ensemble.
Outputs to artifacts/: transformer_s{42,7}.pt + bundle.joblib.

Reproduce: python train/fetch_data.py && python train/train_final.py
"""
import json, os, pathlib, random, sys

import joblib
import numpy as np
import torch
import torch.nn as nn

# Cap CPU threads so a nightly CPU retrain doesn't saturate the box and starve
# the live serving miner sharing it (which must answer validators within the
# nonce window). Only applied when P44_TORCH_THREADS is set (the autopilot cron).
_torch_threads = os.getenv("P44_TORCH_THREADS")
if _torch_threads:
    torch.set_num_threads(int(_torch_threads))

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from morphodetect.calibration import FPRCalibrator, rank01
from morphodetect.features import chunk_features_v2
from morphodetect.net import ChunkNet, collate, predict, tokenize_hand

import os

DATA = pathlib.Path(__file__).parent / "data"
ART = ROOT / "artifacts"
# Seeds/epochs are overridable via env so the nightly autopilot can run a
# lighter single-seed pass on CPU-only boxes where the GPU is unavailable.
SEEDS = tuple(int(s) for s in os.getenv("P44_SEEDS", "42,7").split(",") if s.strip())
EPOCHS = int(os.getenv("P44_EPOCHS", "50"))
PATIENCE = int(os.getenv("P44_PATIENCE", "8"))
TARGET_FPR = 0.04
MAX_POS_FRAC = 0.25


def load_groups():
    groups, y, dates = [], [], []
    for f in sorted(DATA.glob("2*.json")):
        for rec in json.loads(f.read_text()):
            for group, label in zip(rec["chunks"], rec["groundTruth"]):
                if group:
                    groups.append(group)
                    y.append(int(label))
                    dates.append(rec["sourceDate"])
    return groups, np.array(y), np.array(dates)


def augment_chunk(chunk, pool, rng):
    hands = list(chunk)
    if pool and rng.random() < 0.5:
        for extra in rng.sample(pool, k=min(len(pool), rng.randint(1, 2))):
            hands.extend(extra)
    target = rng.randint(25, 105)
    if len(hands) > target:
        hands = rng.sample(hands, target)
    return hands


AUG_FRAC = float(os.getenv("P44_AUG_FRAC", "0.5"))  # domain-randomization fraction
_SHIFT_KEYS = ("normalized_amount_bb", "amount", "raise_to", "call_to",
               "pot_before", "pot_after")


def _live_shift_group(group, rng):
    """Domain randomization: a benchmark chunk warped toward the live regime with
    RANDOM magnitude (small pots/bets, more passivity). Teaches the model to
    discriminate across table regimes, not just at benchmark scale. Validated:
    stress-test retention 0.78 -> 0.92 at AUG_FRAC=0.5."""
    pot_mul = rng.uniform(0.05, 0.35)
    pass_p = rng.uniform(0.2, 0.6)
    out = []
    for hand in group:
        h = json.loads(json.dumps(hand))
        for a in h.get("actions") or []:
            for k in _SHIFT_KEYS:
                if a.get(k):
                    try:
                        a[k] = float(a[k]) * pot_mul
                    except (TypeError, ValueError):
                        pass
            if a.get("action_type") in ("bet", "raise") and rng.random() < pass_p:
                a["action_type"] = "call" if rng.random() < 0.6 else "check"
        out.append(h)
    return out


def train_net(tokens, y, tr, va, seed, device, epochs=EPOCHS, patience=PATIENCE, bs=24):
    from sklearn.metrics import average_precision_score
    torch.manual_seed(seed); rng = random.Random(seed)
    model = ChunkNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.BCEWithLogitsLoss()
    by_label = {0: [], 1: []}
    for i in tr:
        by_label[int(y[i])].append(tokens[i])
    best_ap, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        idx = list(tr); rng.shuffle(idx)
        for s in range(0, len(idx), bs):
            sel = idx[s:s + bs]
            bc = [augment_chunk(tokens[i], by_label[int(y[i])], rng) for i in sel]
            tok, am, hm = collate(bc, device)
            yy = torch.tensor([float(y[i]) for i in sel], device=device)
            opt.zero_grad()
            loss = lossf(model(tok, am, hm), yy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        vp = predict(model, [tokens[i] for i in va], device)
        ap = average_precision_score(y[va], vp)
        print(f"  seed {seed} ep {ep}: val AP {ap:.4f}")
        if ap > best_ap + 1e-4:
            best_ap, bad = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_ap


def build_artifacts(out_dir=ART, exclude_dates=()):
    """Train the full production bundle into out_dir.

    exclude_dates: release dates dropped entirely (used by the autopilot to
    hold out the newest date for a head-to-head guard evaluation).
    Returns metrics dict."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    groups, y, dates = load_groups()
    keep = ~np.isin(dates, list(exclude_dates))
    groups = [g for g, k in zip(groups, keep) if k]
    y, dates = y[keep], dates[keep]
    udates = sorted(set(dates.tolist()))
    cal_dates = udates[-2:]

    # Domain randomization: add live-shifted copies of TRAINING chunks (tagged
    # "<date>_aug" so batch-rank groups them among fellow shifted chunks, as a
    # live batch would be). Calibration dates are left untouched.
    a_rng = random.Random(20260717)
    aug_g, aug_y, aug_d = [], [], []
    for g, lbl, d in zip(groups, y.tolist(), dates.tolist()):
        aug_g.append(g); aug_y.append(lbl); aug_d.append(d)
        if AUG_FRAC > 0 and d < cal_dates[0] and a_rng.random() < AUG_FRAC:
            aug_g.append(_live_shift_group(g, a_rng)); aug_y.append(lbl)
            aug_d.append(f"{d}_aug")
    groups, y, dates = aug_g, np.array(aug_y), np.array(aug_d)
    va = np.flatnonzero(np.isin(dates, cal_dates))
    tr = np.flatnonzero(~np.isin(dates, cal_dates))
    print(f"{len(groups)} chunks ({int((dates.astype(str)=='').sum())} n/a) | "
          f"train {len(tr)} (incl. {sum(1 for d in dates if str(d).endswith('_aug'))} aug) | "
          f"calibration {len(va)} ({cal_dates})"
          + (f" | excluded {sorted(exclude_dates)}" if exclude_dates else ""))

    print("tokenizing…")
    tokens = [[tokenize_hand(h) for h in g] for g in groups]
    print("featurizing…")
    from morphodetect.calibration import batch_rank
    rows = [chunk_features_v2(g) for g in groups]
    feature_names = sorted(rows[0].keys())
    X = np.array([[r.get(k, 0.0) for k in feature_names] for r in rows], dtype=np.float32)
    # OOD-robust: rank features WITHIN each date-batch (mirrors serving, which
    # ranks within the validator's request batch) so absolute scale/mean shifts
    # between benchmark and live don't matter.
    Xr = np.empty_like(X, dtype=np.float64)
    for d in np.unique(dates):
        m = dates == d
        Xr[m] = batch_rank(X[m])
    X = Xr.astype(np.float32)

    net_val = []
    for seed in SEEDS:
        model, vap = train_net(tokens, y, tr, va, seed, device)
        torch.save(model.state_dict(), out_dir / f"transformer_s{seed}.pt")
        net_val.append(predict(model, [tokens[i] for i in va], device))
        print(f"seed {seed}: best val AP {vap:.4f}")
    net_val = np.mean(net_val, axis=0)

    from sklearn.ensemble import HistGradientBoostingClassifier
    gbm = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_depth=6,
                                         l2_regularization=1.0, random_state=42)
    gbm.fit(X[tr], y[tr])
    gbm_val = gbm.predict_proba(X[va])[:, 1]

    blend_val = (rank01(net_val) + rank01(gbm_val)) / 2
    calibrator = FPRCalibrator(TARGET_FPR).fit(blend_val, y[va])
    from sklearn.metrics import average_precision_score
    cal_ap = float(average_precision_score(y[va], blend_val))
    joblib.dump({
        "gbm": gbm,
        "feature_names": feature_names,
        "calibrator": calibrator,
        "max_pos_frac": MAX_POS_FRAC,
        "batch_rank_gbm": True,
        "cal_dates": cal_dates,
        "train_dates": [d for d in udates if d < cal_dates[0]],
        "cal_ap": cal_ap,
    }, out_dir / "bundle.joblib")

    print(f"calibration-set blend AP: {cal_ap:.4f}")
    print(f"artifacts -> {out_dir}")
    return {"cal_ap": cal_ap, "cal_dates": cal_dates, "latest_date": udates[-1]}


if __name__ == "__main__":
    build_artifacts()
