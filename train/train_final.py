"""Train production artifacts: a heterogeneous TREE-STACK on within-batch
rank-normalized, sanitize-invariant features, with domain-randomization
augmentation. No neural net (the 6-max transformer hurt live transfer).

Reproduce: python train/fetch_data.py && python train/train_final.py
Outputs artifacts/bundle.joblib.
"""
import json, os, pathlib, random, sys

import joblib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from morphodetect.calibration import batch_rank
from morphodetect.features import chunk_features_v2

DATA = pathlib.Path(__file__).parent / "data"
ART = ROOT / "artifacts"
AUG_FRAC = float(os.getenv("P44_AUG_FRAC", "0.5"))   # domain-randomization fraction
AUG9_FRAC = float(os.getenv("P44_AUG9_FRAC", "0.4"))  # synthetic 9-max aug fraction
POS_FRAC = float(os.getenv("P44_POS_FRAC", "0.10"))  # rank-budget positive fraction
_SHIFT_KEYS = ("normalized_amount_bb", "amount", "raise_to", "call_to",
               "pot_before", "pot_after")


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


def _live_shift_group(group, rng):
    """Domain randomization: warp a benchmark chunk toward the live regime with
    RANDOM magnitude (small pots/bets, more passivity)."""
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


def _to_9max_hand(hand, rng, target_seats=9):
    """Grow a 6-max hand to ~9 seats and inject the extra players' early preflop
    folds — adds the live 9-max STRUCTURE (measured n_players 6->9, more early
    folds) without touching the real players' bot/human behavior. Structural
    features (n_players, more actors) survive the sanitizer's action window."""
    import numpy as _np
    h = json.loads(json.dumps(hand))
    md = h.setdefault("metadata", {})
    players = h.get("players") or []
    base_seats = sorted({p.get("seat") for p in players if p.get("seat") is not None})
    if not base_seats:
        return h
    stacks = [p.get("starting_stack") for p in players if p.get("starting_stack")]
    med = float(_np.median(stacks)) if stacks else 5.0
    new_seats, s = [], max(base_seats) + 1
    for _ in range(max(0, target_seats - len(players))):
        players.append({"player_uid": f"seat_{s}", "seat": s,
                        "starting_stack": med * rng.uniform(0.7, 1.3),
                        "hole_cards": None, "showed_hand": False})
        new_seats.append(s); s += 1
    h["players"] = players
    md["max_seats"] = max(int(md.get("max_seats") or len(base_seats)), target_seats)
    actions = h.get("actions") or []
    first_pot = next((float(a.get("pot_before") or 0.0) for a in actions
                      if a.get("pot_before") is not None), 0.0)
    inj = []
    for i, seat in enumerate(new_seats):
        at = "fold" if rng.random() < 0.8 else ("call" if rng.random() < 0.6 else "check")
        inj.append({"action_id": f"x{i}", "street": "preflop", "actor_seat": seat,
                    "action_type": at, "amount": 0.0 if at != "call" else 0.02,
                    "raise_to": None, "call_to": None,
                    "normalized_amount_bb": 0.0 if at != "call" else 1.0,
                    "pot_before": first_pot, "pot_after": first_pot})
    h["actions"] = inj + actions
    return h


def _live_9max_group(group, rng):
    """Live-shift then grow to 9-max — the synthetic 9-max augmentation
    (validated: +0.042 STRESS9 on the stack, no bench/plain-stress regression)."""
    return [_to_9max_hand(hand, rng) for hand in _live_shift_group(group, rng)]


def build_stack():
    """Heterogeneous tree-stack (LGBM/XGB/CatBoost/ExtraTrees/RF -> logistic meta),
    the architecture the top miners use. Skips any lib that isn't installed."""
    from sklearn.ensemble import (HistGradientBoostingClassifier as HGB,
                                  ExtraTreesClassifier, RandomForestClassifier,
                                  StackingClassifier)
    from sklearn.linear_model import LogisticRegression
    ests = [
        ("hgb", HGB(max_iter=400, learning_rate=0.06, max_depth=6,
                    l2_regularization=1.0, random_state=1)),
        ("et", ExtraTreesClassifier(n_estimators=400, max_depth=14, n_jobs=-1, random_state=2)),
        ("rf", RandomForestClassifier(n_estimators=400, max_depth=16, n_jobs=-1, random_state=3)),
    ]
    try:
        from lightgbm import LGBMClassifier
        ests.append(("lgb", LGBMClassifier(n_estimators=400, num_leaves=101, learning_rate=0.0221,
                                           subsample=0.697, colsample_bytree=0.529,
                                           min_child_samples=34, reg_lambda=0.00824,
                                           random_state=4, verbose=-1)))  # Optuna-tuned on stress
    except Exception:
        pass
    try:
        from xgboost import XGBClassifier
        ests.append(("xgb", XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                          subsample=0.85, colsample_bytree=0.8, random_state=5,
                                          eval_metric="logloss", verbosity=0)))
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier
        ests.append(("cat", CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                                               random_seed=6, verbose=0)))
    except Exception:
        pass
    return StackingClassifier(estimators=ests,
                              final_estimator=LogisticRegression(max_iter=1000),
                              cv=3, n_jobs=-1, stack_method="predict_proba")


def build_artifacts(out_dir=ART, exclude_dates=()):
    """Train the production bundle into out_dir. Returns metrics dict."""
    from sklearn.metrics import average_precision_score
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    groups, y, dates = load_groups()
    keep = ~np.isin(dates, list(exclude_dates))
    groups = [g for g, k in zip(groups, keep) if k]
    y, dates = y[keep], dates[keep]
    udates = sorted(set(dates.tolist()))
    cal_dates = udates[-2:]

    # Domain randomization: add live-shifted copies of TRAINING chunks, tagged
    # "<date>_aug" so batch-rank groups them among fellow shifted chunks.
    a_rng = random.Random(20260717)
    aug_g, aug_y, aug_d = [], [], []
    for g, lbl, d in zip(groups, y.tolist(), dates.tolist()):
        aug_g.append(g); aug_y.append(lbl); aug_d.append(d)
        if AUG_FRAC > 0 and d < cal_dates[0] and a_rng.random() < AUG_FRAC:
            aug_g.append(_live_shift_group(g, a_rng)); aug_y.append(lbl); aug_d.append(f"{d}_aug")
        # synthetic 9-max aug: warp 6-max training chunks to the live 9-max regime
        if AUG9_FRAC > 0 and d < cal_dates[0] and a_rng.random() < AUG9_FRAC:
            aug_g.append(_live_9max_group(g, a_rng)); aug_y.append(lbl); aug_d.append(f"{d}_9m")
    groups, y, dates = aug_g, np.array(aug_y), np.array(aug_d)
    tr = np.flatnonzero(~np.isin(dates, cal_dates))
    va = np.flatnonzero(np.isin(dates, cal_dates))
    n_aug = sum(1 for d in dates if str(d).endswith("_aug"))
    n_9m = sum(1 for d in dates if str(d).endswith("_9m"))
    print(f"{len(groups)} chunks | train {len(tr)} (incl. {n_aug} live-aug, {n_9m} 9max-aug) | "
          f"calibration {len(va)} ({cal_dates})"
          + (f" | excluded {sorted(exclude_dates)}" if exclude_dates else ""))

    print("featurizing…")
    rows = [chunk_features_v2(g) for g in groups]
    feature_names = sorted(rows[0].keys())
    X = np.array([[r.get(k, 0.0) for k in feature_names] for r in rows], dtype=np.float32)
    Xr = np.empty_like(X, dtype=np.float64)
    for d in np.unique(dates):
        m = dates == d
        Xr[m] = batch_rank(X[m])
    X = Xr.astype(np.float32)

    print("training tree-stack…")
    model = build_stack().fit(X[tr], y[tr])
    val_p = model.predict_proba(X[va])[:, 1]
    cal_ap = float(average_precision_score(y[va], val_p))

    joblib.dump({
        "model": model,
        "feature_names": feature_names,
        "batch_rank_gbm": True,
        "pos_frac": POS_FRAC,
        "cal_dates": cal_dates,
        "train_dates": [d for d in udates if d < cal_dates[0]],
        "cal_ap": cal_ap,
    }, out_dir / "bundle.joblib")
    print(f"calibration-set AP: {cal_ap:.4f}")
    print(f"artifacts -> {out_dir}")
    return {"cal_ap": cal_ap, "cal_dates": cal_dates, "latest_date": udates[-1]}


if __name__ == "__main__":
    build_artifacts()
