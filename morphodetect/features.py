"""Sanitize-invariant chunk features (v2) for Poker44.

Own implementation of publicly known ideas: features are computed on hands
projected through the validator's sanitizer (train == serve), hero appears
only as a share, per-hand scalars are aggregated with order statistics so
30-hand and 105-hand chunks look alike. Adds cross-hand repeat signatures
and compression-based diversity metrics (strongest bot-tells).
"""
import json, math, pathlib, sys, zlib
from collections import Counter
import numpy as np

from poker44.validator.payload_view import prepare_hand_for_miner, _VISIBLE_BB_BUCKETS

BB = 0.02
BUCKETS = np.asarray(_VISIBLE_BB_BUCKETS, dtype=float)
ACT_TYPES = ["small_blind", "big_blind", "ante", "check", "call", "bet", "raise", "fold"]
STREETS = ["preflop", "flop", "turn", "river"]
ORDER_STATS = ("mean", "std", "min", "max", "q10", "q50", "q90")


def _order_stats(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return dict.fromkeys(ORDER_STATS, 0.0)
    return {
        "mean": float(v.mean()), "std": float(v.std()),
        "min": float(v.min()), "max": float(v.max()),
        "q10": float(np.quantile(v, 0.1)), "q50": float(np.quantile(v, 0.5)),
        "q90": float(np.quantile(v, 0.9)),
    }


def _entropy(counter, n):
    if n <= 0:
        return 0.0
    ps = np.array([c / n for c in counter.values() if c > 0])
    ent = float(-(ps * np.log(ps)).sum())
    return ent / math.log(max(len(counter), 2))


def _max_run_share(seq):
    if not seq:
        return 0.0
    best = run = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best / len(seq)


def _lz76(s):
    """Normalized Lempel-Ziv 76 phrase complexity of a token string."""
    n = len(s)
    if n < 2:
        return 0.0
    phrases, i, k = 1, 0, 1
    while i + k <= n:
        if s[i:i + k] in s[:i + k - 1]:
            k += 1
        else:
            phrases += 1
            i += k
            k = 1
    denom = n / max(math.log2(n), 1e-9)
    return phrases / max(denom, 1e-9)


def hand_view(hand):
    """Per-hand token sequences and scalars from a sanitized hand."""
    hand = prepare_hand_for_miner(hand)
    actions = hand.get("actions") or []
    md = hand.get("metadata") or {}
    hero = md.get("hero_seat")
    toks, rich, actors = [], [], []
    amounts, pots = [], []
    hero_actions = 0
    for a in actions:
        at = a.get("action_type") or "?"
        st = a.get("street") or "?"
        amt = float(a.get("normalized_amount_bb") or 0.0)
        ab = int(np.searchsorted(BUCKETS, amt, side="right"))
        toks.append(at[:2])
        rich.append(f"{st[:1]}{at[:2]}{ab}")
        actors.append(a.get("actor_seat"))
        amounts.append(ab)
        pots.append(float(a.get("pot_after") or 0.0) / BB)
        if hero is not None and a.get("actor_seat") == hero:
            hero_actions += 1
    n = len(actions)
    cnt = Counter(a.get("action_type") for a in actions)
    streets_seen = {a.get("street") for a in actions}
    out = hand.get("outcome") or {}
    scal = {
        "n_actions": float(n),
        "n_actors": float(len(set(actors))),
        "hero_share": hero_actions / max(n, 1),
        "action_entropy": _entropy(cnt, n),
        "actor_entropy": _entropy(Counter(actors), n),
        "act_run_share": _max_run_share(toks),
        "actor_run_share": _max_run_share(actors),
        "actor_switch": float(np.mean([a != b for a, b in zip(actors, actors[1:])])) if n > 1 else 0.0,
        "amt_bucket_mean": float(np.mean(amounts)) if amounts else 0.0,
        "amt_bucket_max": float(np.max(amounts)) if amounts else 0.0,
        "amt_nonzero": float(np.mean([a > 0 for a in amounts])) if amounts else 0.0,
        "pot_final_bb": min(pots[-1], 1000.0) if pots else 0.0,
        "pot_growth": (pots[-1] - pots[0]) if len(pots) > 1 else 0.0,
        "street_depth": len(streets_seen & set(STREETS)) / 4.0,
        "showdown": 1.0 if out.get("showdown") else 0.0,
        "n_players": float(len(hand.get("players") or [])),
    }
    for t in ACT_TYPES:
        scal[f"share_{t}"] = cnt.get(t, 0) / max(n, 1)
    for s in STREETS:
        scal[f"onstreet_{s}"] = sum(1 for a in actions if a.get("street") == s) / max(n, 1)
    return toks, rich, scal


def _hand_response(hand):
    """Reaction of each actor to facing a bet/raise, + per-hand aggression."""
    hand = prepare_hand_for_miner(hand)
    actions = hand.get("actions") or []
    faced_bet = False
    fold = call = raise_ = faced = aggr = passive = 0
    for a in actions:
        at = a.get("action_type")
        if faced_bet and at in ("fold", "call", "raise"):
            faced += 1
            if at == "fold": fold += 1
            elif at == "call": call += 1
            else: raise_ += 1
        if at in ("bet", "raise"):
            faced_bet = True; aggr += 1
        elif at in ("call", "check"):
            passive += 1
    f = max(faced, 1)
    return {"resp_fold": fold / f, "resp_call": call / f, "resp_raise": raise_ / f,
            "faced_aggr_rate": faced / max(len(actions), 1),
            "aggr_factor": aggr / max(passive, 1)}


def _norm_entropy(counter):
    n = sum(counter.values())
    if n <= 0:
        return 0.0
    ps = np.array([c / n for c in counter.values() if c > 0])
    return float(-(ps * np.log(ps)).sum()) / math.log(max(len(counter), 2))


def _lag1(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or np.std(x[:-1]) < 1e-9 or np.std(x[1:]) < 1e-9:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def chunk_features_v2(hands):
    views = [hand_view(h) for h in hands]
    feat = {}
    # order-stat aggregation of per-hand scalars
    keys = views[0][2].keys()
    for k in keys:
        vals = [v[2][k] for v in views]
        for sname, sval in _order_stats(vals).items():
            feat[f"{k}_{sname}"] = sval
    # cross-hand repeat signatures
    for name, idx in (("act", 0), ("rich", 1)):
        sigs = [",".join(map(str, v[idx])) for v in views]
        c = Counter(sigs)
        feat[f"sig_{name}_top"] = c.most_common(1)[0][1] / len(sigs)
        feat[f"sig_{name}_uniq"] = len(c) / len(sigs)
    # compression diversity
    hand_strs = [",".join(v[1]) for v in views]
    joined = "|".join(hand_strs).encode()
    per_hand = sum(len(zlib.compress(h.encode())) for h in hand_strs)
    feat["gzip_ratio"] = len(zlib.compress(joined)) / max(per_hand, 1)
    stream = "".join(hand_strs)
    feat["lz76"] = _lz76(stream[:4000])
    # conditional entropy H(tok_t | tok_{t-1}) over the action stream
    big = Counter(); uni = Counter()
    for v in views:
        t = v[0]
        uni.update(t)
        big.update(zip(t, t[1:]))
    tot = sum(big.values())
    hrate = 0.0
    if tot:
        for (a, b), c in big.items():
            p_ab = c / tot
            p_b_a = c / uni[a]
            hrate -= p_ab * math.log(max(p_b_a, 1e-12))
    feat["entropy_rate"] = hrate
    # pairwise jaccard of rich bigrams (sampled)
    rng = np.random.default_rng(0)
    bs = [set(zip(v[1], v[1][1:])) for v in views]
    pairs = min(60, len(bs) * (len(bs) - 1) // 2)
    if pairs:
        js = []
        for _ in range(pairs):
            i, j = rng.choice(len(bs), 2, replace=False)
            u = bs[i] | bs[j]
            js.append(len(bs[i] & bs[j]) / len(u) if u else 0.0)
        feat["jaccard_bigram"] = float(np.mean(js))
    else:
        feat["jaccard_bigram"] = 0.0
    feat["hand_count"] = float(len(hands))
    feat["hand_count_log"] = math.log1p(len(hands))

    # --- v3: richer discriminative signals for live 9-max ranking ---
    resp = [_hand_response(h) for h in hands]
    for k in ("resp_fold", "resp_call", "resp_raise", "faced_aggr_rate", "aggr_factor"):
        vv = np.array([r[k] for r in resp], dtype=float)
        feat[f"{k}_mean"] = float(vv.mean())
        feat[f"{k}_std"] = float(vv.std())
        feat[f"{k}_q90"] = float(np.quantile(vv, 0.9))
    tri = Counter()
    for v in views:
        tri.update(zip(v[0], v[0][1:], v[0][2:]))
    feat["uni_entropy"] = _norm_entropy(uni)
    feat["bigram_entropy"] = _norm_entropy(big)
    feat["trigram_entropy"] = _norm_entropy(tri)
    feat["bigram_top_share"] = (big.most_common(1)[0][1] / max(sum(big.values()), 1)) if big else 0.0
    feat["trigram_top_share"] = (tri.most_common(1)[0][1] / max(sum(tri.values()), 1)) if tri else 0.0
    feat["bigram_uniq_ratio"] = len(big) / max(sum(big.values()), 1)
    aggr_series = [r["aggr_factor"] for r in resp]
    nact_series = [v[2].get("n_actions", 0.0) for v in views]
    feat["aggr_lag1_autocorr"] = _lag1(aggr_series)
    feat["nact_lag1_autocorr"] = _lag1(nact_series)
    feat["aggr_cv"] = float(np.std(aggr_series) / (np.mean(aggr_series) + 1e-9))
    utils = []
    for h in hands:
        hh = prepare_hand_for_miner(h)
        md = hh.get("metadata") or {}
        ms = float(md.get("max_seats") or 0) or len(hh.get("players") or []) or 6
        utils.append(len(hh.get("players") or []) / max(ms, 1))
    feat["seat_util_mean"] = float(np.mean(utils))
    feat["seat_util_std"] = float(np.std(utils))
    return feat
