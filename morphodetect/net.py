"""Hierarchical chunk transformer: action tokens -> hand encoder -> chunk pooling."""
import math
import numpy as np
import torch
import torch.nn as nn

from poker44.validator.payload_view import prepare_hand_for_miner, _VISIBLE_BB_BUCKETS

BB = 0.02
N_STREET, N_ATYPE, N_HERO, N_POT, N_SEAT = 5, 9, 2, 12, 8
N_AMT = len(_VISIBLE_BB_BUCKETS) + 1
STREET_ID = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
ATYPE_ID = {"small_blind": 0, "big_blind": 1, "ante": 2, "check": 3, "call": 4,
            "bet": 5, "raise": 6, "fold": 7}
MAX_ACTIONS = 16
BUCKETS = np.asarray(_VISIBLE_BB_BUCKETS, dtype=float)


def tokenize_hand(hand):
    """Project a hand through the validator's sanitizer and emit action tokens."""
    hand = prepare_hand_for_miner(hand)
    md = hand.get("metadata") or {}
    hero = md.get("hero_seat")
    rows = []
    for a in (hand.get("actions") or [])[:MAX_ACTIONS]:
        street = STREET_ID.get(a.get("street"), 4)
        atype = ATYPE_ID.get(a.get("action_type"), 8)
        is_hero = 1 if a.get("actor_seat") == hero else 0
        amt = float(a.get("normalized_amount_bb") or 0.0)
        amt_b = int(np.searchsorted(BUCKETS, amt, side="right"))
        pot_bb = max(0.0, float(a.get("pot_before") or 0.0) / BB)
        pot_b = min(N_POT - 1, int(math.log2(pot_bb + 1.0)) + 1) if pot_bb > 0 else 0
        # Absolute seat is not transferable (live is 9-max, the validator
        # reseats players, and benchmark seat 7-9 embeddings are untrained) —
        # neutralize it and rely on action sequences + the hero flag.
        seat = 0
        rows.append((street, atype, is_hero, amt_b, pot_b, seat))
    if not rows:
        rows = [(4, 8, 0, 0, 0, 7)]
    return np.asarray(rows, dtype=np.int16)


class ChunkNet(nn.Module):
    def __init__(self, d=96, heads=4, drop=0.15):
        super().__init__()
        self.emb = nn.ModuleList([
            nn.Embedding(n, d) for n in (N_STREET, N_ATYPE, N_HERO, N_AMT, N_POT, N_SEAT)
        ])
        self.pos = nn.Embedding(MAX_ACTIONS, d)
        enc = lambda: nn.TransformerEncoderLayer(
            d, heads, dim_feedforward=2 * d, dropout=drop,
            batch_first=True, norm_first=True)
        self.hand_enc = nn.TransformerEncoder(enc(), num_layers=2)
        self.chunk_enc = nn.TransformerEncoder(enc(), num_layers=2)  # no pos emb => perm-invariant
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, 1))

    def forward(self, tok, act_mask, hand_mask):
        B, H, A, _ = tok.shape
        x = sum(e(tok[..., i]) for i, e in enumerate(self.emb))
        x = x + self.pos(torch.arange(A, device=tok.device))
        x = x.view(B * H, A, -1)
        am = act_mask.view(B * H, A)
        safe = am.clone(); safe[:, 0] = True
        x = self.hand_enc(x, src_key_padding_mask=~safe)
        x = (x * safe.unsqueeze(-1)).sum(1) / safe.sum(1, keepdim=True).clamp(min=1)
        x = x.view(B, H, -1)
        hm = hand_mask.clone(); hm[:, 0] = True
        x = self.chunk_enc(x, src_key_padding_mask=~hm)
        x = (x * hm.unsqueeze(-1)).sum(1) / hm.sum(1, keepdim=True).clamp(min=1)
        return self.head(x).squeeze(-1)


def collate(batch_chunks, device):
    H = max(len(c) for c in batch_chunks)
    A = MAX_ACTIONS
    B = len(batch_chunks)
    tok = torch.zeros(B, H, A, 6, dtype=torch.long)
    am = torch.zeros(B, H, A, dtype=torch.bool)
    hm = torch.zeros(B, H, dtype=torch.bool)
    for b, chunk in enumerate(batch_chunks):
        hm[b, :len(chunk)] = True
        for h, hand in enumerate(chunk):
            n = min(len(hand), A)
            tok[b, h, :n] = torch.from_numpy(hand[:n].astype(np.int64))
            am[b, h, :n] = True
    return tok.to(device), am.to(device), hm.to(device)


@torch.no_grad()
def predict(model, token_chunks, device, bs=16):
    model.eval()
    out = []
    for s in range(0, len(token_chunks), bs):
        tok, am, hm = collate(token_chunks[s:s + bs], device)
        out.append(torch.sigmoid(model(tok, am, hm)).float().cpu().numpy())
    return np.concatenate(out)
