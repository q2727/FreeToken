"""Expert-overlap metrics for DSpark speculative decoding on FreeToken.

Enabled with FT_EXPERT_METRICS_DIR=<dir> and greedy sampling. Records, per
verify round and per MoE layer, the router's top-k expert sets for every
window slot, plus the accept/reject outcome of each slot. Emits two metrics:

Metric 2 (cross-round locality): for each rejected slot j of round t, find the
slot j of round t+1 (same position, same window) and, if that slot's proposal
was ACCEPTED, compute the per-layer overlap between the rejected token's
router-activated expert set and the accepted token's set.

Metric 1 (cache-only rerun): rejected tokens are re-run with MoE restricted to
shared + GPU-cache-resident experts; the normal full-space router selection is
recorded per layer and compared against the original (full-MoE) activations.
Wiring for the rerun lands in engine.py; this module stores the recorded sets.
"""
import json
import os
import time
from pathlib import Path

import torch

_DIR = os.environ.get("FT_EXPERT_METRICS_DIR", "")
_ACTIVE = bool(_DIR)
_STATE: dict = {"round": -1, "layers": {}, "batch": None}


def active() -> bool:
    return _ACTIVE


def reset_round_capture() -> None:
    _STATE["layers"] = {}


def note_router(layer_id: int, indices: torch.Tensor) -> None:
    """Called from DSV4 MoE.forward with the router's topk ids [rows, topk]."""
    if not _ACTIVE or _STATE["batch"] is None:
        return
    if layer_id in _STATE["layers"]:
        return
    _STATE["layers"][layer_id] = indices.detach().to("cpu", non_blocking=False)


def begin_verify_round(batch) -> None:
    """Capture the batch geometry before the verify forward."""
    if not _ACTIVE:
        return
    reset_round_capture()
    k = int(batch.spec_block)
    n = len(batch.reqs)
    _STATE["batch"] = {
        "k": k,
        "n_reqs": n,
        "span": 1 + k,
        "rows": (1 + k) * n,
        "draft_tokens": batch.draft_tokens.detach().to("cpu", non_blocking=False)
        if batch.draft_tokens is not None
        else None,
        "req_uids": [r.uid for r in batch.reqs],
    }


def finish_verify_round(batch, accepted_counts) -> None:
    """Called after the accept loop with per-request accepted counts."""
    if not _ACTIVE or _STATE["batch"] is None:
        return
    geo = _STATE["batch"]
    k = geo["k"]
    layers = dict(_STATE["layers"])
    _STATE["batch"] = None
    _STATE["round"] += 1
    rnd = _STATE["round"]

    slots = []  # per req: list of {slot, token_id, accepted, sets:{layer:[ids]}}
    for i in range(geo["n_reqs"]):
        n_acc = int(accepted_counts[i])
        base = i * geo["span"]
        req_slots = []
        for j in range(k):
            tok = (
                int(geo["draft_tokens"][i * k + j])
                if geo["draft_tokens"] is not None
                else None
            )
            sets = {}
            for lid, idx in layers.items():
                row = base + j
                if row < idx.shape[0]:
                    sets[str(lid)] = idx[row].tolist()
            req_slots.append(
                {"slot": j, "token_id": tok, "accepted": j < n_acc, "sets": sets}
            )
        slots.append({"req": geo["req_uids"][i], "n_acc": n_acc, "slots": req_slots})

    rec = {
        "round": rnd,
        "ts": time.time(),
        "k": k,
        "reqs": slots,
        "prev": _STATE.get("prev_round"),
    }
    _metric2(rnd, rec)
    out = Path(_DIR)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rounds.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    _STATE["prev_round"] = rec


def _overlap(a, b):
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    return {
        "inter": inter,
        "jaccard": round(inter / max(1, len(sa | sb)), 4),
        "frac_a": round(inter / max(1, len(sa)), 4),
    }


def _metric2(rnd, rec):
    """Match round r's rejected slot j with round r-1's ... no: the spec wants
    round t's rejected slot j vs round t+1's ACCEPTED slot j. Rounds arrive in
    order, so when finishing round t we compare against the PREVIOUS round's
    rejected slots (stored in rec['prev'])."""
    prev = rec.get("prev")
    if not prev:
        return
    pairs = []
    for cur, old in zip(rec["reqs"], prev["reqs"]):
        if cur["req"] != old["req"]:
            continue
        for j in range(min(rec["k"], prev["k"])):
            old_slot = old["slots"][j]
            cur_slot = cur["slots"][j]
            if old_slot["accepted"] or not cur_slot["accepted"]:
                continue
            per_layer = {}
            for lid in old_slot["sets"]:
                if lid in cur_slot["sets"]:
                    per_layer[lid] = _overlap(old_slot["sets"][lid], cur_slot["sets"][lid])
            pairs.append(
                {
                    "slot": j,
                    "rejected_token": old_slot["token_id"],
                    "accepted_token": cur_slot["token_id"],
                    "overlap": per_layer,
                }
            )
    if pairs:
        rec["metric2_pairs"] = pairs


def summary(rows) -> str:
    """Aggregate metric2_pairs from rounds.jsonl rows into per-layer means."""
    import collections

    acc = collections.defaultdict(lambda: [0.0, 0.0, 0])
    n_pairs = 0
    for r in rows:
        for p in r.get("metric2_pairs", []):
            n_pairs += 1
            for lid, ov in p["overlap"].items():
                a = acc[lid]
                a[0] += ov["inter"] / 6.0  # topk=6
                a[1] += ov["jaccard"]
                a[2] += 1
    lines = [f"pairs={n_pairs}"]
    for lid in sorted(acc, key=lambda x: int(x)):
        a = acc[lid]
        if a[2]:
            lines.append(
                f"layer {lid}: mean_inter_top6={a[0]/a[2]:.3f} mean_jaccard={a[1]/a[2]:.4f} n={a[2]}"
            )
    return "\n".join(lines)
