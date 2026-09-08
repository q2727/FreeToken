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
_TOP4 = os.environ.get("FT_TOP4_METRICS", "0") == "1"
_STATE: dict = {"round": -1, "layers": {}, "batch": None}

# Metric-1 shadow rerun: while _SHADOW["active"] is set, note_router diverts into
# _SHADOW["layers"] (the rerun's full-space router sets A') and the original round
# capture in _STATE stays untouched for finish_verify_round.
_SHADOW: dict = {"active": False, "layers": {}, "failed_rows": None}

def shadow_layer_capture():
    return {lid: idx.clone() for lid, idx in _SHADOW["layers"].items()}

def write_top4_record(round_id: int, accepted_counts, top4_tokens, captures) -> None:
    if not _ACTIVE or _STATE["batch"] is None:
        return
    geo, k = _STATE["batch"], _STATE["batch"]["k"]
    top4_tokens = top4_tokens.reshape(geo["n_reqs"], k, 4)
    assert len(captures) == 4
    reqs = []
    for i, n_acc in enumerate(accepted_counts):
        items = []
        for j in range(n_acc, k):
            row = i * geo["span"] + j + 1
            candidates = []
            for rank, layers in enumerate(captures):
                sets = {str(lid): idx[row].tolist() for lid, idx in layers.items()}
                candidates.append({"rank": rank, "token_id": int(top4_tokens[i, j, rank]), "sets": sets})
            items.append({"slot": j, "position": int(geo["positions"][row]), "candidates": candidates})
        reqs.append({"req": geo["req_uids"][i], "n_acc": int(n_acc),
                     "top4_tokens": top4_tokens[i].tolist(), "rejected": items})
    out = Path(_DIR); out.mkdir(parents=True, exist_ok=True)
    with open(out / "top4_shadow.jsonl", "a") as f:
        f.write(json.dumps({"schema": 2, "round": round_id, "k": k,
                           "context": "rank_cohort", "reqs": reqs}) + "\n")


def write_tree_record(round_id: int, accepted_counts, tree_tokens, tree_sets) -> None:
    """Persist cache-only expert unions for the pruned top4 tree."""
    if not _ACTIVE or _STATE["batch"] is None:
        return
    geo, k = _STATE["batch"], _STATE["batch"]["k"]
    reqs = []
    for i, n_acc in enumerate(accepted_counts):
        rejected = []
        for j in range(n_acc, k):
            sets = {str(lid): sorted(values) for lid, values in tree_sets[i][j].items()}
            rejected.append({"slot": j, "token_id": int(tree_tokens[i][j]), "sets": sets})
        reqs.append({"req": geo["req_uids"][i], "n_acc": int(n_acc), "rejected": rejected})
    out = Path(_DIR); out.mkdir(parents=True, exist_ok=True)
    with open(out / "tree_shadow.jsonl", "a") as f:
        f.write(json.dumps({"schema": 3, "round": round_id, "k": k,
                            "context": "top4_tree_pruned", "reqs": reqs}) + "\n")


def top4_enabled() -> bool:
    return _TOP4 and _ACTIVE


def begin_decode_observation(batch):
    if top4_enabled() and not batch.speculative and batch.is_decode:
        _STATE["layers"] = {}
        _STATE["observing"] = True


def finish_decode_observation(batch):
    if not _STATE.get("observing"):
        return
    _STATE["observing"] = False
    records = []
    for i, req in enumerate(batch.reqs):
        records.append({"req": req.uid, "position": int(batch.positions.reshape(-1)[i]),
            "token_id": int(batch.input_ids.reshape(-1)[i]),
            "sets": {str(lid): idx[i].tolist() for lid, idx in _STATE["layers"].items()}})
    with open(Path(_DIR) / "actual_decode.jsonl", "a") as stream:
        stream.write(json.dumps({"schema": 2, "after_round": _STATE["round"], "tokens": records}) + "\n")


def active() -> bool:
    return _ACTIVE


def reset_round_capture() -> None:
    _STATE["layers"] = {}


def note_router(layer_id: int, indices: torch.Tensor) -> None:
    """Called from DSV4 MoE.forward with the router's topk ids [rows, topk]."""
    if not _ACTIVE:
        return
    if _SHADOW["active"]:
        if layer_id not in _SHADOW["layers"]:
            _SHADOW["layers"][layer_id] = indices.detach().to("cpu", non_blocking=False)
        return
    if _STATE["batch"] is None and not _STATE.get("observing"):
        return
    if layer_id in _STATE["layers"]:
        return
    _STATE["layers"][layer_id] = indices.detach().to("cpu", non_blocking=False)


def shadow_active() -> bool:
    return _ACTIVE and _SHADOW["active"]


def begin_shadow(failed_rows=None) -> None:
    _SHADOW["active"] = True
    _SHADOW["layers"] = {}
    _SHADOW["failed_rows"] = None if failed_rows is None else tuple(failed_rows)


def end_shadow() -> None:
    _SHADOW["active"] = False
    _SHADOW["failed_rows"] = None


def shadow_failed_rows():
    """Flat input rows whose routed output is the failed-token shadow."""
    return _SHADOW.get("failed_rows") if _SHADOW.get("active") else None


def current_round() -> int:
    """Id of the last finished round; the in-flight round is current_round() + 1."""
    return _STATE["round"]


def resident_mask(cache, layer_id: int, num_experts: int, top_k: int):
    """Bool [num_experts] mask of the experts currently resident in ``layer_id``'s
    GPU slot cache, or None when the layer cannot honor a cache-only restriction
    (no offload cache, or fewer resident experts than top_k).

    ``id_of_slot`` holds ``layer_id * num_experts + expert_id`` per slot (-1 = free)."""
    if cache is None or getattr(cache, "id_of_slot", None) is None:
        return None
    ids = cache.id_of_slot
    lo = layer_id * num_experts
    own = ids[(ids >= lo) & (ids < lo + num_experts)] - lo
    if own.numel() < top_k:
        return None
    mask = torch.zeros(num_experts, dtype=torch.bool, device=ids.device)
    mask[own.long()] = True
    return mask


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
        "positions": batch.positions.detach().cpu().reshape(-1).clone() if _TOP4 else None,
        "input_ids": batch.input_ids.detach().cpu().reshape(-1).clone() if _TOP4 else None,
        "top4_tokens": getattr(batch, "draft_top4", None),
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
                row = base + j + int(_TOP4)
                if row < idx.shape[0]:
                    sets[str(lid)] = idx[row].tolist()
            req_slots.append(
                {"slot": j, "token_id": tok, "accepted": j < n_acc, "sets": sets,
                 **({"position": int(geo["positions"][base + j + 1])} if _TOP4 else {})}
            )
        request_record = {"req": geo["req_uids"][i], "n_acc": n_acc, "slots": req_slots}
        if _TOP4:
            request_record["anchor"] = {"position": int(geo["positions"][base]),
                "token_id": int(geo["input_ids"][base]),
                "sets": {str(lid): idx[base].tolist() for lid, idx in layers.items()}}
            request_record["top4_tokens"] = geo["top4_tokens"][i].tolist()
        slots.append(request_record)

    rec = {
        "round": rnd,
        "ts": time.time(),
        "k": k,
        "reqs": slots,
        "prev": None if _TOP4 else _STATE.get("prev_round"),
    }
    if _TOP4:
        rec["schema"] = 2
    else:
        _metric2(rnd, rec)
    out = Path(_DIR)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rounds.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    if not _TOP4:
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


def write_shadow_record(round_id: int, accepted_counts) -> int:
    """Write the metric-1 shadow-rerun record for the in-flight round.

    Pairs, per rejected slot (``slot j >= n_acc``), using the actual draft input row
    ``base + j + 1`` (the anchor is row ``base``), the original round's full-space router
    set A (from the live _STATE capture, consumed later by finish_verify_round)
    against the shadow rerun's full-space set A' (_SHADOW["layers"], recorded while
    the MoE compute itself was restricted to cache-resident experts). Emits
    shadow.jsonl next to rounds.jsonl; returns the number of rejected slots."""
    if not _ACTIVE or _STATE["batch"] is None:
        return 0
    geo = _STATE["batch"]
    layers = _STATE["layers"]
    shadow = _SHADOW["layers"]
    k = geo["k"]
    reqs = []
    pairs = []
    n_rejected = 0
    for i in range(geo["n_reqs"]):
        n_acc = int(accepted_counts[i])
        base = i * geo["span"]
        slots = []
        for j in range(n_acc, k):
            n_rejected += 1
            # Verify input is [anchor, draft_0, ..., draft_{k-1}].  The normal
            # round record keeps legacy slot labels, but shadow rows must follow
            # the actual draft token input position.
            row = base + j + 1
            tok = (
                int(geo["draft_tokens"][i * k + j])
                if geo["draft_tokens"] is not None
                else None
            )
            sets = {}
            per_layer = {}
            for lid, idx in shadow.items():
                if row < idx.shape[0]:
                    sets[str(lid)] = idx[row].tolist()
            for lid, idx in layers.items():
                if lid in shadow and row < idx.shape[0] and row < shadow[lid].shape[0]:
                    per_layer[str(lid)] = _overlap(idx[row].tolist(), shadow[lid][row].tolist())
            slots.append({"slot": j, "token_id": tok, "sets": sets})
            pairs.append(
                {
                    "req": geo["req_uids"][i],
                    "slot": j,
                    "token_id": tok,
                    "overlap": per_layer,
                }
            )
        reqs.append({"req": geo["req_uids"][i], "n_acc": n_acc, "rejected": slots})
    rec = {
        "round": round_id,
        "ts": time.time(),
        "k": k,
        "reqs": reqs,
        "metric1_pairs": pairs,
    }
    out = Path(_DIR)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "shadow.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return n_rejected


def summary_shadow(rows) -> str:
    """Aggregate metric1_pairs from shadow.jsonl rows into per-layer means."""
    import collections

    acc = collections.defaultdict(lambda: [0.0, 0.0, 0])
    n_pairs = 0
    for r in rows:
        for p in r.get("metric1_pairs", []):
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
