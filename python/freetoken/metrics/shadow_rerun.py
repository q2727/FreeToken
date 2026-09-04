"""Metric 1: cache-only shadow rerun of the rejected verify tokens.

After a DSpark verify forward (whose router sets A are captured by
``expert_overlap``) but BEFORE acceptance mutates any request/pool state, this
module re-runs the SAME verify forward with every MoE layer's routed-expert
selection restricted to the experts resident in that layer's GPU slot cache
(see ``Gate.forward(restrict=...)`` and ``expert_overlap.resident_mask``). The
router itself still chooses from the full expert space; those full-space sets
A' are recorded per layer and paired against A per rejected slot.

The rerun is not state-free: attention rewrites the window-KV rows, the
compressor/indexer carry rings, and the compressed pools of the block's
positions, and the carry journal would grow a second set of rows. Everything
the rerun touches is snapshotted beforehand and restored in a ``finally``, so
the subsequent accept loop / ``_restore_speculative_carry`` observe exactly the
state the original verify left behind.

Greedy only: acceptance is recomputed here from the same argmax tensors the
accept loop consumes (``accepted_prefix`` over ``proposed``/``target``), which
is only defined for the greedy branch. Sampled requests skip the rerun.

Placement note: the rerun MUST happen before the accept loop, which calls
``release_tail`` (freeing the rejected positions' pages) and truncates the
requests -- either would invalidate the slot maps the rerun's attention reads.
"""

from __future__ import annotations

import torch

from freetoken.core import get_global_ctx
from freetoken.metrics import expert_overlap as eo
from freetoken.utils import init_logger

logger = init_logger(__name__)


def maybe_run_shadow(engine, batch, target_cpu, proposed_cpu) -> None:
    """Run the cache-only shadow verify for ``batch`` if it has rejected slots."""
    if not eo.active() or not getattr(batch, "speculative", False):
        return
    if any(not req.sampling_params.is_greedy for req in batch.reqs):
        return  # acceptance below mirrors the greedy branch only
    if target_cpu is None or proposed_cpu is None:
        return
    try:
        accepted_counts = _greedy_accepted_counts(batch, target_cpu, proposed_cpu)
        k = int(batch.spec_block)
        if all(n_acc >= k for n_acc in accepted_counts):
            return  # nothing rejected this round
        _run(engine, batch, accepted_counts)
    except Exception:
        # Measurement must never take down the serving path. Pool state is
        # restored in _run's finally, so a failure here leaves a lost shadow
        # record, not corruption.
        logger.warning("metric-1 shadow rerun failed; skipping this round", exc_info=True)


def _greedy_accepted_counts(batch, target_cpu, proposed_cpu) -> list[int]:
    """Mirror of the accept loop's greedy arithmetic (no state mutation)."""
    from freetoken.models.deepseek_v4.dspark import accepted_prefix

    k = int(batch.spec_block)
    counts = []
    off = 0
    for i, req in enumerate(batch.reqs):
        span = 1 + k
        start = req.input_ids.numel() - k
        budget = req.max_device_len - start - 1
        width = max(0, min(k, budget))
        proposed = proposed_cpu[i * k:(i + 1) * k]
        n_acc, _bonus = accepted_prefix(proposed[:width], target_cpu[off:off + width + 1])
        counts.append(n_acc)
        off += span
    return counts


def _run(engine, batch, accepted_counts) -> None:
    snapshot = _snapshot_pools(engine, batch)
    saved_journal = batch.spec_carry_states
    batch.spec_carry_states = {}
    round_id = eo.current_round() + 1  # finish_verify_round assigns this id next
    eo.begin_shadow()
    try:
        with engine.ctx.forward_batch(batch):
            engine.model.forward()  # logits discarded; only router sets are kept
    finally:
        eo.end_shadow()
        batch.spec_carry_states = saved_journal
        _restore_pools(engine, batch, snapshot)
    n_rejected = eo.write_shadow_record(round_id, accepted_counts)
    logger.info("metric-1 shadow rerun: round=%d rejected_slots=%d", round_id, n_rejected)


# ---------------------------------------------------------------------------
# Pool snapshot / restore
# ---------------------------------------------------------------------------


def _snapshot_pools(engine, batch):
    """Clone every pool row the rerun can overwrite, addressed exactly the way
    the model's prefill path addresses it (live ``full_loc_map`` per segment)."""
    backend = get_global_ctx().attn_backend
    pool = get_global_ctx().kv_cache
    segments = batch.attn_metadata.segments
    slot_lists = [
        backend.window_slots_of(ti, start, start + n) for (_off, n, ti, start) in segments
    ]
    slots = torch.cat(slot_lists) if slot_lists else None
    valid = slots[slots >= 0] if slots is not None else slots
    snap = {"valid": valid, "window": [], "rings": [], "cmp": []}
    if valid is None or valid.numel() == 0:
        return snap
    transformer = getattr(engine.model, "_transformer", None)
    layers = getattr(transformer, "layers", None) if transformer is not None else None
    if not layers:
        return snap
    for layer in layers:
        attn = layer.attn
        lid = attn.layer_id
        snap["window"].append((lid, pool.window_pool[lid][valid].clone()))
        compressor = getattr(attn, "compressor", None)
        if compressor is None:
            continue
        ratio = attn.compress_ratio
        rows = _completed_cmp_rows(pool, segments, ratio)
        # The per-page ring blocks at the block's window slots (attn + idx tiers).
        blocks = backend.read_carry_blocks(lid, "attn", valid, compressor.ring_size).clone()
        snap["rings"].append((lid, "attn", compressor.ring_size, blocks))
        indexer = getattr(attn, "indexer", None)
        if indexer is not None:
            icomp = indexer.compressor
            blocks = backend.read_carry_blocks(lid, "idx", valid, icomp.ring_size).clone()
            snap["rings"].append((lid, "idx", icomp.ring_size, blocks))
        if rows is not None and rows.numel():
            snap["cmp"].append(("attn", lid, rows, pool.cmp_pool[lid][rows].clone()))
            if indexer is not None and pool.idx_pool[lid] is not None:
                snap["cmp"].append(("idx", lid, rows, pool.idx_pool[lid][rows].clone()))
    return snap


def _restore_pools(engine, batch, snap) -> None:
    backend = get_global_ctx().attn_backend
    pool = get_global_ctx().kv_cache
    valid = snap["valid"]
    if valid is None or valid.numel() == 0:
        return
    for lid, data in snap["window"]:
        pool.window_pool[lid].index_copy_(0, valid, data)
    for lid, tier, ring_size, blocks in snap["rings"]:
        # ``valid`` may repeat a page within one window page; every repeated
        # block carries the same saved bytes, so the scatter is unambiguous.
        backend.write_carry_blocks(lid, tier, valid, ring_size, blocks)
    for tier, lid, rows, data in snap["cmp"]:
        p = pool.cmp_pool[lid] if tier == "attn" else pool.idx_pool[lid]
        p.index_copy_(0, rows, data)


def _completed_cmp_rows(pool, segments, ratio: int):
    """Compressed-pool rows the block's tokens can complete: positions p in the
    segment with (p + 1) % ratio == 0, at the arithmetic row full_loc(p) // ratio."""
    rows = []
    for (_off, n, ti, start) in segments:
        pos = [p for p in range(start, start + n) if (p + 1) % ratio == 0]
        if pos:
            t = torch.tensor(pos, dtype=torch.int64, device=pool.full_loc_map.device)
            rows.append(pool.cmp_rows(pool.full_loc_map[ti, t], ratio))
    return torch.cat(rows) if rows else None
