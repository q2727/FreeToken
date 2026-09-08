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

import os
import torch
import traceback

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
        if eo.top4_enabled():
            if os.environ.get("FT_DSPARK_TREE_METRICS", "0") == "1":
                _run_tree(engine, batch, accepted_counts, batch.draft_top4)
            else:
                _run_top4(engine, batch, accepted_counts, batch.draft_top4)
        else:
            _run(engine, batch, accepted_counts)
    except Exception:
        # Measurement must never take down the serving path. Pool state is
        # restored in _run's finally, so a failure here leaves a lost shadow
        # record, not corruption.
        logger.warning("metric-1 shadow rerun failed; skipping this round", exc_info=True)
        if eo.top4_enabled():
            logger.error("top4 measurement failed: %s", traceback.format_exc())
            raise  # A failed measurement must not silently enter the result set.


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
    failed_rows = [i * (int(batch.spec_block) + 1) + j + 1
                   for i, n in enumerate(accepted_counts)
                   for j in range(n, int(batch.spec_block))]
    eo.begin_shadow(failed_rows)
    try:
        with engine.ctx.forward_batch(batch):
            engine.model.forward()  # logits discarded; only router sets are kept
    finally:
        eo.end_shadow()
        batch.spec_carry_states = saved_journal
        _restore_pools(engine, batch, snapshot)
    n_rejected = eo.write_shadow_record(round_id, accepted_counts)
    logger.info("metric-1 shadow rerun: round=%d rejected_slots=%d", round_id, n_rejected)

def _run_capture(engine, batch, failed_rows=None):
    snapshot = _snapshot_pools(engine, batch)
    saved_journal = batch.spec_carry_states
    transformer = engine.model._transformer
    saved_features = transformer._target_features
    batch.spec_carry_states = {}
    eo.begin_shadow(failed_rows)
    try:
        with engine.ctx.forward_batch(batch):
            engine.model.forward()
        return eo.shadow_layer_capture()
    finally:
        eo.end_shadow()
        batch.spec_carry_states = saved_journal
        transformer._target_features = saved_features
        _restore_pools(engine, batch, snapshot)

def _run_top4(engine, batch, accepted_counts, top4):
    if top4 is None:
        raise RuntimeError("missing top4 draft candidates")
    top4 = top4.reshape(-1, 4)
    original = batch.input_ids.clone()
    captures = []
    round_id = eo.current_round() + 1
    k = int(batch.spec_block); span = k + 1
    assert k == 8 and top4.shape == (k * len(batch.reqs), 4)
    assert torch.equal(top4[:, 0], batch.draft_tokens.detach().cpu().reshape(-1))
    caches = {id(layer.ffn.experts.offload_cache): layer.ffn.experts.offload_cache
              for layer in engine.model._transformer.layers}
    cache_before = [(c, {name: getattr(c, name).clone() for name in
                         ("slot_for_id", "id_of_slot", "usage", "step", "expert_recency")})
                    for c in caches.values()]
    try:
        for rank in range(4):
            batch.input_ids.copy_(original)
            for i, n_acc in enumerate(accepted_counts):
                for j in range(n_acc, k):
                    row = i * span + j + 1
                    batch.input_ids.reshape(-1)[row] = int(top4[i * k + j, rank])
            captures.append(_run_capture(engine, batch))
        for cache, old in cache_before:
            assert all(torch.equal(getattr(cache, name), val) for name, val in old.items()), "shadow changed cache state"
    finally:
        batch.input_ids.copy_(original)
    eo.write_top4_record(round_id, accepted_counts, top4, captures)


def _run_tree(engine, batch, accepted_counts, top4):
    """Verify pruned top4-tree leaf paths and union each depth's router sets."""
    if top4 is None:
        raise RuntimeError("missing top4 draft candidates")
    from freetoken.metrics.draft_tree_mask import compile_tree, leaf_paths
    top4 = top4.reshape(len(batch.reqs), int(batch.spec_block), 4)
    original = batch.input_ids.clone()
    k, span = int(batch.spec_block), int(batch.spec_block) + 1
    round_id = eo.current_round() + 1
    budget = max(1, int(os.environ.get("FT_DSPARK_TREE_BUDGET", "16")))
    tokens, unions = [], []
    try:
        for i, n_acc in enumerate(accepted_counts):
            rows = [dict() for _ in range(k)]
            seen_nodes = set()
            token_rows = [int(top4[i, j, 0]) for j in range(k)]
            for j in range(n_acc, k):
                token_rows[j] = int(top4[i, j, 0])
            tree = compile_tree(
                torch.empty(0, dtype=torch.long), top4[i, n_acc:].cpu(),
                torch.arange(4, 0, -1, dtype=torch.float32).expand(k - n_acc, 4), budget,
            )
            for path in leaf_paths(tree["parents"]):
                batch.input_ids.copy_(original)
                failed_rows = []
                for node in path:
                    depth = int(tree["depths"][node]) + n_acc
                    row = i * span + depth + 1
                    batch.input_ids.reshape(-1)[row] = int(tree["tokens"][node])
                    failed_rows.append(row)
                capture = _run_capture(engine, batch, failed_rows)
                for node in path:
                    depth = int(tree["depths"][node]) + n_acc
                    row = i * span + depth + 1
                    if node in seen_nodes:
                        continue
                    seen_nodes.add(node)
                    rank = int(tree["cols"][node])
                    for lid, idx in capture.items():
                        rows[depth].setdefault(str(lid), []).append((rank, set(idx[row].tolist())))
            tokens.append(token_rows)
            unions.append(rows)
    finally:
        batch.input_ids.copy_(original)
    eo.write_tree_record(round_id, accepted_counts, tokens, unions)


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
