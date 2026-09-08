"""DSV4 MoE: sqrtsoftplus/hash router, shared SwiGLU expert, offloaded FP4 routed
experts (GPU slot-cache / cpu / hybrid decode paths)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator
from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
from freetoken.kernel.triton.dsv4.swiglu import fused_swiglu
from freetoken.layers import OffloadMoELayer

from .args import DeepseekV4Args
from .layers import Linear

from .parallel import div_tp, tp_size


class Gate(nn.Module):
    """MoE router: sqrtsoftplus scoring + hash routing (first ``n_hash_layers``)."""

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        super().__init__()
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim, dtype=torch.bfloat16), requires_grad=False)
        if self.hash:
            self.tid2eid = nn.Parameter(
                torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int64), requires_grad=False
            )
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor, restrict: torch.Tensor | None = None, topk: int | None = None):
        scores = bf16_linear_fp32(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash and restrict is None:
            indices = self.tid2eid[input_ids]
        else:
            if restrict is not None:
                # Metric-1 shadow rerun: choose top-k only among ``restrict``'s
                # True entries (the layer's cache-resident experts). Hash layers
                # fall back to score-based selection here -- the tid2eid table
                # cannot express a resident-set restriction.
                scores = scores.masked_fill(~restrict, float("-inf"))
            indices = scores.topk(self.topk if topk is None else topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices


class Expert(nn.Module):
    """Dense SwiGLU expert (the shared expert; routed experts are offloaded FP4).

    Under TP the intermediate dimension splits: ``w1``/``w3`` are column-parallel and
    ``w2`` is row-parallel, so the output is a partial sum. ``MoE.forward`` owns the
    single all-reduce that completes it together with the routed half.
    """

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float):
        super().__init__()
        inter_local = div_tp(inter_dim, "moe_inter_dim", multiple_of=128)
        self.w1 = Linear(dim, inter_local, kind="fp8")
        self.w2 = Linear(inter_local, dim, kind="fp8")
        self.w3 = Linear(dim, inter_local, kind="fp8")
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = fused_swiglu(self.w1(x), self.w3(x), self.swiglu_limit, x.dtype)
        return self.w2(h)


class DSV4OffloadMoELayer(OffloadMoELayer):
    """Routed FP4 experts on the shared offload cache: the base whole-layer
    streaming prefill (grouped inline-dequant GEMM for dense chunks, GEMV
    below the route crossover) and slot-cache / cpu / hybrid decode paths
    (per-route dequant GEMV)."""

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        super().__init__(
            layer_id=layer_id,
            num_experts=args.n_routed_experts,
            top_k=args.n_activated_experts,
            hidden_size=args.dim,
            intermediate_size=args.moe_inter_dim,
            renormalize=True,
            activation="silu",
        )
        self.swiglu_limit = args.swiglu_limit

    def _maybe_all_reduce(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Suppress the base class's per-call collective.

        The routed and the shared expert are both partial sums over the same output
        dim, so DSV4 adds them first and all-reduces ONCE in ``MoE.forward``. Letting
        the base reduce here would cost a second collective per layer and would also
        double-count the shared expert.
        """
        return hidden_states

    def _prefill_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Whole-layer streaming moves all num_experts rows per layer; a small
        # chunk touches at most T*top_k of them, so below that crossover the
        # decode-style on-demand slot path strictly moves fewer bytes (and
        # keeps short-prompt slot residency -- hence hybrid decode's GPU/CPU
        # route split -- unchanged). Mixing modes across chunks is safe: the
        # streaming buffers disown their borrowed slots on invalidation.
        cache = self.offload_cache
        assert cache is not None
        # unpinned (LOCKED) layers must take the base materialize path: their copy_missing is the whole-layer pageable branch with position == expert id, which ensure_experts's LRU slot remap would contradict (the GEMM would gather other experts' weights)
        if (
            hidden_states.shape[0] * self.top_k >= self.num_experts
            or cache.is_unpinned_layer(self.layer_id)
        ):
            return super()._prefill_routed(hidden_states, topk_weights, topk_ids)

        # A speculative verify is a decode wearing a prefill's clothes. The scheduler
        # marks the batch "prefill" because the block is an extend over several
        # positions, but it carries block_size rows per request, not a prompt -- and it
        # runs on the critical path of every decode step.
        #
        # The path below fetches EVERY missing expert over PCIe, uncapped, with no CPU
        # overlap. That is the right trade for a prompt, where the fetch amortizes over
        # hundreds of tokens. For a 5-row block it moves up to T*top_k experts per layer
        # with nothing hiding the latency: measured at ~0.4s per block against a 0.06s
        # single-token step, which is the whole reason speculation lost to plain decode
        # here rather than a low acceptance rate.
        #
        # Hybrid decode caps the fetch and overlaps the overflow on the CPU pool, which
        # is what a handful of rows wants.
        if (
            getattr(get_global_ctx().batch, "speculative", False)
            and cache.decode_target == "hybrid"
        ):
            return self._decode_routed(hidden_states, topk_weights, topk_ids)
        cache.ensure_experts(self.layer_id, topk_ids)  # in-place expert-id -> slot
        cache.copy_missing()
        if cache.collect_stats:
            cache.record_decode_stats(self.layer_id)
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=True,
        )


class MoE(nn.Module):
    """Sparse MoE: hash/score router -> offloaded FP4 routed experts + shared expert."""

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        super().__init__()
        self.dim = args.dim
        self.gate = Gate(layer_id, args)
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, args.swiglu_limit)
        self.experts = DSV4OffloadMoELayer(layer_id, args)
        self._comm = DistributedCommunicator() if tp_size() > 1 else None

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        input_ids = input_ids.flatten()
        weights, indices = self.gate(x, input_ids)
        # Expert-overlap metrics (FT_EXPERT_METRICS_DIR): record the router's
        # full-space top-k per MoE layer for the active verify round. During a
        # metric-1 shadow rerun this is A' -- the unrestricted choice under the
        # shadow hidden states -- diverted into the shadow capture.
        from freetoken.metrics.expert_overlap import (
            note_router, resident_mask, shadow_active, shadow_failed_rows,
        )
        note_router(self.experts.layer_id, indices)
        from freetoken.metrics.expert_overlap import top4_enabled
        if shadow_active() and top4_enabled() and shadow_failed_rows() is None:
            # Measurement reads resident banks directly. Do not ensure/copy experts,
            # submit CPU work, or update LRU: all four candidates see one cache state.
            cache = self.experts.offload_cache
            if cache is None:
                raise RuntimeError("top4 cache-only measurement requires an offload cache")
            slots = cache.slot_for_id[self.experts.layer_id]
            mask = slots >= 0
            count = min(self.experts.top_k, int(mask.sum().item()))
            shared = self.shared_experts(x)
            if count:
                weights, ids = self.gate(x, input_ids, restrict=mask, topk=count)
                slot_ids = slots[ids.long()].to(torch.int32).contiguous()
                assert bool((slot_ids >= 0).all())
                routed = self.experts._expert_gemm(
                    cache, x, weights.float().contiguous(), slot_ids,
                    views=cache.bank_views(), n=None,
                    alphas=cache.alphas_for_slots(self.experts.layer_id), is_prefill=False,
                )
                out = shared + routed
            else:
                out = shared
            if self._comm is not None:
                out = self._comm.all_reduce(out)
            return out.view(shape)
        failed_rows = shadow_failed_rows()
        if shadow_active() and failed_rows is not None:
            # Keep the accepted prefix as ordinary full-MoE context, while only
            # the rejected tail uses the cache-only route.
            failed = torch.tensor(failed_rows, dtype=torch.long, device=x.device)
            failed = failed[failed < x.shape[0]]
            normal_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            normal_mask[failed] = False
            shared = self.shared_experts(x)
            routed = torch.zeros_like(x)
            if bool(normal_mask.any()):
                routed[normal_mask] = self.experts.routed_forward(
                    x[normal_mask], weights[normal_mask].float().contiguous(),
                    indices[normal_mask].to(torch.int32).contiguous())
            if bool(failed.numel()):
                # Early rounds can have fewer resident experts than model top-k.
                # Restrict to the currently resident subset; an empty cache has
                # zero routed contribution until the normal prefix warms it.
                cache = self.experts.offload_cache
                mask = resident_mask(cache, self.experts.layer_id,
                                     self.experts.num_experts, 1)
                if mask is not None:
                    count = min(self.experts.top_k, int(mask.sum().item()))
                    fw, fi = self.gate(x[failed], input_ids[failed],
                                        restrict=mask, topk=count)
                    # routed_forward's executor buffer is fixed at model top-k;
                    # pad partial-cache routes with zero-weight resident IDs.
                    if count < self.experts.top_k:
                        pad = self.experts.top_k - count
                        fill = fi[:, :1].expand(-1, pad)
                        fi = torch.cat((fi, fill), dim=1)
                        fw = torch.cat((fw, torch.zeros_like(fw[:, :pad])), dim=1)
                    routed[failed] = self.experts.routed_forward(
                        x[failed], fw.float().contiguous(),
                        fi.to(torch.int32).contiguous())
            out = routed + shared
            if self._comm is not None:
                out = self._comm.all_reduce(out)
            return out.view(shape)
        if shadow_active():
            # Metric-1 cache-only rerun: recompute the routing restricted to the
            # layer's GPU-cache-resident experts for the actual MoE compute
            # (ensure_experts then hits on every route, so the rerun never
            # fetches and the resident set is stable through the shadow pass).
            mask = resident_mask(
                self.experts.offload_cache,
                self.experts.layer_id,
                self.experts.num_experts,
                self.experts.top_k,
            )
            if mask is not None:
                weights, indices = self.gate(x, input_ids, restrict=mask)
        # Shared expert enqueued before routed_forward: hybrid decode blocks on the
        # CPU pool inside routed_forward, so this GEMM must already be on the stream
        # to overlap the CPU overflow compute.
        shared = self.shared_experts(x)
        # routed_forward may mutate the ids in place (offload decode slot remap);
        # indices.to(int32) always copies (int64 source), so no clone needed here.
        routed = self.experts.routed_forward(
            x, weights.float().contiguous(), indices.to(torch.int32).contiguous()
        )
        out = routed + shared
        # Both halves are partial sums over the split intermediate dim; one collective
        # completes the layer (see DSV4OffloadMoELayer._maybe_all_reduce).
        if self._comm is not None:
            out = self._comm.all_reduce(out)
        return out.view(shape)
