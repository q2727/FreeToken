"""Compact tree layout for DSpark top-k draft verification.

Nodes are laid out depth first.  ``parents`` is the flattened parent index;
the attention mask lets a node attend to its ancestors and itself only.
"""
from __future__ import annotations

import torch


def build_tree(tokens: torch.Tensor, logprobs: torch.Tensor, budget: int):
    """Build a best-first top-k tree from ``[depth, branch]`` candidates.

    The returned tensors are suitable for a packed tree batch.  A node at depth
    ``d`` uses only the candidates from position ``d``; path scores are summed.
    """
    if budget <= 0 or tokens.numel() == 0:
        z = torch.empty(0, dtype=torch.long, device=tokens.device)
        return {"tokens": z, "parents": z, "depths": z, "scores": z.float()}
    tokens, logprobs = tokens.detach(), logprobs.detach()
    frontier = [(-1, -1, 0.0)]
    out = []
    for depth in range(tokens.shape[0]):
        candidates = []
        for parent, _old_depth, score in frontier:
            for col in range(tokens.shape[1]):
                candidates.append((score + float(logprobs[depth, col]), parent, depth, col))
        candidates.sort(key=lambda x: (-x[0], x[1], x[3]))
        keep = candidates[: max(0, budget - len(out))]
        if not keep:
            break
        next_frontier = []
        for score, parent, d, col in keep:
            idx = len(out)
            out.append((int(tokens[d, col]), parent, d, score))
            next_frontier.append((idx, d, score))
        frontier = next_frontier
    if not out:
        return {"tokens": torch.empty(0, dtype=torch.long), "parents": torch.empty(0, dtype=torch.long),
                "depths": torch.empty(0, dtype=torch.long), "scores": torch.empty(0)}
    return {k: torch.tensor([row[i] for row in out], device=tokens.device,
                            dtype=torch.float32 if k == "scores" else torch.long)
            for k, i in (("tokens", 0), ("parents", 1), ("depths", 2), ("scores", 3))}


def ancestor_mask(parents: torch.Tensor) -> torch.Tensor:
    """Return the packed causal tree mask (True means attention is allowed)."""
    n = int(parents.numel())
    mask = torch.zeros((n, n), dtype=torch.bool, device=parents.device)
    for node in range(n):
        cur = node
        while cur >= 0:
            mask[node, cur] = True
            cur = int(parents[cur])
    return mask
