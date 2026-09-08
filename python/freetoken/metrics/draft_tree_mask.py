"""Compact tree layout for DSpark top-k draft verification.

Nodes are laid out depth first.  ``parents`` is the flattened parent index;
the attention mask lets a node attend to its ancestors and itself only.
"""
from __future__ import annotations

import heapq
import torch


def build_tree(tokens: torch.Tensor, logprobs: torch.Tensor, budget: int):
    """Build a best-first top-k tree from ``[depth, branch]`` candidates.

    The returned tensors are suitable for a packed tree batch.  A node at depth
    ``d`` uses only the candidates from position ``d``; path scores are summed.
    """
    if budget <= 0 or tokens.numel() == 0:
        z = torch.empty(0, dtype=torch.long, device=tokens.device)
        return {"tokens": z, "parents": z, "depths": z, "scores": z.float(), "cols": z}
    tokens, logprobs = tokens.detach(), logprobs.detach()
    heap = [(-float(logprobs[0, col]), -1, 0, col)
            for col in range(tokens.shape[1])]
    heapq.heapify(heap)
    out = []
    while heap and len(out) < budget:
        neg_score, parent, depth, col = heapq.heappop(heap)
        score, idx = -neg_score, len(out)
        out.append((int(tokens[depth, col]), parent, depth, score, col))
        if depth + 1 < tokens.shape[0]:
            for child in range(tokens.shape[1]):
                heapq.heappush(heap, (-(score + float(logprobs[depth + 1, child])),
                                      idx, depth + 1, child))
    if not out:
        z = torch.empty(0, dtype=torch.long, device=tokens.device)
        return {"tokens": z, "parents": z, "depths": z, "scores": z.float(), "cols": z}
    return {k: torch.tensor([row[i] for row in out], device=tokens.device,
                            dtype=torch.float32 if k == "scores" else torch.long)
            for k, i in (("tokens", 0), ("parents", 1), ("depths", 2),
                         ("scores", 3), ("cols", 4))}


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


def leaf_paths(parents: torch.Tensor) -> list[list[int]]:
    """Return root-to-leaf node paths, so each path can be verified linearly."""
    n = int(parents.numel())
    children = [[] for _ in range(n)]
    for node, parent in enumerate(parents.tolist()):
        if parent >= 0:
            children[parent].append(node)
    leaves = [i for i, c in enumerate(children) if not c]
    paths = []
    for leaf in leaves:
        path = []
        while leaf >= 0:
            path.append(leaf)
            leaf = int(parents[leaf])
        paths.append(path[::-1])
    return paths


def compile_tree(prefix: torch.Tensor, tokens: torch.Tensor,
                 logprobs: torch.Tensor, budget: int):
    """Pack one request's prefix and tree nodes for a tree-attention forward.

    The prefix is shared once.  Tree nodes are the only newly computed tokens;
    rows after a node are never materialized.  ``mask`` is boolean attention
    visibility for the packed rows.
    """
    tree = build_tree(tokens, logprobs, budget)
    p = int(prefix.numel())
    n = int(tree["tokens"].numel())
    parents = tree["parents"] + p
    full_parents = torch.cat([
        torch.arange(p, device=prefix.device, dtype=torch.long) - 1,
        parents,
    ])
    mask = ancestor_mask(full_parents)
    if p:
        mask[:p, :p] = torch.tril(torch.ones((p, p), dtype=torch.bool,
                                              device=prefix.device))
        mask[p:, :p] = True
    return {"input_ids": torch.cat((prefix, tree["tokens"])),
            "positions": torch.cat((torch.arange(p, device=prefix.device),
                                     p + tree["depths"])),
            "parents": full_parents, "mask": mask, **tree}
