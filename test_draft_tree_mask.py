import torch

from freetoken.metrics.draft_tree_mask import ancestor_mask, build_tree


def test_tree_budget_and_ancestors():
    ids = torch.tensor([[10, 11], [20, 21]])
    scores = torch.tensor([[-.1, -.2], [-.1, -.2]])
    tree = build_tree(ids, scores, 6)
    assert tree["tokens"].numel() == 6
    mask = ancestor_mask(tree["parents"])
    assert torch.all(torch.diag(mask))
    for i, p in enumerate(tree["parents"].tolist()):
        if p >= 0:
            assert mask[i, p]


def test_empty_budget():
    tree = build_tree(torch.ones(2, 4, dtype=torch.long), torch.zeros(2, 4), 0)
    assert all(v.numel() == 0 for v in tree.values())
