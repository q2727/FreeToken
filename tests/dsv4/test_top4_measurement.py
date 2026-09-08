"""Contracts specific to the top4 measurement (CPU tests)."""
import json
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from freetoken.metrics import expert_overlap as eo
from freetoken.metrics import shadow_rerun as shadow
from freetoken.models.deepseek_v4.dspark import DSparkDrafter
from freetoken.models.deepseek_v4.moe import MoE


def test_raw_top4_preserves_greedy_and_tie_argmax(monkeypatch):
    class Markov:
        def embed(self, x):
            return x[:, None].float()
        def bias(self, x):
            return torch.zeros((x.shape[0], 12))
    drafter = NS(markov_head=Markov(), confidence_head=lambda h, m: h[:, 0])
    logits = torch.zeros(2, 8, 12)
    logits[:, :, 6] = 5
    params = [NS(temperature=0, top_p=1, top_k=-1, is_greedy=True)] * 2
    monkeypatch.setattr(eo, 'top4_enabled', lambda: False)
    original = DSparkDrafter.sample_block(drafter, logits, torch.ones(2, 8, 3), torch.tensor([1, 2]), params)
    assert drafter.last_top4 is None
    monkeypatch.setattr(eo, 'top4_enabled', lambda: True)
    observed = DSparkDrafter.sample_block(drafter, logits, torch.ones(2, 8, 3), torch.tensor([1, 2]), params)
    assert all(torch.equal(a, b) for a, b in zip(original, observed))
    assert drafter.last_top4.shape == (2, 8, 4)
    assert torch.equal(drafter.last_top4[:, :, 0].flatten(), original[0])
    assert all(len(set(row)) == 4 for req in drafter.last_top4.tolist() for row in req)


def test_writer_uses_candidate_input_row_and_includes_final_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(eo, '_ACTIVE', True)
    monkeypatch.setattr(eo, '_DIR', str(tmp_path))
    monkeypatch.setattr(eo, '_STATE', {'batch': {'k': 8, 'n_reqs': 1, 'span': 9,
        'req_uids': [3], 'positions': torch.arange(100, 109)}})
    rows = torch.arange(54).reshape(9, 6)
    eo.write_top4_record(0, [6], torch.arange(32).reshape(1, 8, 4), [{3: rows}] * 4)
    result = json.loads((tmp_path / 'top4_shadow.jsonl').read_text())
    slots = result['reqs'][0]['rejected']
    assert slots[0]['candidates'][0]['sets']['3'] == rows[7].tolist()
    assert slots[1]['candidates'][0]['sets']['3'] == rows[8].tolist()
    assert slots[1]['position'] == 108


def test_candidate_exception_restores_original_input(monkeypatch):
    inputs = torch.arange(18)
    original = inputs.clone()
    proposed = torch.cat([inputs[1:9], inputs[10:18]])
    candidates = torch.stack([proposed + 100 * i for i in range(4)], dim=-1)
    batch = NS(input_ids=inputs, spec_block=8, reqs=[1, 2], draft_tokens=proposed)
    cache = NS(**{k: torch.tensor([0]) for k in ('slot_for_id', 'id_of_slot', 'usage', 'step', 'expert_recency')})
    engine = NS(model=NS(_transformer=NS(layers=[NS(ffn=NS(experts=NS(offload_cache=cache)))])))
    calls = []
    def capture(engine, batch):
        calls.append(batch.input_ids.clone())
        if len(calls) == 2:
            raise RuntimeError('injected')
        return {}
    monkeypatch.setattr(shadow, '_run_capture', capture)
    with pytest.raises(RuntimeError, match='injected'):
        shadow._run_top4(engine, batch, [2, 7], candidates)
    assert torch.equal(batch.input_ids, original)
    assert torch.equal(calls[0], original)
    assert torch.equal(calls[1][:3], original[:3])
    assert int(calls[1][3]) == int(original[3]) + 100
    assert int(calls[1][17]) == int(original[17]) + 100


def test_capture_exception_restores_journal_features_and_pools(monkeypatch):
    journal, features = object(), object()
    batch = NS(spec_carry_states=journal)
    transformer = NS(_target_features=features)
    def forward():
        transformer._target_features = object()
        raise RuntimeError('injected forward')
    engine = NS(model=NS(_transformer=transformer, forward=forward), ctx=NS(forward_batch=lambda _: nullcontext()))
    restored = []
    monkeypatch.setattr(shadow, '_snapshot_pools', lambda e, b: 'snapshot')
    monkeypatch.setattr(shadow, '_restore_pools', lambda e, b, s: restored.append(s))
    with pytest.raises(RuntimeError, match='injected forward'):
        shadow._run_capture(engine, batch)
    assert batch.spec_carry_states is journal
    assert transformer._target_features is features
    assert restored == ['snapshot']
    assert not eo._SHADOW['active']


@pytest.mark.parametrize('resident', [0, 2, 6])
def test_shadow_computes_directly_from_resident_banks(monkeypatch, resident):
    monkeypatch.setattr(eo, 'shadow_active', lambda: True)
    monkeypatch.setattr(eo, 'top4_enabled', lambda: True)
    monkeypatch.setattr(eo, 'note_router', lambda *a: None)
    mapping = torch.full((1, 8), -1, dtype=torch.int32)
    mapping[0, :resident] = torch.arange(10, 10 + resident)
    cache = NS(slot_for_id=mapping, bank_views=lambda: (), alphas_for_slots=lambda l: None)
    calls = []
    def gate(x, ids, restrict=None, topk=None):
        count = 6 if topk is None else topk
        return torch.ones(x.shape[0], count), torch.arange(count).expand(x.shape[0], -1)
    def gemm(cache, x, weights, ids, **kwargs):
        calls.append(ids.clone())
        assert ids.min() >= 10 and ids.max() < 10 + resident
        return torch.ones_like(x)
    experts = NS(layer_id=0, top_k=6, offload_cache=cache, _expert_gemm=gemm)
    layer = NS(dim=3, gate=gate, experts=experts, shared_experts=lambda x: x * 2, _comm=None)
    out = MoE.forward(layer, torch.ones(1, 2, 3), torch.tensor([1, 2]))
    assert torch.equal(out, torch.full((1, 2, 3), 3.0 if resident else 2.0))
    assert len(calls) == int(resident > 0)
