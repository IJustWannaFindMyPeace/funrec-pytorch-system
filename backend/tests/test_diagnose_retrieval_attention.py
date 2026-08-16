import pytest
import torch

from offline.evaluation.diagnose_retrieval_attention import (
    recent_position_mask,
    summarize_weights,
)


def test_recent_position_mask_excludes_left_padding():
    ids = torch.tensor([[0, 0, 4, 5, 6], [0, 7, 8, 9, 10]])

    mask = recent_position_mask(ids, recent_count=2)

    assert mask.tolist() == [
        [False, False, False, True, True],
        [False, False, False, True, True],
    ]


def test_summarize_weights_reports_normalized_recent_mass():
    ids = torch.tensor([[0, 3, 4, 5]])
    weights = torch.tensor([[0.0, 0.2, 0.3, 0.5]])

    summary = summarize_weights(weights, ids, recent_count=2)

    assert summary["examples"] == 1
    assert summary["mean_non_padding_length"] == 3.0
    assert summary["mean_recent_weight_mass"] == pytest.approx(0.8)
    assert summary["mean_peak_attention_weight"] == pytest.approx(0.5)
