import numpy as np
import pytest
import torch

from offline.evaluation.diagnose_validation_models import (
    grouped_ranking_metrics,
    grouped_retrieval_metrics,
    ranking_metrics,
    safe_auc,
)


def test_safe_auc_returns_none_for_single_class():
    assert safe_auc([1, 1], [0.8, 0.9]) is None


def test_ranking_metrics_reports_auc_and_logloss():
    result = ranking_metrics(
        labels=np.array([0, 1]),
        scores=np.array([0.1, 0.9]),
        losses=np.array([0.105, 0.105]),
    )
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["logloss"] == pytest.approx(0.105)
    assert result["positive_rate"] == pytest.approx(0.5)


def test_grouped_ranking_metrics_keeps_all_examples():
    result = grouped_ranking_metrics(
        values=np.arange(8),
        labels=np.array([0, 1] * 4),
        scores=np.linspace(0.1, 0.8, 8),
        losses=np.ones(8),
    )
    assert sum(row["examples"] for row in result) == 8


def test_grouped_retrieval_metrics_uses_single_target_protocol():
    recommendations = torch.arange(1, 51).repeat(4, 1)
    targets = torch.tensor([1, 6, 1, 10])
    result = grouped_retrieval_metrics(
        values=np.array([1, 2, 3, 4]),
        recommendations=recommendations,
        targets=targets,
    )
    assert sum(row["examples"] for row in result) == 4
    assert all("recall@5" in row for row in result)
