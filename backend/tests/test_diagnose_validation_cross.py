import numpy as np
import pytest

from offline.evaluation.diagnose_validation_cross import (
    binary_entropy,
    calibrated_ranking_metrics,
    cross_masks,
    safe_pr_auc,
)


def test_binary_entropy_is_log_two_for_balanced_labels():
    assert binary_entropy(0.5) == pytest.approx(np.log(2.0))


def test_calibrated_metrics_compare_with_constant_baseline():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    losses = -(
        labels * np.log(scores)
        + (1 - labels) * np.log(1 - scores)
    )
    result = calibrated_ranking_metrics(labels, scores, losses)
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["normalized_logloss"] < 1.0
    assert result["logloss_improvement_over_constant"] > 0.0


def test_pr_auc_returns_none_for_single_class():
    assert safe_pr_auc([1, 1], [0.8, 0.9]) is None


def test_cross_masks_partition_every_example_once():
    activity = np.arange(16)
    popularity = np.arange(16)[::-1]
    masks = list(cross_masks(activity, popularity))
    coverage = sum(mask.astype(np.int64) for _, _, mask in masks)
    assert np.all(coverage == 1)
