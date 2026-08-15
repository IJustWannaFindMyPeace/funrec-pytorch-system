import numpy as np
import pytest

from offline.evaluation.diagnose_validation import (
    concentration_summary,
    gini,
    quantile_labels,
    reconstruct_raw_user_activity,
    validation_only,
)


def test_validation_only_never_requires_test():
    validation = {"movie_id": np.array([1, 2])}
    assert validation_only({"validation": validation}) is validation


def test_validation_only_rejects_missing_validation():
    with pytest.raises(ValueError, match="validation"):
        validation_only({"test": {"movie_id": np.array([1])}})


def test_quantile_labels_handles_constant_values():
    assert quantile_labels([3, 3, 3]).tolist() == [0, 0, 0]


def test_reconstruct_raw_activity_adds_three_protocol_events():
    counts = reconstruct_raw_user_activity(
        {"user_id": np.array([1, 1, 2])}
    )
    assert counts.tolist() == [0, 5, 4]


def test_gini_is_zero_for_equal_weights():
    assert gini([4, 4, 4, 4]) == pytest.approx(0.0)


def test_concentration_reports_top_user_share():
    result = concentration_summary(np.array([0, 7, 1, 1, 1]))
    assert result["users"] == 4
    assert result["maximum"] == 7
    assert result["top_10_percent_user_share"] == pytest.approx(0.7)

