import numpy as np
import pytest

from offline.evaluation.diagnose_retrieval_history_mask import (
    mask_history_split,
    non_padding_summary,
    subtract_summaries,
)


def validation_split():
    return {
        "user_id": np.array([1]),
        "hist_movie_id": np.arange(1, 21).reshape(1, 20),
        "hist_genres": np.arange(101, 121).reshape(1, 20),
    }


def test_recent_only_keeps_rightmost_values_without_mutation():
    original = validation_split()
    masked = mask_history_split(original, "recent_10_only", 10)
    assert masked["hist_movie_id"].tolist() == [
        [0] * 10 + list(range(11, 21))
    ]
    assert masked["hist_genres"].tolist() == [
        [0] * 10 + list(range(111, 121))
    ]
    assert original["hist_movie_id"].tolist() == [list(range(1, 21))]


def test_older_only_keeps_left_half_and_masks_recent_half():
    masked = mask_history_split(
        validation_split(), "older_10_only", 10
    )
    assert masked["hist_movie_id"].tolist() == [
        list(range(1, 11)) + [0] * 10
    ]
    assert masked["hist_genres"].tolist() == [
        list(range(101, 111)) + [0] * 10
    ]


def test_full_condition_returns_an_independent_copy():
    original = validation_split()
    masked = mask_history_split(original, "full_20", 10)
    masked["hist_movie_id"][0, 0] = 999
    assert original["hist_movie_id"][0, 0] == 1


def test_mask_rejects_unknown_condition_and_invalid_length():
    with pytest.raises(ValueError, match="Unknown"):
        mask_history_split(validation_split(), "future", 10)
    with pytest.raises(ValueError, match="smaller"):
        mask_history_split(validation_split(), "full_20", 20)


def test_subtract_summaries_uses_candidate_minus_reference():
    result = subtract_summaries(
        {"recall@10": 0.3, "activity_recall@10_gap": 0.1},
        {"recall@10": 0.2, "activity_recall@10_gap": 0.15},
    )
    assert result["recall@10"] == pytest.approx(0.1)
    assert result["activity_recall@10_gap"] == pytest.approx(-0.05)


def test_non_padding_summary_reports_effective_history_size():
    value = validation_split()
    value["hist_movie_id"][0, :2] = 0
    summary = non_padding_summary(value)
    assert summary["hist_movie_id"] == {
        "minimum": 18,
        "maximum": 18,
        "mean": 18.0,
    }
