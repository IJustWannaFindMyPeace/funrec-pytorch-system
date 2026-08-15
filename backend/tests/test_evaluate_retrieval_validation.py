import numpy as np
import pytest

from offline.evaluation.evaluate_retrieval_validation import (
    infer_sequence_length,
    validate_selection_samples,
)


def split(length=3):
    return {
        "hist_movie_id": np.zeros((2, length), dtype=np.int64),
        "hist_genres": np.zeros((2, length), dtype=np.int64),
    }


def test_selection_artifact_rejects_embedded_test():
    with pytest.raises(ValueError, match="Test must be sealed"):
        validate_selection_samples({
            "train": split(),
            "validation": split(),
            "test": split(),
        })


def test_selection_artifact_accepts_only_train_and_validation():
    train, validation = validate_selection_samples({
        "train": split(),
        "validation": split(),
    })
    assert train is not validation


def test_sequence_length_is_inferred_from_both_histories():
    assert infer_sequence_length(split(20)) == 20


def test_sequence_length_rejects_misaligned_histories():
    value = split(20)
    value["hist_genres"] = np.zeros((2, 10), dtype=np.int64)
    with pytest.raises(ValueError, match="lengths differ"):
        infer_sequence_length(value)
