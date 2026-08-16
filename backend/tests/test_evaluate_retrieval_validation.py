import numpy as np
import pytest
import torch

from modeling.youtubednn import YouTubeDNN
from modeling.youtubednn import SCORING_CONTRACT_SCALED_COSINE_V2

from offline.evaluation.evaluate_retrieval_validation import (
    infer_sequence_length,
    item_embeddings_for_scoring,
    validate_selection_samples,
)


FEATURE_DICT = {
    "user_id": 4,
    "age": 3,
    "gender": 3,
    "occupation": 3,
    "zip_code": 3,
    "movie_id": 5,
    "genres": 4,
}


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


def test_training_raw_item_scoring_matches_full_softmax_columns():
    model = YouTubeDNN(FEATURE_DICT, embedding_dim=4)
    with torch.no_grad():
        model.movie_embedding.weight[1:].copy_(torch.tensor([
            [3.0, 4.0, 0.0, 0.0],
            [1.0, 2.0, 2.0, 1.0],
            [2.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]))

    raw = item_embeddings_for_scoring(model, "training_raw")
    normalized = item_embeddings_for_scoring(model, "exported_normalized")

    assert torch.equal(raw, model.movie_embedding.weight[1:])
    assert torch.allclose(torch.linalg.vector_norm(normalized, dim=1), torch.ones(4))
    assert not torch.allclose(raw, normalized)


def test_scaled_cosine_training_logits_match_exported_item_scores():
    model = YouTubeDNN(
        FEATURE_DICT,
        embedding_dim=4,
        scoring_contract=SCORING_CONTRACT_SCALED_COSINE_V2,
        logit_scale=7.0,
    )
    features = {
        "user_id": torch.tensor([1, 2]),
        "age": torch.tensor([1, 1]),
        "gender": torch.tensor([1, 1]),
        "occupation": torch.tensor([1, 1]),
        "zip_code": torch.tensor([1, 1]),
        "hist_movie_id": torch.tensor([[1, 2, 0], [2, 3, 4]]),
        "hist_genres": torch.tensor([[1, 2, 0], [2, 3, 1]]),
    }
    exported_items = item_embeddings_for_scoring(
        model, "exported_normalized"
    )

    assert torch.allclose(
        model.compute_full_logits(features),
        model.score_user_to_item_embeddings(
            model.encode_user(features), exported_items
        ),
        atol=1e-6,
    )
