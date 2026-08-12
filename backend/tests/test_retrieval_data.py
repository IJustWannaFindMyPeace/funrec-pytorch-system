import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from offline.training.retrieval_data import RetrievalDataset


def build_data():
    return {
        "user_id": np.array([1, 2, 3], dtype=np.int32),
        "age": np.array([2, 3, 4], dtype=np.int32),
        "gender": np.array([1, 2, 1], dtype=np.int32),
        "occupation": np.array([4, 5, 6], dtype=np.int32),
        "zip_code": np.array([7, 8, 9], dtype=np.int32),
        "hist_movie_id": np.array(
            [
                [0, 2, 3],
                [0, 4, 5],
                [6, 7, 8],
            ],
            dtype=np.int32,
        ),
        "hist_genres": np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [1, 3, 4],
            ],
            dtype=np.int32,
        ),
        "movie_id": np.array([4, 6, 9], dtype=np.int32),
        "unused_field": np.array([100, 200, 300]),
    }


def test_retrieval_dataset_returns_required_features_and_target():
    dataset = RetrievalDataset(build_data())

    features, target = dataset[1]

    assert len(dataset) == 3
    assert set(features) == {
        "user_id",
        "age",
        "gender",
        "occupation",
        "zip_code",
        "hist_movie_id",
        "hist_genres",
    }
    assert target.item() == 6
    assert features["hist_movie_id"].tolist() == [0, 4, 5]


def test_retrieval_dataloader_builds_expected_batch_shapes():
    dataset = RetrievalDataset(build_data())
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    features, targets = next(iter(loader))

    assert targets.shape == (2,)
    assert features["user_id"].shape == (2,)
    assert features["hist_movie_id"].shape == (2, 3)
    assert features["hist_genres"].shape == (2, 3)
    assert targets.dtype == torch.int32


def test_retrieval_dataset_rejects_missing_fields():
    data = build_data()
    del data["hist_genres"]

    with pytest.raises(
        KeyError,
        match="Missing retrieval fields: hist_genres",
    ):
        RetrievalDataset(data)


def test_retrieval_dataset_rejects_inconsistent_lengths():
    data = build_data()
    data["movie_id"] = np.array([4, 6], dtype=np.int32)

    with pytest.raises(
        ValueError,
        match="inconsistent lengths",
    ):
        RetrievalDataset(data)


def test_retrieval_dataset_rejects_mismatched_history_lengths():
    data = build_data()
    data["hist_genres"] = np.ones((3, 4), dtype=np.int32)

    with pytest.raises(
        ValueError,
        match="equal sequence lengths",
    ):
        RetrievalDataset(data)