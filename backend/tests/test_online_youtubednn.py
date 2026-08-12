import asyncio

import torch

from online.recall.youtubednn import YouTubeDNNRecallStrategy


class FakeUserModel:
    def encode_user(self, features):
        assert set(features) == {
            "user_id",
            "age",
            "gender",
            "occupation",
            "zip_code",
            "hist_movie_id",
            "hist_genres",
        }
        return torch.tensor(
            [[1.0, 0.0]],
            dtype=torch.float32,
        )


class FakeResourceManager:
    def __init__(self):
        self.device = torch.device("cpu")
        self.user_model = FakeUserModel()

        # Row 0 is padding. Raw movies are 10, 20 and 30.
        self.item_embedding_tensor = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
            ],
            dtype=torch.float32,
        )
        self.all_movie_ids = ["10", "20", "30"]
        self.movie_genre_map = {
            10: [1],
            20: [2],
            30: [1, 2],
        }

    def _ensure_resources_loaded(self):
        return True

    def encode_feature(self, feature_name, raw_value):
        if raw_value is None:
            return 0

        if feature_name == "movie_id":
            mapping = {
                "10": 1,
                "20": 2,
                "30": 3,
            }
            return mapping.get(str(raw_value), 0)

        return 1


def build_strategy():
    strategy = YouTubeDNNRecallStrategy.__new__(
        YouTubeDNNRecallStrategy
    )
    strategy.resource_manager = FakeResourceManager()
    return strategy


def test_preprocess_user_builds_expected_tensors():
    strategy = build_strategy()

    inputs = strategy.preprocess_user(
        {
            "user_id": "1",
            "age": "25",
            "gender": "F",
            "occupation": "2",
            "zip_code": "10001",
            "hist_movie_ids": [10, 20],
        },
        max_hist_len=4,
    )

    assert all(
        value.dtype == torch.long
        for value in inputs.values()
    )
    assert inputs["user_id"].tolist() == [1]
    assert inputs["hist_movie_id"].tolist() == [
        [0, 0, 1, 2]
    ]
    assert inputs["hist_genres"].tolist() == [
        [0, 0, 1, 2]
    ]


def test_preprocess_user_rejects_invalid_history_length():
    strategy = build_strategy()

    try:
        strategy.preprocess_user({}, max_hist_len=0)
    except ValueError as error:
        assert str(error) == (
            "max_hist_len must be greater than zero"
        )
    else:
        raise AssertionError("ValueError was not raised")


def test_recall_uses_pytorch_and_filters_history():
    strategy = build_strategy()

    results = asyncio.run(
        strategy.recall(
            {
                "user_id": "1",
                "age": "25",
                "gender": "F",
                "occupation": "2",
                "zip_code": "10001",
                "hist_movie_ids": [10],
            },
            k=2,
        )
    )

    assert [item["movie_id"] for item in results] == [20, 30]
    assert all(
        item["recall_type"] == "youtube_dnn"
        for item in results
    )
    assert results[0]["score"] > results[1]["score"]
    assert 10 not in {
        item["movie_id"]
        for item in results
    }


def test_recall_caps_k_to_available_candidates():
    strategy = build_strategy()

    results = asyncio.run(
        strategy.recall(
            {
                "user_id": "1",
                "hist_movie_ids": [10, 20],
            },
            k=10,
        )
    )

    assert [item["movie_id"] for item in results] == [30]


def test_recall_returns_empty_for_non_positive_k():
    strategy = build_strategy()

    assert asyncio.run(strategy.recall({}, k=0)) == []
    assert asyncio.run(strategy.recall({}, k=-1)) == []