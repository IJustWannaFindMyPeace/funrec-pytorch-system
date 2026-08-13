import torch
import pytest

from online.ranking.deepfm import (
    DeepFMRankingStrategy,
    FallbackRankingStrategy,
)


class FakeModel:
    def predict_proba(self, features):
        return (
            features["movie_id"].to(torch.float32)
            / 10.0
        )


class FakeResourceManager:
    user_features = [
        "user_id",
        "gender",
        "age",
        "occupation",
        "zip_code",
    ]
    item_features = [
        "movie_id",
        "genres",
        "isAdult",
        "startYear",
    ]

    def __init__(self):
        self.device = torch.device("cpu")
        self.ranking_model = FakeModel()
        self.seen_values = []

    def _ensure_resources_loaded(self):
        return True

    def encode_feature(self, name, value):
        self.seen_values.append((name, value))

        if value is None:
            return 0

        if name == "movie_id":
            return int(value)

        mappings = {
            "Action": 1,
            "Drama": 2,
            "False": 1,
            "True": 2,
            False: 1,
            True: 2,
            2000: 1,
            2001: 2,
        }
        return mappings.get(value, 1)


def build_strategy():
    strategy = DeepFMRankingStrategy()
    strategy.resource_manager = FakeResourceManager()
    return strategy


def test_prepare_batch_inputs_normalizes_item_fields():
    strategy = build_strategy()

    inputs = strategy._prepare_batch_inputs(
        {
            "user_id": "1",
            "gender": "F",
            "age": "1",
            "occupation": "10",
            "zip_code": "48067",
        },
        [
            {
                "movie_id": 2,
                "genres": ["Action", "Drama"],
                "is_adult": False,
                "year": 2000,
            },
            {
                "movie_id": 8,
                "genres": "Drama|Comedy",
                "isAdult": True,
                "startYear": 2001,
            },
        ],
    )

    assert set(inputs) == {
        *FakeResourceManager.user_features,
        *FakeResourceManager.item_features,
    }
    assert all(
        value.shape == (2,)
        for value in inputs.values()
    )
    assert all(
        value.dtype == torch.long
        for value in inputs.values()
    )
    assert inputs["movie_id"].tolist() == [2, 8]

    seen = (
        strategy.resource_manager.seen_values
    )
    assert ("genres", "Action") in seen
    assert ("genres", "Drama") in seen
    assert ("isAdult", False) in seen
    assert ("startYear", 2000) in seen


@pytest.mark.asyncio
async def test_rank_uses_pytorch_probabilities():
    strategy = build_strategy()
    candidates = [
        {
            "movie_id": 2,
            "score": 0.9,
            "recall_type": "youtube_dnn",
            "genres": ["Action"],
            "is_adult": False,
            "year": 2000,
        },
        {
            "movie_id": 8,
            "score": 0.1,
            "recall_type": "youtube_dnn",
            "genres": ["Drama"],
            "is_adult": False,
            "year": 2001,
        },
    ]

    results = await strategy.rank(
        {"user_id": "1"},
        candidates,
    )

    assert [
        item["movie_id"]
        for item in results
    ] == [8, 2]
    assert results[0]["score"] == pytest.approx(0.8)
    assert results[0]["recall_score"] == 0.1
    assert results[0]["recall_type"] == "youtube_dnn"


@pytest.mark.asyncio
async def test_rank_returns_empty_for_no_candidates():
    strategy = build_strategy()

    assert await strategy.rank({}, []) == []


@pytest.mark.asyncio
async def test_fallback_orders_by_recall_score():
    strategy = FallbackRankingStrategy()

    results = await strategy.rank(
        {},
        [
            {
                "movie_id": 1,
                "score": 0.2,
            },
            {
                "movie_id": 2,
                "score": 0.9,
            },
        ],
    )

    assert [
        item["movie_id"]
        for item in results
    ] == [2, 1]
    assert results[0]["recall_score"] == 0.9