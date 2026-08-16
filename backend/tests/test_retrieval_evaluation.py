import pytest
import torch
from torch import nn

from offline.evaluation.retrieval import (
    calculate_single_target_metrics,
    recommend_popular_top_k,
    recommend_top_k,
    recommend_top_k_multi_interest,
)


class FixedUserModel(nn.Module):
    def __init__(self, user_embeddings):
        super().__init__()
        self.register_buffer(
            "user_embeddings",
            user_embeddings,
        )

    def encode_user(self, features):
        return self.user_embeddings


def test_recommend_top_k_filters_history_and_returns_encoded_ids():
    model = FixedUserModel(
        torch.tensor([[1.0, 0.0]])
    )
    item_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.0, 1.0],
        ]
    )
    features = {
        "hist_movie_id": torch.tensor([[1, 2, 0]])
    }

    recommendations = recommend_top_k(
        model,
        features,
        item_embeddings,
        k=2,
    )

    assert recommendations.tolist() == [[3, 4]]


def test_multi_interest_top_k_filters_history():
    class Model:
        scoring_contract = "scaled_cosine_v2"
        logit_scale = 10.0
        def encode_user_interests(self, features):
            return torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    result = recommend_top_k_multi_interest(Model(), {"hist_movie_id": torch.tensor([[1, 0]])}, torch.eye(2), 1)
    assert result.tolist() == [[2]]


def test_single_target_metrics_use_true_rank():
    recommendations = torch.tensor(
        [
            [2, 3, 4, 5],
            [5, 4, 3, 2],
            [1, 2, 3, 4],
        ]
    )
    targets = torch.tensor([2, 4, 9])

    metrics = calculate_single_target_metrics(
        recommendations,
        targets,
        k_values=(1, 2, 4),
    )

    assert metrics["recall@1"] == pytest.approx(1 / 3)
    assert metrics["hit_rate@2"] == pytest.approx(2 / 3)
    assert metrics["recall@4"] == pytest.approx(2 / 3)

    expected_ndcg_at_2 = (
        1.0 + 1.0 / torch.log2(torch.tensor(3.0)).item()
    ) / 3
    assert metrics["ndcg@2"] == pytest.approx(
        expected_ndcg_at_2
    )


def test_recommend_top_k_rejects_invalid_k():
    model = FixedUserModel(torch.tensor([[1.0, 0.0]]))
    features = {
        "hist_movie_id": torch.tensor([[0, 0]])
    }
    item_embeddings = torch.eye(2)

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        recommend_top_k(
            model,
            features,
            item_embeddings,
            k=0,
        )


def test_metrics_reject_insufficient_recommendations():
    with pytest.raises(
        ValueError,
        match="enough columns",
    ):
        calculate_single_target_metrics(
            torch.tensor([[1, 2]]),
            torch.tensor([1]),
            k_values=(5,),
        )

def test_popularity_recommendations_exclude_padding_and_history():
    popularity = torch.tensor(
        [1000.0, 100.0, 90.0, 80.0, 70.0]
    )
    histories = torch.tensor(
        [
            [1, 0, 0],
            [2, 1, 0],
        ]
    )

    recommendations = recommend_popular_top_k(
        popularity=popularity,
        history_ids=histories,
        k=2,
    )

    assert recommendations.tolist() == [
        [2, 3],
        [3, 4],
    ]


def test_popularity_recommendations_handle_duplicate_history():
    popularity = torch.tensor(
        [0.0, 100.0, 90.0, 80.0]
    )
    histories = torch.tensor(
        [[1, 1, 0]]
    )

    recommendations = recommend_popular_top_k(
        popularity=popularity,
        history_ids=histories,
        k=2,
    )

    assert recommendations.tolist() == [[2, 3]]
