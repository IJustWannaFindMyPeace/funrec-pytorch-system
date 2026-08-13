import torch
import pytest
from torch.nn import functional as F

from modeling.deepfm import DeepFM


FEATURE_DICT = {
    "user_id": 8,
    "gender": 3,
    "age": 5,
    "occupation": 6,
    "zip_code": 7,
    "movie_id": 10,
    "genres": 5,
    "isAdult": 4,
    "startYear": 6,
}


def make_features() -> dict[str, torch.Tensor]:
    return {
        "user_id": torch.tensor([1, 2, 3]),
        "gender": torch.tensor([1, 2, 1]),
        "age": torch.tensor([1, 2, 3]),
        "occupation": torch.tensor([1, 2, 3]),
        "zip_code": torch.tensor([1, 2, 3]),
        "movie_id": torch.tensor([1, 2, 3]),
        "genres": torch.tensor([1, 2, 3]),
        "isAdult": torch.tensor([1, 2, 3]),
        "startYear": torch.tensor([1, 2, 3]),
    }


def test_forward_returns_one_logit_per_example() -> None:
    model = DeepFM(
        FEATURE_DICT,
        embedding_dim=4,
        dnn_hidden_units=(8, 4),
        dropout=0.0,
    )

    logits = model(make_features())
    probabilities = model.predict_proba(make_features())

    assert logits.shape == (3,)
    assert probabilities.shape == (3,)
    assert torch.isfinite(logits).all()
    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)


def test_compute_fm_term_matches_pairwise_dot_products() -> None:
    embeddings = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ]
    )

    fm_term = DeepFM.compute_fm_term(embeddings)

    expected = (
        torch.dot(embeddings[0, 0], embeddings[0, 1])
        + torch.dot(embeddings[0, 0], embeddings[0, 2])
        + torch.dot(embeddings[0, 1], embeddings[0, 2])
    )

    assert fm_term.shape == (1,)
    assert torch.allclose(fm_term[0], expected)
    assert expected.item() == pytest.approx(67.0)


def test_loss_is_finite_and_optimizer_updates_model() -> None:
    torch.manual_seed(42)

    model = DeepFM(
        FEATURE_DICT,
        embedding_dim=4,
        dnn_hidden_units=(8, 4),
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )
    labels = torch.tensor([1.0, 0.0, 1.0])

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    optimizer.zero_grad()
    logits = model(make_features())
    loss = F.binary_cross_entropy_with_logits(
        logits,
        labels,
    )
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )


def test_padding_rows_remain_zero_after_optimizer_step() -> None:
    model = DeepFM(
        FEATURE_DICT,
        embedding_dim=4,
        dnn_hidden_units=(8, 4),
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    features = {
        name: torch.zeros(3, dtype=torch.long)
        for name in DeepFM.FEATURE_NAMES
    }
    labels = torch.tensor([1.0, 0.0, 1.0])

    optimizer.zero_grad()
    loss = F.binary_cross_entropy_with_logits(
        model(features),
        labels,
    )
    loss.backward()
    optimizer.step()

    for embedding in model.linear_embeddings.values():
        assert torch.equal(
            embedding.weight[0],
            torch.zeros_like(embedding.weight[0]),
        )

    for embedding in model.feature_embeddings.values():
        assert torch.equal(
            embedding.weight[0],
            torch.zeros_like(embedding.weight[0]),
        )


def test_forward_rejects_missing_feature() -> None:
    model = DeepFM(FEATURE_DICT)
    features = make_features()
    del features["movie_id"]

    with pytest.raises(
        ValueError,
        match="missing required fields",
    ):
        model(features)


def test_forward_rejects_non_vector_feature() -> None:
    model = DeepFM(FEATURE_DICT)
    features = make_features()
    features["movie_id"] = features["movie_id"].unsqueeze(1)

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        model(features)


def test_forward_rejects_inconsistent_batch_sizes() -> None:
    model = DeepFM(FEATURE_DICT)
    features = make_features()
    features["movie_id"] = torch.tensor([1, 2])

    with pytest.raises(
        ValueError,
        match="same batch size",
    ):
        model(features)


@pytest.mark.parametrize(
    ("invalid_values", "message"),
    [
        (torch.tensor([1, -1, 2]), "negative ID"),
        (
            torch.tensor([1, FEATURE_DICT["movie_id"], 2]),
            "out-of-range ID",
        ),
    ],
)
def test_forward_rejects_invalid_feature_ids(
    invalid_values: torch.Tensor,
    message: str,
) -> None:
    model = DeepFM(FEATURE_DICT)
    features = make_features()
    features["movie_id"] = invalid_values

    with pytest.raises(ValueError, match=message):
        model(features)