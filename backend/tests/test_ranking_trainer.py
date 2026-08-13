import torch
import pytest

import math

from torch.utils.data import DataLoader

from modeling.deepfm import DeepFM
from offline.training.ranking_data import RankingDataset
from offline.training.ranking_trainer import (
    calculate_binary_auc,
    evaluate,
    load_checkpoint,
    move_batch_to_device,
    resolve_device,
    save_checkpoint,
    train_one_epoch,
)


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


def build_dataset() -> RankingDataset:
    samples = {
        "user_id": [1, 2, 3, 4, 5, 6, 7, 1],
        "gender": [1, 2, 1, 2, 1, 2, 1, 2],
        "age": [1, 2, 3, 4, 1, 2, 3, 4],
        "occupation": [1, 2, 3, 4, 5, 1, 2, 3],
        "zip_code": [1, 2, 3, 4, 5, 6, 1, 2],
        "movie_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "genres": [1, 2, 3, 4, 1, 2, 3, 4],
        "isAdult": [1, 2, 3, 1, 2, 3, 1, 2],
        "startYear": [1, 2, 3, 4, 5, 1, 2, 3],
        "is_click": [1, 0, 1, 0, 1, 0, 1, 0],
    }
    return RankingDataset(samples)


def build_model() -> DeepFM:
    return DeepFM(
        FEATURE_DICT,
        embedding_dim=4,
        dnn_hidden_units=(8, 4),
        dropout=0.0,
    )


def test_resolve_device_returns_requested_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_move_batch_to_device_preserves_shapes() -> None:
    loader = DataLoader(
        build_dataset(),
        batch_size=2,
    )
    features, labels = next(iter(loader))

    moved_features, moved_labels = move_batch_to_device(
        features,
        labels,
        torch.device("cpu"),
    )

    assert moved_labels.shape == (2,)
    assert moved_labels.dtype == torch.float32
    assert all(
        values.shape == (2,)
        for values in moved_features.values()
    )


@pytest.mark.parametrize(
    ("labels", "scores", "expected"),
    [
        (
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0.1, 0.2, 0.8, 0.9]),
            1.0,
        ),
        (
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0.9, 0.8, 0.2, 0.1]),
            0.0,
        ),
        (
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([0.5, 0.5, 0.5, 0.5]),
            0.5,
        ),
    ],
)
def test_calculate_binary_auc(
    labels: torch.Tensor,
    scores: torch.Tensor,
    expected: float,
) -> None:
    assert calculate_binary_auc(
        labels,
        scores,
    ) == pytest.approx(expected)


def test_auc_rejects_single_class() -> None:
    with pytest.raises(
        ValueError,
        match="both positive and negative",
    ):
        calculate_binary_auc(
            torch.tensor([1, 1]),
            torch.tensor([0.2, 0.8]),
        )

def test_evaluate_reports_nan_auc_for_single_class() -> None:
    dataset = build_dataset()
    dataset.labels.fill_(1)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    stats = evaluate(
        build_model(),
        loader,
        torch.device("cpu"),
        max_batches=1,
    )

    assert stats.examples == 4
    assert torch.isfinite(torch.tensor(stats.loss))
    assert math.isnan(stats.auc)

def test_train_one_epoch_updates_model() -> None:
    torch.manual_seed(7)

    model = build_model()
    loader = DataLoader(
        build_dataset(),
        batch_size=4,
        shuffle=False,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    stats = train_one_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        max_batches=1,
    )

    assert stats.batches == 1
    assert stats.examples == 4
    assert torch.isfinite(torch.tensor(stats.loss))
    assert 0.0 <= stats.auc <= 1.0
    assert any(
        not torch.equal(
            before[name],
            value.detach(),
        )
        for name, value in model.state_dict().items()
    )

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


def test_evaluate_preserves_model_parameters() -> None:
    torch.manual_seed(13)

    model = build_model()
    loader = DataLoader(
        build_dataset(),
        batch_size=4,
        shuffle=False,
    )

    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    stats = evaluate(
        model,
        loader,
        torch.device("cpu"),
        max_batches=1,
    )

    assert stats.batches == 1
    assert stats.examples == 4
    assert torch.isfinite(torch.tensor(stats.loss))
    assert 0.0 <= stats.auc <= 1.0
    assert model.training is False

    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])

    assert all(
        parameter.grad is None
        for parameter in model.parameters()
    )


@pytest.mark.parametrize(
    ("function_name", "message"),
    [
        ("train", "greater than zero"),
        ("evaluate", "greater than zero"),
    ],
)
def test_epoch_functions_reject_invalid_batch_limit(
    function_name: str,
    message: str,
) -> None:
    model = build_model()
    loader = DataLoader(
        build_dataset(),
        batch_size=4,
    )

    if function_name == "train":
        optimizer = torch.optim.Adam(
            model.parameters()
        )
        with pytest.raises(ValueError, match=message):
            train_one_epoch(
                model,
                loader,
                optimizer,
                torch.device("cpu"),
                max_batches=0,
            )
    else:
        with pytest.raises(ValueError, match=message):
            evaluate(
                model,
                loader,
                torch.device("cpu"),
                max_batches=0,
            )


def test_checkpoint_round_trip_restores_state(
    tmp_path,
) -> None:
    torch.manual_seed(11)

    model = build_model()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    loader = DataLoader(
        build_dataset(),
        batch_size=4,
    )
    train_one_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        max_batches=1,
    )

    original_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    checkpoint_path = tmp_path / "ranking.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=2,
        feature_dict=FEATURE_DICT,
        metrics={
            "train_loss": 0.5,
            "validation_auc": 0.75,
        },
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    restored_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )
    checkpoint = load_checkpoint(
        checkpoint_path,
        model,
        restored_optimizer,
    )

    assert checkpoint["epoch"] == 2
    assert checkpoint["feature_dict"] == FEATURE_DICT
    assert checkpoint["model_config"] == {
        "embedding_dim": 4,
        "dnn_hidden_units": (8, 4),
        "dropout": 0.0,
    }
    assert (
        checkpoint["metrics"]["validation_auc"]
        == 0.75
    )

    for name, value in model.state_dict().items():
        assert torch.equal(
            value,
            original_state[name],
        )

    assert (
        len(restored_optimizer.state)
        == len(optimizer.state)
    )