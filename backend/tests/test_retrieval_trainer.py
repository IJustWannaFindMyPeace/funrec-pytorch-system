import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from modeling.youtubednn import YouTubeDNN
from offline.training.retrieval_data import RetrievalDataset
from offline.training.retrieval_trainer import (
    load_checkpoint,
    move_batch_to_device,
    resolve_device,
    save_checkpoint,
    train_one_epoch,
)


FEATURE_DICT = {
    "user_id": 6,
    "age": 5,
    "gender": 3,
    "occupation": 7,
    "zip_code": 8,
    "movie_id": 12,
    "genres": 6,
}


def build_dataset():
    data = {
        "user_id": np.array([1, 2, 3, 4], dtype=np.int32),
        "age": np.array([1, 2, 3, 4], dtype=np.int32),
        "gender": np.array([1, 2, 1, 2], dtype=np.int32),
        "occupation": np.array([1, 2, 3, 4], dtype=np.int32),
        "zip_code": np.array([1, 2, 3, 4], dtype=np.int32),
        "hist_movie_id": np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [1, 3, 4],
                [2, 4, 5],
            ],
            dtype=np.int32,
        ),
        "hist_genres": np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [1, 3, 4],
                [2, 4, 5],
            ],
            dtype=np.int32,
        ),
        "movie_id": np.array([3, 4, 5, 6], dtype=np.int32),
    }
    return RetrievalDataset(data)


def test_resolve_device_returns_requested_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_move_batch_to_device_preserves_shapes():
    loader = DataLoader(build_dataset(), batch_size=2)
    features, targets = next(iter(loader))

    moved_features, moved_targets = move_batch_to_device(
        features,
        targets,
        torch.device("cpu"),
    )

    assert moved_targets.shape == (2,)
    assert moved_features["hist_movie_id"].shape == (2, 3)
    assert all(
        value.device.type == "cpu"
        for value in moved_features.values()
    )


def test_train_one_epoch_updates_model_and_respects_batch_limit():
    torch.manual_seed(7)

    model = YouTubeDNN(FEATURE_DICT, embedding_dim=4)
    loader = DataLoader(
        build_dataset(),
        batch_size=2,
        shuffle=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    before = model.movie_embedding.weight.detach().clone()

    stats = train_one_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        max_batches=1,
    )

    assert stats.batches == 1
    assert stats.examples == 2
    assert np.isfinite(stats.loss)
    assert not torch.equal(
        before,
        model.movie_embedding.weight.detach(),
    )
    assert torch.equal(
        model.movie_embedding.weight[0],
        torch.zeros(4),
    )


def test_train_one_epoch_rejects_invalid_batch_limit():
    model = YouTubeDNN(FEATURE_DICT, embedding_dim=4)
    loader = DataLoader(build_dataset(), batch_size=2)
    optimizer = torch.optim.Adam(model.parameters())

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        train_one_epoch(
            model,
            loader,
            optimizer,
            torch.device("cpu"),
            max_batches=0,
        )


def test_checkpoint_round_trip_restores_model_and_optimizer(tmp_path):
    torch.manual_seed(11)

    model = YouTubeDNN(FEATURE_DICT, embedding_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    original_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    checkpoint_path = tmp_path / "retrieval.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=2,
        feature_dict=FEATURE_DICT,
        metrics={"train_loss": 1.25},
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    checkpoint = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
    )

    assert checkpoint["epoch"] == 2
    assert checkpoint["feature_dict"] == FEATURE_DICT
    assert checkpoint["metrics"]["train_loss"] == 1.25

    for name, value in model.state_dict().items():
        assert torch.equal(value, original_state[name])