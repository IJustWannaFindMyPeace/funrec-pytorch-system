import numpy as np
import pytest
import torch

import offline.training.train_retrieval as training
from modeling.youtubednn import YouTubeDNN
from offline.config import config


FEATURE_DICT = {
    "user_id": 6,
    "age": 5,
    "gender": 3,
    "occupation": 7,
    "zip_code": 8,
    "movie_id": 5,
    "genres": 6,
}


def test_activity_balanced_weights_use_only_train_user_activity():
    train = {"user_id": [1, 1, 2, 3, 3, 3, 4, 4, 4, 4]}

    weights = training.build_activity_balanced_user_weights(train)

    assert weights.shape[0] == 5
    assert weights[0].item() == 0.0
    assert torch.isfinite(weights).all()
    assert weights[1:].min().item() > 0.0


def test_training_rejects_artifacts_with_an_embedded_test_split():
    with pytest.raises(ValueError, match="exactly Train and Validation"):
        training.validate_training_selection_samples(
            {"train": {}, "validation": {}, "test": {}}
        )

    valid = {"train": {}, "validation": {}}
    assert training.validate_training_selection_samples(valid) is valid


def configure_temporary_export_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(
        training,
        "USER_MODEL_PATH",
        tmp_path / "retrieval_user_model.pt",
    )
    monkeypatch.setattr(
        config,
        "SAVED_MODELS_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        config,
        "ITEM_EMB_PATH",
        tmp_path / "item_embeddings.npy",
    )
    monkeypatch.setattr(
        config,
        "MOVIE_IDS_PATH",
        tmp_path / "movie_ids.npy",
    )


def test_export_retrieval_artifacts_are_safe_and_aligned(
    monkeypatch,
    tmp_path,
):
    configure_temporary_export_paths(monkeypatch, tmp_path)

    model = YouTubeDNN(FEATURE_DICT, embedding_dim=4)
    vocab_dict = {
        "movie_id": np.array(
            ["10", "20", "30", "40"],
            dtype=object,
        )
    }

    training.export_retrieval_artifacts(
        model=model,
        feature_dict=FEATURE_DICT,
        vocab_dict=vocab_dict,
    )

    item_embeddings = np.load(config.ITEM_EMB_PATH)
    movie_ids = np.load(config.MOVIE_IDS_PATH)
    user_artifact = torch.load(
        training.USER_MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    assert item_embeddings.shape == (4, 4)
    assert movie_ids.shape == (4,)
    assert movie_ids.dtype.kind in {"U", "S"}
    assert movie_ids.tolist() == ["10", "20", "30", "40"]
    assert np.allclose(
        np.linalg.norm(item_embeddings, axis=1),
        1.0,
        atol=1e-6,
    )
    assert user_artifact["feature_dict"] == FEATURE_DICT
    assert user_artifact["embedding_dim"] == 4


def test_export_rejects_misaligned_movie_vocabulary(
    monkeypatch,
    tmp_path,
):
    configure_temporary_export_paths(monkeypatch, tmp_path)

    model = YouTubeDNN(FEATURE_DICT, embedding_dim=4)
    vocab_dict = {
        "movie_id": np.array(["10", "20", "30"], dtype=object)
    }

    with pytest.raises(
        ValueError,
        match="different lengths",
    ):
        training.export_retrieval_artifacts(
            model=model,
            feature_dict=FEATURE_DICT,
            vocab_dict=vocab_dict,
        )


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("epochs", 0, "epochs must be greater than zero"),
        ("batch_size", 0, "batch_size must be greater than zero"),
        (
            "learning_rate",
            0,
            "learning_rate must be greater than zero",
        ),
        ("patience", 0, "patience must be greater than zero"),
        (
            "num_workers",
            -1,
            "num_workers must not be negative",
        ),
    ],
)
def test_training_rejects_invalid_configuration(
    argument,
    value,
    message,
):
    arguments = {
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "device_name": "cpu",
        "patience": 1,
        "num_workers": 0,
    }
    arguments[argument] = value

    with pytest.raises(ValueError, match=message):
        training.run_retrieval_training(**arguments)
