import pickle

import pytest
import torch

import offline.training.train_ranking as training
from modeling.deepfm import DeepFM
from offline.config import config


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

VOCAB_DICT = {
    name: list(range(FEATURE_DICT[name] - 1))
    for name in DeepFM.FEATURE_NAMES
}


def configure_temporary_paths(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        training,
        "RANKING_MODEL_PATH",
        tmp_path / "ranking_model.pt",
    )
    monkeypatch.setattr(
        training,
        "MODEL_CONFIG_PATH",
        tmp_path / "ranking_model_config.pkl",
    )
    monkeypatch.setattr(
        config,
        "SAVED_MODELS_DIR",
        tmp_path,
    )


def build_model() -> DeepFM:
    return DeepFM(
        feature_dict=FEATURE_DICT,
        embedding_dim=4,
        dnn_hidden_units=(8, 4),
        dropout=0.0,
    )


def test_export_ranking_artifacts_are_loadable(
    monkeypatch,
    tmp_path,
) -> None:
    configure_temporary_paths(
        monkeypatch,
        tmp_path,
    )

    model = build_model()

    training.export_ranking_artifacts(
        model=model,
        feature_dict=FEATURE_DICT,
        vocab_dict=VOCAB_DICT,
    )

    artifact = torch.load(
        training.RANKING_MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    with open(
        training.MODEL_CONFIG_PATH,
        "rb",
    ) as file:
        model_config = pickle.load(file)

    assert artifact["feature_dict"] == FEATURE_DICT
    assert artifact["model_config"] == {
        "feature_names": list(
            DeepFM.FEATURE_NAMES
        ),
        "embedding_dim": 4,
        "dnn_hidden_units": (8, 4),
        "dropout": 0.0,
    }
    assert (
        model_config["feature_dict"]
        == FEATURE_DICT
    )
    assert model_config["feature_names"] == list(
        DeepFM.FEATURE_NAMES
    )

    restored = DeepFM(
        feature_dict=artifact["feature_dict"],
        embedding_dim=(
            artifact["model_config"][
                "embedding_dim"
            ]
        ),
        dnn_hidden_units=(
            artifact["model_config"][
                "dnn_hidden_units"
            ]
        ),
        dropout=(
            artifact["model_config"]["dropout"]
        ),
    )
    restored.load_state_dict(
        artifact["model_state_dict"]
    )

    for name, value in model.state_dict().items():
        assert torch.equal(
            value,
            restored.state_dict()[name],
        )


def test_export_rejects_missing_vocabulary(
    monkeypatch,
    tmp_path,
) -> None:
    configure_temporary_paths(
        monkeypatch,
        tmp_path,
    )

    vocab_dict = dict(VOCAB_DICT)
    del vocab_dict["movie_id"]

    with pytest.raises(
        ValueError,
        match="missing features",
    ):
        training.export_ranking_artifacts(
            model=build_model(),
            feature_dict=FEATURE_DICT,
            vocab_dict=vocab_dict,
        )


def test_history_round_trip(tmp_path) -> None:
    history_path = tmp_path / "history.json"
    history = [
        {
            "epoch": 1,
            "train_loss": 0.5,
            "validation_auc": 0.75,
        }
    ]

    training.save_history(
        history_path,
        history,
    )

    assert training.load_history(
        history_path
    ) == history


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        (
            "epochs",
            0,
            "epochs must be greater than zero",
        ),
        (
            "batch_size",
            0,
            "batch_size must be greater than zero",
        ),
        (
            "learning_rate",
            0,
            "learning_rate must be greater than zero",
        ),
        (
            "patience",
            0,
            "patience must be greater than zero",
        ),
        (
            "num_workers",
            -1,
            "num_workers must not be negative",
        ),
        (
            "max_train_batches",
            0,
            "max_train_batches must be greater than zero",
        ),
        (
            "max_eval_batches",
            0,
            "max_eval_batches must be greater than zero",
        ),
    ],
)
def test_training_rejects_invalid_configuration(
    argument,
    value,
    message,
) -> None:
    arguments = {
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "device_name": "cpu",
        "patience": 1,
        "num_workers": 0,
        "max_train_batches": 1,
        "max_eval_batches": 1,
    }
    arguments[argument] = value

    with pytest.raises(
        ValueError,
        match=message,
    ):
        training.run_ranking_training(
            **arguments
        )