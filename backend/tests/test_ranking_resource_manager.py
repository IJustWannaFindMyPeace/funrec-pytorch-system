import torch
import pickle

import numpy as np
import pytest

from modeling.deepfm import DeepFM
from online.ranking.resource_manager import (
    RankingResourceManager,
)


FEATURE_DICT = {
    "user_id": 4,
    "gender": 3,
    "age": 4,
    "occupation": 4,
    "zip_code": 4,
    "movie_id": 5,
    "genres": 4,
    "isAdult": 3,
    "startYear": 4,
}


def build_vocab_dict():
    return {
        name: np.asarray(
            [
                str(index)
                for index in range(
                    1,
                    FEATURE_DICT[name],
                )
            ]
        )
        for name in DeepFM.FEATURE_NAMES
    }


def write_artifacts(root) -> None:
    ranking_dir = root / "ranking"
    ranking_dir.mkdir(parents=True)

    model = DeepFM(
        feature_dict=FEATURE_DICT,
        embedding_dim=4,
        dnn_hidden_units=(8, 4),
        dropout=0.0,
    )
    model_config = {
        "feature_names": list(
            DeepFM.FEATURE_NAMES
        ),
        "embedding_dim": 4,
        "dnn_hidden_units": (8, 4),
        "dropout": 0.0,
    }

    with open(
        ranking_dir / "vocab_dict.pkl",
        "wb",
    ) as file:
        pickle.dump(build_vocab_dict(), file)

    with open(
        ranking_dir / "feature_dict.pkl",
        "wb",
    ) as file:
        pickle.dump(FEATURE_DICT, file)

    with open(
        ranking_dir / "model_config.pkl",
        "wb",
    ) as file:
        pickle.dump(
            {
                "feature_dict": FEATURE_DICT,
                "feature_names": list(
                    DeepFM.FEATURE_NAMES
                ),
                "model_config": model_config,
            },
            file,
        )

    torch.save(
        {
            "feature_dict": FEATURE_DICT,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
        },
        ranking_dir / "ranking_model.pt",
    )


@pytest.fixture(autouse=True)
def reset_singleton():
    RankingResourceManager._instance = None
    yield
    RankingResourceManager._instance = None


def test_resource_manager_loads_pytorch_model(
    monkeypatch,
    tmp_path,
) -> None:
    write_artifacts(tmp_path)

    monkeypatch.setenv(
        "MODEL_DEPLOY_DIR",
        str(tmp_path),
    )
    monkeypatch.setenv(
        "RANKING_DEVICE",
        "cpu",
    )

    manager = RankingResourceManager()

    assert manager._ensure_resources_loaded()
    assert manager.device == torch.device("cpu")
    assert isinstance(
        manager.ranking_model,
        DeepFM,
    )
    assert manager.ranking_model.training is False
    assert manager.feature_dict == FEATURE_DICT
    assert manager.encode_feature(
        "movie_id",
        "1",
    ) == 1
    assert manager.encode_feature(
        "movie_id",
        1,
    ) == 1
    assert manager.encode_feature(
        "movie_id",
        "unknown",
    ) == 0


def test_resource_manager_rejects_missing_artifacts(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(
        "MODEL_DEPLOY_DIR",
        str(tmp_path),
    )
    monkeypatch.setenv(
        "RANKING_DEVICE",
        "cpu",
    )

    manager = RankingResourceManager()

    with pytest.raises(
        FileNotFoundError,
        match="artifacts are missing",
    ):
        manager._ensure_resources_loaded()


def test_resource_manager_rejects_vocab_mismatch(
    monkeypatch,
    tmp_path,
) -> None:
    write_artifacts(tmp_path)

    feature_path = (
        tmp_path
        / "ranking"
        / "feature_dict.pkl"
    )
    invalid_feature_dict = dict(FEATURE_DICT)
    invalid_feature_dict["movie_id"] += 1

    with open(feature_path, "wb") as file:
        pickle.dump(
            invalid_feature_dict,
            file,
        )

    monkeypatch.setenv(
        "MODEL_DEPLOY_DIR",
        str(tmp_path),
    )
    monkeypatch.setenv(
        "RANKING_DEVICE",
        "cpu",
    )

    manager = RankingResourceManager()

    with pytest.raises(
        ValueError,
        match="Vocabulary size mismatch",
    ):
        manager._ensure_resources_loaded()