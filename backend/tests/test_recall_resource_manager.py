import pickle

import numpy as np
import torch

from modeling.youtubednn import (
    SCORING_CONTRACT_SCALED_COSINE_V2,
    YouTubeDNN,
)
from online.recall.resource_manager import RecallResourceManager


def build_deployed_artifacts(tmp_path):
    deploy_dir = tmp_path / "deployed_models"
    recall_dir = deploy_dir / "recall"
    recall_dir.mkdir(parents=True)

    feature_dict = {
        "user_id": 3,
        "gender": 3,
        "age": 3,
        "occupation": 3,
        "zip_code": 3,
        "movie_id": 4,
        "genres": 3,
        "isAdult": 2,
        "startYear": 3,
    }

    vocab_dict = {
        "user_id": np.array(["1", "2"]),
        "gender": np.array(["F", "M"]),
        "age": np.array(["18", "25"]),
        "occupation": np.array(["1", "2"]),
        "zip_code": np.array(["10001", "10002"]),
        "movie_id": np.array(["1", "2", "3"]),
        "genres": np.array(["Action", "Comedy"]),
    }

    with open(recall_dir / "vocab_dict.pkl", "wb") as file:
        pickle.dump(vocab_dict, file)

    movie_ids = np.array(["1", "2", "3"])
    item_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.6, 0.8],
        ],
        dtype=np.float32,
    )

    np.save(recall_dir / "movie_ids.npy", movie_ids)
    np.save(recall_dir / "item_embeddings.npy", item_embeddings)

    model = YouTubeDNN(
        feature_dict=feature_dict,
        embedding_dim=2,
    )
    torch.save(
        {
            "feature_dict": feature_dict,
            "embedding_dim": 2,
            "model_state_dict": model.state_dict(),
        },
        recall_dir / "retrieval_user_model.pt",
    )

    return deploy_dir, item_embeddings


def reset_resource_manager():
    RecallResourceManager._instance = None


def test_resource_manager_loads_aligned_pytorch_artifacts(
    tmp_path,
    monkeypatch,
):
    deploy_dir, expected_embeddings = build_deployed_artifacts(
        tmp_path
    )

    monkeypatch.setenv("MODEL_DEPLOY_DIR", str(deploy_dir))
    monkeypatch.setenv("RECALL_DEVICE", "cpu")
    reset_resource_manager()

    manager = RecallResourceManager()

    assert manager._ensure_resources_loaded() is True
    assert manager.device == torch.device("cpu")
    assert isinstance(manager.user_model, YouTubeDNN)
    assert manager.user_model.training is False

    np.testing.assert_array_equal(
        manager.item_embeddings,
        expected_embeddings,
    )
    assert manager.item_embedding_matrix.shape == (4, 2)
    assert manager.item_embedding_tensor.shape == (4, 2)
    assert manager.item_embedding_tensor.device.type == "cpu"
    np.testing.assert_array_equal(
        manager.item_embedding_matrix[0],
        np.zeros(2, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        manager.item_embedding_matrix[1:],
        expected_embeddings,
    )

    assert manager.all_movie_ids == ["1", "2", "3"]
    assert manager.encode_feature("movie_id", "1") == 1
    assert manager.encode_feature("movie_id", 1) == 1
    assert manager.encode_feature("movie_id", "missing") == 0

    reset_resource_manager()


def test_resource_manager_rejects_misaligned_movie_ids(
    tmp_path,
    monkeypatch,
):
    deploy_dir, _ = build_deployed_artifacts(tmp_path)
    recall_dir = deploy_dir / "recall"

    np.save(
        recall_dir / "movie_ids.npy",
        np.array(["1", "2"]),
    )

    monkeypatch.setenv("MODEL_DEPLOY_DIR", str(deploy_dir))
    monkeypatch.setenv("RECALL_DEVICE", "cpu")
    reset_resource_manager()

    manager = RecallResourceManager()

    assert manager._ensure_resources_loaded() is False
    assert manager.user_model is None
    assert manager.item_embedding_matrix is None
    assert manager.item_embedding_tensor is None

    reset_resource_manager()


def test_resource_manager_reports_missing_artifacts(
    tmp_path,
    monkeypatch,
):
    deploy_dir = tmp_path / "deployed_models"
    deploy_dir.mkdir()

    monkeypatch.setenv("MODEL_DEPLOY_DIR", str(deploy_dir))
    monkeypatch.setenv("RECALL_DEVICE", "cpu")
    reset_resource_manager()

    manager = RecallResourceManager()

    assert manager._ensure_resources_loaded() is False
    assert manager.user_model is None
    assert manager.item_embedding_matrix is None

    reset_resource_manager()


def test_resource_manager_rejects_v2_artifacts_without_manifest(
    tmp_path,
    monkeypatch,
):
    deploy_dir, _ = build_deployed_artifacts(tmp_path)
    model_path = deploy_dir / "recall" / "retrieval_user_model.pt"
    artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    artifact["scoring_contract"] = SCORING_CONTRACT_SCALED_COSINE_V2
    artifact["logit_scale"] = 10.0
    torch.save(artifact, model_path)

    monkeypatch.setenv("MODEL_DEPLOY_DIR", str(deploy_dir))
    monkeypatch.setenv("RECALL_DEVICE", "cpu")
    reset_resource_manager()

    assert RecallResourceManager()._ensure_resources_loaded() is False
    reset_resource_manager()
