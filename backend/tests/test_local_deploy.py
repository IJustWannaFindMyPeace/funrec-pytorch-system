from offline.config import config
from offline.storage.local_deploy import (
    deploy_ranking_models,
    deploy_recall_models,
)

def test_deploy_recall_models_copies_pytorch_artifacts(
    monkeypatch,
    tmp_path,
):
    source_dir = tmp_path / "source"
    saved_models_dir = source_dir / "saved_models"
    deploy_dir = tmp_path / "deployed_models"

    saved_models_dir.mkdir(parents=True)

    vocab_path = source_dir / "vocab_dict.pkl"
    item_embeddings_path = source_dir / "item_embeddings.npy"
    movie_ids_path = source_dir / "movie_ids.npy"
    user_model_path = (
        saved_models_dir / "retrieval_user_model.pt"
    )

    source_files = {
        vocab_path: b"vocabulary",
        item_embeddings_path: b"item embeddings",
        movie_ids_path: b"movie ids",
        user_model_path: b"pytorch user model",
    }

    for path, content in source_files.items():
        path.write_bytes(content)

    monkeypatch.setattr(
        config,
        "VOCAB_DICT_PATH",
        vocab_path,
    )
    monkeypatch.setattr(
        config,
        "ITEM_EMB_PATH",
        item_embeddings_path,
    )
    monkeypatch.setattr(
        config,
        "MOVIE_IDS_PATH",
        movie_ids_path,
    )
    monkeypatch.setattr(
        config,
        "SAVED_MODELS_DIR",
        saved_models_dir,
    )

    deploy_recall_models(deploy_dir)

    expected_recall_files = {
        "vocab_dict.pkl": b"vocabulary",
        "item_embeddings.npy": b"item embeddings",
        "movie_ids.npy": b"movie ids",
        "retrieval_user_model.pt": b"pytorch user model",
    }

    for name, expected_content in expected_recall_files.items():
        deployed_path = deploy_dir / "recall" / name

        assert deployed_path.exists()
        assert deployed_path.read_bytes() == expected_content

    # The three legacy root-level data files remain for compatibility.
    for name in (
        "vocab_dict.pkl",
        "item_embeddings.npy",
        "movie_ids.npy",
    ):
        assert (deploy_dir / name).exists()

    # TensorFlow SavedModel deployment metadata must not be created.
    assert not (
        deploy_dir
        / "model"
        / "user_recall"
        / "active.json"
    ).exists()

def test_deploy_ranking_models_copies_pytorch_artifacts(
    monkeypatch,
    tmp_path,
):
    source_dir = tmp_path / "source"
    saved_models_dir = source_dir / "saved_models"
    deploy_dir = tmp_path / "deployed_models"

    saved_models_dir.mkdir(parents=True)

    vocab_path = source_dir / "ranking_vocab_dict.pkl"
    feature_dict_path = (
        source_dir / "ranking_feature_dict.pkl"
    )
    model_config_path = (
        source_dir / "ranking_model_config.pkl"
    )
    ranking_model_path = (
        saved_models_dir / "ranking_model.pt"
    )

    source_files = {
        vocab_path: b"ranking vocabulary",
        feature_dict_path: b"ranking feature dictionary",
        model_config_path: b"ranking model config",
        ranking_model_path: b"pytorch ranking model",
    }

    for path, content in source_files.items():
        path.write_bytes(content)

    monkeypatch.setattr(
        config,
        "RANKING_VOCAB_DICT_PATH",
        vocab_path,
    )
    monkeypatch.setattr(
        config,
        "RANKING_FEATURE_DICT_PATH",
        feature_dict_path,
    )
    monkeypatch.setattr(
        config,
        "TEMP_DIR",
        source_dir,
    )
    monkeypatch.setattr(
        config,
        "SAVED_MODELS_DIR",
        saved_models_dir,
    )

    legacy_dir = (
        deploy_dir
        / "model"
        / "ranking"
        / "v1"
        / "ranking_model"
    )
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "saved_model.pb").write_bytes(
        b"legacy tensorflow model"
    )

    deploy_ranking_models(deploy_dir)

    expected_files = {
        "vocab_dict.pkl": b"ranking vocabulary",
        "feature_dict.pkl": (
            b"ranking feature dictionary"
        ),
        "model_config.pkl": (
            b"ranking model config"
        ),
        "ranking_model.pt": (
            b"pytorch ranking model"
        ),
    }

    for name, expected_content in (
        expected_files.items()
    ):
        deployed_path = (
            deploy_dir / "ranking" / name
        )

        assert deployed_path.exists()
        assert (
            deployed_path.read_bytes()
            == expected_content
        )

    assert not (
        deploy_dir / "model" / "ranking"
    ).exists()


def test_deploy_ranking_models_rejects_missing_artifact(
    monkeypatch,
    tmp_path,
):
    source_dir = tmp_path / "source"
    saved_models_dir = source_dir / "saved_models"
    deploy_dir = tmp_path / "deployed_models"

    saved_models_dir.mkdir(parents=True)

    vocab_path = source_dir / "ranking_vocab_dict.pkl"
    feature_dict_path = (
        source_dir / "ranking_feature_dict.pkl"
    )
    model_config_path = (
        source_dir / "ranking_model_config.pkl"
    )

    vocab_path.write_bytes(b"vocabulary")
    feature_dict_path.write_bytes(b"feature dictionary")
    model_config_path.write_bytes(b"model config")

    monkeypatch.setattr(
        config,
        "RANKING_VOCAB_DICT_PATH",
        vocab_path,
    )
    monkeypatch.setattr(
        config,
        "RANKING_FEATURE_DICT_PATH",
        feature_dict_path,
    )
    monkeypatch.setattr(
        config,
        "TEMP_DIR",
        source_dir,
    )
    monkeypatch.setattr(
        config,
        "SAVED_MODELS_DIR",
        saved_models_dir,
    )

    import pytest

    with pytest.raises(
        FileNotFoundError,
        match="精排部署工件缺失",
    ):
        deploy_ranking_models(deploy_dir)