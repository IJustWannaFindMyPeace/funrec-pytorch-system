from offline.config import config
from offline.storage.local_deploy import deploy_recall_models


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