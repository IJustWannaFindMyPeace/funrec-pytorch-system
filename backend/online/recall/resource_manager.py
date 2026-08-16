"""PyTorch retrieval resource manager for online recall."""

import logging
import hashlib
import json
import os
import pickle
from pathlib import Path
from threading import Lock
from typing import Any

# Import torch before NumPy/scikit-learn on Windows.
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

from modeling.youtubednn import YouTubeDNN


logger = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RecallResourceManager:
    """Lazily load shared PyTorch retrieval artifacts."""

    _instance = None
    _instance_lock = Lock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return

        self.deploy_dir = Path(
            os.getenv(
                "MODEL_DEPLOY_DIR",
                "/app/tmp/web_project/deployed_models",
            )
        )
        self.recall_dir = self.deploy_dir / "recall"

        requested_device = os.getenv("RECALL_DEVICE", "cpu").lower()
        if requested_device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "RECALL_DEVICE=cuda，但 CUDA 不可用，回退到 CPU"
            )
            requested_device = "cpu"

        self.device = torch.device(requested_device)

        self.user_model = None
        self.encoders = {}
        self.all_movie_ids = []
        self.item_embeddings = None
        self.item_embedding_matrix = None
        self.item_embedding_tensor = None
        self.scoring_contract = None
        self.logit_scale = None
        self.movie_genre_map = {}

        self._resources_loaded = False
        self._load_lock = Lock()
        self.initialized = True

    def _ensure_resources_loaded(self) -> bool:
        """Load resources once and report whether they are ready."""
        if self._resources_loaded:
            return True

        with self._load_lock:
            if self._resources_loaded:
                return True

            self._resources_loaded = self.load_resources()

        return self._resources_loaded

    def load_resources(self) -> bool:
        """Load and validate all deployed retrieval artifacts."""
        logger.info(
            "从本地目录加载 PyTorch 召回资源: %s",
            self.recall_dir,
        )

        vocab_path = self.recall_dir / "vocab_dict.pkl"
        embedding_path = self.recall_dir / "item_embeddings.npy"
        movie_ids_path = self.recall_dir / "movie_ids.npy"
        model_path = self.recall_dir / "retrieval_user_model.pt"
        manifest_path = self.recall_dir / "retrieval_manifest.json"

        required_paths = (
            vocab_path,
            embedding_path,
            movie_ids_path,
            model_path,
        )
        missing_paths = [
            path for path in required_paths if not path.is_file()
        ]

        if missing_paths:
            logger.error(
                "召回工件缺失: %s",
                ", ".join(str(path) for path in missing_paths),
            )
            return False

        try:
            with open(vocab_path, "rb") as file:
                vocab_dict = pickle.load(file)

            encoders = {}
            for feature_name, classes in vocab_dict.items():
                encoder = LabelEncoder()
                encoder.classes_ = np.asarray(classes)
                encoders[feature_name] = encoder

            item_embeddings = np.load(
                embedding_path,
                allow_pickle=False,
            ).astype(np.float32, copy=False)
            movie_ids = np.load(
                movie_ids_path,
                allow_pickle=False,
            )

            artifact = torch.load(
                model_path,
                map_location="cpu",
                weights_only=True,
            )

            feature_dict = artifact["feature_dict"]
            embedding_dim = int(artifact["embedding_dim"])
            history_pooling = artifact.get(
                "history_pooling", "masked_mean"
            )
            max_sequence_length = int(
                artifact.get("max_sequence_length", 10)
            )
            recent_history_length = int(
                artifact.get("recent_history_length", 5)
            )
            scoring_contract = artifact.get(
                "scoring_contract", "legacy_raw_item_v1"
            )
            logit_scale = float(artifact.get("logit_scale", 1.0))

            if scoring_contract == "scaled_cosine_v2" and not manifest_path.is_file():
                raise ValueError(
                    "scaled_cosine_v2 requires retrieval_manifest.json"
                )
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected_files = manifest.get("files", {})
                actual_paths = {
                    "vocab_dict.pkl": vocab_path,
                    "item_embeddings.npy": embedding_path,
                    "movie_ids.npy": movie_ids_path,
                    "retrieval_user_model.pt": model_path,
                }
                if manifest.get("scoring_contract") != scoring_contract:
                    raise ValueError("retrieval manifest scoring contract mismatch")
                if float(manifest.get("logit_scale")) != logit_scale:
                    raise ValueError("retrieval manifest logit scale mismatch")
                for name, path in actual_paths.items():
                    if expected_files.get(name) != file_sha256(path):
                        raise ValueError(f"retrieval artifact hash mismatch: {name}")

            if item_embeddings.ndim != 2:
                raise ValueError(
                    "item_embeddings must be two-dimensional"
                )

            if len(movie_ids) != len(item_embeddings):
                raise ValueError(
                    "movie_ids and item_embeddings are misaligned"
                )

            if item_embeddings.shape[1] != embedding_dim:
                raise ValueError(
                    "item embedding dimension does not match model"
                )

            if scoring_contract == "scaled_cosine_v2":
                norms = np.linalg.norm(item_embeddings, axis=1)
                if not np.allclose(norms, 1.0, atol=1e-5):
                    raise ValueError(
                        "scaled_cosine_v2 requires normalized item embeddings"
                    )

            expected_item_count = feature_dict["movie_id"] - 1
            if len(item_embeddings) != expected_item_count:
                raise ValueError(
                    "item count does not match model feature_dict"
                )

            movie_encoder = encoders.get("movie_id")
            if movie_encoder is None:
                raise ValueError(
                    "movie_id encoder is missing from vocabulary"
                )

            if len(movie_encoder.classes_) != len(movie_ids):
                raise ValueError(
                    "movie vocabulary and movie_ids are misaligned"
                )

            model = YouTubeDNN(
                feature_dict=feature_dict,
                embedding_dim=embedding_dim,
                history_pooling=history_pooling,
                max_sequence_length=max_sequence_length,
                recent_history_length=recent_history_length,
                scoring_contract=scoring_contract,
                logit_scale=logit_scale,
            )
            model.load_state_dict(artifact["model_state_dict"])
            model.to(self.device)
            model.eval()

            padded_embeddings = np.zeros(
                (
                    len(item_embeddings) + 1,
                    embedding_dim,
                ),
                dtype=np.float32,
            )
            padded_embeddings[1:] = item_embeddings

            self.encoders = encoders
            self.all_movie_ids = movie_ids.tolist()
            self.item_embeddings = item_embeddings
            self.item_embedding_matrix = padded_embeddings
            self.item_embedding_tensor = torch.from_numpy(
                padded_embeddings
            ).to(self.device)
            self.user_model = model
            self.scoring_contract = scoring_contract
            self.logit_scale = logit_scale

            logger.info(
                "PyTorch 召回资源加载完成，设备=%s，物品向量形状=%s",
                self.device,
                self.item_embedding_matrix.shape,
            )
            return True

        except Exception:
            logger.exception("加载 PyTorch 召回资源失败")
            self.user_model = None
            self.encoders = {}
            self.all_movie_ids = []
            self.item_embeddings = None
            self.item_embedding_matrix = None
            self.item_embedding_tensor = None
            self.scoring_contract = None
            self.logit_scale = None
            return False

    def encode_feature(
        self,
        feature_name: str,
        raw_value: Any,
    ) -> int:
        """Encode a raw value, reserving encoded ID 0 for unknowns."""
        encoder = self.encoders.get(feature_name)
        if encoder is None or raw_value is None:
            return 0

        classes = encoder.classes_
        candidates = [raw_value]

        if classes.dtype.kind in {"U", "S", "O"}:
            candidates.append(str(raw_value))
        elif classes.dtype.kind in {"i", "u"}:
            try:
                candidates.append(int(raw_value))
            except (TypeError, ValueError):
                pass

        for candidate in candidates:
            try:
                return int(encoder.transform([candidate])[0]) + 1
            except (TypeError, ValueError):
                continue

        return 0

    def set_movie_genre_map(self, movie_db_objects):
        """Build the raw movie ID to encoded genre ID mapping."""
        if not self._ensure_resources_loaded():
            logger.warning(
                "召回资源未就绪，无法构建电影类型映射"
            )
            return

        genre_encoder = self.encoders.get("genres")
        if genre_encoder is None:
            return

        for movie in movie_db_objects:
            raw_movie_id = movie.movie_id
            encoded_genres = []

            try:
                if getattr(movie, "genres", None):
                    raw_genres = (
                        movie.genres
                        if isinstance(movie.genres, list)
                        else str(movie.genres).split("|")
                    )

                    encoded_genres = [
                        encoded
                        for genre in raw_genres
                        if (
                            encoded := self.encode_feature(
                                "genres",
                                genre,
                            )
                        )
                        > 0
                    ]
            except Exception as error:
                logger.debug(
                    "映射电影 %s 的类型时出错: %s",
                    raw_movie_id,
                    error,
                )

            self.movie_genre_map[raw_movie_id] = encoded_genres
