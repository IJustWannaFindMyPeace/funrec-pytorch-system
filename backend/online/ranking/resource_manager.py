"""Resource manager for online PyTorch DeepFM ranking."""

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

# PyTorch must be imported before NumPy-dependent modules on Windows.
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

from modeling.deepfm import DeepFM


logger = logging.getLogger(__name__)


class RankingResourceManager:
    """Load and cache deployed PyTorch ranking artifacts."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        deploy_root = Path(
            os.getenv(
                "MODEL_DEPLOY_DIR",
                "/app/tmp/web_project/deployed_models",
            )
        )
        self.ranking_dir = deploy_root / "ranking"

        requested_device = os.getenv(
            "RANKING_DEVICE",
            "cpu",
        )
        self.device = torch.device(requested_device)

        if (
            self.device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "RANKING_DEVICE requests CUDA, "
                "but CUDA is unavailable"
            )

        self.ranking_model: Optional[DeepFM] = None
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_dict: Dict[str, int] = {}
        self.model_config: Dict[str, Any] = {}

        self.user_features = [
            "user_id",
            "gender",
            "age",
            "occupation",
            "zip_code",
        ]
        self.item_features = [
            "movie_id",
            "genres",
            "isAdult",
            "startYear",
        ]
        self.all_features = (
            self.user_features
            + self.item_features
        )

        self._resources_loaded = False
        self._initialized = True

    def _ensure_resources_loaded(self) -> bool:
        """Load resources once and report readiness."""
        if not self._resources_loaded:
            self._load_resources()
            self._resources_loaded = True

        return self.is_ready

    def _required_paths(self) -> Dict[str, Path]:
        return {
            "vocab": (
                self.ranking_dir / "vocab_dict.pkl"
            ),
            "feature_dict": (
                self.ranking_dir / "feature_dict.pkl"
            ),
            "model_config": (
                self.ranking_dir / "model_config.pkl"
            ),
            "model": (
                self.ranking_dir / "ranking_model.pt"
            ),
        }

    def _load_resources(self) -> None:
        """Load and validate every deployed ranking artifact."""
        logger.info(
            "从本地目录加载 PyTorch 精排资源: %s",
            self.ranking_dir,
        )

        paths = self._required_paths()
        missing_paths = [
            path
            for path in paths.values()
            if not path.exists()
        ]
        if missing_paths:
            missing = "\n".join(
                str(path)
                for path in missing_paths
            )
            raise FileNotFoundError(
                "PyTorch ranking artifacts are missing:\n"
                f"{missing}"
            )

        with open(paths["vocab"], "rb") as file:
            vocab_dict = pickle.load(file)

        with open(
            paths["feature_dict"],
            "rb",
        ) as file:
            feature_dict = pickle.load(file)

        with open(
            paths["model_config"],
            "rb",
        ) as file:
            deployed_config = pickle.load(file)

        artifact = torch.load(
            paths["model"],
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(feature_dict, dict):
            raise ValueError(
                "ranking feature_dict must be a dictionary"
            )
        if not isinstance(vocab_dict, dict):
            raise ValueError(
                "ranking vocab_dict must be a dictionary"
            )

        missing_features = [
            name
            for name in DeepFM.FEATURE_NAMES
            if (
                name not in feature_dict
                or name not in vocab_dict
            )
        ]
        if missing_features:
            raise ValueError(
                "Ranking artifacts are missing features: "
                + ", ".join(missing_features)
            )

        for name in DeepFM.FEATURE_NAMES:
            expected_size = len(vocab_dict[name]) + 1
            if feature_dict[name] != expected_size:
                raise ValueError(
                    f"Vocabulary size mismatch for {name!r}: "
                    f"feature_dict={feature_dict[name]}, "
                    f"vocabulary={expected_size}"
                )

        if (
            artifact.get("feature_dict")
            != feature_dict
        ):
            raise ValueError(
                "Deployed model feature_dict does not "
                "match feature_dict.pkl"
            )

        artifact_config = artifact.get(
            "model_config",
            {},
        )
        config_feature_dict = deployed_config.get(
            "feature_dict"
        )
        if config_feature_dict != feature_dict:
            raise ValueError(
                "model_config.pkl feature_dict does not "
                "match feature_dict.pkl"
            )

        expected_feature_names = list(
            DeepFM.FEATURE_NAMES
        )
        if (
            artifact_config.get("feature_names")
            != expected_feature_names
        ):
            raise ValueError(
                "Deployed model feature order is invalid"
            )

        model = DeepFM(
            feature_dict=feature_dict,
            embedding_dim=int(
                artifact_config["embedding_dim"]
            ),
            dnn_hidden_units=tuple(
                artifact_config[
                    "dnn_hidden_units"
                ]
            ),
            dropout=float(
                artifact_config["dropout"]
            ),
        )
        model.load_state_dict(
            artifact["model_state_dict"]
        )
        model.to(self.device)
        model.eval()

        encoders = {}
        for name in DeepFM.FEATURE_NAMES:
            classes = vocab_dict[name]
            encoder = LabelEncoder()
            encoder.classes_ = (
                classes
                if isinstance(classes, np.ndarray)
                else np.asarray(classes)
            )
            encoders[name] = encoder

        self.ranking_model = model
        self.encoders = encoders
        self.feature_dict = dict(feature_dict)
        self.model_config = dict(
            artifact_config
        )

        logger.info(
            "PyTorch 精排资源加载完成，设备=%s",
            self.device,
        )

    def encode_feature(
        self,
        feature_name: str,
        value: Any,
    ) -> int:
        """Encode one raw feature; return 0 when unknown."""
        self._ensure_resources_loaded()

        if value is None:
            return 0

        encoder = self.encoders.get(feature_name)
        if encoder is None:
            return 0

        try:
            classes = encoder.classes_
            kind = classes.dtype.kind

            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and kind in {"U", "S", "O"}
            ):
                value = str(
                    int(value)
                    if isinstance(value, float)
                    else value
                )
            elif (
                isinstance(value, str)
                and kind in {"i", "u", "f"}
            ):
                value = (
                    int(value)
                    if value.isdigit()
                    else value
                )

            if value not in classes:
                return 0

            return (
                int(
                    encoder.transform([value])[0]
                )
                + 1
            )
        except (TypeError, ValueError):
            return 0

    @property
    def is_ready(self) -> bool:
        """Whether every ranking resource is ready."""
        return (
            self.ranking_model is not None
            and set(self.encoders)
            == set(DeepFM.FEATURE_NAMES)
            and bool(self.feature_dict)
        )