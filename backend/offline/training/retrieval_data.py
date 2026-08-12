"""Dataset utilities for PyTorch retrieval training."""

from collections.abc import Mapping
from typing import Dict, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


USER_FEATURE_NAMES = (
    "user_id",
    "age",
    "gender",
    "occupation",
    "zip_code",
    "hist_movie_id",
    "hist_genres",
)

TARGET_NAME = "movie_id"


class RetrievalDataset(Dataset):
    """Zero-copy dataset wrapper around preprocessed NumPy arrays."""

    def __init__(self, data: Mapping[str, object]) -> None:
        required_names = (*USER_FEATURE_NAMES, TARGET_NAME)
        missing_names = [
            name for name in required_names if name not in data
        ]

        if missing_names:
            missing = ", ".join(missing_names)
            raise KeyError(f"Missing retrieval fields: {missing}")

        tensors = {
            name: torch.as_tensor(data[name])
            for name in required_names
        }

        lengths = {
            name: len(tensor)
            for name, tensor in tensors.items()
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Retrieval fields have inconsistent lengths: {lengths}"
            )

        for name in USER_FEATURE_NAMES[:5]:
            if tensors[name].ndim != 1:
                raise ValueError(
                    f"{name} must be a one-dimensional array"
                )

        for name in USER_FEATURE_NAMES[5:]:
            if tensors[name].ndim != 2:
                raise ValueError(
                    f"{name} must be a two-dimensional array"
                )

        if tensors[TARGET_NAME].ndim != 1:
            raise ValueError(
                f"{TARGET_NAME} must be a one-dimensional array"
            )

        history_lengths = {
            tensors["hist_movie_id"].shape[1],
            tensors["hist_genres"].shape[1],
        }
        if len(history_lengths) != 1:
            raise ValueError(
                "hist_movie_id and hist_genres must have equal sequence lengths"
            )

        self.features: Dict[str, Tensor] = {
            name: tensors[name]
            for name in USER_FEATURE_NAMES
        }
        self.targets = tensors[TARGET_NAME]
        self.size = lengths[TARGET_NAME]

    def __len__(self) -> int:
        return self.size

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        features = {
            name: tensor[index]
            for name, tensor in self.features.items()
        }
        return features, self.targets[index]