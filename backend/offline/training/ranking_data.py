"""Dataset utilities for PyTorch ranking training."""

from collections.abc import Mapping, Sequence
from typing import Dict, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


RANKING_FEATURE_NAMES = (
    "user_id",
    "gender",
    "age",
    "occupation",
    "zip_code",
    "movie_id",
    "genres",
    "isAdult",
    "startYear",
)

LABEL_NAME = "is_click"


class RankingDataset(
    Dataset[Tuple[Dict[str, Tensor], Tensor]]
):
    """Pointwise ranking dataset for DeepFM training."""

    def __init__(
        self,
        samples: Mapping[str, Sequence],
    ) -> None:
        if not isinstance(samples, Mapping):
            raise ValueError("samples must be a mapping")

        required_fields = (
            *RANKING_FEATURE_NAMES,
            LABEL_NAME,
        )
        missing_fields = [
            name
            for name in required_fields
            if name not in samples
        ]
        if missing_fields:
            raise ValueError(
                "samples are missing required fields: "
                + ", ".join(missing_fields)
            )

        tensors = {
            name: torch.as_tensor(samples[name])
            for name in required_fields
        }

        for name, values in tensors.items():
            if values.ndim != 1:
                raise ValueError(
                    f"field {name!r} must be one-dimensional"
                )

        lengths = {
            name: values.shape[0]
            for name, values in tensors.items()
        }
        expected_length = lengths[required_fields[0]]

        inconsistent_fields = {
            name: length
            for name, length in lengths.items()
            if length != expected_length
        }
        if inconsistent_fields:
            raise ValueError(
                "all sample fields must have the same length: "
                f"{inconsistent_fields}"
            )

        for name in RANKING_FEATURE_NAMES:
            values = tensors[name]

            if values.dtype == torch.bool:
                raise ValueError(
                    f"feature {name!r} must contain integer IDs"
                )

            if values.is_floating_point():
                raise ValueError(
                    f"feature {name!r} must contain integer IDs"
                )

            if torch.any(values < 0):
                raise ValueError(
                    f"feature {name!r} contains a negative ID"
                )

        labels = tensors[LABEL_NAME]

        if labels.dtype == torch.bool:
            labels = labels.to(torch.float32)
        elif labels.is_floating_point():
            labels = labels.to(torch.float32)
        else:
            labels = labels.to(torch.float32)

        if torch.any((labels != 0) & (labels != 1)):
            raise ValueError(
                "is_click labels must contain only 0 or 1"
            )

        self.features = {
            name: tensors[name]
            for name in RANKING_FEATURE_NAMES
        }
        self.labels = labels
        self._length = expected_length

    def __len__(self) -> int:
        return self._length

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        features = {
            name: values[index]
            for name, values in self.features.items()
        }
        return features, self.labels[index]


def build_ranking_dataloader(
    samples: Mapping[str, Sequence],
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Build a DataLoader for pointwise ranking samples."""
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")

    if (
        not isinstance(num_workers, int)
        or isinstance(num_workers, bool)
        or num_workers < 0
    ):
        raise ValueError(
            "num_workers must be a non-negative integer"
        )

    dataset = RankingDataset(samples)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )