"""PyTorch implementation of the DeepFM ranking model."""

from collections.abc import Mapping, Sequence
from typing import Dict, Tuple

import torch
from torch import Tensor, nn


class DeepFM(nn.Module):
    """DeepFM model for pointwise binary ranking."""

    FEATURE_NAMES: Tuple[str, ...] = (
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

    def __init__(
        self,
        feature_dict: Dict[str, int],
        embedding_dim: int = 16,
        dnn_hidden_units: Sequence[int] = (128, 64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        missing_features = [
            name
            for name in self.FEATURE_NAMES
            if name not in feature_dict
        ]
        if missing_features:
            raise ValueError(
                "feature_dict is missing required features: "
                + ", ".join(missing_features)
            )

        invalid_vocab_sizes = {
            name: feature_dict[name]
            for name in self.FEATURE_NAMES
            if (
                not isinstance(feature_dict[name], int)
                or isinstance(feature_dict[name], bool)
                or feature_dict[name] <= 0
            )
        }
        if invalid_vocab_sizes:
            raise ValueError(
                "feature vocabulary sizes must be positive integers: "
                f"{invalid_vocab_sizes}"
            )

        if (
            not isinstance(embedding_dim, int)
            or isinstance(embedding_dim, bool)
            or embedding_dim <= 0
        ):
            raise ValueError("embedding_dim must be a positive integer")

        hidden_units = tuple(dnn_hidden_units)
        if not hidden_units:
            raise ValueError("dnn_hidden_units must not be empty")

        if any(
            not isinstance(unit, int)
            or isinstance(unit, bool)
            or unit <= 0
            for unit in hidden_units
        ):
            raise ValueError(
                "dnn_hidden_units must contain positive integers"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.feature_dict = {
            name: feature_dict[name]
            for name in self.FEATURE_NAMES
        }
        self.embedding_dim = embedding_dim
        self.dnn_hidden_units = hidden_units
        self.dropout = float(dropout)

        self.linear_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(
                    num_embeddings=self.feature_dict[name],
                    embedding_dim=1,
                    padding_idx=0,
                )
                for name in self.FEATURE_NAMES
            }
        )

        self.feature_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(
                    num_embeddings=self.feature_dict[name],
                    embedding_dim=embedding_dim,
                    padding_idx=0,
                )
                for name in self.FEATURE_NAMES
            }
        )

        dnn_layers = []
        input_dim = len(self.FEATURE_NAMES) * embedding_dim

        for hidden_dim in hidden_units:
            dnn_layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                ]
            )
            input_dim = hidden_dim

        dnn_layers.append(nn.Linear(input_dim, 1))
        self.dnn = nn.Sequential(*dnn_layers)

        self.bias = nn.Parameter(torch.zeros(1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize trainable parameters and zero padding rows."""
        for embedding in self.linear_embeddings.values():
            nn.init.zeros_(embedding.weight)

        for embedding in self.feature_embeddings.values():
            nn.init.normal_(
                embedding.weight,
                mean=0.0,
                std=0.01,
            )

        for module in self.dnn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        with torch.no_grad():
            for embedding in self.linear_embeddings.values():
                embedding.weight[0].zero_()

            for embedding in self.feature_embeddings.values():
                embedding.weight[0].zero_()

            self.bias.zero_()

    def _validate_features(
        self,
        features: Mapping[str, Tensor],
    ) -> int:
        if not isinstance(features, Mapping):
            raise ValueError("features must be a mapping of tensors")

        missing_features = [
            name
            for name in self.FEATURE_NAMES
            if name not in features
        ]
        if missing_features:
            raise ValueError(
                "features are missing required fields: "
                + ", ".join(missing_features)
            )

        batch_size = None

        for name in self.FEATURE_NAMES:
            values = features[name]

            if not isinstance(values, Tensor):
                raise ValueError(
                    f"feature {name!r} must be a torch.Tensor"
                )

            if values.ndim != 1:
                raise ValueError(
                    f"feature {name!r} must be one-dimensional"
                )

            if batch_size is None:
                batch_size = values.shape[0]
            elif values.shape[0] != batch_size:
                raise ValueError(
                    "all features must have the same batch size"
                )

            if torch.any(values < 0):
                raise ValueError(
                    f"feature {name!r} contains a negative ID"
                )

            if torch.any(values >= self.feature_dict[name]):
                raise ValueError(
                    f"feature {name!r} contains an out-of-range ID"
                )

        if batch_size is None:
            raise ValueError("features must not be empty")

        return batch_size

    @staticmethod
    def compute_fm_term(embeddings: Tensor) -> Tensor:
        """
        Compute the standard second-order FM interaction term.

        Args:
            embeddings: Tensor with shape
                [batch_size, field_count, embedding_dim].

        Returns:
            Tensor with shape [batch_size].
        """
        if embeddings.ndim != 3:
            raise ValueError(
                "embeddings must have shape "
                "[batch_size, field_count, embedding_dim]"
            )

        summed_embeddings = embeddings.sum(dim=1)
        squared_sum = summed_embeddings.square()
        sum_of_squares = embeddings.square().sum(dim=1)

        return 0.5 * (
            squared_sum - sum_of_squares
        ).sum(dim=1)

    def compute_components(
        self,
        features: Mapping[str, Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the linear, FM and DNN logit components.

        Returns:
            Tuple containing three tensors, each with shape [batch_size].
        """
        self._validate_features(features)

        encoded_features = {
            name: features[name].long()
            for name in self.FEATURE_NAMES
        }

        linear_terms = [
            self.linear_embeddings[name](
                encoded_features[name]
            ).squeeze(-1)
            for name in self.FEATURE_NAMES
        ]
        linear_logit = torch.stack(
            linear_terms,
            dim=1,
        ).sum(dim=1)

        feature_embeddings = torch.stack(
            [
                self.feature_embeddings[name](
                    encoded_features[name]
                )
                for name in self.FEATURE_NAMES
            ],
            dim=1,
        )

        fm_logit = self.compute_fm_term(
            feature_embeddings
        )

        dnn_input = feature_embeddings.flatten(start_dim=1)
        dnn_logit = self.dnn(dnn_input).squeeze(-1)

        return linear_logit, fm_logit, dnn_logit

    def forward(
        self,
        features: Mapping[str, Tensor],
    ) -> Tensor:
        """Return raw binary-classification logits."""
        linear_logit, fm_logit, dnn_logit = (
            self.compute_components(features)
        )

        return (
            self.bias
            + linear_logit
            + fm_logit
            + dnn_logit
        )

    def predict_proba(
        self,
        features: Mapping[str, Tensor],
    ) -> Tensor:
        """Return click probabilities."""
        return torch.sigmoid(self(features))