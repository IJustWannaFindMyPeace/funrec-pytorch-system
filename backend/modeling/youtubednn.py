"""PyTorch implementation of the YouTubeDNN two-tower retrieval model."""

from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MaskedMeanPooling(nn.Module):
    """Average sequence embeddings while excluding padding ID 0."""

    def forward(self, embeddings: Tensor, ids: Tensor) -> Tensor:
        mask = ids.ne(0).unsqueeze(-1).to(embeddings.dtype)
        summed = (embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1.0)
        return summed / counts


class YouTubeDNN(nn.Module):
    """YouTubeDNN two-tower retrieval model."""

    USER_FEATURES = (
        "user_id",
        "age",
        "gender",
        "occupation",
        "zip_code",
    )

    def __init__(
        self,
        feature_dict: Dict[str, int],
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()

        self.feature_dict = dict(feature_dict)
        self.embedding_dim = embedding_dim

        self.user_embeddings = nn.ModuleDict(
            {
                feature_name: nn.Embedding(
                    num_embeddings=self.feature_dict[feature_name],
                    embedding_dim=embedding_dim,
                    padding_idx=0,
                )
                for feature_name in self.USER_FEATURES
            }
        )

        self.movie_embedding = nn.Embedding(
            num_embeddings=self.feature_dict["movie_id"],
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        self.genre_embedding = nn.Embedding(
            num_embeddings=self.feature_dict["genres"],
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.sequence_pooling = MaskedMeanPooling()

        user_input_dim = (len(self.USER_FEATURES) + 2) * embedding_dim
        self.user_dnn = nn.Sequential(
            nn.Linear(user_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.user_embeddings.values():
            nn.init.normal_(embedding.weight, mean=0.0, std=0.01)
            with torch.no_grad():
                embedding.weight[0].zero_()

        for embedding in (self.movie_embedding, self.genre_embedding):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.01)
            with torch.no_grad():
                embedding.weight[0].zero_()

        for module in self.user_dnn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_user(self, features: Dict[str, Tensor]) -> Tensor:
        static_embeddings = [
            self.user_embeddings[name](features[name].long())
            for name in self.USER_FEATURES
        ]

        history_movie_ids = features["hist_movie_id"].long()
        history_genre_ids = features["hist_genres"].long()

        history_movie_embedding = self.sequence_pooling(
            self.movie_embedding(history_movie_ids),
            history_movie_ids,
        )
        history_genre_embedding = self.sequence_pooling(
            self.genre_embedding(history_genre_ids),
            history_genre_ids,
        )

        user_input = torch.cat(
            [
                *static_embeddings,
                history_movie_embedding,
                history_genre_embedding,
            ],
            dim=-1,
        )
        return F.normalize(self.user_dnn(user_input), p=2, dim=-1)

    def encode_item(self, movie_ids: Tensor) -> Tensor:
        item_embedding = self.movie_embedding(movie_ids.long())
        return F.normalize(item_embedding, p=2, dim=-1)

    def compute_full_logits(
        self,
        features: Dict[str, Tensor],
    ) -> Tensor:
        """Compute logits against every non-padding movie class."""
        user_embedding = self.encode_user(features)

        # Keep the training behavior of the original implementation:
        # normalized user vectors against raw movie embedding weights.
        item_weights = self.movie_embedding.weight[1:]

        return user_embedding @ item_weights.transpose(0, 1)

    def compute_full_softmax_loss(
        self,
        features: Dict[str, Tensor],
        movie_ids: Tensor,
    ) -> Tensor:
        """Compute exact softmax loss over all non-padding movies."""
        movie_ids = movie_ids.long()

        if torch.any(movie_ids <= 0):
            raise ValueError("movie_ids must not contain padding ID 0")

        if torch.any(movie_ids >= self.feature_dict["movie_id"]):
            raise ValueError("movie_ids contain an out-of-range class ID")

        logits = self.compute_full_logits(features)

        # Logit column 0 represents encoded movie ID 1.
        targets = movie_ids - 1

        return F.cross_entropy(logits, targets)

    def forward(
        self,
        features: Dict[str, Tensor],
        movie_ids: Tensor,
    ) -> Tensor:
        user_embedding = self.encode_user(features)
        item_embedding = self.encode_item(movie_ids)
        return (user_embedding * item_embedding).sum(dim=-1)