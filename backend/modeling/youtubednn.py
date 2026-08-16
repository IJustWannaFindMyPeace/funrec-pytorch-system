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


class PersonalizedPositionAwareAttentionPooling(nn.Module):
    """Pool history with user-conditioned, position-aware attention."""

    def __init__(
        self,
        embedding_dim: int,
        query_dim: int,
        max_sequence_length: int,
    ) -> None:
        super().__init__()
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        self.max_sequence_length = max_sequence_length
        self.value_projection = nn.Linear(
            embedding_dim, embedding_dim, bias=False
        )
        self.query_projection = nn.Linear(
            query_dim, embedding_dim, bias=False
        )
        self.position_embedding = nn.Embedding(
            max_sequence_length, embedding_dim
        )
        self.score_projection = nn.Linear(embedding_dim, 1, bias=False)

    def attention_weights(
        self,
        embeddings: Tensor,
        ids: Tensor,
        query: Tensor,
    ) -> Tensor:
        """Return normalized weights with padding positions set to zero."""
        sequence_length = embeddings.shape[1]
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                "History is longer than configured max_sequence_length"
            )
        positions = torch.arange(
            sequence_length,
            device=embeddings.device,
        )
        position_embeddings = self.position_embedding(positions)[None, :, :]
        hidden = torch.tanh(
            self.value_projection(embeddings)
            + self.query_projection(query)[:, None, :]
            + position_embeddings
        )
        scores = self.score_projection(hidden).squeeze(-1)
        mask = ids.ne(0)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1) * mask.to(scores.dtype)
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def forward(
        self,
        embeddings: Tensor,
        ids: Tensor,
        query: Tensor,
    ) -> Tensor:
        weights = self.attention_weights(embeddings, ids, query)
        return (embeddings * weights.unsqueeze(-1)).sum(dim=1)


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
        history_pooling: str = "masked_mean",
        max_sequence_length: int = 10,
    ) -> None:
        super().__init__()

        self.feature_dict = dict(feature_dict)
        self.embedding_dim = embedding_dim
        if history_pooling not in {"masked_mean", "personalized_attention"}:
            raise ValueError(f"Unsupported history_pooling: {history_pooling}")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        self.history_pooling = history_pooling
        self.max_sequence_length = max_sequence_length

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

        if history_pooling == "masked_mean":
            self.sequence_pooling = MaskedMeanPooling()
            self.movie_attention_pooling = None
            self.genre_attention_pooling = None
        else:
            query_dim = len(self.USER_FEATURES) * embedding_dim
            self.sequence_pooling = None
            self.movie_attention_pooling = (
                PersonalizedPositionAwareAttentionPooling(
                    embedding_dim,
                    query_dim,
                    max_sequence_length,
                )
            )
            self.genre_attention_pooling = (
                PersonalizedPositionAwareAttentionPooling(
                    embedding_dim,
                    query_dim,
                    max_sequence_length,
                )
            )

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

        for pooling in (
            self.movie_attention_pooling,
            self.genre_attention_pooling,
        ):
            if pooling is None:
                continue
            nn.init.normal_(pooling.position_embedding.weight, std=0.01)
            for projection in (
                pooling.value_projection,
                pooling.query_projection,
                pooling.score_projection,
            ):
                nn.init.xavier_uniform_(projection.weight)

    def encode_user(self, features: Dict[str, Tensor]) -> Tensor:
        static_embeddings = [
            self.user_embeddings[name](features[name].long())
            for name in self.USER_FEATURES
        ]

        history_movie_ids = features["hist_movie_id"].long()
        history_genre_ids = features["hist_genres"].long()

        movie_embeddings = self.movie_embedding(history_movie_ids)
        genre_embeddings = self.genre_embedding(history_genre_ids)
        if self.history_pooling == "masked_mean":
            history_movie_embedding = self.sequence_pooling(
                movie_embeddings, history_movie_ids
            )
            history_genre_embedding = self.sequence_pooling(
                genre_embeddings, history_genre_ids
            )
        else:
            static_query = torch.cat(static_embeddings, dim=-1)
            history_movie_embedding = self.movie_attention_pooling(
                movie_embeddings,
                history_movie_ids,
                static_query,
            )
            history_genre_embedding = self.genre_attention_pooling(
                genre_embeddings,
                history_genre_ids,
                static_query,
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
