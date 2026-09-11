"""PyTorch implementation of the YouTubeDNN two-tower retrieval model."""

from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


SCORING_CONTRACT_LEGACY_RAW_ITEM = "legacy_raw_item_v1"
SCORING_CONTRACT_SCALED_COSINE_V2 = "scaled_cosine_v2"
SUPPORTED_SCORING_CONTRACTS = {
    SCORING_CONTRACT_LEGACY_RAW_ITEM,
    SCORING_CONTRACT_SCALED_COSINE_V2,
}


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
        history_pooling: str = "masked_mean",
        max_sequence_length: int = 10,
        recent_history_length: int = 5,
        scoring_contract: str = SCORING_CONTRACT_LEGACY_RAW_ITEM,
        logit_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.feature_dict = dict(feature_dict)
        self.embedding_dim = embedding_dim
        if history_pooling != "masked_mean":
            raise ValueError("Baseline V2 only supports masked_mean history pooling")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        self.history_pooling = history_pooling
        self.max_sequence_length = max_sequence_length
        self.recent_history_length = recent_history_length
        if scoring_contract not in SUPPORTED_SCORING_CONTRACTS:
            raise ValueError(
                f"Unsupported scoring_contract: {scoring_contract}"
            )
        if logit_scale <= 0:
            raise ValueError("logit_scale must be positive")
        self.scoring_contract = scoring_contract
        self.logit_scale = float(logit_scale)

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

    def score_user_to_item_embeddings(
        self,
        user_embeddings: Tensor,
        item_embeddings: Tensor,
    ) -> Tensor:
        """Score item vectors under this model's declared contract."""
        scores = user_embeddings @ item_embeddings.transpose(0, 1)
        if self.scoring_contract == SCORING_CONTRACT_SCALED_COSINE_V2:
            scores = scores * self.logit_scale
        return scores

    def compute_full_logits(
        self,
        features: Dict[str, Tensor],
    ) -> Tensor:
        """Compute logits against every non-padding movie class."""
        user_embedding = self.encode_user(features)

        if self.scoring_contract == SCORING_CONTRACT_LEGACY_RAW_ITEM:
            # Retain V1 checkpoint behavior for reproducibility only.
            item_weights = self.movie_embedding.weight[1:]
        else:
            encoded_movie_ids = torch.arange(
                1,
                self.feature_dict["movie_id"],
                dtype=torch.long,
                device=user_embedding.device,
            )
            item_weights = self.encode_item(encoded_movie_ids)

        return self.score_user_to_item_embeddings(
            user_embedding,
            item_weights,
        )

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
        scores = (user_embedding * item_embedding).sum(dim=-1)
        if self.scoring_contract == SCORING_CONTRACT_SCALED_COSINE_V2:
            scores = scores * self.logit_scale
        return scores
