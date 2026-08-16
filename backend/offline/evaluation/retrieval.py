"""Offline evaluation utilities for retrieval models."""

from typing import Dict, Iterable, Sequence

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from offline.training.retrieval_trainer import move_batch_to_device


@torch.no_grad()
def recommend_top_k(
    model: nn.Module,
    features: Dict[str, Tensor],
    item_embeddings: Tensor,
    k: int,
) -> Tensor:
    """Recommend encoded movie IDs while excluding user history."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if item_embeddings.ndim != 2:
        raise ValueError("item_embeddings must be two-dimensional")

    if k > item_embeddings.shape[0]:
        raise ValueError("k must not exceed the number of items")

    user_embeddings = model.encode_user(features)
    scores = user_embeddings @ item_embeddings.transpose(0, 1)

    history_ids = features["hist_movie_id"].long()
    valid_history = (
        (history_ids > 0)
        & (history_ids <= item_embeddings.shape[0])
    )

    history_columns = (history_ids - 1).clamp_min(0)

    history_mask = torch.zeros_like(
        scores,
        dtype=torch.int32,
    )
    history_mask.scatter_add_(
        1,
        history_columns,
        valid_history.to(torch.int32),
    )

    scores = scores.masked_fill(
        history_mask > 0,
        float("-inf"),
    )

    # Column 0 corresponds to encoded movie ID 1.
    return torch.topk(scores, k=k, dim=1).indices + 1


@torch.no_grad()
def recommend_top_k_multi_interest(model, features, item_embeddings, k):
    """Top-K with max-over-interest scoring and standard history exclusion."""
    interests = model.encode_user_interests(features)
    scores = torch.einsum("bkd,nd->bkn", interests, item_embeddings).max(1).values
    if getattr(model, "scoring_contract", None) == "scaled_cosine_v2":
        scores = scores * model.logit_scale
    history = features["hist_movie_id"].long()
    valid = (history > 0) & (history <= item_embeddings.shape[0])
    mask = torch.zeros_like(scores, dtype=torch.int32)
    mask.scatter_add_(1, (history - 1).clamp_min(0), valid.to(torch.int32))
    scores = scores.masked_fill(mask > 0, float("-inf"))
    return torch.topk(scores, k=k, dim=1).indices + 1

def recommend_popular_top_k(
    popularity: Tensor,
    history_ids: Tensor,
    k: int,
) -> Tensor:
    """Recommend globally popular encoded IDs, excluding history."""
    if popularity.ndim != 1:
        raise ValueError("popularity must be one-dimensional")

    if history_ids.ndim != 2:
        raise ValueError("history_ids must be two-dimensional")

    item_count = popularity.shape[0] - 1

    if item_count <= 0:
        raise ValueError(
            "popularity must include padding and item classes"
        )

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if k > item_count:
        raise ValueError("k must not exceed the number of items")

    scores = popularity.to(torch.float32).unsqueeze(0).expand(
        history_ids.shape[0],
        -1,
    ).clone()

    # Encoded class 0 is padding and must never be recommended.
    scores[:, 0] = float("-inf")

    valid_history = (
        (history_ids > 0)
        & (history_ids <= item_count)
    )

    history_mask = torch.zeros_like(
        scores,
        dtype=torch.int32,
    )
    history_mask.scatter_add_(
        1,
        history_ids.clamp(min=0, max=item_count),
        valid_history.to(torch.int32),
    )

    scores = scores.masked_fill(
        history_mask > 0,
        float("-inf"),
    )

    # Score columns directly correspond to encoded movie IDs.
    return torch.topk(scores, k=k, dim=1).indices

def calculate_single_target_metrics(
    recommendations: Tensor,
    targets: Tensor,
    k_values: Sequence[int],
) -> Dict[str, float]:
    """Calculate retrieval metrics when each user has one target."""
    if recommendations.ndim != 2:
        raise ValueError("recommendations must be two-dimensional")

    targets = targets.reshape(-1).long()

    if recommendations.shape[0] != targets.shape[0]:
        raise ValueError(
            "recommendations and targets have different batch sizes"
        )

    metrics: Dict[str, float] = {}

    for k in sorted(set(k_values)):
        if k <= 0:
            raise ValueError("k values must be greater than zero")
        if k > recommendations.shape[1]:
            raise ValueError(
                "recommendations do not contain enough columns"
            )

        top_k = recommendations[:, :k]
        matches = top_k.eq(targets.unsqueeze(1))
        hits = matches.any(dim=1)

        ranks = torch.argmax(
            matches.to(torch.int64),
            dim=1,
        ) + 1

        ndcg = torch.where(
            hits,
            1.0 / torch.log2(ranks.to(torch.float32) + 1.0),
            torch.zeros_like(ranks, dtype=torch.float32),
        )

        hit_rate = hits.to(torch.float32).mean().item()

        metrics[f"recall@{k}"] = hit_rate
        metrics[f"hit_rate@{k}"] = hit_rate
        metrics[f"ndcg@{k}"] = ndcg.mean().item()

    return metrics


@torch.no_grad()
def evaluate_retrieval(
    model: nn.Module,
    data_loader: Iterable,
    item_embeddings: Tensor,
    device: torch.device,
    k_values: Sequence[int] = (5, 10),
    description: str = "Retrieval evaluation",
) -> Dict[str, float]:
    """Evaluate retrieval metrics over the complete test set."""
    if not k_values:
        raise ValueError("k_values must not be empty")

    max_k = max(k_values)
    model.eval()

    total_examples = 0
    metric_sums = {
        f"{metric}@{k}": 0.0
        for k in sorted(set(k_values))
        for metric in ("recall", "hit_rate", "ndcg")
    }

    progress = tqdm(
        data_loader,
        total=len(data_loader),
        desc=description,
    )

    for features, targets in progress:
        features, targets = move_batch_to_device(
            features,
            targets,
            device,
        )

        recommendations = recommend_top_k(
            model=model,
            features=features,
            item_embeddings=item_embeddings,
            k=max_k,
        )
        batch_metrics = calculate_single_target_metrics(
            recommendations,
            targets,
            k_values,
        )

        batch_size = targets.shape[0]
        total_examples += batch_size

        for name, value in batch_metrics.items():
            metric_sums[name] += value * batch_size

    if total_examples == 0:
        raise ValueError("The data loader produced no evaluation batches")

    return {
        name: value / total_examples
        for name, value in metric_sums.items()
    }
