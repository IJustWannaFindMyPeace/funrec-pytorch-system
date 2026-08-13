"""Training utilities for the PyTorch DeepFM ranking model."""

from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm


@dataclass(frozen=True)
class RankingStats:
    """Aggregated statistics from one ranking pass."""

    loss: float
    auc: float
    examples: int
    batches: int


def resolve_device(
    requested_device: Optional[str] = None,
) -> torch.device:
    """Resolve and validate the training device."""
    if requested_device is None:
        requested_device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(requested_device)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is not available"
        )

    return device


def move_batch_to_device(
    features: Dict[str, Tensor],
    labels: Tensor,
    device: torch.device,
) -> Tuple[Dict[str, Tensor], Tensor]:
    """Move one ranking batch to the selected device."""
    moved_features = {
        name: value.to(
            device,
            non_blocking=True,
        )
        for name, value in features.items()
    }
    moved_labels = labels.to(
        device,
        dtype=torch.float32,
        non_blocking=True,
    )
    return moved_features, moved_labels


def build_progress_iterator(
    data_loader: Iterable,
    max_batches: Optional[int],
    description: Optional[str],
):
    """Limit a loader and optionally display progress."""
    iterator = iter(data_loader)

    try:
        total_batches = len(data_loader)
    except TypeError:
        total_batches = None

    if max_batches is not None:
        iterator = islice(iterator, max_batches)
        total_batches = (
            min(total_batches, max_batches)
            if total_batches is not None
            else max_batches
        )

    return tqdm(
        iterator,
        total=total_batches,
        desc=description,
        leave=True,
        disable=description is None,
    )


def calculate_binary_auc(
    labels: Tensor,
    scores: Tensor,
) -> float:
    """
    Calculate binary ROC-AUC using average ranks for ties.

    AUC is the probability that a randomly selected positive
    example receives a higher score than a randomly selected
    negative example.
    """
    labels = labels.detach().reshape(-1).cpu()
    scores = scores.detach().reshape(-1).cpu()

    if labels.numel() == 0:
        raise ValueError("labels and scores must not be empty")

    if labels.shape != scores.shape:
        raise ValueError(
            "labels and scores must have the same shape"
        )

    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")

    if torch.any((labels != 0) & (labels != 1)):
        raise ValueError(
            "labels must contain only 0 or 1"
        )

    labels = labels.to(torch.int64)
    scores = scores.to(torch.float64)

    positive_count = int(labels.sum().item())
    negative_count = labels.numel() - positive_count

    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "ROC-AUC requires both positive and negative labels"
        )

    order = torch.argsort(
        scores,
        stable=True,
    )
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    _, tie_counts = torch.unique_consecutive(
        sorted_scores,
        return_counts=True,
    )

    tie_ends = tie_counts.cumsum(dim=0)
    tie_starts = tie_ends - tie_counts + 1
    average_ranks = (
        tie_starts.to(torch.float64)
        + tie_ends.to(torch.float64)
    ) / 2.0

    ranks = torch.repeat_interleave(
        average_ranks,
        tie_counts,
    )

    positive_rank_sum = ranks[
        sorted_labels == 1
    ].sum()

    auc = (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)

    return float(auc.item())


def _finalize_stats(
    total_loss: float,
    total_examples: int,
    total_batches: int,
    labels: list[Tensor],
    scores: list[Tensor],
    pass_name: str,
) -> RankingStats:
    if total_batches == 0:
        raise ValueError(
            f"The data loader produced no {pass_name} batches"
        )

    all_labels = torch.cat(labels)
    all_scores = torch.cat(scores)

    has_positive = bool(torch.any(all_labels == 1))
    has_negative = bool(torch.any(all_labels == 0))

    auc = (
        calculate_binary_auc(
            all_labels,
            all_scores,
        )
        if has_positive and has_negative
        else float("nan")
    )

    return RankingStats(
        loss=total_loss / total_examples,
        auc=auc,
        examples=total_examples,
        batches=total_batches,
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: Optional[int] = None,
    progress_description: Optional[str] = None,
) -> RankingStats:
    """Train DeepFM for one pass over the loader."""
    if max_batches is not None and max_batches <= 0:
        raise ValueError(
            "max_batches must be greater than zero"
        )

    model.train()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_examples = 0
    total_batches = 0
    collected_labels = []
    collected_scores = []

    progress = build_progress_iterator(
        data_loader,
        max_batches,
        progress_description,
    )

    for batch_index, (features, labels) in enumerate(
        progress
    ):
        features, labels = move_batch_to_device(
            features,
            labels,
            device,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(features)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite training loss at "
                f"batch {batch_index}"
            )

        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_examples += batch_size
        total_batches += 1

        collected_labels.append(
            labels.detach().cpu()
        )
        collected_scores.append(
            torch.sigmoid(logits.detach()).cpu()
        )

        progress.set_postfix(
            loss=f"{total_loss / total_examples:.4f}"
        )

    return _finalize_stats(
        total_loss,
        total_examples,
        total_batches,
        collected_labels,
        collected_scores,
        "training",
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    max_batches: Optional[int] = None,
    progress_description: Optional[str] = None,
) -> RankingStats:
    """Evaluate DeepFM loss and ROC-AUC."""
    if max_batches is not None and max_batches <= 0:
        raise ValueError(
            "max_batches must be greater than zero"
        )

    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_examples = 0
    total_batches = 0
    collected_labels = []
    collected_scores = []

    progress = build_progress_iterator(
        data_loader,
        max_batches,
        progress_description,
    )

    for batch_index, (features, labels) in enumerate(
        progress
    ):
        features, labels = move_batch_to_device(
            features,
            labels,
            device,
        )

        logits = model(features)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite evaluation loss at "
                f"batch {batch_index}"
            )

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        total_batches += 1

        collected_labels.append(labels.cpu())
        collected_scores.append(
            torch.sigmoid(logits).cpu()
        )

        progress.set_postfix(
            loss=f"{total_loss / total_examples:.4f}"
        )

    return _finalize_stats(
        total_loss,
        total_examples,
        total_batches,
        collected_labels,
        collected_scores,
        "evaluation",
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    feature_dict: Dict[str, int],
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Save a resumable DeepFM checkpoint."""
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_config = {
        "embedding_dim": getattr(
            model,
            "embedding_dim",
            None,
        ),
        "dnn_hidden_units": tuple(
            getattr(
                model,
                "dnn_hidden_units",
                (),
            )
        ),
        "dropout": getattr(
            model,
            "dropout",
            None,
        ),
    }

    torch.save(
        {
            "epoch": epoch,
            "feature_dict": dict(feature_dict),
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": dict(metrics or {}),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[
        torch.optim.Optimizer
    ] = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load model state and optionally optimizer state."""
    checkpoint = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint