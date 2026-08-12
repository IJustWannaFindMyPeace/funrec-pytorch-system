"""Training utilities for the PyTorch YouTubeDNN model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TrainingStats:
    """Aggregated statistics from one training pass."""

    loss: float
    examples: int
    batches: int


def resolve_device(requested_device: Optional[str] = None) -> torch.device:
    """Resolve and validate the training device."""
    if requested_device is None:
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(requested_device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    return device


def move_batch_to_device(
    features: Dict[str, Tensor],
    targets: Tensor,
    device: torch.device,
) -> Tuple[Dict[str, Tensor], Tensor]:
    """Move one retrieval batch to the selected device."""
    moved_features = {
        name: value.to(device, non_blocking=True)
        for name, value in features.items()
    }
    moved_targets = targets.to(device, non_blocking=True)
    return moved_features, moved_targets


def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> TrainingStats:
    """Train for one pass, optionally stopping after a fixed batch count."""
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be greater than zero")

    model.train()

    total_loss = 0.0
    total_examples = 0
    total_batches = 0

    for batch_index, (features, targets) in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        features, targets = move_batch_to_device(
            features,
            targets,
            device,
        )

        optimizer.zero_grad(set_to_none=True)

        loss = model.compute_full_softmax_loss(features, targets)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at batch {batch_index}"
            )

        loss.backward()
        optimizer.step()

        batch_size = targets.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_examples += batch_size
        total_batches += 1

    if total_batches == 0:
        raise ValueError("The data loader produced no training batches")

    return TrainingStats(
        loss=total_loss / total_examples,
        examples=total_examples,
        batches=total_batches,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    feature_dict: Dict[str, int],
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Save a resumable training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "feature_dict": dict(feature_dict),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": dict(metrics or {}),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load model state and optionally optimizer state."""
    checkpoint = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint