"""Model-performance diagnostics on Validation only."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

from modeling.youtubednn import YouTubeDNN
from offline.config import config
from offline.evaluation.diagnose_validation import (
    count_by_id,
    lookup,
    quantile_labels,
    reconstruct_raw_user_activity,
    validation_only,
)
from offline.evaluation.evaluate_ranking import build_model as build_ranking_model
from offline.evaluation.retrieval import (
    calculate_single_target_metrics,
    recommend_top_k,
)
from offline.training.ranking_data import RankingDataset
from offline.training.retrieval_data import RetrievalDataset
from offline.training.retrieval_trainer import (
    move_batch_to_device,
    resolve_device,
)
from offline.training.train_ranking import (
    BEST_CHECKPOINT_PATH as RANKING_BEST_CHECKPOINT_PATH,
)
from offline.training.train_retrieval import (
    BEST_CHECKPOINT_PATH as RETRIEVAL_BEST_CHECKPOINT_PATH,
)


K_VALUES = (5, 10, 20, 50)


def safe_auc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def ranking_metrics(labels, scores, losses):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    losses = np.asarray(losses)
    return {
        "examples": int(len(labels)),
        "positives": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "roc_auc": safe_auc(labels, scores),
        "logloss": float(losses.mean()),
        "mean_positive_score": (
            float(scores[labels == 1].mean())
            if np.any(labels == 1)
            else None
        ),
        "mean_negative_score": (
            float(scores[labels == 0].mean())
            if np.any(labels == 0)
            else None
        ),
    }


def grouped_ranking_metrics(values, labels, scores, losses):
    values = np.asarray(values)
    groups = quantile_labels(values)
    result = []
    for group in np.unique(groups):
        mask = groups == group
        row = {
            "quantile": int(group),
            "slice_minimum": int(values[mask].min()),
            "slice_maximum": int(values[mask].max()),
        }
        row.update(ranking_metrics(
            np.asarray(labels)[mask],
            np.asarray(scores)[mask],
            np.asarray(losses)[mask],
        ))
        result.append(row)
    return result


def grouped_retrieval_metrics(values, recommendations, targets):
    values = np.asarray(values)
    groups = quantile_labels(values)
    recommendations = torch.as_tensor(recommendations)
    targets = torch.as_tensor(targets)
    result = []
    for group in np.unique(groups):
        mask = groups == group
        torch_mask = torch.as_tensor(mask, dtype=torch.bool)
        metrics = calculate_single_target_metrics(
            recommendations[torch_mask],
            targets[torch_mask],
            K_VALUES,
        )
        result.append({
            "quantile": int(group),
            "examples": int(mask.sum()),
            "slice_minimum": int(values[mask].min()),
            "slice_maximum": int(values[mask].max()),
            **metrics,
        })
    return result


def load_samples():
    with open(config.TRAIN_DATA_PATH, "rb") as file:
        retrieval = pickle.load(file)
    with open(config.RANKING_TRAIN_DATA_PATH, "rb") as file:
        ranking = pickle.load(file)
    return retrieval, ranking


@torch.no_grad()
def retrieval_predictions(samples, device, batch_size):
    checkpoint = torch.load(
        RETRIEVAL_BEST_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model = YouTubeDNN(
        feature_dict=checkpoint["feature_dict"],
        embedding_dim=config.EMB_DIM,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    item_embeddings = torch.as_tensor(
        np.load(config.ITEM_EMB_PATH),
        dtype=torch.float32,
        device=device,
    )
    loader = DataLoader(
        RetrievalDataset(validation_only(samples)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    recommendations = []
    targets = []
    for features, batch_targets in loader:
        features, batch_targets = move_batch_to_device(
            features, batch_targets, device
        )
        recommendations.append(
            recommend_top_k(
                model, features, item_embeddings, max(K_VALUES)
            ).cpu()
        )
        targets.append(batch_targets.cpu())
    return (
        torch.cat(recommendations),
        torch.cat(targets),
        int(checkpoint["epoch"]),
    )


@torch.no_grad()
def ranking_predictions(samples, device, batch_size):
    checkpoint = torch.load(
        RANKING_BEST_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model = build_ranking_model(checkpoint).to(device).eval()
    loader = DataLoader(
        RankingDataset(validation_only(samples)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    labels, scores, losses = [], [], []
    for features, batch_labels in loader:
        features = {
            name: value.to(device, non_blocking=True)
            for name, value in features.items()
        }
        batch_labels = batch_labels.to(device, non_blocking=True)
        logits = model(features)
        batch_losses = F.binary_cross_entropy_with_logits(
            logits, batch_labels, reduction="none"
        )
        labels.append(batch_labels.cpu())
        scores.append(torch.sigmoid(logits).cpu())
        losses.append(batch_losses.cpu())
    return (
        torch.cat(labels).numpy(),
        torch.cat(scores).numpy(),
        torch.cat(losses).numpy(),
        int(checkpoint["epoch"]),
    )


def run(output, device_name=None, batch_size=512):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostics: {output}")

    retrieval, ranking = load_samples()
    retrieval_validation = validation_only(retrieval)
    ranking_validation = validation_only(ranking)
    device = resolve_device(device_name)

    recommendations, retrieval_targets, retrieval_epoch = (
        retrieval_predictions(retrieval, device, batch_size)
    )
    labels, scores, losses, ranking_epoch = ranking_predictions(
        ranking, device, batch_size
    )

    raw_user_counts = reconstruct_raw_user_activity(retrieval["train"])
    retrieval_activity = lookup(
        raw_user_counts,
        retrieval_validation["user_id"],
        "Retrieval Validation user",
    )
    ranking_activity = lookup(
        raw_user_counts,
        ranking_validation["user_id"],
        "Ranking Validation user",
    )
    movie_popularity = count_by_id(retrieval["train"]["movie_id"])
    retrieval_popularity = lookup(
        movie_popularity,
        retrieval_validation["movie_id"],
        "Retrieval Validation movie",
    )
    ranking_popularity = lookup(
        movie_popularity,
        ranking_validation["movie_id"],
        "Ranking Validation movie",
    )

    result = {
        "protocol": {
            "version": 3,
            "split": "validation",
            "test_accessed": False,
            "device": str(device),
            "batch_size": batch_size,
            "k_values": list(K_VALUES),
            "candidate_funnel_available": False,
            "candidate_funnel_reason": (
                "DeepFM Validation is an independently sampled candidate set; "
                "it is not the YouTubeDNN Top-K candidate set."
            ),
        },
        "retrieval": {
            "checkpoint_epoch": retrieval_epoch,
            "overall": calculate_single_target_metrics(
                recommendations, retrieval_targets, K_VALUES
            ),
            "by_raw_user_activity": grouped_retrieval_metrics(
                retrieval_activity, recommendations, retrieval_targets
            ),
            "by_target_train_popularity": grouped_retrieval_metrics(
                retrieval_popularity, recommendations, retrieval_targets
            ),
        },
        "ranking": {
            "checkpoint_epoch": ranking_epoch,
            "overall": ranking_metrics(labels, scores, losses),
            "by_raw_user_activity": grouped_ranking_metrics(
                ranking_activity, labels, scores, losses
            ),
            "by_movie_train_popularity": grouped_ranking_metrics(
                ranking_popularity, labels, scores, losses
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run model diagnostics on Validation only"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            config.TEMP_DIR
            / "evaluation"
            / "baseline_v1_validation_model_diagnostics_v3.json"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(
        run(args.output, args.device, args.batch_size),
        indent=2,
        allow_nan=False,
    ))


if __name__ == "__main__":
    main()
