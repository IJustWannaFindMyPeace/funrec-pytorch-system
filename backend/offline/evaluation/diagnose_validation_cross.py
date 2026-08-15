"""Cross-slice model diagnostics on Validation only."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from offline.config import config
from offline.evaluation.diagnose_validation import (
    count_by_id,
    lookup,
    quantile_labels,
    reconstruct_raw_user_activity,
    validation_only,
)
from offline.evaluation.diagnose_validation_models import (
    K_VALUES,
    load_samples,
    ranking_predictions,
    retrieval_predictions,
    safe_auc,
)
from offline.evaluation.retrieval import calculate_single_target_metrics
from offline.training.retrieval_trainer import resolve_device


def safe_pr_auc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return None
    return float(average_precision_score(labels, scores))


def binary_entropy(positive_rate):
    positive_rate = float(positive_rate)
    if not 0.0 <= positive_rate <= 1.0:
        raise ValueError("positive_rate must be between zero and one")
    if positive_rate in (0.0, 1.0):
        return 0.0
    return float(
        -positive_rate * np.log(positive_rate)
        - (1.0 - positive_rate) * np.log(1.0 - positive_rate)
    )


def calibrated_ranking_metrics(labels, scores, losses):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    losses = np.asarray(losses)
    positive_rate = float(labels.mean())
    logloss = float(losses.mean())
    entropy = binary_entropy(positive_rate)
    normalized = logloss / entropy if entropy > 0.0 else None
    return {
        "examples": int(len(labels)),
        "positives": int(labels.sum()),
        "positive_rate": positive_rate,
        "roc_auc": safe_auc(labels, scores),
        "pr_auc": safe_pr_auc(labels, scores),
        "prevalence_pr_auc_baseline": positive_rate,
        "logloss": logloss,
        "constant_prevalence_logloss": entropy,
        "normalized_logloss": normalized,
        "logloss_improvement_over_constant": (
            1.0 - normalized if normalized is not None else None
        ),
    }


def cross_masks(activity, popularity):
    activity = np.asarray(activity)
    popularity = np.asarray(popularity)
    if activity.shape != popularity.shape or activity.ndim != 1:
        raise ValueError("slice values must be aligned one-dimensional arrays")
    activity_groups = quantile_labels(activity)
    popularity_groups = quantile_labels(popularity)
    for activity_group in np.unique(activity_groups):
        for popularity_group in np.unique(popularity_groups):
            mask = (
                (activity_groups == activity_group)
                & (popularity_groups == popularity_group)
            )
            if np.any(mask):
                yield int(activity_group), int(popularity_group), mask


def cross_retrieval_metrics(
    activity, popularity, recommendations, targets
):
    recommendations = torch.as_tensor(recommendations)
    targets = torch.as_tensor(targets)
    rows = []
    for activity_group, popularity_group, mask in cross_masks(
        activity, popularity
    ):
        torch_mask = torch.as_tensor(mask, dtype=torch.bool)
        metrics = calculate_single_target_metrics(
            recommendations[torch_mask], targets[torch_mask], K_VALUES
        )
        rows.append({
            "activity_quantile": activity_group,
            "popularity_quantile": popularity_group,
            "examples": int(mask.sum()),
            "activity_minimum": int(np.asarray(activity)[mask].min()),
            "activity_maximum": int(np.asarray(activity)[mask].max()),
            "popularity_minimum": int(np.asarray(popularity)[mask].min()),
            "popularity_maximum": int(np.asarray(popularity)[mask].max()),
            **metrics,
        })
    return rows


def cross_ranking_metrics(
    activity, popularity, labels, scores, losses
):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    losses = np.asarray(losses)
    rows = []
    for activity_group, popularity_group, mask in cross_masks(
        activity, popularity
    ):
        rows.append({
            "activity_quantile": activity_group,
            "popularity_quantile": popularity_group,
            "activity_minimum": int(np.asarray(activity)[mask].min()),
            "activity_maximum": int(np.asarray(activity)[mask].max()),
            "popularity_minimum": int(np.asarray(popularity)[mask].min()),
            "popularity_maximum": int(np.asarray(popularity)[mask].max()),
            **calibrated_ranking_metrics(
                labels[mask], scores[mask], losses[mask]
            ),
        })
    return rows


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
    recommendations, targets, retrieval_epoch = retrieval_predictions(
        retrieval, device, batch_size
    )
    labels, scores, losses, ranking_epoch = ranking_predictions(
        ranking, device, batch_size
    )

    raw_activity = reconstruct_raw_user_activity(retrieval["train"])
    movie_popularity = count_by_id(retrieval["train"]["movie_id"])
    retrieval_activity = lookup(
        raw_activity, retrieval_validation["user_id"],
        "Retrieval Validation user",
    )
    retrieval_popularity = lookup(
        movie_popularity, retrieval_validation["movie_id"],
        "Retrieval Validation movie",
    )
    ranking_activity = lookup(
        raw_activity, ranking_validation["user_id"],
        "Ranking Validation user",
    )
    ranking_popularity = lookup(
        movie_popularity, ranking_validation["movie_id"],
        "Ranking Validation movie",
    )

    result = {
        "protocol": {
            "version": 4,
            "split": "validation",
            "test_accessed": False,
            "device": str(device),
            "batch_size": batch_size,
            "activity_axis": "reconstructed raw user interactions",
            "popularity_axis": "Retrieval Train target count",
            "quantiles_per_axis": 4,
            "ranking_logloss_note": (
                "Raw logloss is prevalence-dependent. normalized_logloss "
                "divides it by the same slice's constant-prevalence entropy."
            ),
        },
        "retrieval": {
            "checkpoint_epoch": retrieval_epoch,
            "activity_x_target_popularity": cross_retrieval_metrics(
                retrieval_activity,
                retrieval_popularity,
                recommendations.numpy(),
                targets.numpy(),
            ),
        },
        "ranking": {
            "checkpoint_epoch": ranking_epoch,
            "overall": calibrated_ranking_metrics(labels, scores, losses),
            "activity_x_movie_popularity": cross_ranking_metrics(
                ranking_activity,
                ranking_popularity,
                labels,
                scores,
                losses,
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run 4x4 cross diagnostics on Validation only"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            config.TEMP_DIR
            / "evaluation"
            / "baseline_v1_validation_cross_diagnostics_v4.json"
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
