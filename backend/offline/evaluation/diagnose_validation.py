"""Validation-only structural diagnostics for Baseline V1."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from offline.config import config


def validation_only(samples):
    if "validation" not in samples:
        raise ValueError("samples do not contain a validation split")
    return samples["validation"]


def quantile_labels(values, bins=4):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return np.zeros(len(values), dtype=np.int64)
    return np.digitize(values, edges[1:-1], right=True)


def gini(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or np.any(values < 0):
        raise ValueError("values must be non-empty, one-dimensional and non-negative")
    if values.sum() == 0:
        return 0.0
    ordered = np.sort(values)
    index = np.arange(1, len(ordered) + 1)
    return float(
        2 * np.sum(index * ordered) / (len(ordered) * ordered.sum())
        - (len(ordered) + 1) / len(ordered)
    )


def count_by_id(ids):
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0 or np.any(ids < 0):
        raise ValueError("IDs must be non-empty, one-dimensional and non-negative")
    return np.bincount(ids)


def lookup(counts, ids, name):
    ids = np.asarray(ids, dtype=np.int64)
    if np.any(ids < 0) or np.any(ids >= len(counts)):
        raise ValueError(f"{name} contains IDs absent from Train")
    return counts[ids]


def reconstruct_raw_user_activity(retrieval_train):
    """N_raw(u) = N_retrieval_train(u) + 3 under the frozen protocol."""
    counts = count_by_id(retrieval_train["user_id"])
    counts[counts > 0] += 3
    return counts


def concentration_summary(counts):
    weights = np.asarray(counts, dtype=np.int64)
    weights = weights[weights > 0]
    ordered = np.sort(weights)[::-1]
    top_n = max(1, int(np.ceil(len(ordered) * 0.10)))
    return {
        "users": int(len(weights)),
        "minimum": int(weights.min()),
        "median": float(np.median(weights)),
        "maximum": int(weights.max()),
        "gini": gini(weights),
        "top_10_percent_user_share": float(
            ordered[:top_n].sum() / ordered.sum()
        ),
    }


def grouped(values, labels=None):
    values = np.asarray(values)
    groups = quantile_labels(values)
    rows = []
    for group in np.unique(groups):
        mask = groups == group
        row = {
            "quantile": int(group),
            "examples": int(mask.sum()),
            "mean": float(values[mask].mean()),
            "minimum": int(values[mask].min()),
            "maximum": int(values[mask].max()),
        }
        if labels is not None:
            row["positive_rate"] = float(
                np.asarray(labels, dtype=np.float64)[mask].mean()
            )
        rows.append(row)
    return rows


def summarize_retrieval(train, validation):
    train_users = count_by_id(train["user_id"])
    raw_counts = reconstruct_raw_user_activity(train)
    val_users = np.asarray(validation["user_id"], dtype=np.int64)
    histories = np.asarray(validation["hist_movie_id"], dtype=np.int64)
    observed = (histories != 0).sum(axis=1)
    max_window = histories.shape[1]

    train_popularity = count_by_id(train["movie_id"])
    target_popularity = lookup(
        train_popularity,
        validation["movie_id"],
        "Validation target movie",
    )
    raw_activity = lookup(raw_counts, val_users, "Validation user")

    return {
        "examples": int(len(val_users)),
        "observed_history_window": {
            "maximum_configured_length": int(max_window),
            "minimum": int(observed.min()),
            "median": float(np.median(observed)),
            "maximum": int(observed.max()),
            "saturation_rate": float(np.mean(observed == max_window)),
            "meaning": "Truncated model input; not raw user activity.",
        },
        "reconstructed_raw_user_activity": {
            "formula": "retrieval_train_examples_per_user + 3",
            "slices": grouped(raw_activity),
        },
        "target_train_popularity": {
            "slices": grouped(target_popularity),
        },
        "retrieval_train_sample_concentration": (
            concentration_summary(train_users)
        ),
    }, raw_counts


def summarize_ranking(train, validation, raw_counts):
    train_sample_counts = count_by_id(train["user_id"])
    val_users = np.asarray(validation["user_id"], dtype=np.int64)
    labels = np.asarray(validation["is_click"], dtype=np.float64)
    raw_activity = lookup(raw_counts, val_users, "Ranking Validation user")
    sampled_weight = lookup(
        train_sample_counts, val_users, "Ranking Validation user"
    )
    return {
        "examples": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "raw_user_activity_slices": grouped(raw_activity, labels),
        "ranking_sample_weight_slices": grouped(sampled_weight, labels),
        "ranking_train_sample_concentration": (
            concentration_summary(train_sample_counts)
        ),
        "meaning": {
            "raw_user_activity": (
                "Original interactions reconstructed from Retrieval protocol."
            ),
            "ranking_sample_weight": (
                "Sampled DeepFM Train rows; only a gradient-weight proxy."
            ),
        },
    }


def run(output):
    with open(config.TRAIN_DATA_PATH, "rb") as file:
        retrieval = pickle.load(file)
    with open(config.RANKING_TRAIN_DATA_PATH, "rb") as file:
        ranking = pickle.load(file)

    retrieval_summary, raw_counts = summarize_retrieval(
        retrieval["train"], validation_only(retrieval)
    )
    result = {
        "protocol": {
            "version": 2,
            "split": "validation",
            "test_accessed": False,
            "purpose": "structural diagnostics before model slicing",
        },
        "definitions": {
            "observed_history_window": (
                "Non-padding events in the length-limited model input."
            ),
            "raw_user_activity": (
                "Original interaction count reconstructed by protocol."
            ),
            "ranking_sample_weight": (
                "Per-user sampled rows in DeepFM Train."
            ),
        },
        "retrieval": retrieval_summary,
        "ranking": summarize_ranking(
            ranking["train"], validation_only(ranking), raw_counts
        ),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run structural diagnostics on Validation only"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            config.TEMP_DIR
            / "evaluation"
            / "baseline_v1_validation_diagnostics_v2.json"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

