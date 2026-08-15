"""Diagnose equal-weight history dilution on Retrieval Validation only."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from modeling.youtubednn import YouTubeDNN
from offline.config import config
from offline.evaluation.diagnose_validation import (
    count_by_id,
    lookup,
    reconstruct_raw_user_activity,
)
from offline.evaluation.diagnose_validation_cross import (
    cross_retrieval_metrics,
)
from offline.evaluation.diagnose_validation_models import (
    K_VALUES,
    grouped_retrieval_metrics,
)
from offline.evaluation.evaluate_retrieval_validation import (
    infer_sequence_length,
    validate_selection_samples,
)
from offline.evaluation.retrieval import (
    calculate_single_target_metrics,
    recommend_top_k,
)
from offline.training.retrieval_data import RetrievalDataset
from offline.training.retrieval_trainer import (
    move_batch_to_device,
    resolve_device,
)
from offline.training.train_retrieval import BEST_CHECKPOINT_PATH


HISTORY_FIELDS = ("hist_movie_id", "hist_genres")
CONDITIONS = ("full_20", "recent_10_only", "older_10_only")


def mask_history_split(validation, condition, keep_length=10):
    """Return a copied Validation split with one history region masked."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown history-mask condition: {condition}")
    if keep_length <= 0:
        raise ValueError("keep_length must be positive")

    masked = {
        name: np.asarray(values).copy()
        for name, values in validation.items()
    }
    sequence_length = infer_sequence_length(masked)
    if keep_length >= sequence_length:
        raise ValueError("keep_length must be smaller than sequence length")

    if condition == "recent_10_only":
        for name in HISTORY_FIELDS:
            masked[name][:, :-keep_length] = 0
    elif condition == "older_10_only":
        for name in HISTORY_FIELDS:
            masked[name][:, -keep_length:] = 0
    return masked


def load_model_and_items(device):
    checkpoint = torch.load(
        BEST_CHECKPOINT_PATH,
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
    return model, item_embeddings, int(checkpoint["epoch"])


@torch.no_grad()
def predict(model, item_embeddings, validation, device, batch_size):
    loader = DataLoader(
        RetrievalDataset(validation),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    recommendations, targets = [], []
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
    return torch.cat(recommendations), torch.cat(targets)


def condition_metrics(
    recommendations,
    targets,
    activity,
    popularity,
):
    return {
        "overall": calculate_single_target_metrics(
            recommendations, targets, K_VALUES
        ),
        "by_raw_user_activity": grouped_retrieval_metrics(
            activity, recommendations, targets
        ),
        "by_target_train_popularity": grouped_retrieval_metrics(
            popularity, recommendations, targets
        ),
        "activity_x_target_popularity": cross_retrieval_metrics(
            activity, popularity, recommendations, targets
        ),
    }


def quantile_metric(condition, section, quantile, metric):
    rows = condition[section]
    matches = [row for row in rows if int(row["quantile"]) == quantile]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {section} row for quantile {quantile}"
        )
    return float(matches[0][metric])


def summarize_condition(condition):
    low = quantile_metric(
        condition, "by_raw_user_activity", 0, "recall@10"
    )
    high = quantile_metric(
        condition, "by_raw_user_activity", 3, "recall@10"
    )
    return {
        "recall@10": float(condition["overall"]["recall@10"]),
        "ndcg@10": float(condition["overall"]["ndcg@10"]),
        "low_activity_aq0_recall@10": low,
        "high_activity_aq3_recall@10": high,
        "activity_recall@10_gap": low - high,
        "tail_pq0_recall@10": quantile_metric(
            condition,
            "by_target_train_popularity",
            0,
            "recall@10",
        ),
    }


def subtract_summaries(candidate, reference):
    return {
        name: float(candidate[name] - reference[name])
        for name in reference
    }


def non_padding_summary(validation):
    result = {}
    for name in HISTORY_FIELDS:
        counts = (np.asarray(validation[name]) != 0).sum(axis=1)
        result[name] = {
            "minimum": int(counts.min()),
            "maximum": int(counts.max()),
            "mean": float(counts.mean()),
        }
    return result


def run(
    output,
    expected_seq_len=20,
    keep_length=10,
    device_name=None,
    batch_size=512,
):
    if expected_seq_len <= 0:
        raise ValueError("expected_seq_len must be positive")
    if keep_length <= 0 or keep_length >= expected_seq_len:
        raise ValueError(
            "keep_length must be positive and smaller than expected_seq_len"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostics: {output}")

    with open(config.TRAIN_DATA_PATH, "rb") as file:
        samples = pickle.load(file)
    train, validation = validate_selection_samples(samples)
    actual_seq_len = infer_sequence_length(validation)
    if actual_seq_len != expected_seq_len:
        raise ValueError(
            f"Expected history length {expected_seq_len}, "
            f"found {actual_seq_len}"
        )

    device = resolve_device(device_name)
    model, item_embeddings, checkpoint_epoch = load_model_and_items(device)
    activity = lookup(
        reconstruct_raw_user_activity(train),
        validation["user_id"],
        "Retrieval Validation user",
    )
    popularity = lookup(
        count_by_id(train["movie_id"]),
        validation["movie_id"],
        "Retrieval Validation movie",
    )

    conditions = {}
    non_padding_counts = {}
    for name in CONDITIONS:
        masked = mask_history_split(validation, name, keep_length)
        non_padding_counts[name] = non_padding_summary(masked)
        recommendations, targets = predict(
            model,
            item_embeddings,
            masked,
            device,
            batch_size,
        )
        conditions[name] = condition_metrics(
            recommendations,
            targets,
            activity,
            popularity,
        )

    summaries = {
        name: summarize_condition(value)
        for name, value in conditions.items()
    }
    full = summaries["full_20"]
    comparisons = {
        "recent_10_only_minus_full_20": subtract_summaries(
            summaries["recent_10_only"], full
        ),
        "older_10_only_minus_full_20": subtract_summaries(
            summaries["older_10_only"], full
        ),
        "recent_10_only_minus_older_10_only": subtract_summaries(
            summaries["recent_10_only"], summaries["older_10_only"]
        ),
    }
    result = {
        "protocol": {
            "version": 1,
            "experiment": "retrieval-history-mask-diagnostic",
            "split": "validation",
            "test_accessed": False,
            "selection_artifact_contains_test": False,
            "checkpoint_retrained": False,
            "checkpoint_epoch": checkpoint_epoch,
            "history_length": actual_seq_len,
            "keep_length": keep_length,
            "sequence_layout": "left padding; newer events on the right",
            "conditions": list(CONDITIONS),
            "history_pooling": "masked_mean",
            "device": str(device),
            "batch_size": batch_size,
            "k_values": list(K_VALUES),
        },
        "summaries": summaries,
        "comparisons": comparisons,
        "non_padding_counts": non_padding_counts,
        "conditions": conditions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose equal-weight history dilution on Validation only"
        )
    )
    parser.add_argument("--expected-seq-len", type=int, default=20)
    parser.add_argument("--keep-length", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        run(
            args.output,
            args.expected_seq_len,
            args.keep_length,
            args.device,
            args.batch_size,
        ),
        indent=2,
        allow_nan=False,
    ))


if __name__ == "__main__":
    main()
