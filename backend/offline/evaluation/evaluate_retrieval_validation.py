"""Evaluate one retrieval ablation on Validation without loading Test."""

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


def validate_selection_samples(samples):
    if set(samples) != {"train", "validation"}:
        raise ValueError(
            "Selection artifact must contain exactly Train and Validation; "
            "Test must be sealed separately"
        )
    return samples["train"], samples["validation"]


def infer_sequence_length(split):
    movie_history = np.asarray(split["hist_movie_id"])
    genre_history = np.asarray(split["hist_genres"])
    if movie_history.ndim != 2 or genre_history.ndim != 2:
        raise ValueError("History arrays must be two-dimensional")
    if movie_history.shape[1] != genre_history.shape[1]:
        raise ValueError("Movie and genre history lengths differ")
    if movie_history.shape[1] <= 0:
        raise ValueError("History length must be positive")
    return int(movie_history.shape[1])


@torch.no_grad()
def predict_validation(validation, device, batch_size):
    checkpoint = torch.load(
        BEST_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    checkpoint_metrics = checkpoint.get("metrics", {})
    history_pooling = checkpoint_metrics.get(
        "history_pooling", "masked_mean"
    )
    max_sequence_length = int(
        checkpoint_metrics.get(
            "max_sequence_length",
            np.asarray(validation["hist_movie_id"]).shape[1],
        )
    )
    recent_history_length = int(checkpoint_metrics.get("recent_history_length", 5))
    model = YouTubeDNN(
        feature_dict=checkpoint["feature_dict"],
        embedding_dim=config.EMB_DIM,
        history_pooling=history_pooling,
        max_sequence_length=max_sequence_length,
        recent_history_length=recent_history_length,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    item_embeddings = torch.as_tensor(
        np.load(config.ITEM_EMB_PATH),
        dtype=torch.float32,
        device=device,
    )
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
    return (
        torch.cat(recommendations),
        torch.cat(targets),
        int(checkpoint["epoch"]),
        history_pooling,
    )


def run(output, expected_seq_len, device_name=None, batch_size=512):
    if expected_seq_len <= 0:
        raise ValueError("expected_seq_len must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite metrics: {output}")

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
    recommendations, targets, checkpoint_epoch, history_pooling = predict_validation(
        validation, device, batch_size
    )
    raw_user_counts = reconstruct_raw_user_activity(train)
    activity = lookup(
        raw_user_counts,
        validation["user_id"],
        "Retrieval Validation user",
    )
    movie_popularity = count_by_id(train["movie_id"])
    popularity = lookup(
        movie_popularity,
        validation["movie_id"],
        "Retrieval Validation movie",
    )

    result = {
        "protocol": {
            "version": 1,
            "experiment": "retrieval-history-length-ablation",
            "split": "validation",
            "test_accessed": False,
            "selection_artifact_contains_test": False,
            "history_length": actual_seq_len,
            "history_pooling": history_pooling,
            "movie_history_semantics": "most recent N movies",
            "genre_history_semantics": "most recent N flattened genre tokens",
            "device": str(device),
            "batch_size": batch_size,
            "k_values": list(K_VALUES),
        },
        "checkpoint_epoch": checkpoint_epoch,
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval ablation on Validation only"
    )
    parser.add_argument("--expected-seq-len", type=int, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        run(
            args.output,
            args.expected_seq_len,
            args.device,
            args.batch_size,
        ),
        indent=2,
        allow_nan=False,
    ))


if __name__ == "__main__":
    main()
