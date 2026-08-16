"""Diagnose personalized attention behavior on Retrieval Validation only."""

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
    lookup,
    quantile_labels,
    reconstruct_raw_user_activity,
)
from offline.evaluation.evaluate_retrieval_validation import (
    infer_sequence_length,
    validate_selection_samples,
)
from offline.training.retrieval_data import RetrievalDataset
from offline.training.retrieval_trainer import (
    move_batch_to_device,
    resolve_device,
)
from offline.training.train_retrieval import BEST_CHECKPOINT_PATH


def recent_position_mask(ids, recent_count):
    """Mark at most `recent_count` non-padding events at each row's right."""
    if recent_count <= 0:
        raise ValueError("recent_count must be positive")
    valid = ids.ne(0)
    reverse_rank = valid.flip(1).cumsum(1).flip(1)
    return valid & reverse_rank.le(recent_count)


def summarize_weights(weights, ids, recent_count):
    """Summarize concentration without inspecting targets or Test."""
    valid = ids.ne(0)
    counts = valid.sum(dim=1).clamp_min(1).to(weights.dtype)
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)
    recent_mass = (weights * recent_position_mask(ids, recent_count)).sum(dim=1)
    return {
        "examples": int(weights.shape[0]),
        "mean_non_padding_length": float(counts.mean().item()),
        "mean_attention_entropy": float(entropy.mean().item()),
        "mean_normalized_attention_entropy": float(
            (entropy / counts.log().clamp_min(1.0)).mean().item()
        ),
        "mean_effective_history_length": float(entropy.exp().mean().item()),
        "mean_peak_attention_weight": float(
            weights.max(dim=1).values.mean().item()
        ),
        "mean_recent_weight_mass": float(recent_mass.mean().item()),
    }


@torch.no_grad()
def collect_attention_weights(model, validation, device, batch_size):
    loader = DataLoader(
        RetrievalDataset(validation),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    output = {"movie": ([], []), "genre": ([], [])}
    for features, targets in loader:
        features, _ = move_batch_to_device(features, targets, device)
        static_embeddings = [
            model.user_embeddings[name](features[name].long())
            for name in model.USER_FEATURES
        ]
        query = torch.cat(static_embeddings, dim=-1)
        for name, embedding, pooling, field in (
            ("movie", model.movie_embedding, model.movie_attention_pooling, "hist_movie_id"),
            ("genre", model.genre_embedding, model.genre_attention_pooling, "hist_genres"),
        ):
            ids = features[field].long()
            output[name][0].append(
                pooling.attention_weights(embedding(ids), ids, query).cpu()
            )
            output[name][1].append(ids.cpu())
    return {
        name: (torch.cat(values[0]), torch.cat(values[1]))
        for name, values in output.items()
    }


def grouped_summaries(weights, ids, groups, recent_count):
    result = {}
    for group in sorted(np.unique(groups)):
        mask = torch.as_tensor(groups == group, dtype=torch.bool)
        result[str(int(group))] = summarize_weights(
            weights[mask], ids[mask], recent_count
        )
    return result


def run(output, expected_seq_len=20, recent_count=5, device_name=None, batch_size=512):
    if expected_seq_len <= 0:
        raise ValueError("expected_seq_len must be positive")
    if recent_count <= 0 or recent_count > expected_seq_len:
        raise ValueError("recent_count must be in [1, expected_seq_len]")
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
            f"Expected history length {expected_seq_len}, found {actual_seq_len}"
        )

    checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if checkpoint.get("metrics", {}).get("history_pooling") != "personalized_attention":
        raise ValueError("Checkpoint is not personalized_attention")
    device = resolve_device(device_name)
    model = YouTubeDNN(
        feature_dict=checkpoint["feature_dict"],
        embedding_dim=config.EMB_DIM,
        history_pooling="personalized_attention",
        max_sequence_length=actual_seq_len,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    attention = collect_attention_weights(model, validation, device, batch_size)
    activity = lookup(
        reconstruct_raw_user_activity(train),
        validation["user_id"],
        "Retrieval Validation user",
    )
    groups = quantile_labels(activity)

    channels = {}
    for name, (weights, ids) in attention.items():
        channels[name] = {
            "overall": summarize_weights(weights, ids, recent_count),
            "by_raw_user_activity_quantile": grouped_summaries(
                weights, ids, groups, recent_count
            ),
        }
    result = {
        "protocol": {
            "version": 1,
            "experiment": "retrieval-attention-mechanism-diagnostic",
            "split": "validation",
            "test_accessed": False,
            "test_deserialized": False,
            "selection_artifact_contains_test": False,
            "checkpoint_retrained": False,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "history_length": actual_seq_len,
            "history_pooling": "personalized_attention",
            "recent_position_count": recent_count,
            "sequence_layout": "left padding; newer events on the right",
            "device": str(device),
            "batch_size": batch_size,
        },
        "channels": channels,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose personalized attention on Retrieval Validation only"
    )
    parser.add_argument("--expected-seq-len", type=int, default=20)
    parser.add_argument("--recent-count", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        run(args.output, args.expected_seq_len, args.recent_count, args.device, args.batch_size),
        indent=2,
        allow_nan=False,
    ))


if __name__ == "__main__":
    main()
