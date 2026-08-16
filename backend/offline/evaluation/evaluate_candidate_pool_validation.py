"""Validation-only two-stage candidate-pool evaluator (V2)."""

import json
from pathlib import Path

import numpy as np
import torch

from offline.evaluation.candidate_pool import append_unique_candidates
from offline.evaluation.retrieval import calculate_single_target_metrics


def rank_ids(model, static_features, movie_features, ids):
    """Return candidate IDs sorted by fixed DeepFM score."""
    if not ids:
        return []
    device = next(model.parameters()).device
    features = {
        name: torch.full((len(ids),), int(value), device=device, dtype=torch.long)
        for name, value in static_features.items()
    }
    for name, values in movie_features.items():
        features[name] = torch.as_tensor(values, device=device, dtype=torch.long)
    order = torch.argsort(model(features), descending=True).detach().cpu().tolist()
    return [ids[index] for index in order]


def summarize(common_targets, baseline_ranked, candidate_ranked, tail_mask):
    targets = torch.as_tensor(common_targets)
    baseline = torch.as_tensor(baseline_ranked)
    candidate = torch.as_tensor(candidate_ranked)
    metrics = {
        "baseline": calculate_single_target_metrics(baseline, targets, (10,)),
        "candidate": calculate_single_target_metrics(candidate, targets, (10,)),
    }
    mask = torch.as_tensor(tail_mask, dtype=torch.bool)
    metrics["baseline_tail_pq0_recall_at_10"] = float(
        (baseline[mask, :10] == targets[mask, None]).any(1).float().mean()
    )
    metrics["candidate_tail_pq0_recall_at_10"] = float(
        (candidate[mask, :10] == targets[mask, None]).any(1).float().mean()
    )
    return metrics


def write_result(output, result):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite metrics: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
