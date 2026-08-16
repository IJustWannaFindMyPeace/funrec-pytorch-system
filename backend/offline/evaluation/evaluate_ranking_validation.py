"""Evaluate a DeepFM ranking checkpoint on Validation only."""

import json
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from offline.config import config
from offline.evaluation.evaluate_ranking import build_model
from offline.training.ranking_data import RankingDataset
from offline.training.ranking_trainer import evaluate, resolve_device
from offline.training.train_ranking import BEST_CHECKPOINT_PATH


def load_validation_artifacts(checkpoint_path=BEST_CHECKPOINT_PATH):
    with open(config.RANKING_TRAIN_DATA_PATH, "rb") as file:
        samples = pickle.load(file)
    if set(samples) != {"train", "validation"}:
        raise ValueError(
            "Ranking selection artifact must contain exactly Train and "
            "Validation; Test must remain sealed"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return samples["validation"], checkpoint


def run_validation_evaluation(
    output_path,
    device_name=None,
    batch_size=512,
    checkpoint_path=BEST_CHECKPOINT_PATH,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite metrics: {output_path}")
    validation, checkpoint = load_validation_artifacts(checkpoint_path)
    device = resolve_device(device_name)
    dataset = RankingDataset(validation)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_model(checkpoint).to(device).eval()
    stats = evaluate(
        model,
        loader,
        device,
        progress_description="DeepFM Validation evaluation",
    )
    result = {
        "protocol": {"split": "validation", "test_accessed": False, "selection_artifact_contains_test": False, "batch_size": batch_size},
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "metrics": {"loss": stats.loss, "roc_auc": stats.auc},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result
