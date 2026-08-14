"""Evaluate the best DeepFM checkpoint on the held-out test split."""

import argparse
import json
import math
import pickle
from pathlib import Path

# PyTorch must load before NumPy-dependent modules on Windows.
import torch
from torch.utils.data import DataLoader

from modeling.deepfm import DeepFM
from offline.config import config
from offline.training.ranking_data import RankingDataset
from offline.training.ranking_trainer import evaluate, resolve_device
from offline.training.train_ranking import BEST_CHECKPOINT_PATH


DEFAULT_OUTPUT_PATH = config.TEMP_DIR / "evaluation" / "ranking_test_metrics.json"


def load_evaluation_artifacts():
    required_paths = (
        config.RANKING_TRAIN_DATA_PATH,
        BEST_CHECKPOINT_PATH,
    )
    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        missing = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Ranking evaluation artifacts are missing:\n" f"{missing}"
        )

    with open(config.RANKING_TRAIN_DATA_PATH, "rb") as file:
        samples = pickle.load(file)

    if "test" not in samples:
        raise ValueError("Ranking samples do not contain a test split")

    checkpoint = torch.load(
        BEST_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    return samples, checkpoint


def build_model(checkpoint: dict) -> DeepFM:
    model_config = checkpoint["model_config"]
    model = DeepFM(
        feature_dict=checkpoint["feature_dict"],
        embedding_dim=model_config["embedding_dim"],
        dnn_hidden_units=tuple(model_config["dnn_hidden_units"]),
        dropout=model_config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def run_evaluation(
    device_name: str | None = None,
    batch_size: int = 512,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    overwrite: bool = False,
) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite final ranking test result: {output_path}"
        )

    samples, checkpoint = load_evaluation_artifacts()
    device = resolve_device(device_name)
    model = build_model(checkpoint).to(device)
    model.eval()

    test_dataset = RankingDataset(samples["test"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    stats = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        progress_description="DeepFM test evaluation",
    )
    if not math.isfinite(stats.auc):
        raise ValueError("DeepFM test AUC is not finite")

    labels = test_dataset.labels
    result = {
        "protocol": {
            "evaluation_split": "test",
            "test_examples": len(test_dataset),
            "positive_examples": int((labels == 1).sum().item()),
            "negative_examples": int((labels == 0).sum().item()),
            "batch_size": batch_size,
        },
        "model": {
            "name": "deepfm",
            "best_checkpoint_epoch": int(checkpoint["epoch"]),
            "selection_metrics": checkpoint.get("metrics", {}),
            "test_metrics": {
                "loss": stats.loss,
                "roc_auc": stats.auc,
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f"精排 Test 结果已保存: {output_path}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DeepFM on the held-out test split"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_evaluation(
        device_name=args.device,
        batch_size=args.batch_size,
        output_path=args.output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
