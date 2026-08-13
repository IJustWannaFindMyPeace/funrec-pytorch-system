"""Evaluate YouTubeDNN against a global-popularity baseline."""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, Sequence

# PyTorch must load before NumPy in the current Windows environment.
import torch
import numpy as np
from torch.utils.data import DataLoader

from modeling.youtubednn import YouTubeDNN
from offline.config import config
from offline.evaluation.retrieval import (
    calculate_single_target_metrics,
    evaluate_retrieval,
    recommend_popular_top_k,
)
from offline.training.retrieval_data import RetrievalDataset
from offline.training.train_retrieval import (
    BEST_CHECKPOINT_PATH,
    USER_MODEL_PATH,
)


DEFAULT_OUTPUT_PATH = config.TEMP_DIR / "retrieval_metrics.json"


def load_evaluation_artifacts():
    """Load test samples and trained retrieval artifacts."""
    required_paths = (
        config.TRAIN_DATA_PATH,
        config.ITEM_EMB_PATH,
        config.MOVIE_IDS_PATH,
        USER_MODEL_PATH,
        BEST_CHECKPOINT_PATH,
    )
    missing_paths = [
        path for path in required_paths if not path.exists()
    ]

    if missing_paths:
        missing = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Retrieval evaluation artifacts are missing:\n"
            f"{missing}"
        )

    with open(config.TRAIN_DATA_PATH, "rb") as file:
        samples = pickle.load(file)

    user_artifact = torch.load(
        USER_MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )
    best_checkpoint = torch.load(
        BEST_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    item_embeddings = np.load(config.ITEM_EMB_PATH)
    movie_ids = np.load(config.MOVIE_IDS_PATH)

    return (
        samples,
        user_artifact,
        best_checkpoint,
        item_embeddings,
        movie_ids,
    )


def evaluate_popularity_baseline(
    train_data: dict,
    test_data: dict,
    item_count: int,
    k_values: Sequence[int],
    batch_size: int,
) -> Dict[str, float]:
    """Evaluate global popularity with the same history filtering."""
    train_targets = torch.as_tensor(
        train_data["movie_id"]
    ).long()
    test_targets = torch.as_tensor(
        test_data["movie_id"]
    ).long()
    test_histories = torch.as_tensor(
        test_data["hist_movie_id"]
    ).long()

    popularity = torch.bincount(
        train_targets,
        minlength=item_count + 1,
    ).to(torch.float32)

    max_k = max(k_values)
    recommendations = []

    for start_index in range(
        0,
        len(test_targets),
        batch_size,
    ):
        histories = test_histories[
            start_index : start_index + batch_size
        ]
        recommendations.append(
            recommend_popular_top_k(
                popularity=popularity,
                history_ids=histories,
                k=max_k,
            )
        )

    recommendations = torch.cat(
        recommendations,
        dim=0,
    )

    return calculate_single_target_metrics(
        recommendations=recommendations,
        targets=test_targets,
        k_values=k_values,
    )


def calculate_comparison(
    model_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    test_users: int,
) -> Dict[str, dict]:
    """Calculate absolute and relative improvements."""
    comparison = {}

    for name, model_value in model_metrics.items():
        baseline_value = baseline_metrics[name]

        comparison[name] = {
            "youtube_dnn": model_value,
            "popularity": baseline_value,
            "absolute_improvement": (
                model_value - baseline_value
            ),
            "relative_lift": (
                model_value / baseline_value
                if baseline_value > 0
                else None
            ),
        }

        if name.startswith("hit_rate@"):
            comparison[name]["youtube_dnn_hits"] = round(
                model_value * test_users
            )
            comparison[name]["popularity_hits"] = round(
                baseline_value * test_users
            )

    return comparison


def run_evaluation(
    device_name: str | None = None,
    batch_size: int = 512,
    k_values: Sequence[int] = (5, 10),
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict:
    """Run model and popularity evaluation and save JSON results."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    if not k_values:
        raise ValueError("k_values must not be empty")

    if any(k <= 0 for k in k_values):
        raise ValueError("k values must be greater than zero")

    device = torch.device(
        device_name
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    (
        samples,
        user_artifact,
        best_checkpoint,
        item_embeddings_array,
        movie_ids,
    ) = load_evaluation_artifacts()

    model = YouTubeDNN(
        feature_dict=user_artifact["feature_dict"],
        embedding_dim=user_artifact["embedding_dim"],
    )
    model.load_state_dict(
        user_artifact["model_state_dict"]
    )
    model = model.to(device)
    model.eval()

    item_embeddings = torch.from_numpy(
        item_embeddings_array
    ).to(device)

    test_dataset = RetrievalDataset(samples["test"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    start = time.perf_counter()
    model_metrics = evaluate_retrieval(
        model=model,
        data_loader=test_loader,
        item_embeddings=item_embeddings,
        device=device,
        k_values=k_values,
        description="YouTubeDNN evaluation",
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    model_seconds = time.perf_counter() - start

    start = time.perf_counter()
    popularity_metrics = evaluate_popularity_baseline(
        train_data=samples["train"],
        test_data=samples["test"],
        item_count=len(movie_ids),
        k_values=k_values,
        batch_size=batch_size,
    )
    popularity_seconds = time.perf_counter() - start

    result = {
        "protocol": {
            "test_users": len(test_dataset),
            "relevant_items_per_user": 1,
            "history_filtered": True,
            "item_count": len(movie_ids),
            "embedding_dimension": (
                item_embeddings_array.shape[1]
            ),
            "k_values": list(k_values),
        },
        "model": {
            "name": "youtube_dnn",
            "best_checkpoint_epoch": int(
                best_checkpoint["epoch"]
            ),
            "metrics": model_metrics,
            "elapsed_seconds": model_seconds,
        },
        "baseline": {
            "name": "global_popularity",
            "metrics": popularity_metrics,
            "elapsed_seconds": popularity_seconds,
        },
        "comparison": calculate_comparison(
            model_metrics=model_metrics,
            baseline_metrics=popularity_metrics,
            test_users=len(test_dataset),
        ),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))
    print(f"评估结果已保存: {output_path}")

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate YouTubeDNN and popularity retrieval"
        )
    )
    parser.add_argument(
        "--device",
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[5, 10],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_evaluation(
        device_name=args.device,
        batch_size=args.batch_size,
        k_values=tuple(args.k),
        output_path=args.output,
    )


if __name__ == "__main__":
    main()