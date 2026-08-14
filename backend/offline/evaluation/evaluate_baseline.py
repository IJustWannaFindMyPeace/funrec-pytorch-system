"""Run and record the final held-out Baseline V0 evaluation."""

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from offline.config import config
from offline.evaluation.evaluate_ranking import run_evaluation as run_ranking
from offline.evaluation.evaluate_retrieval import run_evaluation as run_retrieval


DEFAULT_OUTPUT_PATH = config.TEMP_DIR / "evaluation" / "baseline_v0_results.json"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_manifest(
    seed: int,
    retrieval: dict | None = None,
    ranking: dict | None = None,
) -> dict:
    retrieval_best_epoch = (
        retrieval.get("model", {}).get("best_checkpoint_epoch")
        if retrieval is not None
        else None
    )
    ranking_best_epoch = (
        ranking.get("model", {}).get("best_checkpoint_epoch")
        if ranking is not None
        else None
    )
    return {
        "experiment": "baseline-v0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "seed": seed,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "configuration": {
            "max_sequence_length": config.MAX_SEQ_LEN,
            "embedding_dimension": config.EMB_DIM,
            "negative_sample_size": config.NEG_SAMPLE_SIZE,
            "batch_size": config.BATCH_SIZE,
            "configured_default_epochs": config.EPOCHS,
            "learning_rate": config.LEARNING_RATE,
        },
        "training_selection": {
            "retrieval_best_checkpoint_epoch": retrieval_best_epoch,
            "ranking_best_checkpoint_epoch": ranking_best_epoch,
        },
        "artifact_root": "<FUNREC_PROCESSED_DATA_PATH>/web_project",
    }


def run_baseline_evaluation(
    device_name: str | None = None,
    batch_size: int = 512,
    k_values: tuple[int, ...] = (5, 10),
    seed: int = 42,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    overwrite: bool = False,
) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k values must be positive")

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite final Baseline V0 result: {output_path}"
        )

    component_dir = output_path.parent / "components"
    retrieval = run_retrieval(
        device_name=device_name,
        batch_size=batch_size,
        k_values=k_values,
        output_path=component_dir / "retrieval_test_metrics.json",
    )
    ranking = run_ranking(
        device_name=device_name,
        batch_size=batch_size,
        output_path=component_dir / "ranking_test_metrics.json",
        overwrite=True,
    )

    result = {
        "manifest": build_manifest(
            seed,
            retrieval=retrieval,
            ranking=ranking,
        ),
        "retrieval": retrieval,
        "ranking": ranking,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f"Baseline V0 最终结果已保存: {output_path}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run final held-out Baseline V0 evaluation"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_baseline_evaluation(
        device_name=args.device,
        batch_size=args.batch_size,
        k_values=tuple(args.k),
        seed=args.seed,
        output_path=args.output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
