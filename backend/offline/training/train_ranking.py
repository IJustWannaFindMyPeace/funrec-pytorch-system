"""Train and export the PyTorch DeepFM ranking model."""

import argparse
import json
import math
import pickle
import random
from pathlib import Path
from typing import Optional

# PyTorch must be imported before NumPy on Windows.
import torch
import numpy as np
from torch.utils.data import DataLoader

from modeling.deepfm import DeepFM
from offline.config import config
from offline.training.ranking_data import RankingDataset
from offline.training.ranking_trainer import (
    evaluate,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    train_one_epoch,
)


LAST_CHECKPOINT_PATH = (
    config.SAVED_MODELS_DIR / "ranking_last.pt"
)
BEST_CHECKPOINT_PATH = (
    config.SAVED_MODELS_DIR / "ranking_best.pt"
)
RANKING_MODEL_PATH = (
    config.SAVED_MODELS_DIR / "ranking_model.pt"
)
TRAINING_HISTORY_PATH = (
    config.SAVED_MODELS_DIR / "ranking_history.json"
)
MODEL_CONFIG_PATH = (
    config.TEMP_DIR / "ranking_model_config.pkl"
)

DNN_HIDDEN_UNITS = (128, 64, 32)
DROPOUT = 0.1


def set_random_seed(seed: int) -> None:
    """Configure reproducible random state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_artifacts():
    """Load ranking samples, feature sizes and vocabularies."""
    required_paths = (
        config.RANKING_TRAIN_DATA_PATH,
        config.RANKING_FEATURE_DICT_PATH,
        config.RANKING_VOCAB_DICT_PATH,
    )
    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        missing = "\n".join(
            str(path)
            for path in missing_paths
        )
        raise FileNotFoundError(
            "Ranking preprocessing artifacts are missing:\n"
            f"{missing}\n"
            "Run: python -m "
            "offline.feature.preprocess_ranking"
        )

    with open(
        config.RANKING_TRAIN_DATA_PATH,
        "rb",
    ) as file:
        samples = pickle.load(file)

    with open(
        config.RANKING_FEATURE_DICT_PATH,
        "rb",
    ) as file:
        feature_dict = pickle.load(file)

    with open(
        config.RANKING_VOCAB_DICT_PATH,
        "rb",
    ) as file:
        vocab_dict = pickle.load(file)

    return samples, feature_dict, vocab_dict


def build_data_loaders(
    samples,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
):
    """Build shuffled training and ordered validation loaders."""
    train_dataset = RankingDataset(samples["train"])
    validation_dataset = RankingDataset(samples["test"])

    pin_memory = device.type == "cuda"

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return train_loader, validation_loader


def load_history(path: Path) -> list:
    """Load existing ranking history when resuming."""
    path = Path(path)

    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as file:
        history = json.load(file)

    if not isinstance(history, list):
        raise ValueError(
            "Ranking training history must be a list"
        )

    return history


def save_history(
    path: Path,
    history: list,
) -> None:
    """Persist ranking history as readable JSON."""
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            history,
            file,
            indent=2,
            allow_nan=True,
        )


@torch.no_grad()
def export_ranking_artifacts(
    model: DeepFM,
    feature_dict: dict,
    vocab_dict: dict,
) -> None:
    """Export a portable PyTorch ranking artifact."""
    config.SAVED_MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    MODEL_CONFIG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_cpu = model.to("cpu")
    model_cpu.eval()

    model_config = {
        "feature_names": list(model_cpu.FEATURE_NAMES),
        "embedding_dim": model_cpu.embedding_dim,
        "dnn_hidden_units": tuple(
            model_cpu.dnn_hidden_units
        ),
        "dropout": model_cpu.dropout,
    }

    torch.save(
        {
            "feature_dict": dict(feature_dict),
            "model_config": model_config,
            "model_state_dict": model_cpu.state_dict(),
        },
        RANKING_MODEL_PATH,
    )

    with open(MODEL_CONFIG_PATH, "wb") as file:
        pickle.dump(
            {
                "feature_dict": dict(feature_dict),
                "feature_names": list(
                    model_cpu.FEATURE_NAMES
                ),
                "model_config": model_config,
            },
            file,
        )

    missing_vocabularies = [
        name
        for name in model_cpu.FEATURE_NAMES
        if name not in vocab_dict
    ]
    if missing_vocabularies:
        raise ValueError(
            "Ranking vocabulary is missing features: "
            + ", ".join(missing_vocabularies)
        )

    print("已导出精排模型:")
    print(f"  PyTorch 模型: {RANKING_MODEL_PATH}")
    print(f"  模型配置: {MODEL_CONFIG_PATH}")


def run_ranking_training(
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    device_name: Optional[str] = None,
    resume: bool = False,
    patience: int = 2,
    num_workers: int = 0,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    seed: int = 42,
):
    """Run DeepFM training, validation and export."""
    if epochs <= 0:
        raise ValueError(
            "epochs must be greater than zero"
        )
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero"
        )
    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than zero"
        )
    if patience <= 0:
        raise ValueError(
            "patience must be greater than zero"
        )
    if num_workers < 0:
        raise ValueError(
            "num_workers must not be negative"
        )
    if (
        max_train_batches is not None
        and max_train_batches <= 0
    ):
        raise ValueError(
            "max_train_batches must be greater than zero"
        )
    if (
        max_eval_batches is not None
        and max_eval_batches <= 0
    ):
        raise ValueError(
            "max_eval_batches must be greater than zero"
        )

    set_random_seed(seed)
    device = resolve_device(device_name)

    print("=" * 60)
    print("PyTorch DeepFM 精排模型训练")
    print("=" * 60)
    print(f"设备: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(device)}"
        )

    samples, feature_dict, vocab_dict = (
        load_training_artifacts()
    )
    train_loader, validation_loader = (
        build_data_loaders(
            samples=samples,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            seed=seed,
        )
    )

    model = DeepFM(
        feature_dict=feature_dict,
        embedding_dim=config.EMB_DIM,
        dnn_hidden_units=DNN_HIDDEN_UNITS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    start_epoch = 1
    best_validation_auc = float("-inf")
    epochs_without_improvement = 0
    history = []
    has_best_checkpoint = False

    if resume:
        if not LAST_CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                "Resume requested but checkpoint "
                "does not exist: "
                f"{LAST_CHECKPOINT_PATH}"
            )

        checkpoint = load_checkpoint(
            LAST_CHECKPOINT_PATH,
            model,
            optimizer,
            map_location=device,
        )

        if checkpoint.get("feature_dict") != feature_dict:
            raise ValueError(
                "Checkpoint feature_dict does not match "
                "the current ranking artifacts"
            )

        start_epoch = int(checkpoint["epoch"]) + 1

        best_validation_auc = float(
            checkpoint.get("metrics", {}).get(
                "best_validation_auc",
                float("-inf"),
            )
        )
        epochs_without_improvement = int(
            checkpoint.get("metrics", {}).get(
                "epochs_without_improvement",
                0,
            )
        )
        history = load_history(
            TRAINING_HISTORY_PATH
        )
        has_best_checkpoint = (
            BEST_CHECKPOINT_PATH.exists()
        )

        print(
            f"从 epoch {checkpoint['epoch']} "
            "的 checkpoint 继续"
        )

    if start_epoch > epochs:
        raise ValueError(
            f"Checkpoint 已完成 {start_epoch - 1} "
            f"个 epoch，目标 epochs={epochs}"
        )

    print(
        f"训练样本: {len(train_loader.dataset)}"
    )
    print(
        f"验证样本: {len(validation_loader.dataset)}"
    )
    print(f"Batch size: {batch_size}")
    print(f"目标 epoch: {epochs}")

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):
        train_stats = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            max_batches=max_train_batches,
            progress_description=(
                f"Train {epoch}/{epochs}"
            ),
        )
        validation_stats = evaluate(
            model=model,
            data_loader=validation_loader,
            device=device,
            max_batches=max_eval_batches,
            progress_description=(
                f"Eval {epoch}/{epochs}"
            ),
        )

        validation_auc_is_finite = math.isfinite(
            validation_stats.auc
        )
        improved = (
            not has_best_checkpoint
            or (
                validation_auc_is_finite
                and (
                    not math.isfinite(
                        best_validation_auc
                    )
                    or validation_stats.auc
                    > best_validation_auc
                )
            )
        )

        if improved:
            best_validation_auc = (
                validation_stats.auc
            )
            epochs_without_improvement = 0
            has_best_checkpoint = True
        else:
            epochs_without_improvement += 1

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_stats.loss,
            "train_auc": train_stats.auc,
            "validation_loss": (
                validation_stats.loss
            ),
            "validation_auc": (
                validation_stats.auc
            ),
            "train_examples": (
                train_stats.examples
            ),
            "validation_examples": (
                validation_stats.examples
            ),
            "train_batches": train_stats.batches,
            "validation_batches": (
                validation_stats.batches
            ),
            "improved": improved,
        }
        history.append(epoch_record)
        save_history(
            TRAINING_HISTORY_PATH,
            history,
        )

        checkpoint_metrics = {
            "train_loss": train_stats.loss,
            "train_auc": train_stats.auc,
            "validation_loss": (
                validation_stats.loss
            ),
            "validation_auc": (
                validation_stats.auc
            ),
            "best_validation_auc": (
                best_validation_auc
            ),
            "epochs_without_improvement": (
                epochs_without_improvement
            ),
        }

        save_checkpoint(
            LAST_CHECKPOINT_PATH,
            model,
            optimizer,
            epoch=epoch,
            feature_dict=feature_dict,
            metrics=checkpoint_metrics,
        )

        if improved:
            save_checkpoint(
                BEST_CHECKPOINT_PATH,
                model,
                optimizer,
                epoch=epoch,
                feature_dict=feature_dict,
                metrics=checkpoint_metrics,
            )

        print(
            f"Epoch {epoch}/{epochs} - "
            f"train_loss={train_stats.loss:.6f} - "
            f"train_auc={train_stats.auc:.6f} - "
            f"val_loss={validation_stats.loss:.6f} - "
            f"val_auc={validation_stats.auc:.6f} - "
            f"best_val_auc={best_validation_auc:.6f}"
        )

        if (
            epochs_without_improvement
            >= patience
        ):
            print(
                f"验证 AUC 连续 {patience} 个 epoch "
                "未改善，提前停止。"
            )
            break

    if not BEST_CHECKPOINT_PATH.exists():
        raise RuntimeError(
            "Best ranking checkpoint was not created"
        )

    best_checkpoint = load_checkpoint(
        BEST_CHECKPOINT_PATH,
        model,
        map_location=device,
    )
    print(
        "加载最佳 checkpoint: epoch "
        f"{best_checkpoint['epoch']}"
    )

    export_ranking_artifacts(
        model=model,
        feature_dict=feature_dict,
        vocab_dict=vocab_dict,
    )

    return {
        "model": model,
        "history": history,
        "best_checkpoint": best_checkpoint,
        "device": device,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train PyTorch DeepFM ranking model"
        )
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=config.LEARNING_RATE,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    run_ranking_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device_name=args.device,
        resume=args.resume,
        patience=args.patience,
        num_workers=args.num_workers,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()