"""Train and export the PyTorch YouTubeDNN retrieval model."""

import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path
from typing import Optional

# PyTorch must be imported before NumPy on the current Windows environment.
import torch
import numpy as np
from torch.utils.data import DataLoader

from modeling.youtubednn import (
    MINDYouTubeDNN,
    SCORING_CONTRACT_LEGACY_RAW_ITEM,
    SUPPORTED_SCORING_CONTRACTS,
    YouTubeDNN,
)
from offline.config import config
from offline.evaluation.diagnose_validation import quantile_labels
from offline.training.retrieval_data import RetrievalDataset
from offline.training.retrieval_trainer import (
    evaluate_loss,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    train_one_epoch,
)


LAST_CHECKPOINT_PATH = (
    config.SAVED_MODELS_DIR / "retrieval_last.pt"
)
BEST_CHECKPOINT_PATH = (
    config.SAVED_MODELS_DIR / "retrieval_best.pt"
)
USER_MODEL_PATH = (
    config.SAVED_MODELS_DIR / "retrieval_user_model.pt"
)
TRAINING_HISTORY_PATH = (
    config.SAVED_MODELS_DIR / "retrieval_history.json"
)


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one exported artifact."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_random_seed(seed: int) -> None:
    """Configure reproducible random state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_activity_balanced_user_weights(train):
    """Build Train-only weights with equal total mass per activity quartile."""
    user_ids = np.asarray(train["user_id"], dtype=np.int64)
    counts = np.bincount(user_ids)
    activity = counts.copy()
    activity[activity > 0] += 3
    groups = np.zeros_like(activity)
    groups[1:] = quantile_labels(activity[1:])
    group_counts = np.bincount(groups[user_ids], minlength=4).astype(float)
    weights = np.zeros_like(activity, dtype=np.float32)
    weights[1:] = 1.0 / group_counts[groups[1:]]
    weights[1:] /= weights[user_ids].mean()
    return torch.as_tensor(weights)


def validate_training_selection_samples(samples):
    """Fail closed unless a training artifact contains only selection splits."""
    if not isinstance(samples, dict) or set(samples) != {
        "train",
        "validation",
    }:
        raise ValueError(
            "Training artifact must contain exactly Train and Validation; "
            "Test must remain sealed"
        )
    return samples


def load_training_artifacts():
    """Load preprocessed samples, feature sizes, and raw vocabularies."""
    required_paths = (
        config.TRAIN_DATA_PATH,
        config.FEATURE_DICT_PATH,
        config.VOCAB_DICT_PATH,
    )
    missing_paths = [
        path for path in required_paths if not path.exists()
    ]

    if missing_paths:
        missing = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Retrieval preprocessing artifacts are missing:\n"
            f"{missing}\n"
            "Run: python -m offline.feature.preprocess_retrieval"
        )

    with open(config.TRAIN_DATA_PATH, "rb") as file:
        samples = pickle.load(file)

    with open(config.FEATURE_DICT_PATH, "rb") as file:
        feature_dict = pickle.load(file)

    with open(config.VOCAB_DICT_PATH, "rb") as file:
        vocab_dict = pickle.load(file)

    return validate_training_selection_samples(samples), feature_dict, vocab_dict


def build_data_loaders(
    samples,
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    """Build train and validation data loaders."""
    train_dataset = RetrievalDataset(samples["train"])
    validation_dataset = RetrievalDataset(samples["validation"])

    pin_memory = device.type == "cuda"

    generator = torch.Generator()
    generator.manual_seed(42)

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
    """Load existing training history when resuming."""
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as file:
        history = json.load(file)

    if not isinstance(history, list):
        raise ValueError("Retrieval training history must be a list")

    return history


def save_history(path: Path, history: list) -> None:
    """Persist training history as readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


@torch.no_grad()
def export_retrieval_artifacts(
    model: YouTubeDNN,
    feature_dict: dict,
    vocab_dict: dict,
) -> None:
    """Export the user tower and normalized item embeddings."""
    config.SAVED_MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_cpu = model.to("cpu")
    model_cpu.eval()

    torch.save(
        {
            "model_type": "mind_k2" if isinstance(model_cpu, MINDYouTubeDNN) else "youtube_dnn",
            "feature_dict": dict(feature_dict),
            "embedding_dim": model_cpu.embedding_dim,
            "history_pooling": model_cpu.history_pooling,
            "max_sequence_length": model_cpu.max_sequence_length,
            "recent_history_length": model_cpu.recent_history_length,
            "scoring_contract": model_cpu.scoring_contract,
            "logit_scale": model_cpu.logit_scale,
            "interest_count": getattr(model_cpu, "interest_count", None),
            "routing_iterations": getattr(model_cpu, "routing_iterations", None),
            "model_state_dict": model_cpu.state_dict(),
        },
        USER_MODEL_PATH,
    )

    encoded_movie_ids = torch.arange(
        1,
        feature_dict["movie_id"],
        dtype=torch.long,
    )
    item_embeddings = (
        model_cpu.encode_item(encoded_movie_ids)
        .detach()
        .numpy()
        .astype(np.float32)
    )

    raw_movie_ids = np.asarray(
        vocab_dict["movie_id"],
        dtype=str,
    )

    if len(raw_movie_ids) != len(item_embeddings):
        raise ValueError(
            "Movie vocabulary and exported embeddings have "
            "different lengths"
        )

    np.save(config.ITEM_EMB_PATH, item_embeddings)
    np.save(config.MOVIE_IDS_PATH, raw_movie_ids)

    manifest_path = config.RETRIEVAL_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "artifact_version": 2,
        "model_type": "mind_k2" if isinstance(model_cpu, MINDYouTubeDNN) else "youtube_dnn",
        "scoring_contract": model_cpu.scoring_contract,
        "logit_scale": model_cpu.logit_scale,
        "files": {
            "vocab_dict.pkl": file_sha256(config.VOCAB_DICT_PATH),
            "item_embeddings.npy": file_sha256(config.ITEM_EMB_PATH),
            "movie_ids.npy": file_sha256(config.MOVIE_IDS_PATH),
            "retrieval_user_model.pt": file_sha256(USER_MODEL_PATH),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("已导出召回模型:")
    print(f"  用户模型: {USER_MODEL_PATH}")
    print(f"  物品向量: {config.ITEM_EMB_PATH}")
    print(f"  电影 ID: {config.MOVIE_IDS_PATH}")
    print(f"  物品向量形状: {item_embeddings.shape}")


def run_retrieval_training(
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
    history_pooling: str = "masked_mean",
    max_sequence_length: int = config.MAX_SEQ_LEN,
    recent_history_length: int = 5,
    loss_weighting: str = "uniform",
    scoring_contract: str = SCORING_CONTRACT_LEGACY_RAW_ITEM,
    logit_scale: float = 1.0,
    model_type: str = "youtube_dnn",
):
    """Run training, validation, checkpointing, and export."""
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")
    if patience <= 0:
        raise ValueError("patience must be greater than zero")
    if num_workers < 0:
        raise ValueError("num_workers must not be negative")
    if history_pooling not in {"masked_mean", "personalized_attention", "dual_timescale_attention"}:
        raise ValueError("Unsupported history_pooling")
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    if history_pooling == "dual_timescale_attention" and not 0 < recent_history_length < max_sequence_length:
        raise ValueError("recent_history_length must be within max_sequence_length")
    if loss_weighting not in {"uniform", "activity_balanced"}:
        raise ValueError("Unsupported loss_weighting")
    if scoring_contract not in SUPPORTED_SCORING_CONTRACTS:
        raise ValueError("Unsupported scoring_contract")
    if logit_scale <= 0:
        raise ValueError("logit_scale must be positive")
    if model_type not in {"youtube_dnn", "mind_k2"}:
        raise ValueError("Unsupported model_type")

    set_random_seed(seed)
    device = resolve_device(device_name)

    print("=" * 60)
    print("PyTorch YouTubeDNN 召回模型训练")
    print("=" * 60)
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    samples, feature_dict, vocab_dict = (
        load_training_artifacts()
    )
    actual_sequence_length = int(
        np.asarray(samples["validation"]["hist_movie_id"]).shape[1]
    )
    if actual_sequence_length != max_sequence_length:
        raise ValueError(
            f"Configured max_sequence_length={max_sequence_length}, "
            f"but Validation history length is {actual_sequence_length}"
        )
    train_loader, validation_loader = build_data_loaders(
        samples=samples,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    user_loss_weights = None
    if loss_weighting == "activity_balanced":
        user_loss_weights = build_activity_balanced_user_weights(samples["train"]).to(device)

    model_class = MINDYouTubeDNN if model_type == "mind_k2" else YouTubeDNN
    model = model_class(
        feature_dict=feature_dict,
        embedding_dim=config.EMB_DIM,
        history_pooling=history_pooling,
        max_sequence_length=max_sequence_length,
        recent_history_length=recent_history_length,
        scoring_contract=scoring_contract,
        logit_scale=logit_scale,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    start_epoch = 1
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    if resume:
        if not LAST_CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                "Resume requested but checkpoint does not exist: "
                f"{LAST_CHECKPOINT_PATH}"
            )

        checkpoint_metadata = torch.load(
            LAST_CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=True,
        )
        checkpoint_metrics = checkpoint_metadata.get("metrics", {})
        if checkpoint_metrics.get("history_pooling", "masked_mean") != history_pooling:
            raise ValueError("Checkpoint history_pooling does not match")
        if int(checkpoint_metrics.get(
            "max_sequence_length", max_sequence_length
        )) != max_sequence_length:
            raise ValueError("Checkpoint max_sequence_length does not match")
        checkpoint = load_checkpoint(
            LAST_CHECKPOINT_PATH,
            model,
            optimizer,
            map_location=device,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(
            checkpoint.get("metrics", {}).get(
                "best_validation_loss",
                float("inf"),
            )
        )
        epochs_without_improvement = int(
            checkpoint.get("metrics", {}).get(
                "epochs_without_improvement",
                0,
            )
        )
        history = load_history(TRAINING_HISTORY_PATH)

        print(
            f"从 epoch {checkpoint['epoch']} 的 checkpoint 继续"
        )

    if start_epoch > epochs:
        raise ValueError(
            f"Checkpoint 已完成 {start_epoch - 1} 个 epoch，"
            f"目标 epochs={epochs}"
        )

    print(f"训练样本: {len(train_loader.dataset)}")
    print(f"验证样本: {len(validation_loader.dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"目标 epoch: {epochs}")
    print(f"History pooling: {history_pooling}")
    print(f"History length: {max_sequence_length}")

    for epoch in range(start_epoch, epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            max_batches=max_train_batches,
            progress_description=f"Train {epoch}/{epochs}",
            user_loss_weights=user_loss_weights,
        )
        validation_stats = evaluate_loss(
            model=model,
            data_loader=validation_loader,
            device=device,
            max_batches=max_eval_batches,
            progress_description=f"Eval {epoch}/{epochs}",
        )

        improved = (
            validation_stats.loss < best_validation_loss
        )

        if improved:
            best_validation_loss = validation_stats.loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_stats.loss,
            "validation_loss": validation_stats.loss,
            "train_examples": train_stats.examples,
            "validation_examples": validation_stats.examples,
            "train_batches": train_stats.batches,
            "validation_batches": validation_stats.batches,
            "improved": improved,
        }
        history.append(epoch_record)
        save_history(TRAINING_HISTORY_PATH, history)

        checkpoint_metrics = {
            "train_loss": train_stats.loss,
            "validation_loss": validation_stats.loss,
            "best_validation_loss": best_validation_loss,
            "epochs_without_improvement": (
                epochs_without_improvement
            ),
            "history_pooling": history_pooling,
            "max_sequence_length": max_sequence_length,
            "recent_history_length": recent_history_length,
            "loss_weighting": loss_weighting,
            "scoring_contract": scoring_contract,
            "logit_scale": logit_scale,
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
            f"val_loss={validation_stats.loss:.6f} - "
            f"best_val_loss={best_validation_loss:.6f}"
        )

        if epochs_without_improvement >= patience:
            print(
                f"验证 loss 连续 {patience} 个 epoch 未改善，"
                "提前停止。"
            )
            break

    if not BEST_CHECKPOINT_PATH.exists():
        raise RuntimeError("Best checkpoint was not created")

    best_checkpoint = load_checkpoint(
        BEST_CHECKPOINT_PATH,
        model,
        map_location=device,
    )
    print(
        f"加载最佳 checkpoint: epoch "
        f"{best_checkpoint['epoch']}"
    )

    export_retrieval_artifacts(
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
        description="Train PyTorch YouTubeDNN retrieval model"
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
    parser.add_argument(
        "--history-pooling",
        choices=("masked_mean", "personalized_attention", "dual_timescale_attention"),
        default="masked_mean",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=config.MAX_SEQ_LEN,
    )
    parser.add_argument("--recent-history-length", type=int, default=5)
    parser.add_argument(
        "--loss-weighting",
        choices=("uniform", "activity_balanced"),
        default="uniform",
    )
    parser.add_argument(
        "--scoring-contract",
        choices=tuple(sorted(SUPPORTED_SCORING_CONTRACTS)),
        default=SCORING_CONTRACT_LEGACY_RAW_ITEM,
    )
    parser.add_argument("--logit-scale", type=float, default=1.0)
    parser.add_argument("--model-type", choices=("youtube_dnn", "mind_k2"), default="youtube_dnn")
    return parser.parse_args()


def main():
    args = parse_args()

    run_retrieval_training(
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
        history_pooling=args.history_pooling,
        max_sequence_length=args.max_sequence_length,
        recent_history_length=args.recent_history_length,
        loss_weighting=args.loss_weighting,
        scoring_contract=args.scoring_contract,
        logit_scale=args.logit_scale,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
