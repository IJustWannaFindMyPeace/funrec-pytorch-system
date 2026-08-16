"""
本地部署脚本

将训练好的模型和工件部署到本地共享目录用于在线服务：
- 召回模型：YoutubeDNN 用户模型、物品嵌入向量、词表
- 精排模型：DeepFM 模型、特征字典、词表

使用方法:
    uv run python -m offline.storage.local_deploy
    uv run python -m offline.storage.local_deploy --ranking-only
    uv run python -m offline.storage.local_deploy --recall-only
"""

import shutil
import argparse
from pathlib import Path
from offline.config import config


def deploy_recall_models(deploy_dir: Path):
    """部署召回相关的模型和工件"""
    print("\n" + "=" * 50)
    print("部署召回模型...")
    print("=" * 50)
    
    recall_dir = deploy_dir / "recall"
    recall_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 词表字典
    if config.VOCAB_DICT_PATH.exists():
        shutil.copy2(config.VOCAB_DICT_PATH, recall_dir / "vocab_dict.pkl")
        # 同时复制到根目录以保持向后兼容
        shutil.copy2(config.VOCAB_DICT_PATH, deploy_dir / "vocab_dict.pkl")
        print("  ✓ 复制了 vocab_dict.pkl")
    else:
        print("  ✗ vocab_dict.pkl 不存在")
    
    # 2. 物品嵌入向量
    if config.ITEM_EMB_PATH.exists():
        shutil.copy2(config.ITEM_EMB_PATH, recall_dir / "item_embeddings.npy")
        shutil.copy2(config.ITEM_EMB_PATH, deploy_dir / "item_embeddings.npy")
        print("  ✓ 复制了 item_embeddings.npy")
    else:
        print("  ✗ item_embeddings.npy 不存在")
        
    if config.MOVIE_IDS_PATH.exists():
        shutil.copy2(config.MOVIE_IDS_PATH, recall_dir / "movie_ids.npy")
        shutil.copy2(config.MOVIE_IDS_PATH, deploy_dir / "movie_ids.npy")
        print("  ✓ 复制了 movie_ids.npy")
    else:
        print("  ✗ movie_ids.npy 不存在")
        
    # 3. PyTorch 用户塔模型
    user_model_path = (
        config.SAVED_MODELS_DIR / "retrieval_user_model.pt"
    )
    deployed_user_model_path = (
        recall_dir / "retrieval_user_model.pt"
    )
    manifest_path = config.RETRIEVAL_MANIFEST_PATH

    if user_model_path.exists():
        shutil.copy2(
            user_model_path,
            deployed_user_model_path,
        )
        print("  ✓ 复制了 retrieval_user_model.pt")
    else:
        print(
            "  ✗ PyTorch 用户模型不存在: "
            f"{user_model_path}"
        )

    if manifest_path.exists():
        shutil.copy2(manifest_path, recall_dir / "retrieval_manifest.json")
        print("  Copied retrieval_manifest.json")


def deploy_ranking_models(deploy_dir: Path):
    """部署 PyTorch DeepFM 精排模型及编码工件。"""
    print("\n" + "=" * 50)
    print("部署精排模型...")
    print("=" * 50)

    ranking_dir = deploy_dir / "ranking"
    ranking_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_paths = {
        "vocab_dict.pkl": (
            config.RANKING_VOCAB_DICT_PATH
        ),
        "feature_dict.pkl": (
            config.RANKING_FEATURE_DICT_PATH
        ),
        "model_config.pkl": (
            config.TEMP_DIR
            / "ranking_model_config.pkl"
        ),
        "ranking_model.pt": (
            config.RANKING_MODEL_PATH
        ),
    }

    missing_paths = [
        path
        for path in source_paths.values()
        if not path.exists()
    ]
    if missing_paths:
        missing = "\n".join(
            f"  - {path}"
            for path in missing_paths
        )
        raise FileNotFoundError(
            "精排部署工件缺失:\n"
            f"{missing}"
        )

    for deployed_name, source_path in (
        source_paths.items()
    ):
        shutil.copy2(
            source_path,
            ranking_dir / deployed_name,
        )
        print(
            f"  ✓ 复制了 ranking/{deployed_name}"
        )

    # 清理旧 TensorFlow SavedModel 部署目录，避免在线
    # 服务误加载过期模型。
    legacy_model_dir = (
        deploy_dir / "model" / "ranking"
    )
    if legacy_model_dir.exists():
        shutil.rmtree(legacy_model_dir)


def deploy_local(recall: bool = True, ranking: bool = True):
    """
    主部署函数
    
    Args:
        recall: 是否部署召回模型
        ranking: 是否部署精排模型
    """
    deploy_dir = config.DEPLOY_DIR
    print(f"部署模型到本地目录: {deploy_dir}")
    
    # 确保部署目录存在
    deploy_dir.mkdir(parents=True, exist_ok=True)

    if recall:
        deploy_recall_models(deploy_dir)
        
    if ranking:
        deploy_ranking_models(deploy_dir)
    
    print("\n" + "=" * 50)
    print("部署完成!")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="部署模型到本地目录")
    parser.add_argument("--recall-only", action="store_true", help="只部署召回模型")
    parser.add_argument("--ranking-only", action="store_true", help="只部署精排模型")
    args = parser.parse_args()
    
    if args.recall_only:
        deploy_local(recall=True, ranking=False)
    elif args.ranking_only:
        deploy_local(recall=False, ranking=True)
    else:
        deploy_local(recall=True, ranking=True)
