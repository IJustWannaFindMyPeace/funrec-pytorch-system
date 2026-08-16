"""
精排模型数据预处理 (DeepFM 等)

本模块为精排/CTR 预估生成逐点样本：
- 正样本：用户-物品交互，click=True 或 conversion=True
- 困难负样本：曝光但未点击/转化 (exposure=True, click=False, conversion=False)
- 随机负样本：用户从未交互过的物品

输出格式：
{
    "train": {
        "user_id": [...], "gender": [...], ...,  # 用户特征
        "movie_id": [...], "genres": [...], ..., # 物品特征
        "is_click": [0, 1, 1, 0, ...]             # 二分类标签
    },
    "test": { ... }
}
"""

import sys
import pickle
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from offline.config import config
from offline.evaluation.itemcf import build_itemcf_index, recommend_itemcf


def load_raw_data():
    """加载原始 MovieLens 数据文件"""
    print("加载原始数据...")
    try:
        df_movies = pd.read_pickle(config.DATASET_DIR / "movies.pkl")
        df_ratings = pd.read_pickle(config.DATASET_DIR / "ratings.pkl")
        df_ratings = df_ratings.copy()
        df_ratings["timestamp"] = pd.to_numeric(
            df_ratings["timestamp"],
            errors="raise",
        ).astype(np.int64)
        df_ratings["_source_row"] = np.arange(len(df_ratings), dtype=np.int64)
        df_users = pd.read_pickle(config.DATASET_DIR / "users.pkl")
        return df_movies, df_ratings, df_users
    except FileNotFoundError:
        print(f"数据文件不存在: {config.DATASET_DIR}")
        sys.exit(1)


def process_features_for_ranking(df_movies, df_ratings, df_users):
    """
    为精排模型处理特征
    
    Returns:
        df_merged: 包含所有特征和标签的 DataFrame
        user_vocab: 用户特征词表
        movie_vocab: 电影特征词表
    """
    print("处理特征...")
    
    # 选择列
    user_columns = ["user_id", "gender", "age", "occupation", "zip_code"]
    movie_columns = ["movie_id", "genres", "isAdult", "startYear"]
    ratings_columns = [
        "user_id",
        "movie_id",
        "rating",
        "timestamp",
        "_source_row",
    ]
    
    df_users = df_users[user_columns].copy()
    df_movies = df_movies[["movie_id", "genres", "isAdult", "startYear"]].copy()
    
    # 处理类型 - 为简化精排取第一个类型
    # (DeepFM 期望标量特征，而非序列)
    df_movies['genres'] = df_movies['genres'].str.split("|").str[0]
    df_movies['isAdult'] = df_movies['isAdult'].fillna(False)
    df_movies['startYear'] = df_movies['startYear'].fillna(0)
    
    df_ratings = df_ratings[ratings_columns].copy()
    
    # 编码用户特征
    print("编码用户特征...")
    user_vocab = {}
    user_sparse_feature_columns = ["user_id", "gender", "age", "occupation", "zip_code"]
    
    for feat_name in user_sparse_feature_columns:
        label_encoder = LabelEncoder()
        df_users[feat_name + "_encoded"] = label_encoder.fit_transform(df_users[feat_name]) + 1  # 0 用于填充/未知
        user_vocab[feat_name] = label_encoder.classes_
    
    # 编码电影特征
    print("编码电影特征...")
    movie_vocab = {}
    movie_sparse_feature_columns = ["movie_id", "genres", "isAdult", "startYear"]
    
    for feat_name in movie_sparse_feature_columns:
        label_encoder = LabelEncoder()
        # 处理潜在的 NaN 值
        df_movies[feat_name] = df_movies[feat_name].fillna("unknown" if df_movies[feat_name].dtype == object else 0)
        df_movies[feat_name + "_encoded"] = label_encoder.fit_transform(df_movies[feat_name].astype(str)) + 1
        movie_vocab[feat_name] = label_encoder.classes_
    
    # 合并所有特征
    print("合并特征...")
    df_merged = df_ratings.merge(
        df_users[["user_id", "user_id_encoded", "gender_encoded", "age_encoded", 
                  "occupation_encoded", "zip_code_encoded"]],
        on="user_id", 
        how="left"
    )
    df_merged = df_merged.merge(
        df_movies[["movie_id", "movie_id_encoded", "genres_encoded", 
                   "isAdult_encoded", "startYear_encoded"]],
        on="movie_id",
        how="left"
    )
    
    # 重命名编码列为最终名称
    df_merged = df_merged.rename(columns={
        "user_id_encoded": "user_id_enc",
        "gender_encoded": "gender",
        "age_encoded": "age",
        "occupation_encoded": "occupation",
        "zip_code_encoded": "zip_code",
        "movie_id_encoded": "movie_id_enc",
        "genres_encoded": "genres",
        "isAdult_encoded": "isAdult",
        "startYear_encoded": "startYear",
    })
    
    # 保留原始 ID 用于负采样，编码后的 ID 用于模型
    df_merged["user_id_original"] = df_merged["user_id"]
    df_merged["movie_id_original"] = df_merged["movie_id"]
    df_merged["user_id"] = df_merged["user_id_enc"]
    df_merged["movie_id"] = df_merged["movie_id_enc"]
    
    return df_merged, user_vocab, movie_vocab

def split_interactions_by_time(
    df_interactions,
    validation_ratio=0.1,
    test_ratio=0.2,
):
    """Split every user's interactions chronologically into three sets."""
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio 必须在 0 和 1 之间")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio 必须在 0 和 1 之间")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio + test_ratio 必须小于 1")

    ordered = df_interactions.sort_values(
        ["user_id_original", "timestamp", "_source_row"],
        kind="stable",
    ).copy()
    group_sizes = ordered.groupby("user_id_original")[
        "user_id_original"
    ].transform("size")
    short_user_count = ordered.loc[
        group_sizes < 3,
        "user_id_original",
    ].nunique()
    if short_user_count:
        raise ValueError(
            f"有 {short_user_count} 个用户少于 3 条交互，无法三段切分"
        )

    validation_sizes = np.maximum(
        1,
        np.floor(group_sizes * validation_ratio).astype(int),
    )
    test_sizes = np.maximum(
        1,
        np.floor(group_sizes * test_ratio).astype(int),
    )
    train_sizes = group_sizes - validation_sizes - test_sizes
    if (train_sizes < 1).any():
        raise ValueError("切分比例使部分用户没有训练交互")

    positions = ordered.groupby("user_id_original").cumcount()
    train_mask = positions < train_sizes
    validation_mask = (
        (positions >= train_sizes)
        & (positions < train_sizes + validation_sizes)
    )
    test_mask = positions >= train_sizes + validation_sizes

    return (
        ordered.loc[train_mask].copy(),
        ordered.loc[validation_mask].copy(),
        ordered.loc[test_mask].copy(),
    )


def split_train_validation_with_final_embargo(df_interactions):
    """Return Train/Validation only, before materializing final interactions."""
    ordered = df_interactions.sort_values(
        ["user_id_original", "timestamp", "_source_row"],
        kind="stable",
    ).copy()
    train_parts, validation_parts = [], []
    for _, user_rows in ordered.groupby("user_id_original", sort=False):
        if len(user_rows) < 3:
            continue
        selection_rows = user_rows.iloc[:-1]
        train_parts.append(selection_rows.iloc[:-1])
        validation_parts.append(selection_rows.iloc[[-1]])
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(validation_parts, ignore_index=True),
    )

def assign_labels_from_training_history(
    train_interactions,
    validation_interactions,
    test_interactions,
):
    """Create labels for every split using training-period means only."""
    splits = [
        train_interactions.copy(),
        validation_interactions.copy(),
        test_interactions.copy(),
    ]
    train_user_averages = splits[0].groupby("user_id_original")[
        "rating"
    ].mean()

    for name, interactions in zip(
        ("train", "validation", "test"),
        splits,
    ):
        interactions["user_avg_rating"] = interactions[
            "user_id_original"
        ].map(train_user_averages)
        if interactions["user_avg_rating"].isna().any():
            missing_users = interactions.loc[
                interactions["user_avg_rating"].isna(),
                "user_id_original",
            ].nunique()
            raise ValueError(
                f"{name} 数据中有 {missing_users} 个用户缺少训练期评分均值"
            )
        interactions["conversion"] = (
            interactions["rating"] >= interactions["user_avg_rating"]
        ).astype(int)
        interactions["click"] = (
            interactions["rating"]
            >= interactions["user_avg_rating"] - 1
        ).astype(int)
        interactions["exposure"] = 1
        interactions["is_click"] = interactions["click"]

    return tuple(splits)


def assign_selection_labels(train_interactions, validation_interactions):
    """Label Train/Validation interactions using Train-period averages only."""
    train_average = train_interactions.groupby("user_id_original")[
        "rating"
    ].mean()
    outputs = []
    for name, interactions in (
        ("train", train_interactions),
        ("validation", validation_interactions),
    ):
        frame = interactions.copy()
        frame["user_avg_rating"] = frame["user_id_original"].map(train_average)
        if frame["user_avg_rating"].isna().any():
            raise ValueError(f"{name} contains users without Train labels")
        frame["conversion"] = (frame["rating"] >= frame["user_avg_rating"]).astype(int)
        frame["click"] = (frame["rating"] >= frame["user_avg_rating"] - 1).astype(int)
        frame["exposure"] = 1
        frame["is_click"] = frame["click"]
        outputs.append(frame)
    return tuple(outputs)

def generate_negative_samples(
    df_merged,
    movie_vocab,
    all_interactions=None,
    interaction_history=None,
    excluded_pairs=None,
    neg_ratio_from_exposure=1,
    neg_ratio_random=2,
    include_all_hard_negatives=False,
    random_seed=42,
):
    """
    为精排生成负样本
    
    困难负样本按用户采样：对于每个用户的正样本，
    从该用户曝光但未点击的物品中采样。
    这使它们成为"困难"样本，因为用户实际看到了物品但选择不参与。
    
    随机负样本是用户从未交互过的物品。
    
    Args:
        df_merged: 包含正样本和曝光样本的 DataFrame
        movie_vocab: 电影特征词表
        neg_ratio_from_exposure: 每个正样本的困难负样本数量
        neg_ratio_random: 每个正样本的随机负样本数量
    
    Returns:
        添加了负样本的 DataFrame
    """
    print("生成负样本...")

    if all_interactions is None:
        all_interactions = df_merged
    if interaction_history is None:
        interaction_history = df_merged

    rng = random.Random(random_seed)
    excluded_by_user = defaultdict(set)
    if excluded_pairs:
        for user_id, movie_id in excluded_pairs:
            excluded_by_user[user_id].add(movie_id)
    
    # 获取用于负采样的电影特征
    movie_features = all_interactions[
        ["movie_id", "genres", "isAdult", "startYear", "movie_id_original"]
    ].drop_duplicates(subset="movie_id")
    movie_features_dict = movie_features.set_index("movie_id").to_dict("index")
    all_movie_ids = set(movie_features_dict)
    
    # 分离正样本和困难负样本（曝光但未点击）
    positive_samples = df_merged[df_merged["is_click"] == 1].copy()
    positive_samples["_sample_type"] = "positive"

    hard_negative_pool = df_merged[df_merged["is_click"] == 0].copy()
    
    print(f"  正样本: {len(positive_samples)}")
    print(f"  困难负样本池 (曝光但未点击): {len(hard_negative_pool)}")
    
    # 构建每个用户的困难负样本池
    # Key: user_id_original, Value: 该用户的困难负样本 DataFrame
    user_hard_negatives = {}
    for user_id, group in hard_negative_pool.groupby("user_id_original"):
        user_hard_negatives[user_id] = group
    
    # 构建每个用户的交互历史（用于排除随机负样本）
    user_interactions = (
        interaction_history.groupby("user_id_original")["movie_id"]
        .apply(set)
        .to_dict()
    )
    
    # 统计同时有正样本和困难负样本的用户数
    users_with_hard_neg = set(user_hard_negatives.keys())
    positive_users = set(positive_samples["user_id_original"].unique())
    users_with_both = users_with_hard_neg & positive_users
    print(f"  同时有正样本和困难负样本的用户数: {len(users_with_both)}")
    
    negative_samples = []
    
    # 1. 构造困难负样本（曝光但未点击）
    # 训练集按比例采样以控制类别分布；验证和测试保留全部真实
    # 负反馈，避免只评估在该窗口中至少有一个正样本的用户。
    if include_all_hard_negatives:
        if len(hard_negative_pool) > 0:
            hard_negative_df = hard_negative_pool.copy()
            hard_negative_df["is_click"] = 0
            hard_negative_df["_sample_type"] = "hard_negative"
            negative_samples.append(hard_negative_df)
            print(
                "    保留了 "
                f"{len(hard_negative_df)} 个全部真实困难负样本"
            )
    elif neg_ratio_from_exposure > 0:
        print(f"  为每个正样本采样 {neg_ratio_from_exposure} 个困难负样本...")
        hard_neg_list = []
        hard_neg_count = 0

        for user_id, user_positives in tqdm(
            positive_samples.groupby("user_id_original"),
            desc="困难负样本 (每个用户)",
        ):
            user_hard_neg_pool = user_hard_negatives.get(user_id)

            if user_hard_neg_pool is None or len(user_hard_neg_pool) == 0:
                continue

            n_positives = len(user_positives)
            n_hard_neg_needed = n_positives * neg_ratio_from_exposure
            n_to_sample = min(len(user_hard_neg_pool), n_hard_neg_needed)

            if n_to_sample > 0:
                sampled = user_hard_neg_pool.sample(
                    n=n_to_sample,
                    replace=False,
                    random_state=rng.randrange(2**32),
                )
                hard_neg_list.append(sampled)
                hard_neg_count += len(sampled)

        if hard_neg_list:
            hard_neg_df = pd.concat(hard_neg_list, ignore_index=True)
            hard_neg_df["is_click"] = 0
            hard_neg_df["_sample_type"] = "hard_negative"
            negative_samples.append(hard_neg_df)
            print(f"    添加了 {hard_neg_count} 个困难负样本")

    # 2. 生成随机负样本（用户从未交互过的物品）
    if neg_ratio_random > 0:
        print(f"  为每个正样本采样 {neg_ratio_random} 个随机负样本...")
        random_neg_list = []
        random_shortfall = 0

        grouped_positives = positive_samples.groupby("user_id_original")
        for user_id_orig, user_positives in tqdm(
            grouped_positives,
            total=grouped_positives.ngroups,
            desc="随机负样本 (每个用户)",
        ):
            user_interacted = user_interactions.get(user_id_orig, set())
            user_excluded = excluded_by_user.get(user_id_orig, set())

            available_movies = sorted(
                all_movie_ids - user_interacted - user_excluded
            )

            n_needed = len(user_positives) * neg_ratio_random
            n_to_sample = min(n_needed, len(available_movies))
            random_shortfall += n_needed - n_to_sample

            if n_to_sample == 0:
                continue

            neg_movie_ids = rng.sample(available_movies, n_to_sample)
            anchors = user_positives.reset_index(drop=True)

            for offset, neg_movie_id in enumerate(neg_movie_ids):
                movie_feats = movie_features_dict[neg_movie_id]
                row = anchors.iloc[offset % len(anchors)]

                random_neg_list.append({
                    "user_id": row["user_id"],
                    "user_id_original": user_id_orig,
                    "gender": row["gender"],
                    "age": row["age"],
                    "occupation": row["occupation"],
                    "zip_code": row["zip_code"],
                    "movie_id": neg_movie_id,
                    "movie_id_original": movie_feats.get(
                        "movie_id_original",
                        neg_movie_id,
                    ),
                    "genres": movie_feats.get("genres", 0),
                    "isAdult": movie_feats.get("isAdult", 0),
                    "startYear": movie_feats.get("startYear", 0),
                    "is_click": 0,
                    "rating": 0,
                    "timestamp": row["timestamp"],
                    "_sample_type": "random_negative",
                })

        if random_shortfall:
            print(
                f"    因候选不足少生成了 {random_shortfall} 个随机负样本"
            )
        
        if random_neg_list:
            random_neg_df = pd.DataFrame(random_neg_list)
            negative_samples.append(random_neg_df)
            print(f"    添加了 {len(random_neg_df)} 随机负样本")
    
    # 合并正样本和负样本
    output_cols = [
        "user_id",
        "gender",
        "age",
        "occupation",
        "zip_code",
        "movie_id",
        "genres",
        "isAdult",
        "startYear",
        "is_click",
        "timestamp",
        "user_id_original",
        "_sample_type",
    ]
    all_samples = [positive_samples[output_cols]]
    
    for neg_df in negative_samples:
        # 确保所有必需列存在
        for col in output_cols:
            if col not in neg_df.columns:
                neg_df[col] = 0
        all_samples.append(neg_df[output_cols])
    
    df_final = pd.concat(all_samples, ignore_index=True)
    
    # 确保所有列为整数类型
    feature_cols = ["user_id", "gender", "age", "occupation", "zip_code",
                    "movie_id", "genres", "isAdult", "startYear", "is_click"]
    for col in feature_cols:
        df_final[col] = df_final[col].fillna(0).astype(int)
    
    print(f"  最终数据集: {len(df_final)} 样本")
    print(f"    正样本: {(df_final['is_click'] == 1).sum()}")
    print(f"    负样本 (总数): {(df_final['is_click'] == 0).sum()}")
    
    return df_final

def convert_to_dict(df, feature_columns, label_column="is_click"):
    """将 DataFrame 转换为用于模型训练的字典格式"""
    result = {}
    for col in feature_columns:
        result[col] = df[col].values.astype(np.int32)
    result[label_column] = df[label_column].values.astype(np.int32)
    
    # 保留 user_id_original 用于 gAUC 评估
    if "user_id_original" in df.columns:
        result["user_id_original"] = df["user_id_original"].values
    
    return result


def generate_candidate_aware_negatives(train_df, train_interactions):
    """Add one Train-only ItemCF hard negative for each positive row."""
    neighbors = build_itemcf_index(train_interactions, max_user_items=100)
    feature_columns = ["genres", "isAdult", "startYear", "movie_id_original"]
    movie_features = train_interactions[
        ["movie_id", *feature_columns]
    ].drop_duplicates("movie_id").set_index("movie_id")
    histories = {
        user_id: set(items)
        for user_id, items in train_interactions.groupby("user_id_original")["movie_id"]
    }
    # ItemCF's ranking depends only on the positive item, not on the row.  Cache
    # it once per movie; user-specific filtering happens below.
    ranked_candidates = {
        int(movie_id): recommend_itemcf(neighbors, [movie_id], 50)
        for movie_id in train_df.loc[train_df["is_click"] == 1, "movie_id"].unique()
    }
    positive_rows = train_df.loc[train_df["is_click"] == 1].copy()
    candidate_ids = []
    for user_id, movie_id in zip(
        positive_rows["user_id_original"], positive_rows["movie_id"]
    ):
        seen = histories.get(user_id, set())
        candidate_ids.append(next(
            (candidate for candidate in ranked_candidates[int(movie_id)] if candidate not in seen),
            None,
        ))
    candidate_rows = positive_rows.assign(_candidate_movie_id=candidate_ids)
    candidate_rows = candidate_rows.dropna(subset=["_candidate_movie_id"])
    if candidate_rows.empty:
        return train_df
    candidate_rows["_candidate_movie_id"] = candidate_rows["_candidate_movie_id"].astype(int)
    candidate_rows["movie_id"] = candidate_rows["_candidate_movie_id"]
    for column in feature_columns:
        candidate_rows[column] = candidate_rows["movie_id"].map(movie_features[column])
    candidate_rows = candidate_rows.drop(columns="_candidate_movie_id")
    candidate_rows["is_click"] = 0
    candidate_rows["_sample_type"] = "candidate_aware_negative"
    return pd.concat([train_df, candidate_rows], ignore_index=True)

def run_ranking_preprocessing(
    neg_ratio_from_exposure=1,
    neg_ratio_random=2,
    validation_ratio=0.1,
    test_ratio=0.2,
    selection_only=False,
    candidate_aware_negatives=False,
):
    """Run strict chronological ranking preprocessing."""
    print("=" * 60)
    print("精排模型数据预处理")
    print("=" * 60)

    df_movies, df_ratings, df_users = load_raw_data()
    df_merged, user_vocab, movie_vocab = process_features_for_ranking(
        df_movies,
        df_ratings,
        df_users,
    )

    if selection_only:
        train_interactions, validation_interactions = (
            split_train_validation_with_final_embargo(df_merged)
        )
        train_interactions, validation_interactions = (
            assign_selection_labels(train_interactions, validation_interactions)
        )
        test_interactions = None
    else:
        train_interactions, validation_interactions, test_interactions = (
        split_interactions_by_time(
            df_merged,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )
        )
        train_interactions, validation_interactions, test_interactions = (
        assign_labels_from_training_history(
            train_interactions,
            validation_interactions,
            test_interactions,
        )
        )

    selection_source = train_interactions if selection_only else df_merged
    train_df = generate_negative_samples(
        train_interactions,
        movie_vocab,
        all_interactions=selection_source,
        interaction_history=selection_source,
        neg_ratio_from_exposure=neg_ratio_from_exposure,
        neg_ratio_random=neg_ratio_random,
        random_seed=42,
    )
    if candidate_aware_negatives:
        if not selection_only:
            raise ValueError(
                "candidate_aware_negatives requires selection_only=True"
            )
        train_df = generate_candidate_aware_negatives(
            train_df, train_interactions
        )

    def random_pairs(frame):
        random_rows = frame[frame["_sample_type"] == "random_negative"]
        return set(zip(random_rows["user_id_original"], random_rows["movie_id"]))

    train_random_pairs = random_pairs(train_df)
    validation_df = generate_negative_samples(
        validation_interactions,
        movie_vocab,
        all_interactions=selection_source,
        interaction_history=selection_source,
        excluded_pairs=train_random_pairs,
        neg_ratio_from_exposure=neg_ratio_from_exposure,
        neg_ratio_random=neg_ratio_random,
        include_all_hard_negatives=True,
        random_seed=43,
    )

    if not selection_only:
        validation_random_pairs = random_pairs(validation_df)
        test_df = generate_negative_samples(
        test_interactions,
        movie_vocab,
        all_interactions=df_merged,
        interaction_history=df_merged,
        excluded_pairs=train_random_pairs | validation_random_pairs,
        neg_ratio_from_exposure=neg_ratio_from_exposure,
        neg_ratio_random=neg_ratio_random,
        include_all_hard_negatives=True,
            random_seed=44,
        )

    feature_columns = [
        "user_id", "gender", "age", "occupation", "zip_code",
        "movie_id", "genres", "isAdult", "startYear",
    ]
    train_data = convert_to_dict(train_df, feature_columns, "is_click")
    validation_data = convert_to_dict(
        validation_df,
        feature_columns,
        "is_click",
    )
    samples = {
        "train": train_data,
        "validation": validation_data,
    }
    if not selection_only:
        samples["test"] = convert_to_dict(test_df, feature_columns, "is_click")

    vocab_dict = {**user_vocab, **movie_vocab}
    feature_dict = {key: len(values) + 1 for key, values in vocab_dict.items()}

    print("保存处理后的数据...")
    pickle.dump(samples, open(config.RANKING_TRAIN_DATA_PATH, "wb"))
    pickle.dump(feature_dict, open(config.RANKING_FEATURE_DICT_PATH, "wb"))
    pickle.dump(vocab_dict, open(config.RANKING_VOCAB_DICT_PATH, "wb"))

    print()
    print("数据摘要:")
    print(f"  特征: {feature_columns}")
    print(f"  特征词汇表大小: {feature_dict}")
    for split_name, split_data in samples.items():
        print(f"  {split_name} 样本: {len(split_data['user_id'])}")
        print(
            f"  {split_name} 正样本比例: "
            f"{split_data['is_click'].mean():.2%}"
        )

    return samples, feature_dict

if __name__ == "__main__":
    run_ranking_preprocessing()
