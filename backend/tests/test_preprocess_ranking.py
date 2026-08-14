import numpy as np
import pandas as pd
import pytest

from offline.feature.preprocess_ranking import (
    assign_labels_from_training_history,
    generate_negative_samples,
    split_interactions_by_time,
)


def test_split_interactions_by_time_is_per_user_stable_and_disjoint():
    interactions = pd.DataFrame(
        {
            "user_id_original": [
                "1", "1", "1", "1", "1",
                "2", "2", "2", "2",
            ],
            "timestamp": [20, 10, 20, 20, 30, 5, 6, 7, 8],
            "_source_row": list(range(9)),
        }
    )

    train, validation, test = split_interactions_by_time(
        interactions,
        validation_ratio=0.2,
        test_ratio=0.2,
    )

    assert train.loc[train["user_id_original"] == "1", "_source_row"].tolist() == [1, 0, 2]
    assert validation.loc[validation["user_id_original"] == "1", "_source_row"].tolist() == [3]
    assert test.loc[test["user_id_original"] == "1", "_source_row"].tolist() == [4]
    assert set(train["_source_row"]).isdisjoint(validation["_source_row"])
    assert set(train["_source_row"]).isdisjoint(test["_source_row"])
    assert set(validation["_source_row"]).isdisjoint(test["_source_row"])
    assert len(train) + len(validation) + len(test) == len(interactions)

    train_max = train.groupby("user_id_original")["timestamp"].max()
    validation_min = validation.groupby("user_id_original")["timestamp"].min()
    validation_max = validation.groupby("user_id_original")["timestamp"].max()
    test_min = test.groupby("user_id_original")["timestamp"].min()
    assert (train_max <= validation_min).all()
    assert (validation_max <= test_min).all()


def test_split_rejects_users_without_three_interactions():
    interactions = pd.DataFrame(
        {
            "user_id_original": ["1", "1"],
            "timestamp": [1, 2],
            "_source_row": [0, 1],
        }
    )
    with pytest.raises(ValueError, match="少于 3 条交互"):
        split_interactions_by_time(interactions)

def test_labels_use_training_history_only():
    train = pd.DataFrame(
        {
            "user_id_original": ["1", "1", "2", "2"],
            "rating": [4.0, 8.0, 2.0, 10.0],
        }
    )
    validation = pd.DataFrame(
        {
            "user_id_original": ["1", "2"],
            "rating": [5.0, 5.0],
        }
    )
    test = pd.DataFrame(
        {
            "user_id_original": ["1", "2"],
            "rating": [9.0, 1.0],
        }
    )

    labeled_train, labeled_validation, labeled_test = (
        assign_labels_from_training_history(train, validation, test)
    )

    assert labeled_train["user_avg_rating"].tolist() == [
        6.0,
        6.0,
        6.0,
        6.0,
    ]
    assert labeled_train["is_click"].tolist() == [0, 1, 0, 1]
    assert labeled_train["conversion"].tolist() == [0, 1, 0, 1]

    assert labeled_validation["user_avg_rating"].tolist() == [6.0, 6.0]
    assert labeled_validation["is_click"].tolist() == [1, 1]
    assert labeled_validation["conversion"].tolist() == [0, 0]
    assert labeled_test["user_avg_rating"].tolist() == [6.0, 6.0]
    assert labeled_test["is_click"].tolist() == [1, 0]
    assert labeled_test["conversion"].tolist() == [1, 0]


def test_random_negatives_are_deterministic_unique_and_excludable():
    current_interactions = pd.DataFrame(
        [
            {
                "user_id": 1,
                "user_id_original": "1",
                "gender": 1,
                "age": 1,
                "occupation": 1,
                "zip_code": 1,
                "movie_id": 1,
                "movie_id_original": "1",
                "genres": 1,
                "isAdult": 1,
                "startYear": 1,
                "rating": 8.0,
                "timestamp": 1,
                "is_click": 1,
            },
            {
                "user_id": 1,
                "user_id_original": "1",
                "gender": 1,
                "age": 1,
                "occupation": 1,
                "zip_code": 1,
                "movie_id": 2,
                "movie_id_original": "2",
                "genres": 1,
                "isAdult": 1,
                "startYear": 1,
                "rating": 2.0,
                "timestamp": 2,
                "is_click": 0,
            },
        ]
    )

    catalog_rows = []
    for movie_id in range(3, 9):
        catalog_rows.append(
            {
                "user_id": 2,
                "user_id_original": "2",
                "gender": 1,
                "age": 1,
                "occupation": 1,
                "zip_code": 1,
                "movie_id": movie_id,
                "movie_id_original": str(movie_id),
                "genres": 1,
                "isAdult": 1,
                "startYear": 1,
                "rating": 8.0,
                "timestamp": movie_id,
                "is_click": 1,
            }
        )

    all_interactions = pd.concat(
        [current_interactions, pd.DataFrame(catalog_rows)],
        ignore_index=True,
    )
    movie_vocab = {"movie_id": np.arange(1, 9)}

    first = generate_negative_samples(
        current_interactions,
        movie_vocab,
        all_interactions=all_interactions,
        random_seed=42,
    )
    repeated = generate_negative_samples(
        current_interactions,
        movie_vocab,
        all_interactions=all_interactions,
        random_seed=42,
    )

    assert first.equals(repeated)
    assert first["_sample_type"].value_counts().to_dict() == {
        "random_negative": 2,
        "positive": 1,
        "hard_negative": 1,
    }

    first_random = first[
        first["_sample_type"] == "random_negative"
    ]
    first_pairs = set(
        zip(
            first_random["user_id_original"],
            first_random["movie_id"],
        )
    )

    assert len(first_pairs) == len(first_random)
    assert all(movie_id not in {1, 2} for _, movie_id in first_pairs)

    second = generate_negative_samples(
        current_interactions,
        movie_vocab,
        all_interactions=all_interactions,
        excluded_pairs=first_pairs,
        random_seed=43,
    )
    second_random = second[
        second["_sample_type"] == "random_negative"
    ]
    second_pairs = set(
        zip(
            second_random["user_id_original"],
            second_random["movie_id"],
        )
    )

    assert first_pairs.isdisjoint(second_pairs)
