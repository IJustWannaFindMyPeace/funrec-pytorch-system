import numpy as np
import pandas as pd

from offline.feature.preprocess_ranking import (
    assign_labels_from_training_history,
    generate_negative_samples,
    split_interactions_by_time,
)


def test_split_interactions_by_time_is_per_user_and_stable():
    interactions = pd.DataFrame(
        {
            "user_id_original": ["1", "1", "1", "1", "1", "2", "2"],
            "timestamp": [20, 10, 20, 20, 30, 5, 6],
            "_source_row": [0, 1, 2, 3, 4, 5, 6],
        }
    )

    train, test = split_interactions_by_time(
        interactions,
        test_ratio=0.4,
    )

    user_1_train = train.loc[
        train["user_id_original"] == "1",
        "_source_row",
    ].tolist()
    user_1_test = test.loc[
        test["user_id_original"] == "1",
        "_source_row",
    ].tolist()

    assert user_1_train == [1, 0, 2]
    assert user_1_test == [3, 4]

    assert set(train["user_id_original"]) == {"1", "2"}
    assert set(test["user_id_original"]) == {"1", "2"}
    assert set(train["_source_row"]).isdisjoint(test["_source_row"])
    assert len(train) + len(test) == len(interactions)

    train_max = train.groupby("user_id_original")["timestamp"].max()
    test_min = test.groupby("user_id_original")["timestamp"].min()
    assert (train_max <= test_min).all()


def test_labels_use_training_history_only():
    train = pd.DataFrame(
        {
            "user_id_original": ["1", "1", "2", "2"],
            "rating": [4.0, 8.0, 2.0, 10.0],
        }
    )
    test = pd.DataFrame(
        {
            "user_id_original": ["1", "2"],
            "rating": [5.0, 5.0],
        }
    )

    labeled_train, labeled_test = assign_labels_from_training_history(
        train,
        test,
    )

    assert labeled_train["user_avg_rating"].tolist() == [
        6.0,
        6.0,
        6.0,
        6.0,
    ]
    assert labeled_train["is_click"].tolist() == [0, 1, 0, 1]
    assert labeled_train["conversion"].tolist() == [0, 1, 0, 1]

    assert labeled_test["user_avg_rating"].tolist() == [6.0, 6.0]
    assert labeled_test["is_click"].tolist() == [1, 1]
    assert labeled_test["conversion"].tolist() == [0, 0]


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