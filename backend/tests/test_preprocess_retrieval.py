import pandas as pd

from offline.feature.preprocess_retrieval import (
    generate_train_eval_samples,
)


def test_retrieval_samples_follow_numeric_time_and_source_order():
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1],
            "movie_id": [20, 10, 21, 22],
            "timestamp": [20, 10, 20, 20],
            "_source_row": [0, 1, 2, 3],
        }
    )

    samples = generate_train_eval_samples(
        interactions,
        user_columns=[],
        item_columns=["movie_id"],
        max_hist_seq_len=3,
        max_feat_seq_len=1,
    )

    train = samples["train"]
    test = samples["test"]

    assert train["movie_id"].tolist() == [20, 21]
    assert train["hist_movie_id"].tolist() == [
        [0, 0, 10],
        [0, 10, 20],
    ]

    assert test["movie_id"].tolist() == [22]
    assert test["hist_movie_id"].tolist() == [
        [10, 20, 21],
    ]