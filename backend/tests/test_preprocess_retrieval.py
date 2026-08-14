import pandas as pd

from offline.feature.preprocess_retrieval import (
    generate_train_eval_samples,
)


def test_retrieval_samples_follow_strict_chronological_split():
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1],
            "movie_id": [20, 10, 21, 22, 23],
            "timestamp": [20, 10, 20, 20, 30],
            "_source_row": [0, 1, 2, 3, 4],
        }
    )

    samples = generate_train_eval_samples(
        interactions,
        user_columns=[],
        item_columns=["movie_id"],
        max_hist_seq_len=3,
        max_feat_seq_len=1,
    )

    assert set(samples) == {"train", "validation", "test"}
    assert samples["train"]["movie_id"].tolist() == [20, 21]
    assert samples["train"]["hist_movie_id"].tolist() == [
        [0, 0, 10],
        [0, 10, 20],
    ]
    assert samples["validation"]["movie_id"].tolist() == [22]
    assert samples["validation"]["hist_movie_id"].tolist() == [
        [10, 20, 21],
    ]
    assert samples["test"]["movie_id"].tolist() == [23]
    assert samples["test"]["hist_movie_id"].tolist() == [
        [20, 21, 22],
    ]
