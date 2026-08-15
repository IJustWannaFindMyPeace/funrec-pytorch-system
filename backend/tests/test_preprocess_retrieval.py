import pandas as pd
import pytest

from offline.feature.preprocess_retrieval import (
    add_padding,
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


def test_padding_keeps_most_recent_values_and_left_pads():
    assert add_padding([1, 2, 3, 4], 0, 3) == [2, 3, 4]
    assert add_padding([1, 2], 0, 4) == [0, 0, 1, 2]


def test_retrieval_samples_reject_invalid_sequence_length():
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "movie_id": [10, 20, 30],
            "timestamp": [1, 2, 3],
            "_source_row": [0, 1, 2],
        }
    )
    with pytest.raises(ValueError, match="max_hist_seq_len"):
        generate_train_eval_samples(
            interactions,
            user_columns=[],
            item_columns=["movie_id"],
            max_hist_seq_len=0,
            max_feat_seq_len=1,
        )
