import numpy as np

from offline.evaluation.itemcf import build_itemcf_index, recommend_itemcf


def test_itemcf_uses_train_only_cooccurrence_and_filters_history():
    train = {
        "user_id": np.array([1, 1, 1, 2, 2, 3]),
        "movie_id": np.array([1, 2, 3, 1, 3, 4]),
    }
    neighbors = build_itemcf_index(train)

    assert 4 not in neighbors.get(1, [])
    assert recommend_itemcf(neighbors, [1, 2, 0], k=3) == [3]


def test_itemcf_rejects_non_positive_k():
    neighbors = build_itemcf_index({
        "user_id": np.array([1, 1]), "movie_id": np.array([1, 2])
    })
    try:
        recommend_itemcf(neighbors, [1], k=0)
    except ValueError as error:
        assert "greater than zero" in str(error)
    else:
        raise AssertionError("ValueError was not raised")
