from offline.evaluation.candidate_pool import append_unique_candidates


def test_append_unique_keeps_two_tower_order_then_itemcf_only_items():
    assert append_unique_candidates([4, 2, 2, 0], [2, 5, 4, 6], 5) == [4, 2, 5, 6]


def test_append_unique_respects_pool_cap():
    assert append_unique_candidates([1, 2], [3, 4], 3) == [1, 2, 3]
