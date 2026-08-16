from offline.evaluation.evaluate_candidate_pool_validation import summarize


def test_summary_reports_candidate_and_tail_metrics():
    value = summarize(
        [1, 2], [[1] * 10, [3] * 10], [[3] * 10, [2] * 10], [True, False]
    )
    assert value["baseline"]["recall@10"] == 0.5
    assert value["candidate"]["recall@10"] == 0.5
    assert value["baseline_tail_pq0_recall_at_10"] == 1.0
    assert value["candidate_tail_pq0_recall_at_10"] == 0.0


def test_summary_pads_variable_candidate_pool_lengths_with_reserved_zero():
    value = summarize(
        [4, 5], [[4] * 10, [5] * 10], [[4] * 10, [5] * 12], [True, True]
    )
    assert value["candidate"]["recall@10"] == 1.0
    assert value["candidate_tail_pq0_recall_at_10"] == 1.0
