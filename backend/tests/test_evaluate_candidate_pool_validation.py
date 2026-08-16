from offline.evaluation.evaluate_candidate_pool_validation import summarize


def test_summary_reports_candidate_and_tail_metrics():
    value = summarize(
        [1, 2], [[1] * 10, [3] * 10], [[3] * 10, [2] * 10], [True, False]
    )
    assert value["baseline"]["recall@10"] == 0.5
    assert value["candidate"]["recall@10"] == 0.5
    assert value["baseline_tail_pq0_recall_at_10"] == 1.0
    assert value["candidate_tail_pq0_recall_at_10"] == 0.0
