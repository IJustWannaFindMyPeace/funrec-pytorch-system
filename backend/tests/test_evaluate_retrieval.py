import pytest

from offline.evaluation.evaluate_retrieval import (
    calculate_comparison,
    run_evaluation,
)


def test_calculate_comparison_includes_lift_and_hit_counts():
    model_metrics = {
        "hit_rate@10": 0.10,
        "ndcg@10": 0.05,
    }
    baseline_metrics = {
        "hit_rate@10": 0.02,
        "ndcg@10": 0.01,
    }

    comparison = calculate_comparison(
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
        test_users=1000,
    )

    assert comparison["hit_rate@10"] == {
        "youtube_dnn": 0.10,
        "popularity": 0.02,
        "absolute_improvement": pytest.approx(0.08),
        "relative_lift": pytest.approx(5.0),
        "youtube_dnn_hits": 100,
        "popularity_hits": 20,
    }
    assert comparison["ndcg@10"]["relative_lift"] == pytest.approx(
        5.0
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"batch_size": 0},
            "batch_size must be greater than zero",
        ),
        (
            {"k_values": ()},
            "k_values must not be empty",
        ),
        (
            {"k_values": (0, 10)},
            "k values must be greater than zero",
        ),
    ],
)
def test_run_evaluation_rejects_invalid_configuration(
    arguments,
    message,
):
    with pytest.raises(ValueError, match=message):
        run_evaluation(**arguments)