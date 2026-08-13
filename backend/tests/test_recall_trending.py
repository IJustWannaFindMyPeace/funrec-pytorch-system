import asyncio

import online.recall.trending as module


class UnexpectedElasticsearch:
    def is_available(self):
        raise AssertionError(
            "Elasticsearch should not be checked "
            "without preference genres"
        )


def test_user_preference_skips_elasticsearch_without_genres(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "es_service",
        UnexpectedElasticsearch(),
    )

    strategy = module.UserPreferenceRecallStrategy()

    results = asyncio.run(
        strategy.recall(
            {
                "user_id": 1,
                "hist_movie_ids": [10, 20, 30],
            },
            k=10,
        )
    )

    assert results == []