import torch
import pytest

from offline.training.ranking_data import (
    RANKING_FEATURE_NAMES,
    RankingDataset,
    build_ranking_dataloader,
)


def make_samples() -> dict[str, list[int]]:
    samples = {
        name: [1, 2, 3, 1]
        for name in RANKING_FEATURE_NAMES
    }
    samples["is_click"] = [1, 0, 1, 0]
    samples["user_id_original"] = [10, 20, 30, 40]
    return samples


def test_ranking_dataset_returns_features_and_float_label() -> None:
    dataset = RankingDataset(make_samples())

    features, label = dataset[1]

    assert len(dataset) == 4
    assert set(features) == set(RANKING_FEATURE_NAMES)
    assert all(value.ndim == 0 for value in features.values())
    assert label.ndim == 0
    assert label.dtype == torch.float32
    assert label.item() == 0.0


def test_ranking_dataloader_builds_expected_batch_shapes() -> None:
    loader = build_ranking_dataloader(
        make_samples(),
        batch_size=3,
        shuffle=False,
        num_workers=0,
    )

    features, labels = next(iter(loader))

    assert set(features) == set(RANKING_FEATURE_NAMES)
    assert all(
        values.shape == (3,)
        for values in features.values()
    )
    assert labels.shape == (3,)
    assert labels.dtype == torch.float32
    assert labels.tolist() == [1.0, 0.0, 1.0]


def test_ranking_dataset_ignores_non_model_fields() -> None:
    dataset = RankingDataset(make_samples())

    features, _ = dataset[0]

    assert "user_id_original" not in features


def test_ranking_dataset_rejects_missing_field() -> None:
    samples = make_samples()
    del samples["movie_id"]

    with pytest.raises(
        ValueError,
        match="missing required fields",
    ):
        RankingDataset(samples)


def test_ranking_dataset_rejects_inconsistent_lengths() -> None:
    samples = make_samples()
    samples["genres"] = [1, 2]

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        RankingDataset(samples)


def test_ranking_dataset_rejects_non_vector_field() -> None:
    samples = make_samples()
    samples["movie_id"] = [[1], [2], [3], [4]]

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        RankingDataset(samples)


def test_ranking_dataset_rejects_invalid_labels() -> None:
    samples = make_samples()
    samples["is_click"] = [1, 0, 2, 0]

    with pytest.raises(
        ValueError,
        match="only 0 or 1",
    ):
        RankingDataset(samples)


def test_ranking_dataset_rejects_negative_feature_id() -> None:
    samples = make_samples()
    samples["movie_id"] = [1, -1, 2, 3]

    with pytest.raises(
        ValueError,
        match="negative ID",
    ):
        RankingDataset(samples)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"batch_size": 0}, "positive integer"),
        (
            {"batch_size": 2, "num_workers": -1},
            "non-negative integer",
        ),
    ],
)
def test_ranking_dataloader_rejects_invalid_configuration(
    arguments: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_ranking_dataloader(
            make_samples(),
            shuffle=False,
            **arguments,
        )