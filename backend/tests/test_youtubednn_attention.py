import torch

from modeling.youtubednn import YouTubeDNN


def feature_dict():
    return {
        "user_id": 8,
        "age": 8,
        "gender": 4,
        "occupation": 8,
        "zip_code": 8,
        "movie_id": 16,
        "genres": 12,
    }


def features():
    return {
        "user_id": torch.tensor([1, 2]),
        "age": torch.tensor([1, 2]),
        "gender": torch.tensor([1, 2]),
        "occupation": torch.tensor([1, 2]),
        "zip_code": torch.tensor([1, 2]),
        "hist_movie_id": torch.tensor([
            [0, 0, 1, 2],
            [0, 3, 4, 5],
        ]),
        "hist_genres": torch.tensor([
            [0, 0, 1, 2],
            [0, 3, 4, 5],
        ]),
    }


def test_attention_encoder_returns_finite_normalized_users():
    model = YouTubeDNN(
        feature_dict(),
        embedding_dim=4,
        history_pooling="personalized_attention",
        max_sequence_length=4,
    )
    users = model.encode_user(features())
    assert users.shape == (2, 4)
    assert torch.isfinite(users).all()
    assert torch.allclose(
        torch.linalg.vector_norm(users, dim=1),
        torch.ones(2),
        atol=1e-6,
    )


def test_attention_ignores_padding_embedding_values():
    model = YouTubeDNN(
        feature_dict(),
        embedding_dim=4,
        history_pooling="personalized_attention",
        max_sequence_length=4,
    ).eval()
    value = features()
    before = model.encode_user(value)
    with torch.no_grad():
        model.movie_embedding.weight[0].fill_(999.0)
        model.genre_embedding.weight[0].fill_(999.0)
    after = model.encode_user(value)
    assert torch.allclose(before, after, atol=1e-6)


def test_attention_rejects_history_longer_than_configuration():
    model = YouTubeDNN(
        feature_dict(),
        embedding_dim=4,
        history_pooling="personalized_attention",
        max_sequence_length=3,
    )
    try:
        model.encode_user(features())
    except ValueError as error:
        assert "longer" in str(error)
    else:
        raise AssertionError("Expected long history to be rejected")


def test_masked_mean_default_has_no_attention_parameters():
    model = YouTubeDNN(feature_dict(), embedding_dim=4)
    assert model.history_pooling == "masked_mean"
    assert model.movie_attention_pooling is None
    assert model.genre_attention_pooling is None
    assert not any("attention_pooling" in name for name, _ in model.named_parameters())


def test_unknown_pooling_is_rejected():
    try:
        YouTubeDNN(feature_dict(), history_pooling="transformer")
    except ValueError as error:
        assert "Unsupported" in str(error)
    else:
        raise AssertionError("Expected unsupported pooling to be rejected")
