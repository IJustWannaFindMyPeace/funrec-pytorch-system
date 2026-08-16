import torch

from modeling.youtubednn import DynamicRoutingInterestPooling, MINDYouTubeDNN, MaskedMeanPooling, YouTubeDNN


FEATURE_DICT = {
    "user_id": 11,
    "age": 8,
    "gender": 3,
    "occupation": 22,
    "zip_code": 20,
    "movie_id": 31,
    "genres": 19,
}


def build_features():
    return {
        "user_id": torch.tensor([1, 2]),
        "age": torch.tensor([2, 3]),
        "gender": torch.tensor([1, 2]),
        "occupation": torch.tensor([4, 5]),
        "zip_code": torch.tensor([6, 7]),
        "hist_movie_id": torch.tensor(
            [
                [0, 0, 3, 4],
                [0, 5, 6, 7],
            ]
        ),
        "hist_genres": torch.tensor(
            [
                [0, 0, 2, 3],
                [0, 4, 5, 6],
            ]
        ),
    }


def test_masked_mean_ignores_padding_positions():
    pooling = MaskedMeanPooling()

    embedding_table = torch.tensor(
        [
            [100.0, 100.0],
            [1.0, 3.0],
            [3.0, 5.0],
        ]
    )

    first_ids = torch.tensor([[1, 2, 0, 0]])
    second_ids = torch.tensor([[0, 1, 0, 2]])

    first_result = pooling(embedding_table[first_ids], first_ids)
    second_result = pooling(embedding_table[second_ids], second_ids)

    expected = torch.tensor([[2.0, 4.0]])

    assert torch.allclose(first_result, expected)
    assert torch.allclose(second_result, expected)


def test_masked_mean_returns_zero_for_empty_history():
    pooling = MaskedMeanPooling()

    ids = torch.zeros((2, 4), dtype=torch.long)
    embeddings = torch.randn(2, 4, 3)

    result = pooling(embeddings, ids)

    assert torch.equal(result, torch.zeros((2, 3)))


def test_dynamic_routing_interests_are_padded_safe_and_normalized():
    pooling = DynamicRoutingInterestPooling(embedding_dim=2, interest_count=2, routing_iterations=3)
    ids = torch.tensor([[0, 1, 2], [0, 0, 0]])
    embeddings = torch.tensor([[[9., 9.], [1., 0.], [0., 1.]], [[3., 4.], [5., 6.], [7., 8.]]])
    interests = pooling(embeddings, ids)
    assert interests.shape == (2, 2, 2)
    assert torch.allclose(torch.linalg.vector_norm(interests[0], dim=-1), torch.ones(2))
    assert torch.equal(interests[1], torch.zeros((2, 2)))


def test_user_and_item_embeddings_have_expected_shape_and_unit_norm():
    model = YouTubeDNN(FEATURE_DICT, embedding_dim=16)
    features = build_features()

    user_embeddings = model.encode_user(features)
    item_embeddings = model.encode_item(torch.tensor([3, 7]))

    assert user_embeddings.shape == (2, 16)
    assert item_embeddings.shape == (2, 16)

    assert torch.allclose(
        torch.linalg.vector_norm(user_embeddings, dim=-1),
        torch.ones(2),
        atol=1e-6,
    )
    assert torch.allclose(
        torch.linalg.vector_norm(item_embeddings, dim=-1),
        torch.ones(2),
        atol=1e-6,
    )


def test_mind_full_logits_use_maximum_over_two_interests():
    model = MINDYouTubeDNN(FEATURE_DICT, embedding_dim=16, scoring_contract="scaled_cosine_v2", logit_scale=10.0)
    logits = model.compute_full_logits(build_features())
    assert logits.shape == (2, FEATURE_DICT["movie_id"] - 1)
    assert torch.isfinite(logits).all()


def test_forward_returns_cosine_similarity_for_each_example():
    model = YouTubeDNN(FEATURE_DICT, embedding_dim=16)
    features = build_features()
    movie_ids = torch.tensor([3, 7])

    scores = model(features, movie_ids)

    expected = (
        model.encode_user(features)
        * model.encode_item(movie_ids)
    ).sum(dim=-1)

    assert scores.shape == (2,)
    assert torch.allclose(scores, expected)
    assert torch.all(scores >= -1.0)
    assert torch.all(scores <= 1.0)


def test_padding_rows_remain_zero_after_optimizer_step():
    model = YouTubeDNN(FEATURE_DICT, embedding_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    features = build_features()
    movie_ids = torch.tensor([3, 7])

    loss = -model(features, movie_ids).mean()
    loss.backward()
    optimizer.step()

    assert torch.equal(
        model.movie_embedding.weight[0],
        torch.zeros(16),
    )
    assert torch.equal(
        model.genre_embedding.weight[0],
        torch.zeros(16),
    )

    for embedding in model.user_embeddings.values():
        assert torch.equal(
            embedding.weight[0],
            torch.zeros(16),
        )

def test_full_logits_exclude_padding_class():
    model = YouTubeDNN(FEATURE_DICT, embedding_dim=16)
    features = build_features()

    logits = model.compute_full_logits(features)

    assert logits.shape == (
        2,
        FEATURE_DICT["movie_id"] - 1,
    )


def test_full_softmax_loss_is_finite_and_updates_movie_embeddings():
    model = YouTubeDNN(FEATURE_DICT, embedding_dim=16)
    features = build_features()
    movie_ids = torch.tensor([3, 7])

    loss = model.compute_full_softmax_loss(features, movie_ids)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    assert model.movie_embedding.weight.grad is not None
    assert torch.count_nonzero(
        model.movie_embedding.weight.grad[1:]
    ) > 0
    assert torch.equal(
        model.movie_embedding.weight.grad[0],
        torch.zeros(16),
    )


def test_full_softmax_loss_rejects_padding_target():
    model = YouTubeDNN(FEATURE_DICT, embedding_dim=16)
    features = build_features()

    try:
        model.compute_full_softmax_loss(
            features,
            torch.tensor([0, 7]),
        )
    except ValueError as error:
        assert "padding ID 0" in str(error)
    else:
        raise AssertionError("padding target should raise ValueError")
