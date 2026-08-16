"""Deterministic candidate-pool expansion and pointwise reranking helpers."""

import torch


def append_unique_candidates(primary, supplemental, pool_size):
    """Keep primary order and append unseen non-padding candidates."""
    if pool_size <= 0:
        raise ValueError("pool_size must be greater than zero")
    output, seen = [], set()
    for source in (primary, supplemental):
        for item_id in source:
            item_id = int(item_id)
            if item_id > 0 and item_id not in seen:
                output.append(item_id)
                seen.add(item_id)
                if len(output) == pool_size:
                    return output
    return output


@torch.no_grad()
def rerank_candidate_pool(model, user_features, candidate_movie_features):
    """Score one user's candidate items with a fixed pointwise ranker."""
    if not candidate_movie_features:
        return []
    count = len(candidate_movie_features)
    features = {
        name: torch.full(
            (count,), int(value), dtype=torch.long, device=next(model.parameters()).device
        )
        for name, value in user_features.items()
    }
    for name, values in candidate_movie_features.items():
        features[name] = torch.as_tensor(
            values, dtype=torch.long, device=next(model.parameters()).device
        )
    scores = model(features).detach().cpu()
    return torch.argsort(scores, descending=True).tolist()
