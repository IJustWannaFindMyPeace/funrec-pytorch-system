"""Train-only ItemCF index and deterministic candidate generation."""

from collections import Counter, defaultdict
from math import log, sqrt

import numpy as np


def train_user_sequences(train):
    """Reconstruct chronological Train target sequences without Validation."""
    users = np.asarray(train["user_id"], dtype=np.int64)
    movies = np.asarray(train["movie_id"], dtype=np.int64)
    if users.shape != movies.shape:
        raise ValueError("Train user_id and movie_id lengths differ")
    sequences = defaultdict(list)
    for user_id, movie_id in zip(users.tolist(), movies.tolist()):
        if movie_id > 0:
            sequences[user_id].append(movie_id)
    return sequences


def build_itemcf_index(train):
    """Build classic I2I weights exclusively from Train target sequences."""
    item_counts = Counter()
    cooccurrence = defaultdict(float)
    for sequence in train_user_sequences(train).values():
        # Repeated consumption by a user is one co-occurrence event.
        items = list(dict.fromkeys(sequence))
        if len(items) < 2:
            continue
        weight = 1.0 / log(1.0 + len(items))
        item_counts.update(items)
        for offset, left in enumerate(items[:-1]):
            for right in items[offset + 1:]:
                cooccurrence[left, right] += weight
                cooccurrence[right, left] += weight

    neighbors = defaultdict(list)
    for (left, right), value in cooccurrence.items():
        neighbors[left].append(
            (right, value / sqrt(item_counts[left] * item_counts[right]))
        )
    for item_id in neighbors:
        neighbors[item_id].sort(key=lambda pair: (-pair[1], pair[0]))
    return dict(neighbors)


def recommend_itemcf(neighbors, history, k):
    """Sum I2I weights over a query history, excluding consumed items."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    consumed = {int(item_id) for item_id in history if int(item_id) > 0}
    scores = defaultdict(float)
    for item_id in consumed:
        for candidate, score in neighbors.get(item_id, []):
            if candidate not in consumed:
                scores[candidate] += score
    return [
        item_id
        for item_id, _ in sorted(
            scores.items(), key=lambda pair: (-pair[1], pair[0])
        )[:k]
    ]
