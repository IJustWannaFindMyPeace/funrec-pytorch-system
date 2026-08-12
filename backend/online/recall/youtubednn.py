"""Online PyTorch YouTubeDNN retrieval strategy."""

import asyncio
import logging
from typing import Any, Dict, List

import torch
from torch import Tensor

from .base import RecallStrategy
from .resource_manager import RecallResourceManager


logger = logging.getLogger(__name__)


class YouTubeDNNRecallStrategy(RecallStrategy):
    """Retrieve movies using the trained PyTorch user tower."""

    STATIC_FEATURES = (
        "user_id",
        "age",
        "gender",
        "occupation",
        "zip_code",
    )

    def __init__(self):
        self.resource_manager = RecallResourceManager()

    @staticmethod
    def _pad_sequence(
        values: List[int],
        max_length: int,
    ) -> List[int]:
        values = values[-max_length:]
        return [0] * (max_length - len(values)) + values

    def preprocess_user(
        self,
        user_features: Dict[str, Any],
        max_hist_len: int = 10,
    ) -> Dict[str, Tensor]:
        """Encode raw online features as model-ready tensors."""
        if max_hist_len <= 0:
            raise ValueError("max_hist_len must be greater than zero")

        manager = self.resource_manager
        inputs = {
            feature_name: torch.tensor(
                [
                    manager.encode_feature(
                        feature_name,
                        user_features.get(feature_name),
                    )
                ],
                dtype=torch.long,
                device=manager.device,
            )
            for feature_name in self.STATIC_FEATURES
        }

        raw_history = user_features.get("hist_movie_ids", []) or []
        encoded_movie_history = [
            encoded_id
            for raw_movie_id in raw_history
            if (
                encoded_id := manager.encode_feature(
                    "movie_id",
                    raw_movie_id,
                )
            )
            > 0
        ]

        encoded_genre_history = []
        for raw_movie_id in raw_history:
            genres = manager.movie_genre_map.get(raw_movie_id)

            if genres is None:
                genres = manager.movie_genre_map.get(
                    str(raw_movie_id),
                    [],
                )

            encoded_genre_history.extend(
                int(genre_id)
                for genre_id in genres
                if int(genre_id) > 0
            )

        inputs["hist_movie_id"] = torch.tensor(
            [
                self._pad_sequence(
                    encoded_movie_history,
                    max_hist_len,
                )
            ],
            dtype=torch.long,
            device=manager.device,
        )
        inputs["hist_genres"] = torch.tensor(
            [
                self._pad_sequence(
                    encoded_genre_history,
                    max_hist_len,
                )
            ],
            dtype=torch.long,
            device=manager.device,
        )

        return inputs

    async def recall(
        self,
        user_context: Dict[str, Any],
        k: int,
    ) -> List[Dict[str, Any]]:
        """Run retrieval outside the asynchronous event loop."""
        if k <= 0:
            return []

        if not self.resource_manager._ensure_resources_loaded():
            logger.warning("YouTubeDNN 召回资源未就绪")
            return []

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._recall_sync,
            user_context,
            k,
        )

    def _recall_sync(
        self,
        user_context: Dict[str, Any],
        k: int,
    ) -> List[Dict[str, Any]]:
        """Execute PyTorch user encoding and exact vector search."""
        manager = self.resource_manager

        if (
            manager.user_model is None
            or manager.item_embedding_tensor is None
        ):
            return []

        try:
            model_inputs = self.preprocess_user(user_context)

            with torch.inference_mode():
                user_embedding = manager.user_model.encode_user(
                    model_inputs
                )
                scores = torch.matmul(
                    manager.item_embedding_tensor,
                    user_embedding[0],
                )

            # Convert the inference tensor into a normal tensor before
            # applying in-place padding and history masks.
            scores = scores.clone()

            # Encoded ID 0 is padding.
            scores[0] = float("-inf")

            raw_history = user_context.get("hist_movie_ids", []) or []
            encoded_history = {
                manager.encode_feature("movie_id", raw_movie_id)
                for raw_movie_id in raw_history
            }

            for encoded_movie_id in encoded_history:
                if 0 < encoded_movie_id < scores.numel():
                    scores[encoded_movie_id] = float("-inf")

            candidate_count = int(torch.isfinite(scores).sum().item())
            result_count = min(k, candidate_count)

            if result_count <= 0:
                return []

            top_scores, top_indices = torch.topk(
                scores,
                k=result_count,
            )

            results = []
            for score, encoded_movie_id in zip(
                top_scores.tolist(),
                top_indices.tolist(),
            ):
                raw_index = encoded_movie_id - 1

                if 0 <= raw_index < len(manager.all_movie_ids):
                    results.append(
                        {
                            "movie_id": int(
                                manager.all_movie_ids[raw_index]
                            ),
                            "score": float(score),
                            "recall_type": "youtube_dnn",
                        }
                    )

            logger.info(
                "YouTubeDNNRecallStrategy.recall，召回数量: %s",
                len(results),
            )
            return results

        except Exception:
            logger.exception("YouTubeDNN 在线召回失败")
            return []