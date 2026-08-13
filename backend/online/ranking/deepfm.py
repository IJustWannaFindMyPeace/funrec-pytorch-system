"""Online PyTorch DeepFM ranking strategies."""

import asyncio
import logging
from typing import Any, Dict, List

import torch

from .base import RankingStrategy
from .resource_manager import RankingResourceManager


logger = logging.getLogger(__name__)


class DeepFMRankingStrategy(RankingStrategy):
    """Rank recalled candidates with PyTorch DeepFM."""

    def __init__(self) -> None:
        self.resource_manager = RankingResourceManager()

    @property
    def name(self) -> str:
        return "deepfm"

    @property
    def is_ready(self) -> bool:
        try:
            return (
                self.resource_manager
                ._ensure_resources_loaded()
            )
        except (FileNotFoundError, ValueError, RuntimeError):
            logger.exception(
                "PyTorch DeepFM 资源加载失败"
            )
            return False

    @staticmethod
    def _extract_item_feature(
        candidate: Dict[str, Any],
        feature_name: str,
    ) -> Any:
        """Read and normalize one candidate item feature."""
        if feature_name == "movie_id":
            return candidate.get("movie_id")

        if feature_name == "genres":
            value = candidate.get("genres")

            if isinstance(value, (list, tuple)):
                return value[0] if value else None

            if isinstance(value, str):
                return value.split("|")[0]

            return value

        if feature_name == "isAdult":
            return candidate.get(
                "isAdult",
                candidate.get("is_adult"),
            )

        if feature_name == "startYear":
            return candidate.get(
                "startYear",
                candidate.get("year"),
            )

        return candidate.get(feature_name)

    def _prepare_batch_inputs(
        self,
        user_features: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Build an encoded torch batch for DeepFM."""
        if not candidates:
            raise ValueError(
                "candidates must not be empty"
            )

        manager = self.resource_manager
        batch_size = len(candidates)
        inputs = {}

        for feature_name in manager.user_features:
            encoded_value = manager.encode_feature(
                feature_name,
                user_features.get(feature_name),
            )
            inputs[feature_name] = torch.full(
                (batch_size,),
                encoded_value,
                dtype=torch.long,
                device=manager.device,
            )

        for feature_name in manager.item_features:
            encoded_values = [
                manager.encode_feature(
                    feature_name,
                    self._extract_item_feature(
                        candidate,
                        feature_name,
                    ),
                )
                for candidate in candidates
            ]
            inputs[feature_name] = torch.tensor(
                encoded_values,
                dtype=torch.long,
                device=manager.device,
            )

        return inputs

    def _fallback_results(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return candidates ordered by recall score."""
        ordered = sorted(
            candidates,
            key=lambda item: item.get(
                "score",
                0.0,
            ),
            reverse=True,
        )

        return [
            {
                "movie_id": candidate["movie_id"],
                "score": float(
                    candidate.get("score", 0.0)
                ),
                "recall_score": float(
                    candidate.get("score", 0.0)
                ),
                "recall_type": candidate.get(
                    "recall_type",
                    "unknown",
                ),
            }
            for candidate in ordered
        ]

    def _rank_sync(
        self,
        user_features: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run synchronous PyTorch ranking inference."""
        if not candidates:
            return []

        try:
            inputs = self._prepare_batch_inputs(
                user_features,
                candidates,
            )

            with torch.inference_mode():
                probabilities = (
                    self.resource_manager
                    .ranking_model
                    .predict_proba(inputs)
                    .detach()
                    .cpu()
                )

            if probabilities.ndim != 1:
                raise ValueError(
                    "DeepFM probabilities must be "
                    "one-dimensional"
                )

            if probabilities.shape[0] != len(candidates):
                raise ValueError(
                    "DeepFM prediction count does not "
                    "match candidate count"
                )

            ranked_results = [
                {
                    "movie_id": candidate["movie_id"],
                    "score": float(
                        probabilities[index].item()
                    ),
                    "recall_score": float(
                        candidate.get("score", 0.0)
                    ),
                    "recall_type": candidate.get(
                        "recall_type",
                        "unknown",
                    ),
                }
                for index, candidate in enumerate(
                    candidates
                )
            ]

            ranked_results.sort(
                key=lambda item: item["score"],
                reverse=True,
            )

            logger.info(
                "PyTorch DeepFM 精排完成，候选数=%d",
                len(ranked_results),
            )
            return ranked_results

        except Exception:
            logger.exception(
                "PyTorch DeepFM 精排失败，"
                "使用召回分数降级"
            )
            return self._fallback_results(candidates)

    async def rank(
        self,
        user_features: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rank candidates asynchronously."""
        if not candidates:
            return []

        if not self.is_ready:
            logger.warning(
                "PyTorch DeepFM 未就绪，"
                "使用召回分数降级"
            )
            return self._fallback_results(candidates)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._rank_sync,
            user_features,
            candidates,
        )


class FallbackRankingStrategy(RankingStrategy):
    """Sort candidates using their recall scores."""

    @property
    def name(self) -> str:
        return "fallback"

    async def rank(
        self,
        user_features: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        del user_features

        ordered = sorted(
            candidates,
            key=lambda item: item.get(
                "score",
                0.0,
            ),
            reverse=True,
        )

        return [
            {
                "movie_id": candidate["movie_id"],
                "score": float(
                    candidate.get("score", 0.0)
                ),
                "recall_score": float(
                    candidate.get("score", 0.0)
                ),
                "recall_type": candidate.get(
                    "recall_type",
                    "unknown",
                ),
            }
            for candidate in ordered
        ]