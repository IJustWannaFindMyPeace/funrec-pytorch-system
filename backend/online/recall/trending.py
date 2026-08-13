import logging
import json
from typing import List, Dict, Any
from app.services.elasticsearch_service import es_service
from .base import RecallStrategy

logger = logging.getLogger(__name__)

class GlobalTrendingRecallStrategy(RecallStrategy):
    async def recall(self, user_context: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        """
        基于评分数量和平均评分召回热门电影
        """
        if not es_service.is_available():
            logger.warning("Elasticsearch 不可用，无法进行全局热门召回")
            return []
            
        try:
            # 查询 ES 获取高评分且热门的电影
            # 使用 function score 结合评分数量和平均评分
            # 这里简化处理：按 rating_count * avg_rating 排序（近似）
            # 或者按 rating_count 排序并过滤 avg_rating
            
            query = {
                "bool": {
                    "filter": [
                        {"range": {"avg_rating": {"gte": 5}}},  # 优质电影
                        {"range": {"rating_count": {"gte": 50}}}  # 热门电影
                    ]
                }
            }
            
            sort = [
                {"rating_count": {"order": "desc"}},
                {"avg_rating": {"order": "desc"}}
            ]
            
            # 使用同步 ES 客户端（ESService 的方法是同步的）
            # 但是 ESService.search_movies 使用特定的查询构建器
            # 我们应该直接使用客户端或为 ESService 添加方法
            # 目前直接使用客户端以完全控制查询
            
            res = es_service.client.search(
                index=es_service.INDEX_NAME,
                query=query,
                sort=sort,
                size=k,
                _source=["movie_id", "title", "genres", "avg_rating"]
            )
            
            hits = res.get("hits", {}).get("hits", [])
            results = []
            for hit in hits:
                source = hit["_source"]
                results.append({
                    "movie_id": source.get("movie_id"),
                    "score": 1.0,  # 热门召回的占位分数
                    "recall_type": "global_trending",
                    "title": source.get("title"),
                    "genres": source.get("genres")
                })
            logger.info(f"GlobalTrendingRecallStrategy.recall, 召回数量: {len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"全局热门召回失败: {e}")
            return []

class UserPreferenceRecallStrategy(RecallStrategy):
    async def recall(
        self,
        user_context: Dict[str, Any],
        k: int,
    ) -> List[Dict[str, Any]]:
        """
        基于用户偏好类型召回电影。
        """
        frequent_genres = user_context.get(
            "frequent_genres",
            [],
        )

        # 没有偏好信号时直接返回，不探测 Elasticsearch。
        if not frequent_genres:
            return []

        if not es_service.is_available():
            logger.warning(
                "Elasticsearch 不可用，无法进行用户偏好召回"
            )
            return []

        try:
            query = {
                "bool": {
                    "must": [
                        {
                            "terms": {
                                "genres.keyword": frequent_genres
                            }
                        }
                    ],
                    "filter": [
                        {
                            "range": {
                                "avg_rating": {
                                    "gte": 5
                                }
                            }
                        }
                    ],
                }
            }

            sort = [
                {
                    "year": {
                        "order": "desc"
                    }
                },
                {
                    "avg_rating": {
                        "order": "desc"
                    }
                },
            ]

            response = es_service.client.search(
                index=es_service.INDEX_NAME,
                query=query,
                sort=sort,
                size=k,
                _source=[
                    "movie_id",
                    "title",
                    "genres",
                ],
            )

            hits = response.get(
                "hits",
                {},
            ).get(
                "hits",
                [],
            )

            results = []
            for hit in hits:
                source = hit["_source"]
                results.append(
                    {
                        "movie_id": source.get("movie_id"),
                        "score": 0.8,
                        "recall_type": "user_preference",
                        "title": source.get("title"),
                        "genres": source.get("genres"),
                    }
                )

            logger.info(
                "UserPreferenceRecallStrategy.recall，"
                "召回数量: %s",
                len(results),
            )
            return results

        except Exception as error:
            logger.error(
                "用户偏好召回失败: %s",
                error,
            )
            return []