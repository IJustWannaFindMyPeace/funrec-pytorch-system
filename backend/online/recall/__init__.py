"""Multi-channel online recall package."""

from .base import RecallStrategy


__all__ = [
    "get_recall_service",
    "RecallService",
    "RecallStrategy",
]


def __getattr__(name):
    """Lazily import the recall service and its external clients."""
    if name in {"RecallService", "get_recall_service"}:
        from .service import RecallService, get_recall_service

        exports = {
            "RecallService": RecallService,
            "get_recall_service": get_recall_service,
        }
        return exports[name]

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )