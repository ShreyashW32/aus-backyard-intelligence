"""Data pipeline helpers for the Queensland backyard recommender."""

__all__ = ["SOURCE_MANIFEST", "build_property_records"]


def __getattr__(name):
    if name == "build_property_records":
        from .feature_builder import build_property_records

        return build_property_records
    if name == "SOURCE_MANIFEST":
        from .sources import SOURCE_MANIFEST

        return SOURCE_MANIFEST
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
