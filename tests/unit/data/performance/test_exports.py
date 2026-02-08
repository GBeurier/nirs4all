from nirs4all.data.performance import __all__ as performance_exports


def test_performance_exports_only_active_cache_components() -> None:
    assert set(performance_exports) == {"DataCache", "CacheEntry"}
