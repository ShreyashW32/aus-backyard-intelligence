"""Tests for the optional Redis / in-memory cache backend."""
from __future__ import annotations

import time

import pytest

from aus_backyard_intelligence import cache_backend


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    # Ensure each test starts with no configured Redis and a fresh singleton.
    monkeypatch.delenv("ABI_REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    cache_backend.reset_cache_for_tests()
    yield
    cache_backend.reset_cache_for_tests()


def test_defaults_to_memory_backend_without_redis():
    assert cache_backend.cache_backend_name() == "memory"


def test_set_get_roundtrip_json_values():
    cache = cache_backend.get_cache()
    cache.set("k", {"a": 1, "items": [1, 2, 3]}, ttl=10)
    assert cache.get("k") == {"a": 1, "items": [1, 2, 3]}


def test_missing_key_returns_none():
    assert cache_backend.get_cache().get("absent") is None


def test_ttl_expiry():
    cache = cache_backend.get_cache()
    cache.set("short", "v", ttl=0.1)
    assert cache.get("short") == "v"
    time.sleep(0.15)
    assert cache.get("short") is None


def test_lru_eviction_bounds_memory():
    cache = cache_backend._InMemoryCache(max_entries=3)
    for i in range(5):
        cache.set(f"k{i}", i, ttl=60)
    # Oldest two evicted; newest three retained.
    assert cache.get("k0") is None
    assert cache.get("k1") is None
    assert cache.get("k4") == 4


def test_unreachable_redis_falls_back_to_memory(monkeypatch):
    # A bad Redis URL must not crash — it silently degrades to memory.
    monkeypatch.setenv("ABI_REDIS_URL", "redis://127.0.0.1:6399/0")
    cache_backend.reset_cache_for_tests()
    # Either redis isn't installed, or the connection fails fast: both → memory.
    assert cache_backend.cache_backend_name() == "memory"
    cache = cache_backend.get_cache()
    cache.set("k", "v", ttl=5)
    assert cache.get("k") == "v"
