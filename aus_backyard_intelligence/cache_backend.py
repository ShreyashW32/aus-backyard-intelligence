"""Optional Redis-backed cache with a transparent in-memory fallback.

The viewer's hot paths — OSINT feed proxies, geocoding, and (most importantly)
satellite pool-detection results — are expensive to recompute. When a Redis
server is available (``ABI_REDIS_URL``/``REDIS_URL``), those results are shared
across worker threads *and survive restarts*, so a viewport that was already
detected returns instantly on the next visit instead of re-running YOLO. This
is the "faster detection" lever: the model itself stays at its measured F1 peak,
but repeat work is served from cache.

Everything degrades gracefully: with no Redis URL, or if ``redis`` is not
installed, or if the server is unreachable, the process falls back to a
thread-safe in-memory LRU+TTL cache and behaves exactly as it did before.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any


class _InMemoryCache:
    """Thread-safe LRU cache with per-entry TTL (the always-available fallback)."""

    def __init__(self, max_entries: int = 512) -> None:
        self._lock = threading.Lock()
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._max_entries = max_entries

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at and now >= expires_at:
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl: float) -> None:
        expires_at = time.monotonic() + ttl if ttl and ttl > 0 else 0.0
        with self._lock:
            self._store[key] = (expires_at, value)
            self._store.move_to_end(key)
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def backend(self) -> str:
        return "memory"


class _RedisCache:
    """Redis-backed JSON cache. Any Redis error transparently falls back to the
    in-process memory cache for that call, so a flaky Redis never breaks a
    request."""

    def __init__(self, client: Any, *, namespace: str = "abi") -> None:
        self._client = client
        self._namespace = namespace
        self._fallback = _InMemoryCache()

    def _redis_key(self, key: str) -> str:
        return f"{self._namespace}:{key}"

    def get(self, key: str) -> Any | None:
        try:
            raw = self._client.get(self._redis_key(key))
            if raw is None:
                return self._fallback.get(key)
            return json.loads(raw)
        except Exception:
            return self._fallback.get(key)

    def set(self, key: str, value: Any, ttl: float) -> None:
        try:
            payload = json.dumps(value)
            ttl_ms = int(ttl * 1000) if ttl and ttl > 0 else None
            if ttl_ms:
                self._client.set(self._redis_key(key), payload, px=ttl_ms)
            else:
                self._client.set(self._redis_key(key), payload)
        except Exception:
            # Redis unreachable/oversized value — keep serving from memory.
            self._fallback.set(key, value, ttl)

    def clear(self) -> None:
        self._fallback.clear()
        try:
            for k in self._client.scan_iter(match=f"{self._namespace}:*"):
                self._client.delete(k)
        except Exception:
            pass

    @property
    def backend(self) -> str:
        return "redis"


_cache_lock = threading.Lock()
_cache_instance: _InMemoryCache | _RedisCache | None = None


def _build_cache() -> _InMemoryCache | _RedisCache:
    url = (os.environ.get("ABI_REDIS_URL") or os.environ.get("REDIS_URL") or "").strip()
    if not url:
        return _InMemoryCache()
    try:
        import redis  # type: ignore

        client = redis.Redis.from_url(
            url, socket_connect_timeout=1.5, socket_timeout=1.5, decode_responses=True
        )
        client.ping()  # fail fast if the server is unreachable
        namespace = os.environ.get("ABI_REDIS_NAMESPACE", "abi").strip() or "abi"
        return _RedisCache(client, namespace=namespace)
    except Exception:
        # redis not installed, bad URL, or server down — fall back silently.
        return _InMemoryCache()


def get_cache() -> _InMemoryCache | _RedisCache:
    """Process-wide cache singleton (Redis if configured/reachable, else memory)."""
    global _cache_instance
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = _build_cache()
    return _cache_instance


def cache_backend_name() -> str:
    """'redis' or 'memory' — surfaced in /api/health for operators."""
    return get_cache().backend


def reset_cache_for_tests() -> None:
    """Drop the singleton so tests can re-detect the backend after changing env."""
    global _cache_instance
    with _cache_lock:
        _cache_instance = None
