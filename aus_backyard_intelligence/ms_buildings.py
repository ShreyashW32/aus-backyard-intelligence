"""Australia-wide building footprints from the open Microsoft Building Footprints
dataset, fetched on demand by area.

OpenStreetMap building coverage is sparse across most Australian suburbs, which
starves the lead pipeline (no buildings → no leads). Microsoft's open dataset has
~11M Australian footprints, so this fills the gap everywhere in the country.

The dataset is published as z9 quadkey-partitioned GeoJSONL.gz tiles. For a scan we
compute the quadkey(s) covering the bbox, download just those tiles (cached on disk
so each is fetched once), and return the footprints inside the bbox as OSM-like
elements. It is Australia-only by construction: only tiles whose index Location is
``Australia`` are ever used.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import tempfile
import threading
import time
import urllib.request
from collections import OrderedDict
from typing import Any

_INDEX_URL = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
_USER_AGENT = "QLDBackyardRecommender/1.0 (+https://example.local)"
_TILE_ZOOM = 9
_SYNTHETIC_ID_BASE = 800_000_000  # distinct from OSM and the 950M satellite-detection space

# Safety caps so one scan can't download half the country or blow up memory.
MAX_TILES_PER_REQUEST = 6
MAX_BUILDINGS_PER_REQUEST = 6000

_CACHE_DIR = os.environ.get(
    "QLDR_MS_BUILDING_CACHE", os.path.join(tempfile.gettempdir(), "qldr_ms_buildings")
)

_index_lock = threading.Lock()
_australia_index: dict[str, str] | None = None  # quadkey -> tile url
_tile_locks_lock = threading.Lock()
_tile_locks: dict[str, threading.Lock] = {}
_result_cache_lock = threading.Lock()
_result_cache: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
MAX_RESULT_CACHE = 32


def _http_get(url: str, *, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _tile_xy(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def _quadkey(x: int, y: int, zoom: int) -> str:
    quad = []
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        quad.append(str(digit))
    return "".join(quad)


def _quadkeys_for_bbox(bbox: tuple[float, float, float, float]) -> list[str]:
    south, west, north, east = bbox
    x_min, y_max = _tile_xy(south, west, _TILE_ZOOM)
    x_max, y_min = _tile_xy(north, east, _TILE_ZOOM)
    x_lo, x_hi = sorted((x_min, x_max))
    y_lo, y_hi = sorted((y_min, y_max))
    keys: list[str] = []
    for x in range(x_lo, x_hi + 1):
        for y in range(y_lo, y_hi + 1):
            keys.append(_quadkey(x, y, _TILE_ZOOM))
            if len(keys) >= MAX_TILES_PER_REQUEST:
                return keys
    return keys


def _load_australia_index(*, timeout: float) -> dict[str, str]:
    global _australia_index
    with _index_lock:
        if _australia_index is not None:
            return _australia_index
        index: dict[str, str] = {}
        try:
            data = _http_get(_INDEX_URL, timeout=timeout).decode("utf-8", "replace")
            for row in csv.DictReader(io.StringIO(data)):
                if row.get("Location") == "Australia":
                    quadkey = str(row.get("QuadKey", "")).strip()
                    url = str(row.get("Url", "")).strip()
                    if quadkey and url:
                        # A quadkey can appear more than once; the last (newest) wins.
                        index[quadkey] = url
        except Exception:
            # Leave _australia_index as None so a later call can retry the index.
            return {}
        _australia_index = index
        return index


def _tile_cache_path(quadkey: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"au_{quadkey}.geojsonl.gz")


def _tile_lock(quadkey: str) -> threading.Lock:
    with _tile_locks_lock:
        lock = _tile_locks.get(quadkey)
        if lock is None:
            lock = threading.Lock()
            _tile_locks[quadkey] = lock
        return lock


def _ensure_tile(quadkey: str, url: str, *, timeout: float) -> str | None:
    """Return a path to the cached .gz tile, downloading it once if needed."""
    path = _tile_cache_path(quadkey)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    with _tile_lock(quadkey):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
        try:
            raw = _http_get(url, timeout=timeout)
        except Exception:
            return None
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "wb") as handle:
                handle.write(raw)
            os.replace(tmp, path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            return None
        return path


def _polygon_centroid_and_ring(geometry: dict[str, Any]) -> tuple[float, float, list[list[float]]] | None:
    if not isinstance(geometry, dict):
        return None
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if gtype == "Polygon" and isinstance(coords, list) and coords:
        ring = coords[0]
    elif gtype == "MultiPolygon" and isinstance(coords, list) and coords and coords[0]:
        ring = coords[0][0]
    else:
        return None
    if not isinstance(ring, list) or len(ring) < 3:
        return None
    lons = [float(point[0]) for point in ring if isinstance(point, (list, tuple)) and len(point) >= 2]
    lats = [float(point[1]) for point in ring if isinstance(point, (list, tuple)) and len(point) >= 2]
    if not lons or not lats:
        return None
    return sum(lats) / len(lats), sum(lons) / len(lons), ring


def _parse_tile_for_bbox(
    path: str, bbox: tuple[float, float, float, float], start_id: int
) -> list[dict[str, Any]]:
    south, west, north, east = bbox
    elements: list[dict[str, Any]] = []
    element_id = start_id
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    feature = json.loads(line)
                except ValueError:
                    continue
                parsed = _polygon_centroid_and_ring(feature.get("geometry", {}))
                if parsed is None:
                    continue
                lat, lon, ring = parsed
                if not (south <= lat <= north and west <= lon <= east):
                    continue
                element_id += 1
                props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
                height = props.get("height")
                tags = {
                    "building": "yes",
                    "detector:source": "microsoft-footprint",
                    "building:source": "microsoft_open_buildings",
                }
                try:
                    if height is not None and float(height) > 0:
                        tags["height"] = f"{float(height):.1f}"
                except (TypeError, ValueError):
                    pass
                elements.append(
                    {
                        "type": "way",
                        "id": element_id,
                        "tags": tags,
                        "geometry": [
                            {"lat": round(float(p[1]), 7), "lon": round(float(p[0]), 7)}
                            for p in ring
                            if isinstance(p, (list, tuple)) and len(p) >= 2
                        ],
                    }
                )
                if len(elements) >= MAX_BUILDINGS_PER_REQUEST:
                    break
    except Exception:
        return elements
    return elements


def fetch_australia_buildings(
    bbox: tuple[float, float, float, float], *, timeout: float = 30.0
) -> list[dict[str, Any]]:
    """Return Microsoft building footprints inside ``bbox`` as OSM-like elements.

    Australia-only and best-effort: any network/parse failure returns ``[]`` so the
    caller can fall back to OpenStreetMap. Tiles are cached on disk (downloaded once)
    and results are memoised per rounded bbox.
    """
    south, west, north, east = bbox
    cache_key = f"{south:.4f}:{west:.4f}:{north:.4f}:{east:.4f}"
    with _result_cache_lock:
        if cache_key in _result_cache:
            cached = _result_cache.pop(cache_key)
            _result_cache[cache_key] = cached
            return [dict(element) for element in cached]

    index = _load_australia_index(timeout=timeout)
    if not index:
        return []
    quadkeys = [qk for qk in _quadkeys_for_bbox(bbox) if qk in index]
    if not quadkeys:
        return []  # outside Australia's tiled coverage

    started = time.time()
    elements: list[dict[str, Any]] = []
    for offset, quadkey in enumerate(quadkeys):
        remaining = timeout - (time.time() - started)
        if remaining <= 1.0:
            break
        path = _ensure_tile(quadkey, index[quadkey], timeout=remaining)
        if not path:
            continue
        start_id = _SYNTHETIC_ID_BASE + offset * MAX_BUILDINGS_PER_REQUEST
        elements.extend(_parse_tile_for_bbox(path, bbox, start_id))
        if len(elements) >= MAX_BUILDINGS_PER_REQUEST:
            elements = elements[:MAX_BUILDINGS_PER_REQUEST]
            break

    with _result_cache_lock:
        _result_cache[cache_key] = [dict(element) for element in elements]
        while len(_result_cache) > MAX_RESULT_CACHE:
            _result_cache.popitem(last=False)
    return elements
