"""Selection-scoped pool and solar detection from satellite imagery.

The primary inference path uses pool-trained YOLO weights (plus an optional
second pool model). A strict colour/water supplement is conditional and the
legacy colour-CV detector is opt-in. Generic COCO YOLO is only an optional
context validator; it is not the pool detector.

The module downloads Google satellite tiles for the viewport, stitches them,
runs detection, then converts pixel coordinates back to (lat, lon).

Thread-safe.  Falls back gracefully when ``ultralytics`` or ``cv2`` are absent.
"""
from __future__ import annotations

import io
import hashlib
import importlib.util
import json
import math
import os
import threading
import time
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Lazy imports — these heavy libraries may not be installed
# ---------------------------------------------------------------------------
_cv2 = None
_np = None
_Image = None
_YOLO = None
_ort = None
_samgeo = None
_torchvision = None
_HAS_CV2 = False
_HAS_YOLO = False
_HAS_ORT = False
_HAS_SAMGEO = False
_HAS_MASK_RCNN = False
_import_lock = threading.Lock()
_import_done = False


def _lazy_import():
    global _cv2, _np, _Image, _YOLO, _ort, _samgeo, _torchvision
    global _HAS_CV2, _HAS_YOLO, _HAS_ORT, _HAS_SAMGEO, _HAS_MASK_RCNN, _import_done
    if _import_done:
        return
    with _import_lock:
        if _import_done:
            return
        try:
            import numpy as np
            _np = np
        except Exception:
            pass
        try:
            import cv2
            _cv2 = cv2
            _HAS_CV2 = True
        except Exception:
            pass
        try:
            from PIL import Image
            _Image = Image
        except Exception:
            pass
        # Ultralytics is needed for the YOLO context validator (opt-in) and for
        # the pool-trained detector (whenever pool weights are configured).
        if _ENABLE_YOLO_CONTEXT or resolve_pool_yolo_weights():
            try:
                config_dir = PROJECT_ROOT / ".ultralytics"
                config_dir.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
                from ultralytics import YOLO
                _YOLO = YOLO
                _HAS_YOLO = True
            except Exception:
                pass
        try:
            import onnxruntime as ort
            _ort = ort
            _HAS_ORT = True
        except Exception:
            pass
        try:
            if importlib.util.find_spec("samgeo") is not None:
                _samgeo = True
                _HAS_SAMGEO = True
        except Exception:
            pass
        try:
            if importlib.util.find_spec("torchvision") is not None:
                _torchvision = True
                _HAS_MASK_RCNN = True
        except Exception:
            pass
        _import_done = True


# ---------------------------------------------------------------------------
# Tile download constants
# ---------------------------------------------------------------------------
_TILE_SIZE = 256
_USER_AGENT = os.environ.get(
    "QLDR_HTTP_USER_AGENT",
    "QLD-Backyard-Recommender/1.1 (local research application)",
)

# LRU tile cache  (thread-safe)
_tile_cache_lock = threading.Lock()
_tile_cache: OrderedDict[str, bytes] = OrderedDict()
_MAX_TILE_CACHE = 2048
_MAX_STITCH_TILE_SIDE = 6
_DETECTION_CHUNK_TILE_SIDE = 4
_MAX_DETECTION_CHUNKS = 128
_FAST_VIEWPORT_TILE_BUDGET = 1024
_AUTO_VIEWPORT_TILE_BUDGET = 96
_PRECISE_VIEWPORT_TILE_BUDGET = 2048
_MAX_TILE_DOWNLOAD_WORKERS = 12
_FAST_SCAN_TIME_BUDGET_SECONDS = 18.0
_MAX_FAST_POOL_ELEMENTS = 48
_MAX_AUTO_POOL_ELEMENTS = 160
_ENABLE_YOLO_CONTEXT = os.environ.get("QLDR_ENABLE_YOLO_CONTEXT", "0") == "1"
_SAVE_REJECTED_POOL_CROPS = os.environ.get("QLDR_POOL_SAVE_REJECTED_CROPS", "0") == "1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Pool-trained YOLO weights. Generic COCO models cannot detect pools (no such
# class), so a real visual pool detector requires fine-tuned weights. When a
# weights file is present, YOLO becomes the primary detector for new/unmapped
# pools; the colour-CV path is then only an opt-in experimental fallback.
_POOL_YOLO_WEIGHTS_ENV = os.environ.get("QLDR_POOL_YOLO_WEIGHTS", "").strip()
# Colour-CV produces many false positives (solar panels, roofs, shaded turf),
# so it is OFF by default and must be explicitly enabled for experimentation.
_ENABLE_CV_POOL_FALLBACK = os.environ.get("QLDR_ENABLE_CV_POOL_FALLBACK", "0") == "1"
# Strict-water CV supplement mode: "auto" (default) runs it only when the
# two-model ensemble is NOT active — the second model covers the bright pools
# the supplement was built for, and without it the supplement is mostly cost
# (2026-07-10 eval: 28 of 42 false positives, largely ocean water, vs 10
# unique true positives). "1" forces it on, "0" forces it off.
_STRICT_WATER_SUPPLEMENT_MODE = os.environ.get("QLDR_ENABLE_STRICT_WATER_SUPPLEMENT", "auto").strip().lower()
# Imported pool models tend to score lower on Google tiles than on their native
# training imagery. 0.15 is the measured F1 peak on the Overpass-independent
# eval (2026-07-09, 581 points): recall 0.872/fp 0.130/F1 0.857 vs 0.864/0.127/
# 0.854 at 0.20; below 0.12 recall saturates at 0.880 while fp_rate jumps to
# 0.142 (see docs/accuracy_log.md). The latest larger r23 report is 0.882 recall,
# 0.071 fp_rate, and 0.899 F1 proxy. Tunable via QLDR_POOL_YOLO_CONF.
_POOL_YOLO_MIN_CONFIDENCE = float(os.environ.get("QLDR_POOL_YOLO_CONF", "0.15"))
# Confidence floor for the multi-class model's SOLAR class (independent of the
# pool class above and the pool eval). Solar arrays carry their own 2-2500 m²
# area gate, so a lower floor surfaces more real rooftop arrays without touching
# pool precision. Lowered 0.35→0.30 (r22b). Tunable via QLDR_POOL_YOLO_SOLAR_CONF.
_POOL_YOLO_SOLAR_CONF = float(os.environ.get("QLDR_POOL_YOLO_SOLAR_CONF", "0.30"))
# Two-model ensemble: when secondary weights resolve (models/pool_yolo_b.pt or
# QLDR_POOL_YOLO_WEIGHTS_B), pool boxes from both models are unioned. Costs a
# second inference per chunk; disable with QLDR_POOL_YOLO_ENSEMBLE=0.
_POOL_YOLO_ENSEMBLE = os.environ.get("QLDR_POOL_YOLO_ENSEMBLE", "1") != "0"
# Inference size for the pool model. 0 = auto: use the checkpoint's own
# training imgsz when it records one (e.g. a P2 model trained at 1024), else
# 640. The bundled ONNX has a fixed 640 input so it always resolves to 640.
_POOL_YOLO_IMGSZ = int(os.environ.get("QLDR_POOL_YOLO_IMGSZ", "0"))
# Test-time augmentation (flips/scales averaged per inference). ~2-3x slower;
# recall effect must be proven on the eval harness before enabling by default.
_POOL_YOLO_TTA = os.environ.get("QLDR_POOL_YOLO_TTA", "0") == "1"
_HARD_NEGATIVE_DIR = os.environ.get(
    "QLDR_POOL_HARD_NEGATIVE_DIR",
    str(PROJECT_ROOT / "data" / "hard_negatives"),
)
_MAX_NEGATIVE_CROPS_PER_RUN = int(os.environ.get("QLDR_POOL_MAX_NEGATIVE_CROPS", "80"))

# YOLO model singleton (thread-safe)
_model_lock = threading.Lock()
_loaded_models: dict[str, Any] = {}
# Ultralytics mutates a native PyTorch model during its first ``predict``
# (notably when it fuses Conv+BatchNorm layers).  Concurrent first predictions
# can therefore leave one thread executing ``Conv.forward`` after another has
# removed ``bn``.  Serialize both ensemble members through one lock: it avoids
# that race and bounds peak activation memory on CPU-only deployments.
_pool_yolo_inference_lock = threading.Lock()
_pool_yolo_runtime_lock = threading.Lock()
_pool_yolo_runtime: dict[str, Any] = {
    "last_success": None,
    "last_error": None,
    "last_error_type": None,
    "last_role": None,
    "last_stage": None,
    "last_event_at": None,
    "last_inference_at": None,
    "compatibility_repairs_total": 0,
    "primary": {},
    "secondary": {},
}
_detector_diagnostics_lock = threading.Lock()
_last_detector_diagnostics: dict[str, Any] = {}
_weight_checksum_lock = threading.Lock()
_weight_checksum_cache: dict[tuple[str, int, int], str] = {}
_negative_crop_lock = threading.Lock()
_negative_crops_saved_this_run = 0

# Increment whenever inference/gating semantics change. This value, the active
# weight checksums, thresholds, and ensemble state form the detector signature
# used by server-side detection caches.
DETECTOR_PIPELINE_VERSION = "2026-07-19-pool-accuracy-v3"
_fast_scan_cursor_lock = threading.Lock()
_fast_scan_cursors: dict[str, int] = {}

DEFAULT_DETECTOR_MODEL = "yolov11n"

# Supported lightweight models with download info
SUPPORTED_MODELS: dict[str, dict[str, str]] = {
    "yolov8n": {
        "weight": "yolov8n.pt",
        "label": "YOLOv8 Nano",
        "description": "6MB · Fastest · Mature ecosystem",
    },
    "yolov11n": {
        "weight": "yolo11n.pt",
        "label": "YOLOv11 Nano",
        "description": "5.4MB · Best efficiency · Lowest latency",
    },
    "yolov11s": {
        "weight": "yolo11s.pt",
        "label": "YOLOv11 Small",
        "description": "18MB · Higher accuracy · Still fast",
    },
}


def normalize_detector_model(model_name: str | None) -> str:
    """Return a supported YOLO model key or raise a client-facing error."""
    candidate = (model_name or DEFAULT_DETECTOR_MODEL).strip().lower()
    if candidate not in SUPPORTED_MODELS:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unsupported detector_model '{model_name}'. Supported models: {supported}.")
    return candidate


def _local_model_path(filename: str) -> str:
    """Use bundled model files regardless of the process working directory."""
    path = PROJECT_ROOT / filename
    return str(path) if path.exists() else filename


# ---------------------------------------------------------------------------
# Slippy-map tile math
# ---------------------------------------------------------------------------
def _lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert (lat, lon) to slippy-map tile (x, y) at *zoom*."""
    lat = max(min(lat, 85.05112878), -85.05112878)
    lon = max(min(lon, 180.0), -180.0)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def _tile_to_lat_lon(x: int, y: int, zoom: int) -> tuple[float, float]:
    """Return the NW corner (lat, lon) of tile (x, y) at *zoom*."""
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lat, lon


def _pixel_to_latlon(
    px: float,
    py: float,
    origin_lat: float,
    origin_lon: float,
    img_width: int,
    img_height: int,
    bbox: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Convert pixel (px, py) back to (lat, lon)."""
    south, west, north, east = bbox
    lat = north - (py / img_height) * (north - south)
    lon = west + (px / img_width) * (east - west)
    return lat, lon


def _normalize_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Return bbox as (south, west, north, east), clamped to Web Mercator limits."""
    south, west, north, east = bbox
    south, north = sorted((max(min(south, 85.05112878), -85.05112878), max(min(north, 85.05112878), -85.05112878)))
    west, east = sorted((max(min(west, 180.0), -180.0), max(min(east, 180.0), -180.0)))
    return south, west, north, east


def _tile_range_for_bbox(
    bbox: tuple[float, float, float, float],
    zoom: int,
) -> tuple[int, int, int, int]:
    """Return inclusive x/y tile range as (x_min, x_max, y_min, y_max)."""
    south, west, north, east = _normalize_bbox(bbox)
    x_west, y_south = _lat_lon_to_tile(south, west, zoom)
    x_east, y_north = _lat_lon_to_tile(north, east, zoom)
    x_min, x_max = sorted((x_west, x_east))
    y_min, y_max = sorted((y_north, y_south))
    return x_min, x_max, y_min, y_max


def _tile_block_bbox(
    x_start: int,
    y_start: int,
    cols: int,
    rows: int,
    zoom: int,
) -> tuple[float, float, float, float]:
    """Return the geographic bbox covered by a tile block."""
    nw_lat, nw_lon = _tile_to_lat_lon(x_start, y_start, zoom)
    se_lat, se_lon = _tile_to_lat_lon(x_start + cols, y_start + rows, zoom)
    return _normalize_bbox((se_lat, nw_lon, nw_lat, se_lon))


def _intersect_bbox(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    """Return the intersection of two bboxes, or None if they do not overlap."""
    south_a, west_a, north_a, east_a = _normalize_bbox(first)
    south_b, west_b, north_b, east_b = _normalize_bbox(second)
    south = max(south_a, south_b)
    west = max(west_a, west_b)
    north = min(north_a, north_b)
    east = min(east_a, east_b)
    if south >= north or west >= east:
        return None
    return south, west, north, east


def _planned_chunk_count(
    bbox: tuple[float, float, float, float],
    zoom: int,
    chunk_side: int,
) -> int:
    x_min, x_max, y_min, y_max = _tile_range_for_bbox(bbox, zoom)
    cols = x_max - x_min + 1
    rows = y_max - y_min + 1
    return math.ceil(cols / chunk_side) * math.ceil(rows / chunk_side)


def _planned_tile_count(bbox: tuple[float, float, float, float], zoom: int) -> int:
    x_min, x_max, y_min, y_max = _tile_range_for_bbox(bbox, zoom)
    return max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)


def _normalize_detector_precision(detector_precision: str | None) -> str:
    candidate = (detector_precision or "auto").strip().lower()
    if candidate not in {"auto", "fast", "precise"}:
        return "auto"
    return candidate


def _choose_tile_zoom(
    bbox: tuple[float, float, float, float],
    requested_zoom: int,
    detector_precision: str = "auto",
) -> int:
    """Pick the highest zoom that fits the scan budget for the requested mode.

    The viewer's map zoom is deliberately NOT a cap: backyard pools are ~8 px
    at z18 and near-invisible to the detector (measured 10/55 labelled pools at
    z18 vs 48/55 at z20 on the same cell), so detection always aims for the
    mode's max zoom and lets the tile/chunk budget shrink it for large areas.
    """
    precision = _normalize_detector_precision(detector_precision)
    min_zoom = 16 if precision == "fast" else 17
    max_zoom = 20 if precision in {"fast", "precise"} else 19
    top_zoom = max(min_zoom, max_zoom)
    tile_budget = {
        "fast": _FAST_VIEWPORT_TILE_BUDGET,
        "auto": _AUTO_VIEWPORT_TILE_BUDGET,
        "precise": _PRECISE_VIEWPORT_TILE_BUDGET,
    }[precision]
    for candidate_zoom in range(top_zoom, min_zoom - 1, -1):
        if (
            _planned_chunk_count(bbox, candidate_zoom, _MAX_STITCH_TILE_SIDE) <= _MAX_DETECTION_CHUNKS
            and _planned_tile_count(bbox, candidate_zoom) <= tile_budget
        ):
            return candidate_zoom
    return min_zoom


def _iter_detection_bboxes(
    bbox: tuple[float, float, float, float],
    zoom: int,
) -> list[tuple[float, float, float, float]]:
    """Split a requested bbox into stitched satellite chips that cover it fully."""
    request_bbox = _normalize_bbox(bbox)
    x_min, x_max, y_min, y_max = _tile_range_for_bbox(request_bbox, zoom)
    total_cols = x_max - x_min + 1
    total_rows = y_max - y_min + 1
    chunk_side = _DETECTION_CHUNK_TILE_SIDE
    while (
        math.ceil(total_cols / chunk_side) * math.ceil(total_rows / chunk_side) > _MAX_DETECTION_CHUNKS
        and chunk_side < _MAX_STITCH_TILE_SIDE
    ):
        chunk_side += 1

    chunks: list[tuple[float, float, float, float]] = []
    for y_start in range(y_min, y_max + 1, chunk_side):
        rows = min(chunk_side, y_max - y_start + 1)
        for x_start in range(x_min, x_max + 1, chunk_side):
            cols = min(chunk_side, x_max - x_start + 1)
            tile_bbox = _tile_block_bbox(x_start, y_start, cols, rows, zoom)
            clipped = _intersect_bbox(tile_bbox, request_bbox)
            if clipped:
                chunks.append(clipped)

    return chunks[:_MAX_DETECTION_CHUNKS]


def _set_last_detector_diagnostics(diagnostics: dict[str, Any]) -> None:
    with _detector_diagnostics_lock:
        _last_detector_diagnostics.clear()
        _last_detector_diagnostics.update(diagnostics)


# ---------------------------------------------------------------------------
# Tile download & stitching
# ---------------------------------------------------------------------------
def _download_tile(x: int, y: int, zoom: int) -> bytes | None:
    """Download a single satellite tile, using the LRU cache."""
    cache_key = f"{zoom}/{x}/{y}"
    with _tile_cache_lock:
        if cache_key in _tile_cache:
            _tile_cache.move_to_end(cache_key)
            return _tile_cache[cache_key]

    # Google satellite tile URL
    url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = resp.read(512 * 1024)  # max 512KB per tile
    except Exception:
        return None

    with _tile_cache_lock:
        _tile_cache[cache_key] = data
        _tile_cache.move_to_end(cache_key)
        while len(_tile_cache) > _MAX_TILE_CACHE:
            _tile_cache.popitem(last=False)
    return data


def _stitch_tiles(
    bbox: tuple[float, float, float, float],
    zoom: int = 18,
) -> tuple[Any, tuple[float, float, float, float]] | None:
    """Download and stitch satellite tiles covering *bbox*.

    Returns (numpy_image_bgr, actual_bbox) or None on failure.
    """
    result = _stitch_tiles_with_stats(bbox, zoom=zoom)
    if result is None:
        return None
    image, actual_bbox, _stats = result
    return image, actual_bbox


def _stitch_tiles_with_stats(
    bbox: tuple[float, float, float, float],
    zoom: int = 18,
) -> tuple[Any, tuple[float, float, float, float], dict[str, int]] | None:
    """Download and stitch satellite tiles, returning tile coverage stats."""
    _lazy_import()
    if _np is None or (not _HAS_CV2 and _Image is None):
        return None

    x_min, x_max, y_min, y_max = _tile_range_for_bbox(bbox, zoom)

    # Clamp a single chip to a bounded memory footprint. Large requests are
    # split by detect_pools_in_bbox instead of truncating the viewport.
    cols = min(x_max - x_min + 1, _MAX_STITCH_TILE_SIDE)
    rows = min(y_max - y_min + 1, _MAX_STITCH_TILE_SIDE)
    if cols <= 0 or rows <= 0:
        return None

    img_w = cols * _TILE_SIZE
    img_h = rows * _TILE_SIZE
    canvas = _np.zeros((img_h, img_w, 3), dtype=_np.uint8)
    attempted_tiles = 0
    loaded_tiles = 0

    tile_jobs = [
        (tx, ty, x_min + tx, y_min + ty)
        for ty in range(rows)
        for tx in range(cols)
    ]
    attempted_tiles = len(tile_jobs)
    worker_count = min(_MAX_TILE_DOWNLOAD_WORKERS, max(1, len(tile_jobs)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_download_tile, tile_x, tile_y, zoom): (tx, ty)
            for tx, ty, tile_x, tile_y in tile_jobs
        }
        for future in as_completed(futures):
            tx, ty = futures[future]
            try:
                tile_data = future.result()
            except Exception:
                tile_data = None
            if tile_data is None:
                continue
            try:
                if _HAS_CV2:
                    arr = _np.frombuffer(tile_data, dtype=_np.uint8)
                    tile_img = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
                else:
                    with _Image.open(io.BytesIO(tile_data)) as pil_img:
                        tile_img = _np.array(pil_img.convert("RGB"))
                if tile_img is not None:
                    # Resize if needed (some tiles may be different sizes)
                    if tile_img.shape[0] != _TILE_SIZE or tile_img.shape[1] != _TILE_SIZE:
                        if _HAS_CV2:
                            tile_img = _cv2.resize(tile_img, (_TILE_SIZE, _TILE_SIZE))
                        else:
                            tile_img = _np.array(_Image.fromarray(tile_img).resize((_TILE_SIZE, _TILE_SIZE)))
                    y_off = ty * _TILE_SIZE
                    x_off = tx * _TILE_SIZE
                    canvas[y_off:y_off + _TILE_SIZE, x_off:x_off + _TILE_SIZE] = tile_img
                    loaded_tiles += 1
            except Exception:
                continue

    if loaded_tiles <= 0:
        return None

    # Actual bbox covered by the stitched tiles
    actual_bbox = _tile_block_bbox(x_min, y_min, cols, rows, zoom)
    stats = {"attempted_tiles": attempted_tiles, "loaded_tiles": loaded_tiles}

    return canvas, actual_bbox, stats


def _detect_blue_pools_numpy(
    image: Any,
    *,
    min_area_px: int = 12,
    max_area_px: int = 80000,
) -> list[dict[str, Any]]:
    """Fallback pool-water segmentation using only NumPy and PIL-decoded RGB tiles."""
    if _np is None:
        return []
    arr = _np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return []
    red = arr[:, :, 0].astype("int16")
    green = arr[:, :, 1].astype("int16")
    blue = arr[:, :, 2].astype("int16")
    water_index = (green - red) / _np.maximum(green + red, 1)
    blue_index = (blue - red) / _np.maximum(blue + red, 1)
    mask = (
        (
            (blue >= 82)
            & (green >= 58)
            & (red <= 200)
            & ((blue - red) >= 14)
            & ((green - red) >= -4)
        )
        | (
            (blue >= 65)
            & (green >= 70)
            & (red <= 175)
            & (water_index >= 0.04)
            & (blue_index >= 0.05)
        )
    )

    height, width = mask.shape
    visited = _np.zeros(mask.shape, dtype=bool)
    detections: list[dict[str, Any]] = []
    for y0 in range(height):
        xs = _np.where(mask[y0] & ~visited[y0])[0]
        for x0 in xs.tolist():
            if visited[y0, x0] or not mask[y0, x0]:
                continue
            stack = [(x0, y0)]
            visited[y0, x0] = True
            area = 0
            min_x = max_x = x0
            min_y = max_y = y0
            while stack:
                x, y = stack.pop()
                area += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx] and mask[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            if area < min_area_px or area > max_area_px:
                continue
            w = max_x - min_x + 1
            h = max_y - min_y + 1
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > 10.0:
                continue
            fill = area / max(w * h, 1)
            if fill < 0.18:
                continue
            confidence = max(0.50, min(0.92, 0.52 + fill * 0.25 + min(1.0, area / 2400) * 0.15))
            detections.append(
                {
                    "cx": min_x + w / 2.0,
                    "cy": min_y + h / 2.0,
                    "x": min_x,
                    "y": min_y,
                    "w": w,
                    "h": h,
                    "area_px": area,
                    "confidence": round(confidence, 2),
                    "circularity": round(fill, 3),
                    "aspect_ratio": round(aspect, 2),
                }
            )
    return detections


# ---------------------------------------------------------------------------
# OpenCV HSV blue-water pool segmentation
# ---------------------------------------------------------------------------
def _detect_blue_pools_cv(
    image: Any,
    *,
    min_area_px: int = 12,
    max_area_px: int = 80000,
    detector_precision: str = "auto",
) -> list[dict[str, Any]]:
    """Detect pool-like blue regions via HSV colour segmentation.

    Returns a list of dicts with keys: x, y, w, h, area_px, contour, confidence.
    """
    _lazy_import()
    if not _HAS_CV2 or _np is None:
        return _detect_blue_pools_numpy(image, min_area_px=min_area_px, max_area_px=max_area_px)

    detector_precision = _normalize_detector_precision(detector_precision)
    hsv = _cv2.cvtColor(image, _cv2.COLOR_BGR2HSV)

    b, g, r = _cv2.split(image)
    hue, sat, val = _cv2.split(hsv)
    red = r.astype("int16")
    green = g.astype("int16")
    blue = b.astype("int16")

    # Fast viewport scanning is now intentionally conservative. It only uses
    # clear blue/cyan/teal water signatures. The broader muted-blue fallback is
    # reserved for auto/precise scans where the user is explicitly asking for a
    # slower, manual-quality pass.
    if detector_precision == "fast":
        strict_blue = (
            (hue >= 88)
            & (hue <= 132)
            & (sat >= 42)
            & (val >= 72)
            & ((blue - red) >= 20)
            & ((green - red) >= 8)
            & ((blue - green) >= -28)
        )
        strict_cyan = (
            (hue >= 64)
            & (hue <= 105)
            & (sat >= 44)
            & (val >= 78)
            & ((green - red) >= 20)
            & ((blue - red) >= 8)
            & (_np.abs(blue - green) <= 72)
        )
        deep_blue = (
            (hue >= 96)
            & (hue <= 130)
            & (sat >= 62)
            & (val >= 58)
            & ((blue - red) >= 34)
            & ((blue - green) >= -5)
        )
        vegetation = (
            (hue >= 35)
            & (hue <= 82)
            & ((green - red) >= 28)
            & ((green - blue) >= 18)
        )
        warm_roof = ((red - green) >= 18) & ((red - blue) >= 18)
        low_chroma_roof = (sat < 30) & ((blue - red) < 22)
        mask = ((strict_blue | strict_cyan | deep_blue) & ~vegetation & ~warm_roof & ~low_chroma_roof).astype(_np.uint8) * 255
    else:
        # Blue/cyan water ranges in HSV. Google satellite pool pixels are often
        # muted blue-gray rather than saturated neon blue, so include
        # low-saturation pool tones while keeping hue and RGB dominance checks
        # tight. This branch is deliberately not used for continuous fast scans.
        lower_blue1 = _np.array([88, 45, 60], dtype=_np.uint8)
        upper_blue1 = _np.array([132, 255, 255], dtype=_np.uint8)
        precise_mode = detector_precision == "precise"
        lower_blue2 = _np.array([96, 34 if precise_mode else 18, 85], dtype=_np.uint8)
        upper_blue2 = _np.array([126, 120, 235], dtype=_np.uint8)
        lower_blue3 = _np.array([55, 38 if precise_mode else 24, 85], dtype=_np.uint8)
        upper_blue3 = _np.array([96, 135, 235], dtype=_np.uint8)

        mask1 = _cv2.inRange(hsv, lower_blue1, upper_blue1)
        mask2 = _cv2.inRange(hsv, lower_blue2, upper_blue2)
        mask3 = _cv2.inRange(hsv, lower_blue3, upper_blue3)
        mask = _cv2.bitwise_or(mask1, _cv2.bitwise_or(mask2, mask3))

        # RGB dominance reinforcement. Keep this conservative: broad muted
        # blue-gray thresholds cause roof and driveway false positives.
        blue_gray = (
            (
                (hue >= 96)
                & (hue <= 126)
                & (sat >= (36 if precise_mode else 18))
                & (sat <= 130)
                & (val >= 82)
                & (val <= 238)
                & ((blue - red) >= (18 if precise_mode else 7))
                & ((green - red) >= (4 if precise_mode else -80))
                & ((blue - green) >= (-10 if precise_mode else -8))
            )
        )
        cyan_gray = (
            (hue >= 55)
            & (hue <= 96)
            & (sat >= (40 if precise_mode else 24))
            & (sat <= 145)
            & (val >= 86)
            & (val <= 238)
            & ((green - red) >= (18 if precise_mode else 7))
            & ((blue - red) >= (8 if precise_mode else 0))
            & (_np.abs(blue - green) <= (32 if precise_mode else 38))
        )
        mask4 = (blue_gray | cyan_gray).astype(_np.uint8) * 255
        mask = _cv2.bitwise_or(mask, mask4)

    # Morphological cleanup. A larger CLOSE bridges the lane-marking / ripple gaps
    # that otherwise shatter a big public pool into many sub-threshold fragments, so
    # it reads as one blob; a gentler OPEN still strips isolated noise specks.
    close_kernel = _cv2.getStructuringElement(
        _cv2.MORPH_ELLIPSE, (7, 7) if detector_precision == "fast" else (9, 9)
    )
    open_kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (3, 3))
    mask = _cv2.morphologyEx(mask, _cv2.MORPH_CLOSE, close_kernel, iterations=2 if detector_precision == "fast" else 3)
    mask = _cv2.morphologyEx(mask, _cv2.MORPH_OPEN, open_kernel, iterations=1)

    # Find contours
    contours, _ = _cv2.findContours(mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)

    detections: list[dict[str, Any]] = []
    for contour in contours:
        area = _cv2.contourArea(contour)
        if area < min_area_px or area > max_area_px:
            continue

        # Shape filtering — pools are roughly rectangular or oval
        perimeter = _cv2.arcLength(contour, True)
        if perimeter < 1:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)

        x, y, w, h = _cv2.boundingRect(contour)
        aspect = max(w, h) / max(min(w, h), 1)
        extent = area / max(w * h, 1)
        rect = _cv2.minAreaRect(contour)
        rect_w, rect_h = rect[1]
        rotated_extent = area / max(rect_w * rect_h, 1)

        # Pools are usually compact rectangles/ovals. Surf, shadows, road
        # paint, and blue roof fragments are much more ragged or sparse. Fast
        # mode is stricter because it runs continuously over whole viewports.
        if detector_precision == "fast":
            min_circularity, min_extent, min_rotated_extent, max_aspect = 0.22, 0.30, 0.42, 6.2
        elif detector_precision == "precise":
            min_circularity, min_extent, min_rotated_extent, max_aspect = 0.20, 0.25, 0.34, 9.0
        else:
            min_circularity, min_extent, min_rotated_extent, max_aspect = 0.18, 0.20, 0.30, 8.0
        if circularity < min_circularity:
            continue
        if extent < min_extent or rotated_extent < min_rotated_extent:
            continue

        # Reject extremely elongated rectangles (rivers/roads), but keep lap pools.
        if aspect > max_aspect:
            continue

        # Build the filled contour mask inside a padded window around the blob
        # so per-contour cost scales with the blob, not the whole chunk image.
        # (Allocating a full chunk-sized mask per contour was the dominant cost
        # and caused fast-mode scans to exhaust their time budget mid-viewport.)
        context_pad = max(10, int(max(w, h) * 0.75))
        win_y0 = max(0, y - context_pad)
        win_y1 = min(image.shape[0], y + h + context_pad)
        win_x0 = max(0, x - context_pad)
        win_x1 = min(image.shape[1], x + w + context_pad)
        img_win = image[win_y0:win_y1, win_x0:win_x1]
        hsv_win = hsv[win_y0:win_y1, win_x0:win_x1]
        blob_mask = _np.zeros((win_y1 - win_y0, win_x1 - win_x0), dtype=_np.uint8)
        _cv2.drawContours(blob_mask, [contour], -1, 255, thickness=-1, offset=(-win_x0, -win_y0))
        blob_sel = blob_mask > 0
        candidate_pixels = img_win[blob_sel]
        candidate_hsv = hsv_win[blob_sel]
        if candidate_pixels.size == 0 or candidate_hsv.size == 0:
            continue

        mean_bgr = candidate_pixels.mean(axis=0)
        mean_hsv = candidate_hsv.mean(axis=0)
        std_hsv = candidate_hsv.std(axis=0)
        blue_minus_red = float(mean_bgr[0] - mean_bgr[2])
        blue_minus_green = float(mean_bgr[0] - mean_bgr[1])
        green_minus_red = float(mean_bgr[1] - mean_bgr[2])
        saturation_mean = float(mean_hsv[1])
        value_std = float(std_hsv[2])
        hue_std = float(std_hsv[0])
        pixel_hue = candidate_hsv[:, 0].astype("int16")
        pixel_sat = candidate_hsv[:, 1].astype("int16")
        pixel_val = candidate_hsv[:, 2].astype("int16")
        pixel_blue = candidate_pixels[:, 0].astype("int16")
        pixel_green = candidate_pixels[:, 1].astype("int16")
        pixel_red = candidate_pixels[:, 2].astype("int16")
        value_mean = float(mean_hsv[2])
        if detector_precision == "fast":
            strict_blue_pixels = (
                (pixel_hue >= 88)
                & (pixel_hue <= 132)
                & (pixel_sat >= 40)
                & (pixel_val >= 70)
                & ((pixel_blue - pixel_red) >= 18)
                & ((pixel_green - pixel_red) >= 7)
                & ((pixel_blue - pixel_green) >= -30)
            )
            strict_cyan_pixels = (
                (pixel_hue >= 64)
                & (pixel_hue <= 105)
                & (pixel_sat >= 42)
                & (pixel_val >= 76)
                & ((pixel_green - pixel_red) >= 18)
                & ((pixel_blue - pixel_red) >= 7)
                & (_np.abs(pixel_blue - pixel_green) <= 72)
            )
        else:
            if detector_precision == "precise":
                strict_blue_pixels = (
                    (pixel_hue >= 92)
                    & (pixel_hue <= 130)
                    & (pixel_sat >= 36)
                    & (pixel_val >= 72)
                    & ((pixel_blue - pixel_red) >= 18)
                    & ((pixel_green - pixel_red) >= 4)
                    & ((pixel_blue - pixel_green) >= -10)
                )
                strict_cyan_pixels = (
                    (pixel_hue >= 55)
                    & (pixel_hue <= 95)
                    & (pixel_sat >= 40)
                    & (pixel_val >= 82)
                    & ((pixel_green - pixel_red) >= 18)
                    & ((pixel_blue - pixel_red) >= 8)
                    & (_np.abs(pixel_blue - pixel_green) <= 32)
                )
            else:
                strict_blue_pixels = (
                    (pixel_hue >= 92)
                    & (pixel_hue <= 130)
                    & (pixel_sat >= 24)
                    & (pixel_val >= 72)
                    & ((pixel_blue - pixel_red) >= 10)
                    & ((pixel_blue - pixel_green) >= -10)
                )
                strict_cyan_pixels = (
                    (pixel_hue >= 55)
                    & (pixel_hue <= 95)
                    & (pixel_sat >= 30)
                    & (pixel_val >= 82)
                    & ((pixel_green - pixel_red) >= 12)
                    & ((pixel_blue - pixel_red) >= -2)
                    & (_np.abs(pixel_blue - pixel_green) <= 24)
                )
        water_fraction = float((strict_blue_pixels | strict_cyan_pixels).mean())
        if detector_precision == "fast":
            water_chroma_ok = (
                (blue_minus_red >= 18 and green_minus_red >= 7 and blue_minus_green >= -28)
                or (green_minus_red >= 20 and blue_minus_red >= 7 and abs(blue_minus_green) <= 72)
            )
            min_water_fraction = 0.62
            min_saturation_mean = 36
            max_texture_value_std = 68
            max_texture_hue_std = 28
        else:
            if detector_precision == "precise":
                water_chroma_ok = (
                    (blue_minus_red >= 18 and green_minus_red >= 4 and blue_minus_green >= -12)
                    or (green_minus_red >= 18 and blue_minus_red >= 8 and abs(blue_minus_green) <= 32)
                )
                min_water_fraction = 0.72
                min_saturation_mean = 36
                max_texture_value_std = 64
                max_texture_hue_std = 24
            else:
                water_chroma_ok = (
                    (blue_minus_red >= 9 and blue_minus_green >= -12)
                    or (green_minus_red >= 12 and blue_minus_red >= -3 and abs(blue_minus_green) <= 28)
                )
                min_water_fraction = 0.34
                min_saturation_mean = 22
                max_texture_value_std = 85
                max_texture_hue_std = 35

        if not water_chroma_ok:
            continue
        if green_minus_red > 24 and blue_minus_red < (18 if detector_precision == "fast" else 6):
            continue
        if water_fraction < min_water_fraction:
            continue
        if saturation_mean < min_saturation_mean:
            continue
        if value_std > max_texture_value_std and hue_std > max_texture_hue_std and area < 12000:
            continue

        # Reuse the padded window computed above; context = window minus blob.
        context_mask = ~blob_sel
        context_pixels = img_win[context_mask]
        context_hsv = hsv_win[context_mask]
        context_roof_fraction = 0.0
        context_vegetation_fraction = 0.0
        context_dark_panel_fraction = 0.0
        if context_pixels.size > 0 and context_hsv.size > 0:
            ctx_hue = context_hsv[:, 0].astype("int16")
            ctx_sat = context_hsv[:, 1].astype("int16")
            ctx_val = context_hsv[:, 2].astype("int16")
            ctx_blue = context_pixels[:, 0].astype("int16")
            ctx_green = context_pixels[:, 1].astype("int16")
            ctx_red = context_pixels[:, 2].astype("int16")
            warm_roof = (
                (ctx_sat >= 35)
                & (ctx_val >= 55)
                & (
                    ((ctx_red - ctx_green) >= 10) & ((ctx_red - ctx_blue) >= 16)
                    | ((ctx_hue <= 26) & ((ctx_red - ctx_blue) >= 10))
                )
            )
            vegetation_context = (
                (ctx_hue >= 34)
                & (ctx_hue <= 86)
                & ((ctx_green - ctx_red) >= 12)
                & ((ctx_green - ctx_blue) >= 4)
            )
            dark_panel_context = (
                (ctx_hue >= 86)
                & (ctx_hue <= 132)
                & (ctx_sat >= 24)
                & (ctx_val <= 118)
                & ((ctx_blue - ctx_red) >= 4)
            )
            context_roof_fraction = float(warm_roof.mean())
            context_vegetation_fraction = float(vegetation_context.mean())
            context_dark_panel_fraction = float(dark_panel_context.mean())

        cx = x + w / 2.0
        cy = y + h / 2.0
        # Large natural-water suppression used to happen here, but muted
        # Google tiles make some urban roofs/vegetation merge into huge broad
        # masks. Ranking + result caps are safer than dropping valid pools.

        # Confidence based on how pool-like the shape is
        shape_score = min(1.0, circularity / 0.78)  # rectangle = 1.0
        size_score = min(1.0, area / 2000)  # larger = more likely real
        aspect_score = max(0.0, 1.0 - (aspect - 1.0) / 5.0)
        fill_score = min(1.0, (extent + rotated_extent) / 1.4)
        water_score = min(1.0, water_fraction / 0.72)
        texture_penalty = min(0.18, max(0.0, (value_std - 24) / 220))
        confidence = 0.58 + 0.42 * (
            shape_score * 0.24
            + size_score * 0.16
            + aspect_score * 0.18
            + fill_score * 0.18
            + water_score * 0.24
        )
        confidence -= texture_penalty
        confidence = max(0.50, min(0.99, confidence))

        detections.append({
            "cx": cx,
            "cy": cy,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area_px": area,
            "confidence": round(confidence, 2),
            "circularity": round(circularity, 3),
            "aspect_ratio": round(aspect, 2),
            "extent": round(extent, 3),
            "rotated_extent": round(rotated_extent, 3),
            "water_fraction": round(water_fraction, 3),
            "value_mean": round(value_mean, 1),
            "saturation_mean": round(saturation_mean, 1),
            "green_minus_red": round(green_minus_red, 1),
            "blue_minus_red": round(blue_minus_red, 1),
            "blue_minus_green": round(blue_minus_green, 1),
            "context_roof_fraction": round(context_roof_fraction, 3),
            "context_vegetation_fraction": round(context_vegetation_fraction, 3),
            "context_dark_panel_fraction": round(context_dark_panel_fraction, 3),
        })

    return detections


# ---------------------------------------------------------------------------
# YOLO model loading
# ---------------------------------------------------------------------------
def _get_yolo_model(model_name: str = "yolov8n") -> Any | None:
    """Load (or return cached) YOLO model. Thread-safe."""
    _lazy_import()
    if not _HAS_YOLO or _YOLO is None:
        return None

    with _model_lock:
        try:
            model_name = normalize_detector_model(model_name)
        except ValueError:
            print(f"YOLO: Unsupported model '{model_name}'. Falling back to {DEFAULT_DETECTOR_MODEL}.")
            model_name = DEFAULT_DETECTOR_MODEL

        if model_name in _loaded_models:
            return _loaded_models[model_name]

        info = SUPPORTED_MODELS.get(model_name)
        if info is None:
            return None

        try:
            weight_path = _local_model_path(info["weight"])
            model = _YOLO(weight_path)

            # Try ONNX export for faster CPU inference
            onnx_path = _local_model_path(info["weight"].replace(".pt", ".onnx"))
            if _HAS_ORT and not os.path.exists(onnx_path):
                try:
                    model.export(format="onnx", imgsz=640, simplify=True)
                    print(f"YOLO: Exported {model_name} to ONNX for faster inference")
                except Exception as exc:
                    print(f"YOLO: ONNX export failed (non-fatal): {exc}")

            # Load ONNX model if available
            if _HAS_ORT and os.path.exists(onnx_path):
                try:
                    model = _YOLO(onnx_path)
                    print(f"YOLO: Using ONNX model for {model_name}")
                except Exception:
                    pass  # Fall back to PyTorch model

            _loaded_models[model_name] = model
            return model
        except Exception as exc:
            print(f"YOLO: Failed to load {model_name}: {exc}")
            return None


def _yolo_detect_structures(
    image: Any,
    model_name: str = "yolov8n",
) -> list[dict[str, Any]]:
    """Run YOLO inference to detect buildings/structures near pool candidates.

    Returns a list of detected buildings with their bounding boxes.
    """
    model = _get_yolo_model(model_name)
    if model is None:
        return []

    try:
        results = model.predict(
            image,
            imgsz=640,
            conf=0.25,
            verbose=False,
        )
        structures = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names.get(cls_id, "")
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                structures.append({
                    "class": cls_name,
                    "confidence": conf,
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    "cx": (x1 + x2) / 2,
                    "cy": (y1 + y2) / 2,
                })
        return structures
    except Exception as exc:
        print(f"YOLO inference failed (non-fatal): {exc}")
        return []


# ---------------------------------------------------------------------------
# Pool-trained YOLO detector (primary visual detector for new/unmapped pools)
# ---------------------------------------------------------------------------
def resolve_pool_yolo_weights() -> str | None:
    """Return a path to pool-trained YOLO weights, or None if not configured.

    Checks the ``QLDR_POOL_YOLO_WEIGHTS`` override first, then conventional
    output locations used by ``scripts/pool_yolo_pipeline.py train``.
    """
    candidates = [
        _POOL_YOLO_WEIGHTS_ENV,
        str(PROJECT_ROOT / "pool_yolo.pt"),
        str(PROJECT_ROOT / "pool_yolo.onnx"),
        str(PROJECT_ROOT / "models" / "pool_yolo.pt"),
        str(PROJECT_ROOT / "models" / "pool_yolo.onnx"),
        str(PROJECT_ROOT / "runs" / "pool_train" / "pool_hard_negative" / "weights" / "best.pt"),
        str(PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def pool_yolo_available() -> bool:
    """True when both Ultralytics and pool-trained weights are present."""
    _lazy_import()
    return bool(_HAS_YOLO and _YOLO is not None and resolve_pool_yolo_weights())


def resolve_pool_yolo_weights_b() -> str | None:
    """Optional SECOND pool model for the ensemble, or None.

    Measured 2026-07-10 (7 eval cells, conf 0.15, raw boxes): base recall
    0.864, Google-tile fine-tune 0.891, union 0.950 with no extra false
    positives — the two models miss DIFFERENT pools, so the union beats
    either alone. Set QLDR_POOL_YOLO_ENSEMBLE=0 to disable the second pass.
    """
    env = os.environ.get("QLDR_POOL_YOLO_WEIGHTS_B", "").strip()
    candidates = [env, str(PROJECT_ROOT / "models" / "pool_yolo_b.pt")]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _weight_descriptor(weights: str | None, *, role: str) -> dict[str, Any] | None:
    """Return stable, cache-safe identity metadata for a detector checkpoint."""
    if not weights:
        return None
    path = Path(weights).expanduser()
    try:
        resolved = path.resolve()
        stat = resolved.stat()
    except OSError:
        return {
            "role": role,
            "name": path.name,
            "path": str(path),
            "sha256": None,
            "size_bytes": None,
            "exists": False,
        }

    cache_key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    with _weight_checksum_lock:
        checksum = _weight_checksum_cache.get(cache_key)
    if checksum is None:
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        checksum = digest.hexdigest()
        with _weight_checksum_lock:
            for old_key in list(_weight_checksum_cache):
                if old_key[0] == str(resolved) and old_key != cache_key:
                    _weight_checksum_cache.pop(old_key, None)
            _weight_checksum_cache[cache_key] = checksum
            while len(_weight_checksum_cache) > 8:
                _weight_checksum_cache.pop(next(iter(_weight_checksum_cache)))
    return {
        "role": role,
        "name": resolved.name,
        "path": str(resolved),
        "sha256": checksum,
        "size_bytes": int(stat.st_size),
        "exists": True,
    }


def get_detector_signature() -> dict[str, Any]:
    """Describe the exact pool/solar inference configuration."""
    primary = _weight_descriptor(resolve_pool_yolo_weights(), role="primary")
    secondary_path = resolve_pool_yolo_weights_b() if _POOL_YOLO_ENSEMBLE else None
    secondary = _weight_descriptor(secondary_path, role="secondary")
    configuration = {
        "pipeline_version": DETECTOR_PIPELINE_VERSION,
        "weights": [item for item in (primary, secondary) if item is not None],
        "ensemble_enabled": bool(_POOL_YOLO_ENSEMBLE),
        "ensemble_active": bool(secondary),
        "thresholds": {
            "pool_base": _POOL_YOLO_MIN_CONFIDENCE,
            "pool_auto": max(_POOL_YOLO_MIN_CONFIDENCE, 0.32),
            "pool_fast": max(_POOL_YOLO_MIN_CONFIDENCE, 0.34),
            "solar": _POOL_YOLO_SOLAR_CONF,
        },
        "imgsz": _POOL_YOLO_IMGSZ or "checkpoint-default",
        "tta": bool(_POOL_YOLO_TTA),
        "strict_water_supplement": _STRICT_WATER_SUPPLEMENT_MODE,
        "cv_fallback_enabled": bool(_ENABLE_CV_POOL_FALLBACK),
    }
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        **configuration,
        "signature": hashlib.sha256(encoded).hexdigest()[:20],
    }


def _safe_inference_error(exc: BaseException) -> str:
    """Return a bounded, single-line error suitable for API diagnostics."""
    try:
        message = " ".join(str(exc).split())
    except Exception:
        message = "unprintable inference exception"
    return f"{type(exc).__name__}: {message[:500]}"


def _record_pool_yolo_runtime(
    role: str,
    *,
    success: bool,
    stage: str,
    error: BaseException | str | None = None,
    repairs: int = 0,
    retried: bool = False,
    elapsed_ms: int | None = None,
) -> None:
    safe_error = (
        _safe_inference_error(error)
        if isinstance(error, BaseException)
        else " ".join(str(error).split())[:550] if error else None
    )
    error_type = type(error).__name__ if isinstance(error, BaseException) else None
    entry = {
        "success": bool(success),
        "stage": stage,
        "error": safe_error,
        "error_type": error_type,
        "compatibility_repairs": int(repairs),
        "retried_after_repair": bool(retried),
        "elapsed_ms": elapsed_ms,
        "timestamp": round(time.time(), 3),
    }
    with _pool_yolo_runtime_lock:
        previous = dict(_pool_yolo_runtime.get(role) or {})
        if "class_names" in previous:
            entry["class_names"] = previous["class_names"]
        _pool_yolo_runtime[role] = entry
        _pool_yolo_runtime["last_success"] = bool(success)
        _pool_yolo_runtime["last_error"] = safe_error
        _pool_yolo_runtime["last_error_type"] = error_type
        _pool_yolo_runtime["last_role"] = role
        _pool_yolo_runtime["last_stage"] = stage
        _pool_yolo_runtime["last_event_at"] = entry["timestamp"]
        if stage.startswith("inference"):
            _pool_yolo_runtime["last_inference_at"] = entry["timestamp"]
        _pool_yolo_runtime["compatibility_repairs_total"] = int(
            _pool_yolo_runtime.get("compatibility_repairs_total", 0)
        ) + int(repairs)


def _record_pool_yolo_class_names(role: str, names: list[str]) -> None:
    clean_names = list(dict.fromkeys(str(name) for name in names if str(name)))
    with _pool_yolo_runtime_lock:
        role_entry = dict(_pool_yolo_runtime.get(role) or {})
        role_entry["class_names"] = clean_names
        _pool_yolo_runtime[role] = role_entry


def get_pool_yolo_runtime_status() -> dict[str, Any]:
    """Return a JSON-safe snapshot of actual model load/inference health."""
    with _pool_yolo_runtime_lock:
        return json.loads(json.dumps(_pool_yolo_runtime))


def _update_scan_inference_diagnostics(
    diagnostics: dict[str, Any] | None,
    role: str,
    *,
    success: bool,
    error: BaseException | str | None = None,
    repairs: int = 0,
    retried: bool = False,
) -> None:
    if diagnostics is None:
        return
    diagnostics["inference_attempts"] = int(diagnostics.get("inference_attempts", 0)) + 1
    outcome_key = "inference_successes" if success else "inference_failures"
    diagnostics[outcome_key] = int(diagnostics.get(outcome_key, 0)) + 1
    diagnostics["compatibility_repairs"] = int(diagnostics.get("compatibility_repairs", 0)) + int(repairs)
    if retried:
        diagnostics["inference_retries"] = int(diagnostics.get("inference_retries", 0)) + 1
    if error:
        diagnostics[f"{role}_inference_error"] = (
            _safe_inference_error(error)
            if isinstance(error, BaseException)
            else " ".join(str(error).split())[:550]
        )


def _repair_fused_conv_forwards(yolo_model: Any) -> int:
    """Repair old/partially-fused Ultralytics Conv objects in-place.

    Some checkpoints (and a concurrent first ``predict`` race) contain Conv
    modules whose ``bn`` attribute has already been removed while their active
    ``forward`` still references it.  Switching only those demonstrably broken
    modules to their own ``forward_fuse`` is safe across Ultralytics versions.
    """
    native_model = getattr(yolo_model, "model", yolo_model)
    modules_method = getattr(native_model, "modules", None)
    if not callable(modules_method):
        return 0
    repaired = 0
    try:
        modules = modules_method()
    except Exception:
        return 0
    for module in modules:
        if hasattr(module, "bn"):
            continue
        forward = getattr(module, "forward", None)
        forward_func = getattr(forward, "__func__", forward)
        code = getattr(forward_func, "__code__", None)
        if code is None or "bn" not in getattr(code, "co_names", ()):
            continue
        forward_fuse = getattr(module, "forward_fuse", None)
        if not callable(forward_fuse):
            continue
        try:
            module.forward = forward_fuse
            repaired += 1
        except Exception:
            continue
    return repaired


def _predict_pool_yolo_model(
    model: Any,
    image: Any,
    *,
    role: str,
    imgsz: int,
    conf: float,
    augment: bool = False,
    diagnostics: dict[str, Any] | None = None,
) -> Any:
    """Run one ensemble member safely, repairing the fused-Conv mismatch once."""
    started_at = time.time()
    repairs = 0
    retried = False
    with _pool_yolo_inference_lock:
        repairs += _repair_fused_conv_forwards(model)
        try:
            results = model.predict(
                image,
                imgsz=imgsz,
                conf=conf,
                verbose=False,
                augment=bool(augment),
            )
        except Exception as first_error:
            retry_repairs = _repair_fused_conv_forwards(model)
            repairs += retry_repairs
            if isinstance(first_error, AttributeError) and "bn" in str(first_error) and retry_repairs:
                retried = True
                try:
                    results = model.predict(
                        image,
                        imgsz=imgsz,
                        conf=conf,
                        verbose=False,
                        augment=bool(augment),
                    )
                except Exception as retry_error:
                    elapsed_ms = round((time.time() - started_at) * 1000)
                    _record_pool_yolo_runtime(
                        role,
                        success=False,
                        stage="inference_retry",
                        error=retry_error,
                        repairs=repairs,
                        retried=True,
                        elapsed_ms=elapsed_ms,
                    )
                    _update_scan_inference_diagnostics(
                        diagnostics,
                        role,
                        success=False,
                        error=retry_error,
                        repairs=repairs,
                        retried=True,
                    )
                    raise
            else:
                elapsed_ms = round((time.time() - started_at) * 1000)
                _record_pool_yolo_runtime(
                    role,
                    success=False,
                    stage="inference",
                    error=first_error,
                    repairs=repairs,
                    elapsed_ms=elapsed_ms,
                )
                _update_scan_inference_diagnostics(
                    diagnostics,
                    role,
                    success=False,
                    error=first_error,
                    repairs=repairs,
                )
                raise

    elapsed_ms = round((time.time() - started_at) * 1000)
    _record_pool_yolo_runtime(
        role,
        success=True,
        stage="inference",
        repairs=repairs,
        retried=retried,
        elapsed_ms=elapsed_ms,
    )
    _update_scan_inference_diagnostics(
        diagnostics,
        role,
        success=True,
        repairs=repairs,
        retried=retried,
    )
    return results


def _get_pool_yolo_model() -> Any | None:
    """Load (or return cached) the pool-trained YOLO model. Thread-safe."""
    _lazy_import()
    if not _HAS_YOLO or _YOLO is None:
        return None
    weights = resolve_pool_yolo_weights()
    if not weights:
        return None
    cache_key = f"__pool__:{weights}"
    with _model_lock:
        if cache_key in _loaded_models:
            return _loaded_models[cache_key]
        try:
            model = _YOLO(weights)
            repaired = _repair_fused_conv_forwards(model)
            train_imgsz = 0
            try:
                ckpt = getattr(model, "ckpt", None) or {}
                train_imgsz = int((ckpt.get("train_args") or {}).get("imgsz") or 0)
            except Exception:
                train_imgsz = 0
            model._qldr_imgsz = train_imgsz if 320 <= train_imgsz <= 1920 else 640
            _loaded_models[cache_key] = model
            repair_note = f", repaired {repaired} fused Conv forward(s)" if repaired else ""
            print(f"YOLO: loaded pool-trained weights {weights} (imgsz {model._qldr_imgsz}{repair_note})")
            return model
        except Exception as exc:
            _record_pool_yolo_runtime("primary", success=False, stage="load", error=exc)
            print(f"YOLO: failed to load pool weights {weights}: {_safe_inference_error(exc)}")
            return None


def _get_pool_yolo_model_b() -> Any | None:
    """Load (or return cached) the secondary ensemble model. Thread-safe."""
    _lazy_import()
    if not _HAS_YOLO or _YOLO is None:
        return None
    weights = resolve_pool_yolo_weights_b()
    if not weights:
        return None
    cache_key = f"__pool_b__:{weights}"
    with _model_lock:
        if cache_key in _loaded_models:
            return _loaded_models[cache_key]
        try:
            model = _YOLO(weights)
            repaired = _repair_fused_conv_forwards(model)
            train_imgsz = 0
            try:
                ckpt = getattr(model, "ckpt", None) or {}
                train_imgsz = int((ckpt.get("train_args") or {}).get("imgsz") or 0)
            except Exception:
                train_imgsz = 0
            model._qldr_imgsz = train_imgsz if 320 <= train_imgsz <= 1920 else 640
            _loaded_models[cache_key] = model
            repair_note = f", repaired {repaired} fused Conv forward(s)" if repaired else ""
            print(f"YOLO: loaded secondary pool weights {weights} (imgsz {model._qldr_imgsz}{repair_note})")
            return model
        except Exception as exc:
            _record_pool_yolo_runtime("secondary", success=False, stage="load", error=exc)
            print(f"YOLO: failed to load secondary pool weights {weights}: {_safe_inference_error(exc)}")
            return None


def _model_class_name(names: Any, cls_id: int) -> str:
    """Resolve an Ultralytics class name from dict/list metadata robustly."""
    value: Any = None
    if isinstance(names, dict):
        value = names.get(cls_id)
        if value is None:
            value = names.get(str(cls_id))
    elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        value = names[cls_id]
    return str(value).strip() if value is not None else ""


def _pool_model_class_role(names: Any, cls_id: int) -> str:
    """Classify checkpoint output as ``pool``, ``solar``, or ``other``.

    The bundled two-class checkpoints are explicitly ``0=solarpanel, 1=pool``.
    Name-based parsing remains the source of truth so exported/reordered models
    stay correct; string-keyed mappings are supported as well as integer keys.
    """
    raw_name = _model_class_name(names, cls_id)
    normalized = "".join(ch for ch in raw_name.lower() if ch.isalnum())
    if any(token in normalized for token in ("solar", "photovoltaic", "pvpanel")):
        return "solar"
    if "pool" in normalized or normalized in {"swimming", "aquaticvenue"}:
        return "pool"

    # Legacy fine-tunes may expose an unnamed or generically named single class.
    try:
        class_count = len(names)
    except (TypeError, AttributeError):
        class_count = 0
    if cls_id == 0 and (not raw_name or class_count == 1):
        return "pool"
    return "other"


def _parse_pool_yolo_results(
    image: Any,
    results: Any,
    *,
    model_role: str,
    include_solar: bool,
    conf: float,
) -> tuple[list[dict[str, Any]], list[tuple[float, float, float, float]], list[str]]:
    """Convert Ultralytics results into the detector's common box schema."""
    detections: list[dict[str, Any]] = []
    solar_boxes: list[tuple[float, float, float, float]] = []
    discovered_names: list[str] = []
    for result in results:
        names = getattr(result, "names", {}) or {}
        if isinstance(names, dict):
            discovered_names.extend(str(value) for value in names.values())
        elif isinstance(names, (list, tuple)):
            discovered_names.extend(str(value) for value in names)
        for box in getattr(result, "boxes", []) or []:
            try:
                cls_id = int(box.cls[0])
            except (TypeError, IndexError, ValueError):
                cls_id = 0
            class_role = _pool_model_class_role(names, cls_id)
            if class_role == "other" or (class_role == "solar" and not include_solar):
                continue
            try:
                confidence = float(box.conf[0]) if box.conf is not None else conf
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            if class_role == "solar":
                if confidence < _POOL_YOLO_SOLAR_CONF:
                    continue
                solar_boxes.append((x1, y1, x2, y2))
                detections.append({
                    "cx": (x1 + x2) / 2.0,
                    "cy": (y1 + y2) / 2.0,
                    "x": x1,
                    "y": y1,
                    "w": width,
                    "h": height,
                    "area_px": width * height,
                    "confidence": round(confidence, 2),
                    "circularity": 0.0,
                    "aspect_ratio": round(max(width, height) / max(min(width, height), 1.0), 2),
                    "detector": "yolo_solar",
                    "model_role": model_role,
                    "class_id": cls_id,
                    "class_name": _model_class_name(names, cls_id),
                })
                continue

            detection = {
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "x": x1,
                "y": y1,
                "w": width,
                "h": height,
                "area_px": width * height,
                "confidence": round(confidence, 2),
                "circularity": 0.0,
                "aspect_ratio": round(max(width, height) / max(min(width, height), 1.0), 2),
                "detector": "yolo",
                "model_role": model_role,
                "class_id": cls_id,
                "class_name": _model_class_name(names, cls_id),
            }
            detection.update(_box_colour_stats(image, x1, y1, x2, y2))
            detections.append(detection)
    return detections, solar_boxes, list(dict.fromkeys(discovered_names))


def _merge_secondary_pool_dets(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Union of two models' pool detections: keep every primary det, add each
    secondary det whose centre is not already covered by a primary POOL det
    (within half the larger box side, floor 6 px). Pure function for testing."""
    primary_pools = [d for d in primary if d.get("detector") != "yolo_solar"]
    merged = list(primary)
    for det in secondary:
        radius = max(6.0, 0.5 * max(det.get("w", 0.0), det.get("h", 0.0)))
        duplicate = any(
            abs(det["cx"] - p["cx"]) <= radius and abs(det["cy"] - p["cy"]) <= radius
            for p in primary_pools
        )
        if not duplicate:
            merged.append(det)
    return merged


def _yolo_detect_pools(
    image: Any,
    *,
    conf: float = _POOL_YOLO_MIN_CONFIDENCE,
    imgsz: int | None = None,
    runtime_diagnostics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run the pool-trained YOLO model and return pixel-space pool boxes.

    Returns dicts shaped like the colour-CV detections (x, y, w, h, cx, cy,
    area_px, confidence) so they can flow through the same geometry/merge path.
    Returns an empty list when no pool weights are configured.
    """
    model = _get_pool_yolo_model()
    if model is None:
        runtime = get_pool_yolo_runtime_status().get("primary") or {}
        error = runtime.get("error") or "RuntimeError: primary pool/solar model unavailable"
        _update_scan_inference_diagnostics(
            runtime_diagnostics,
            "primary",
            success=False,
            error=error,
        )
        return []
    if not imgsz:
        imgsz = _POOL_YOLO_IMGSZ or getattr(model, "_qldr_imgsz", 640)
    try:
        # ONNX backends don't support TTA — only augment native .pt checkpoints.
        use_tta = _POOL_YOLO_TTA and getattr(model, "ckpt", None) is not None
        results = _predict_pool_yolo_model(
            model,
            image,
            role="primary",
            imgsz=imgsz,
            conf=conf,
            augment=use_tta,
            diagnostics=runtime_diagnostics,
        )
    except Exception as exc:
        print(f"Pool YOLO inference failed (non-fatal): {_safe_inference_error(exc)}")
        return []

    detections, solar_boxes, class_names = _parse_pool_yolo_results(
        image,
        results,
        model_role="primary",
        include_solar=True,
        conf=conf,
    )
    _record_pool_yolo_class_names("primary", class_names)
    if runtime_diagnostics is not None:
        runtime_diagnostics["primary_class_names"] = class_names

    primary_pool_count = sum(det.get("detector") == "yolo" for det in detections)
    primary_solar_count = sum(det.get("detector") == "yolo_solar" for det in detections)
    if runtime_diagnostics is not None:
        runtime_diagnostics["primary_pool_candidates"] = int(
            runtime_diagnostics.get("primary_pool_candidates", 0)
        ) + primary_pool_count
        runtime_diagnostics["primary_solar_candidates"] = int(
            runtime_diagnostics.get("primary_solar_candidates", 0)
        ) + primary_solar_count

    # Secondary ensemble pass: union in the second model's POOL boxes only.
    # Its solar head is a distillation of the primary's, so solar stays
    # single-model.
    model_b = _get_pool_yolo_model_b() if _POOL_YOLO_ENSEMBLE else None
    if model_b is not None:
        try:
            imgsz_b = _POOL_YOLO_IMGSZ or getattr(model_b, "_qldr_imgsz", 640)
            results_b = _predict_pool_yolo_model(
                model_b,
                image,
                role="secondary",
                imgsz=imgsz_b,
                conf=conf,
                diagnostics=runtime_diagnostics,
            )
            secondary, _ignored_solar, secondary_names = _parse_pool_yolo_results(
                image,
                results_b,
                model_role="secondary",
                include_solar=False,
                conf=conf,
            )
            _record_pool_yolo_class_names("secondary", secondary_names)
            if runtime_diagnostics is not None:
                runtime_diagnostics["secondary_class_names"] = secondary_names
                runtime_diagnostics["secondary_pool_candidates"] = int(
                    runtime_diagnostics.get("secondary_pool_candidates", 0)
                ) + len(secondary)
            detections = _merge_secondary_pool_dets(detections, secondary)
        except Exception as exc:
            print(f"Secondary pool YOLO inference failed (non-fatal): {_safe_inference_error(exc)}")
    elif _POOL_YOLO_ENSEMBLE and resolve_pool_yolo_weights_b():
        runtime = get_pool_yolo_runtime_status().get("secondary") or {}
        error = runtime.get("error") or "RuntimeError: secondary pool model unavailable"
        _update_scan_inference_diagnostics(
            runtime_diagnostics,
            "secondary",
            success=False,
            error=error,
        )

    # A solar prediction is useful output, but it is not validated strongly enough
    # to be a hard pool veto. In particular, the primary can call a blue/turquoise
    # basin "solar" while the complementary model correctly calls it "pool". Every
    # pool proposal still has to pass the downstream water, size, and context gates,
    # so preserve the proposal and record the disagreement for calibration instead
    # of silently deleting a potentially real pool.
    solar_pool_overlaps = sum(
        1
        for det in detections
        if det.get("detector") == "yolo"
        and any(
            sx1 <= det["cx"] <= sx2 and sy1 <= det["cy"] <= sy2
            for sx1, sy1, sx2, sy2 in solar_boxes
        )
    )
    if runtime_diagnostics is not None:
        runtime_diagnostics["solar_pool_overlap_candidates"] = int(
            runtime_diagnostics.get("solar_pool_overlap_candidates", 0)
        ) + solar_pool_overlaps
        runtime_diagnostics["pool_candidates_after_ensemble"] = int(
            runtime_diagnostics.get("pool_candidates_after_ensemble", 0)
        ) + sum(det.get("detector") == "yolo" for det in detections)
    return detections


def _box_colour_stats(image: Any, x1: float, y1: float, x2: float, y2: float) -> dict[str, float]:
    """Colour fractions inside a box, used to verify YOLO pool proposals."""
    if _np is None:
        return {}
    try:
        h, w = image.shape[:2]
        xa, xb = max(0, int(x1)), min(w, int(x2))
        ya, yb = max(0, int(y1)), min(h, int(y2))
        if xb <= xa or yb <= ya:
            return {}
        roi = image[ya:yb, xa:xb]
        # image is BGR (cv2) or RGB (PIL fallback); both expose channel order via
        # split below. Stitched chunks are BGR, matching the colour-CV path.
        blue = roi[:, :, 0].astype("int32")
        green = roi[:, :, 1].astype("int32")
        red = roi[:, :, 2].astype("int32")
        mean_b, mean_g, mean_r = float(blue.mean()), float(green.mean()), float(red.mean())
        max_c = max(mean_b, mean_g, mean_r)
        min_c = min(mean_b, mean_g, mean_r)
        saturation = (max_c - min_c) / max_c * 255.0 if max_c > 0 else 0.0
        pixel_max = _np.maximum(_np.maximum(red, green), blue)
        pixel_min = _np.minimum(_np.minimum(red, green), blue)
        pixel_sat = (pixel_max - pixel_min) / _np.maximum(pixel_max, 1) * 255.0
        water_pixels = (
            (pixel_max >= 70)
            & (pixel_sat >= 18)
            & (
                (
                    (blue - red >= 12)
                    & (green - red >= 6)
                    & (blue - green >= -24)
                )
                | (
                    (green - red >= 16)
                    & (blue - red >= 2)
                    & (_np.abs(blue - green) <= 60)
                )
            )
        )
        strong_water_pixels = (
            (pixel_max >= 92)
            & (pixel_sat >= 28)
            & (blue - red >= 18)
            & (green - red >= 10)
            & (blue - green >= -18)
        )
        vegetation_pixels = (
            (green - red >= 14)
            & (green - blue >= 6)
            & (pixel_max >= 50)
        )
        warm_roof_pixels = (red - green >= 12) & (red - blue >= 14) & (pixel_max >= 70)
        dark_pixels = pixel_max <= 70
        return {
            "green_minus_red": round(mean_g - mean_r, 1),
            "blue_minus_red": round(mean_b - mean_r, 1),
            "blue_minus_green": round(mean_b - mean_g, 1),
            "saturation_mean": round(saturation, 1),
            "value_mean": round(max_c, 1),
            "water_fraction": round(float(water_pixels.mean()), 3),
            "strong_water_fraction": round(float(strong_water_pixels.mean()), 3),
            "vegetation_fraction": round(float(vegetation_pixels.mean()), 3),
            "warm_roof_fraction": round(float(warm_roof_pixels.mean()), 3),
            "dark_fraction": round(float(dark_pixels.mean()), 3),
        }
    except Exception:
        return {}


def _sam2_refine(image: Any, det: dict[str, Any], weights_path: str) -> list[tuple[float, float]] | None:
    """SAM2 box-prompted segmentation for a pool detection. Optional: requires
    ``pip install torch sam2`` and QLDR_SAM2_WEIGHTS pointing at a checkpoint.
    Returns an image-space (x, y) ring, or None when unavailable/unconfident."""
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
    except Exception:
        return None
    predictor = getattr(_sam2_refine, "_predictor", None)
    if predictor is None:
        predictor = SAM2ImagePredictor.from_pretrained(weights_path)
        _sam2_refine._predictor = predictor  # type: ignore[attr-defined]
    rgb = image[:, :, ::-1]  # BGR -> RGB
    predictor.set_image(rgb)
    x, y, w, h = int(det["x"]), int(det["y"]), int(det["w"]), int(det["h"])
    masks, scores, _ = predictor.predict(box=_np.array([x, y, x + w, y + h]), multimask_output=False)
    if masks is None or not len(masks) or float(scores[0]) < 0.5:
        return None
    mask = (masks[0].astype(_np.uint8)) * 255
    contours, _ = _cv2.findContours(mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return _contour_to_ring(max(contours, key=_cv2.contourArea))


def _contour_to_ring(contour: Any, max_points: int = 40) -> list[tuple[float, float]] | None:
    """Simplify an OpenCV contour to a short (x, y) ring for geometry output."""
    try:
        perimeter = _cv2.arcLength(contour, True)
        approx = _cv2.approxPolyDP(contour, max(1.2, 0.02 * perimeter), True)
        points = [(float(p[0][0]), float(p[0][1])) for p in approx]
        if len(points) < 3:
            return None
        return points[:max_points]
    except Exception:
        return None


def _refine_pool_outline_px(image: Any, det: dict[str, Any]) -> list[tuple[float, float]] | None:
    """True pool outline for a YOLO box detection (which is otherwise just a rectangle).

    Preferred refiner: SAM2, when weights are configured via ``QLDR_SAM2_WEIGHTS``
    (optional heavy dependency — never required). Fallback refiner: the largest
    blue/cyan water component inside the box, simplified to a polygon. Returns
    image-space (x, y) ring points, or ``None`` to keep the plain box.
    """
    # --- Optional SAM2 hook (never blocks; requires torch + sam2 + weights) ---
    sam2_weights = os.environ.get("QLDR_SAM2_WEIGHTS", "")
    if sam2_weights:
        try:
            ring = _sam2_refine(image, det, sam2_weights)  # type: ignore[name-defined]
            if ring:
                return ring
        except Exception:
            pass  # fall through to the classical refiner

    # --- Classical refiner: water-colour mask inside the (padded) box ---
    try:
        x, y, w, h = int(det["x"]), int(det["y"]), int(det["w"]), int(det["h"])
        pad = max(2, int(0.15 * max(w, h)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        blue = crop[:, :, 0].astype(_np.int16)  # BGR order from the tile stitcher
        green = crop[:, :, 1].astype(_np.int16)
        red = crop[:, :, 2].astype(_np.int16)
        brightness = crop.max(axis=2)
        # Blue/cyan water: blue or green lifted above red, and not near-black shadow.
        mask = (((blue - red) > 6) | ((green - red) > 8)) & (brightness > 55)
        mask = (mask.astype(_np.uint8)) * 255
        kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (5, 5))
        mask = _cv2.morphologyEx(mask, _cv2.MORPH_CLOSE, kernel)
        mask = _cv2.morphologyEx(mask, _cv2.MORPH_OPEN, kernel)
        contours, _ = _cv2.findContours(mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=_cv2.contourArea)
        # The water blob must fill a meaningful part of the box, or the "outline"
        # is just a sliver (heavy tree cover) — keep the honest rectangle instead.
        if _cv2.contourArea(largest) < 0.25 * (w * h) or _cv2.contourArea(largest) < 12:
            return None
        ring = _contour_to_ring(largest)
        if not ring:
            return None
        return [(px + x0, py + y0) for px, py in ring]
    except Exception:
        return None


def _yolo_box_is_watery(det: dict[str, Any], detector_precision: str = "fast") -> bool:
    """Light water-colour confirmation for a YOLO pool proposal.

    The pool-trained YOLO model still fires on large grey/white flat roofs and
    glass greenhouses. Those are colourless or warm, while pool water is blue or
    cyan, so a cheap colour check removes the dominant YOLO false positives
    without the strict thresholds that the colour-CV detector needs.

    Precision modes differ on purpose:
      - "fast"/"auto" (one-shot realtime): full strictness — there is no voting
        backstop, so a proposal must show clearly visible water.
      - "precise" (area scans): confident YOLO boxes are accepted with much less
        visible water. Real pools are routinely tree-overhung, shaded, or
        dark-bottomed (low water_fraction despite being obvious pools), and the
        scan pipeline has cross-pass voting, an 8-400 m² area gate, and
        building-context filters to keep precision — recall is the scarce
        resource here, so the colour gate only vetoes clear roof signatures.
    """
    green_minus_red = det.get("green_minus_red")
    blue_minus_red = det.get("blue_minus_red")
    saturation = det.get("saturation_mean")
    if green_minus_red is None or blue_minus_red is None or saturation is None:
        return True  # no colour info -> do not block the model
    gr = float(green_minus_red)
    br = float(blue_minus_red)
    sat = float(saturation)
    confidence = float(det.get("confidence", 0.0) or 0.0)
    water_fraction = float(det.get("water_fraction", 0.0) or 0.0)
    strong_water_fraction = float(det.get("strong_water_fraction", 0.0) or 0.0)
    vegetation_fraction = float(det.get("vegetation_fraction", 0.0) or 0.0)
    warm_roof_fraction = float(det.get("warm_roof_fraction", 0.0) or 0.0)
    dark_fraction = float(det.get("dark_fraction", 0.0) or 0.0)
    # Always-on roof vetoes (the historical false-positive disaster class):
    # Warm roofs / terracotta: red clearly dominant.
    if br <= -6.0 and gr <= -2.0:
        return False
    # Colourless grey/white roofs and concrete: almost no chroma and no
    # blue/green lift over red.
    if sat < 18.0 and br < 12.0 and gr < 12.0:
        return False
    # Require at least a hint of cyan/blue water (blue or green above red).
    if br < 6.0 and gr < 6.0:
        return False
    if warm_roof_fraction > 0.26 and water_fraction < 0.24:
        return False

    if detector_precision == "precise":
        # Scan mode: trust the model proportionally to its confidence; demand only
        # enough water pixels to rule out pure-roof boxes. Tree overhang and shade
        # legitimately hide most of a pool's water.
        if confidence >= 0.75:
            return water_fraction >= 0.06
        if confidence >= 0.55:
            return water_fraction >= 0.12
        if confidence >= 0.40:
            return water_fraction >= 0.20 or strong_water_fraction >= 0.08
        return water_fraction >= 0.30 or strong_water_fraction >= 0.12

    # One-shot realtime ("fast"/"auto"): full strictness, no voting backstop.
    if vegetation_fraction > 0.46 and strong_water_fraction < 0.12:
        return False
    if dark_fraction > 0.48 and water_fraction < 0.24:
        return False
    if water_fraction < 0.18:
        return False
    if water_fraction < 0.24 and strong_water_fraction < 0.08:
        return False
    if confidence < 0.45 and water_fraction < 0.34 and strong_water_fraction < 0.14:
        return False
    if confidence < 0.58 and water_fraction < 0.28 and strong_water_fraction < 0.12:
        return False
    return True


def _is_large_clear_water_pool(det: dict[str, Any], area_sqm: float) -> bool:
    """Accept genuinely large public / Olympic pools that the per-mode size caps
    would otherwise reject.

    A municipal pool is a large, bright, uniform, *strongly* blue basin — a colour
    signature that roofs, solar arrays, courts, and sports fields cannot fake. We
    therefore lift the area ceiling only when that strong-water signature is present,
    so large false positives are not admitted along with the real big pools.
    """
    # Public pools fragment along lane markings, so accept large *fragments* too
    # (200 m² is already far bigger than any backyard pool the size caps target).
    if area_sqm < 200.0 or area_sqm > 5000.0:
        return False
    # The large-area exemption must be stricter than the normal candidate path:
    # one confirmed feedback false positive had an extremely high generic water
    # fraction but only a blue-grey (not turquoise) colour signature. Reuse the
    # measured turquoise separation before allowing any size-cap bypass.
    if not _has_turquoise_water_signature(det):
        return False
    water_fraction = float(det.get("water_fraction", 0.0) or 0.0)
    value_mean = float(det.get("value_mean", 0.0) or 0.0)
    saturation_mean = float(det.get("saturation_mean", 0.0) or 0.0)
    green_minus_red = float(det.get("green_minus_red", 0.0) or 0.0)
    blue_minus_red = float(det.get("blue_minus_red", 0.0) or 0.0)
    # value_mean / saturation_mean / strong_water_fraction are only present on some
    # candidate types (colour-CV blobs don't carry strong_water_fraction); gate on
    # each only when it is actually populated. A genuine large basin is almost pure
    # pool-water colour, which roofs / solar / courts / fields cannot reach.
    if value_mean and value_mean < 90.0:
        return False
    if saturation_mean and saturation_mean < 38.0:
        return False
    strong_water = det.get("strong_water_fraction")
    if strong_water is not None and float(strong_water) < 0.18:
        return False
    return (
        water_fraction >= 0.82
        and blue_minus_red >= 30.0
        and green_minus_red >= 8.0
    )


# ---------------------------------------------------------------------------
# Pool size estimation (pixels → sqm)
# ---------------------------------------------------------------------------
def _estimate_pool_area_sqm(
    area_px: float,
    bbox: tuple[float, float, float, float],
    img_width: int,
    img_height: int,
) -> float:
    """Estimate real-world area in m² from pixel area."""
    south, west, north, east = bbox
    lat_mid = (south + north) / 2.0
    height_m = (north - south) * 111_320.0
    width_m = (east - west) * 111_320.0 * math.cos(math.radians(lat_mid))
    px_per_m_x = img_width / max(width_m, 1)
    px_per_m_y = img_height / max(height_m, 1)
    sqm = area_px / (px_per_m_x * px_per_m_y)
    return sqm


def _latlon_distance_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    """Return the approximate ground distance between two WGS84 coordinates."""
    lat_mid = math.radians((lat_a + lat_b) / 2.0)
    dx = (lon_b - lon_a) * 111_320.0 * math.cos(lat_mid)
    dy = (lat_b - lat_a) * 111_320.0
    return math.hypot(dx, dy)


def _pool_element_center(element: dict[str, Any]) -> tuple[float, float] | None:
    """Return a synthetic pool element centre as (lat, lon)."""
    geometry = element.get("geometry")
    if isinstance(geometry, list) and geometry:
        points = geometry[:-1] if len(geometry) > 1 and geometry[0] == geometry[-1] else geometry
        coords = [
            (float(point["lat"]), float(point["lon"]))
            for point in points
            if isinstance(point, dict) and "lat" in point and "lon" in point
        ]
        if coords:
            return (
                sum(lat for lat, _lon in coords) / len(coords),
                sum(lon for _lat, lon in coords) / len(coords),
            )
    if "lat" in element and "lon" in element:
        try:
            return float(element["lat"]), float(element["lon"])
        except (TypeError, ValueError):
            return None
    return None


def _deduplicate_detected_pool_elements(
    elements: list[dict[str, Any]],
    *,
    min_distance_m: float = 4.0,
) -> list[dict[str, Any]]:
    """Collapse duplicate satellite fragments while keeping the strongest box."""
    ranked = sorted(
        elements,
        key=lambda element: (
            float(element.get("tags", {}).get("confidence:yolo", "0") or 0),
            float(element.get("tags", {}).get("confidence:cv", "0") or 0),
            float(element.get("tags", {}).get("pool:area_sqm", "0") or 0),
        ),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    kept_centres: list[tuple[float, float]] = []
    for element in ranked:
        centre = _pool_element_center(element)
        if centre is None:
            kept.append(element)
            continue
        if any(_latlon_distance_m(centre[0], centre[1], lat, lon) < min_distance_m for lat, lon in kept_centres):
            continue
        kept.append(element)
        kept_centres.append(centre)
    return kept


def _has_turquoise_water_signature(det: dict[str, Any]) -> bool:
    """Reject blue-grey solar panels and roofs that mimic muted pool water.

    The single most common satellite false positive is a solar array or
    blue-grey roof: the blue channel is lifted over red, so a pure "blue" test
    accepts it. Real pool water is cyan/turquoise — *both* green and blue are
    clearly elevated over red. Measured against OSM-mapped pools, genuine pools
    sit at ``green_minus_red`` ~47-82 while panel/roof false positives cluster
    at ~9-20, so a green-elevation floor separates them with wide margin.

    Detections that predate this signature (e.g. the NumPy fallback path or
    older fixtures) omit the colour deltas; those default to "pass" so the gate
    only ever tightens, never loosens, behaviour.
    """
    green_minus_red = det.get("green_minus_red")
    blue_minus_red = det.get("blue_minus_red")
    if green_minus_red is None or blue_minus_red is None:
        return True
    # Cyan water needs green meaningfully above red (not just blue above red)
    # and blue genuinely present, ruling out flat-green turf and blue-grey roofs.
    if float(green_minus_red) < 22.0:
        return False
    if float(blue_minus_red) < 12.0:
        return False
    return True


def _passes_precise_pool_candidate(det: dict[str, Any], area_sqm: float) -> bool:
    """Size-aware precision gate for visible satellite pool detections.

    Tiny spa/plunge pools are allowed, but only with very strong water and
    compact-shape evidence. Standard, lap, and large pools get different gates
    so we do not solve false positives by simply deleting whole size classes.
    """
    if not _has_turquoise_water_signature(det):
        return False
    confidence = float(det.get("confidence", 0.0) or 0.0)
    water_fraction = float(det.get("water_fraction", 0.0) or 0.0)
    circularity = float(det.get("circularity", 0.0) or 0.0)
    aspect = float(det.get("aspect_ratio", 99.0) or 99.0)
    extent = float(det.get("extent", 0.0) or 0.0)
    rotated_extent = float(det.get("rotated_extent", 0.0) or 0.0)
    value_mean = float(det.get("value_mean", 0.0) or 0.0)
    saturation_mean = float(det.get("saturation_mean", 0.0) or 0.0)
    roof_context = float(det.get("context_roof_fraction", 0.0) or 0.0)
    dark_panel_context = float(det.get("context_dark_panel_fraction", 0.0) or 0.0)

    if area_sqm < 3.0 or area_sqm > 1800.0:
        return False
    if value_mean and value_mean < 96:
        return False
    if saturation_mean and saturation_mean < 30:
        return False
    if roof_context > 0.46:
        return False
    if dark_panel_context > 0.44 and water_fraction < 0.88:
        return False

    if area_sqm < 8.0:
        if roof_context > 0.22 or dark_panel_context > 0.28:
            return False
        return (
            confidence >= 0.94
            and water_fraction >= 0.84
            and circularity >= 0.50
            and aspect <= 3.8
            and extent >= 0.50
            and rotated_extent >= 0.58
        )

    if area_sqm <= 30.0:
        if roof_context > 0.30 and water_fraction < 0.88:
            return False
        if dark_panel_context > 0.34 and confidence < 0.95:
            return False
        return (
            confidence >= 0.90
            and water_fraction >= 0.72
            and circularity >= 0.34
            and aspect <= 5.2
            and extent >= 0.40
            and rotated_extent >= 0.48
        )

    if area_sqm <= 220.0:
        if roof_context > 0.36 and confidence < 0.94:
            return False
        if dark_panel_context > 0.38 and water_fraction < 0.84:
            return False
        standard_pool = (
            confidence >= 0.88
            and water_fraction >= 0.66
            and circularity >= 0.27
            and aspect <= 6.5
            and extent >= 0.34
            and rotated_extent >= 0.44
        )
        lap_pool = (
            confidence >= 0.92
            and water_fraction >= 0.76
            and 5.0 < aspect <= 9.0
            and extent >= 0.42
            and rotated_extent >= 0.54
        )
        return standard_pool or lap_pool

    if area_sqm <= 650.0:
        if roof_context > 0.34 or dark_panel_context > 0.36:
            return False
        return (
            confidence >= 0.90
            and water_fraction >= 0.70
            and circularity >= 0.22
            and aspect <= 8.0
            and extent >= 0.32
            and rotated_extent >= 0.42
        )

    if roof_context > 0.30 or dark_panel_context > 0.32:
        return False
    return (
        confidence >= 0.94
        and water_fraction >= 0.78
        and circularity >= 0.28
        and aspect <= 6.5
        and extent >= 0.40
        and rotated_extent >= 0.50
    )


def _passes_auto_pool_candidate(det: dict[str, Any], area_sqm: float) -> bool:
    """Default-mode false-positive gate for satellite pool detections.

    Auto scans run the loosest colour segmentation, so blue/grey roofs, dark
    solar arrays, shadows, and blue tarps slip past the bare ``confidence``
    threshold that used to be the only auto-mode filter. This gate rejects the
    common non-pool blobs using the colour/context evidence already attached to
    each detection, while staying deliberately looser than the manual
    ``precise`` verifier so genuine, muted-tile pools still survive.
    """
    confidence = float(det.get("confidence", 0.0) or 0.0)
    water_fraction = float(det.get("water_fraction", 0.0) or 0.0)
    circularity = float(det.get("circularity", 0.0) or 0.0)
    aspect = float(det.get("aspect_ratio", 99.0) or 99.0)
    extent = float(det.get("extent", 0.0) or 0.0)
    rotated_extent = float(det.get("rotated_extent", 0.0) or 0.0)
    value_mean = float(det.get("value_mean", 0.0) or 0.0)
    saturation_mean = float(det.get("saturation_mean", 0.0) or 0.0)
    roof_context = float(det.get("context_roof_fraction", 0.0) or 0.0)
    dark_panel_context = float(det.get("context_dark_panel_fraction", 0.0) or 0.0)

    # Cyan/turquoise colour signature — rejects blue-grey solar panels and roofs.
    if not _has_turquoise_water_signature(det):
        return False
    # Shadows and very dark blobs are not sunlit pool water.
    if value_mean and value_mean < 76:
        return False
    # Washed-out grey roofs / bare concrete carry almost no colour.
    if saturation_mean and saturation_mean < 20:
        return False
    # A blob sitting inside a roof footprint is almost always a blue roof.
    if roof_context > 0.48:
        return False
    # Dark solar arrays read as muted blue; only accept them with strong water.
    if dark_panel_context > 0.48 and water_fraction < 0.85:
        return False
    # The auto colour mask accepts water_fraction down to ~0.34, which is too
    # loose for an unsupervised box. Require a clear majority of water pixels.
    if water_fraction < 0.50:
        return False
    # Reject ragged / sparse / extremely elongated blobs (road paint, surf,
    # tile-edge slivers) that real pools do not produce.
    if circularity < 0.16:
        return False
    if extent < 0.20 or rotated_extent < 0.30:
        return False
    if aspect > 8.0:
        return False
    # A wide-area blob that is only weakly water-coloured is usually shaded
    # turf or a tarp rather than a backyard pool.
    if area_sqm > 250.0 and water_fraction < 0.62 and confidence < 0.85:
        return False
    return True


def _passes_strict_water_supplement_candidate(det: dict[str, Any], area_sqm: float, detector_precision: str) -> bool:
    """Verifier for the high-recall clear-water supplement used with YOLO.

    The colour detector reports ``water_fraction`` over its own mask, so many
    blue roof/panel fragments look perfect unless we also check scene context
    and a genuine pool colour signature.
    """
    confidence = float(det.get("confidence", 0.0) or 0.0)
    water_fraction = float(det.get("water_fraction", 0.0) or 0.0)
    circularity = float(det.get("circularity", 0.0) or 0.0)
    aspect = float(det.get("aspect_ratio", 99.0) or 99.0)
    extent = float(det.get("extent", 0.0) or 0.0)
    rotated_extent = float(det.get("rotated_extent", 0.0) or 0.0)
    value_mean = float(det.get("value_mean", 0.0) or 0.0)
    saturation_mean = float(det.get("saturation_mean", 0.0) or 0.0)
    green_minus_red = float(det.get("green_minus_red", 0.0) or 0.0)
    blue_minus_red = float(det.get("blue_minus_red", 0.0) or 0.0)
    roof_context = float(det.get("context_roof_fraction", 0.0) or 0.0)
    dark_panel_context = float(det.get("context_dark_panel_fraction", 0.0) or 0.0)

    fast = detector_precision == "fast"
    # Measured on Rose Bay/Dover Heights: real pool water is BRIGHT and STRONGLY
    # blue (value_mean med ~179, blue_minus_red med ~64) while the tree / shaded
    # lawn / grey-roof false positives are darker and only weakly blue
    # (value_mean med ~101, blue_minus_red med ~32). The realtime "recovery" pass
    # therefore only accepts obvious bright blue pools; YOLO + OSM cover the rest.
    if area_sqm < 5.0:
        return False
    # A large public/Olympic pool (or a lane-split fragment of one) is conclusively a
    # pool from size + near-pure water colour alone. Accept it here, bypassing the
    # brightness / shape gates that only matter for small backyard pools, but still
    # reject anything with meaningful roof or dark-panel context.
    if _is_large_clear_water_pool(det, area_sqm) and roof_context <= 0.30 and dark_panel_context <= 0.26:
        return True
    # Cap the area for ordinary backyard pools.
    if area_sqm > (260.0 if fast else 650.0):
        return False
    if water_fraction < (0.85 if fast else 0.78):
        return False
    # Bright clear water, OR a dimmer pool that is unmistakably deep blue
    # (shaded backyard pools stay dark but keep a very strong blue cast that
    # tree/roof false positives never reach).
    bright = value_mean >= (135.0 if fast else 112.0)
    deep_blue = value_mean >= (116.0 if fast else 100.0) and blue_minus_red >= 52.0
    if value_mean and not (bright or deep_blue):
        return False
    if saturation_mean and saturation_mean < (58.0 if fast else 46.0):
        return False
    if blue_minus_red < (40.0 if fast else 26.0):
        return False
    if green_minus_red < (14.0 if fast else 10.0):
        return False
    if circularity < (0.24 if fast else 0.20):
        return False
    if extent < (0.30 if fast else 0.24):
        return False
    if rotated_extent < (0.42 if fast else 0.34):
        return False
    if aspect > (6.2 if fast else 8.5):
        return False
    if roof_context > (0.30 if fast else 0.38):
        return False
    if roof_context > 0.22 and blue_minus_red < 50.0:
        return False
    if dark_panel_context > (0.26 if fast else 0.34):
        return False
    if dark_panel_context > 0.18 and blue_minus_red < 50.0:
        return False
    if area_sqm < 12.0 and (confidence < 0.92 or water_fraction < 0.94):
        return False
    return True


def _save_rejected_candidate_crop(
    image: Any,
    det: dict[str, Any],
    *,
    reason: str,
    chunk_index: int,
    tile_zoom: int,
) -> bool:
    """Quarantine a rejected crop for human review without labelling it.

    A runtime rejection is not ground truth: threshold and verifier mistakes
    can reject genuine pools. Captured crops therefore go to a review queue
    with metadata and deliberately receive no YOLO ``.txt`` file. Only a human
    confirmed non-pool may later be promoted into a hard-negative dataset.
    """
    global _negative_crops_saved_this_run
    if not _SAVE_REJECTED_POOL_CROPS or _np is None:
        return False
    with _negative_crop_lock:
        if _negative_crops_saved_this_run >= _MAX_NEGATIVE_CROPS_PER_RUN:
            return False
        _negative_crops_saved_this_run += 1
        sequence = _negative_crops_saved_this_run

    try:
        arr = _np.asarray(image)
        if arr.ndim != 3 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
            return False
        img_h, img_w = arr.shape[:2]
        x = int(max(0, math.floor(float(det.get("x", 0)))))
        y = int(max(0, math.floor(float(det.get("y", 0)))))
        w = int(max(1, math.ceil(float(det.get("w", 1)))))
        h = int(max(1, math.ceil(float(det.get("h", 1)))))
        pad = max(16, int(max(w, h) * 1.25))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(img_w, x + w + pad)
        y1 = min(img_h, y + h + pad)
        if x1 <= x0 or y1 <= y0:
            return False

        crop = arr[y0:y1, x0:x1]
        digest = hashlib.sha1(
            f"{time.time_ns()}:{chunk_index}:{tile_zoom}:{x}:{y}:{w}:{h}:{reason}:{sequence}".encode("utf-8")
        ).hexdigest()[:14]
        safe_reason = "".join(ch if ch.isalnum() else "_" for ch in reason.lower()).strip("_")[:40] or "reject"
        root = Path(_HARD_NEGATIVE_DIR)
        image_dir = root / "review_queue" / "images"
        metadata_dir = root / "review_queue" / "metadata"
        image_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        stem = f"candidate_{safe_reason}_{digest}"
        image_path = image_dir / f"{stem}.jpg"
        metadata_path = metadata_dir / f"{stem}.json"

        if _HAS_CV2 and _cv2 is not None:
            if not _cv2.imwrite(str(image_path), crop):
                return False
        elif _Image is not None:
            _Image.fromarray(crop).save(image_path, quality=92)
        else:
            return False

        numeric_fields = (
            "x", "y", "w", "h", "area_px", "confidence", "circularity",
            "aspect_ratio", "extent", "rotated_extent", "water_fraction",
            "strong_water_fraction", "value_mean", "saturation_mean",
            "green_minus_red", "blue_minus_red", "context_roof_fraction",
            "context_dark_panel_fraction", "context_vegetation_fraction",
        )
        candidate: dict[str, float] = {}
        for key in numeric_fields:
            if det.get(key) is None:
                continue
            try:
                candidate[key] = round(float(det[key]), 6)
            except (TypeError, ValueError):
                continue
        metadata = {
            "review_status": "unreviewed",
            "eligible_for_training": False,
            "training_label_created": False,
            "reason": reason,
            "chunk_index": int(chunk_index),
            "tile_zoom": int(tile_zoom),
            "image": image_path.name,
            "candidate": candidate,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _normalize_analysis_polygon(
    raw_polygon: list[list[float] | tuple[float, float]] | None,
) -> list[tuple[float, float]] | None:
    if raw_polygon is None:
        return None
    if not isinstance(raw_polygon, list) or not 3 <= len(raw_polygon) <= 200:
        raise ValueError("analysis_polygon must contain between 3 and 200 [longitude, latitude] pairs.")
    polygon: list[tuple[float, float]] = []
    for point in raw_polygon:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each analysis_polygon coordinate must be [longitude, latitude].")
        lon, lat = float(point[0]), float(point[1])
        if not (math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("analysis_polygon coordinates must be finite WGS84 coordinates.")
        polygon.append((lon, lat))
    if len(polygon) > 3 and polygon[0] == polygon[-1]:
        polygon.pop()
    return polygon


def _point_in_analysis_polygon(lon: float, lat: float, polygon: list[tuple[float, float]]) -> bool:
    """Inclusive point-in-polygon test used as the final inference boundary."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        cross = (lon - xi) * (yj - yi) - (lat - yi) * (xj - xi)
        if (
            abs(cross) <= 1e-12
            and min(xi, xj) - 1e-12 <= lon <= max(xi, xj) + 1e-12
            and min(yi, yj) - 1e-12 <= lat <= max(yi, yj) + 1e-12
        ):
            return True
        if (yi > lat) != (yj > lat):
            intersect_lon = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < intersect_lon:
                inside = not inside
        j = i
    return inside


def _mask_image_to_analysis_polygon(
    image: Any,
    actual_bbox: tuple[float, float, float, float],
    polygon: list[tuple[float, float]],
) -> tuple[Any, int]:
    """Black out every source pixel outside the selected analysis polygon.

    Pixel/geographic conversion deliberately mirrors ``_pixel_to_latlon`` so
    detections retain their original coordinates. Polygon edges are filled
    inclusively; final centre clipping remains a second boundary safeguard.
    """
    if _np is None:
        raise RuntimeError("NumPy is required to enforce the analysis-polygon mask")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return image, 0
    south, west, north, east = _normalize_bbox(actual_bbox)
    lon_span = east - west
    lat_span = north - south
    if lon_span <= 0 or lat_span <= 0:
        return _np.zeros_like(image), 0

    pixel_polygon = [
        (
            (lon - west) / lon_span * max(0, width - 1),
            (north - lat) / lat_span * max(0, height - 1),
        )
        for lon, lat in polygon
    ]
    mask = _np.zeros((height, width), dtype=_np.uint8)
    try:
        if _HAS_CV2 and _cv2 is not None:
            points = _np.asarray(
                [[round(x), round(y)] for x, y in pixel_polygon],
                dtype=_np.int32,
            )
            _cv2.fillPoly(mask, [points], 255)
        else:
            from PIL import Image as PILImage
            from PIL import ImageDraw

            pil_mask = PILImage.new("L", (width, height), 0)
            ImageDraw.Draw(pil_mask).polygon(pixel_polygon, fill=255, outline=255)
            mask = _np.asarray(pil_mask, dtype=_np.uint8).copy()
    except Exception as exc:
        # Failing closed is essential: never fall back to exposing pixels
        # outside the selected area to an inference backend.
        raise RuntimeError(f"could not create analysis-polygon image mask: {exc}") from exc

    inside_pixels = int(_np.count_nonzero(mask))
    masked = image.copy()
    masked[mask == 0] = 0
    return masked, inside_pixels


def detect_pools_in_bbox(
    bbox: tuple[float, float, float, float],
    *,
    zoom: int = 18,
    detector_model: str = DEFAULT_DETECTOR_MODEL,
    segmentation_mode: str = "auto",
    detector_precision: str = "auto",
    analysis_polygon: list[list[float] | tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    """Detect pools in a geographic bounding box using satellite imagery.

    Returns a list of OSM-style element dicts, each representing a detected
    pool with YOLO metadata tags.

    Parameters
    ----------
    bbox : (south, west, north, east)
    zoom : satellite tile zoom level (17-20 recommended; precise mode can use z20)
    detector_model : YOLO model variant name
    segmentation_mode : 'auto', 'sam', or 'bbox'

    Returns
    -------
    list[dict]  – elements compatible with the ``fetch_osm_features`` format.
    """
    started_at = time.time()
    _lazy_import()
    detector_model = normalize_detector_model(detector_model)
    detector_precision = _normalize_detector_precision(detector_precision)
    request_bbox = _normalize_bbox(bbox)
    selection_polygon = _normalize_analysis_polygon(analysis_polygon)
    detector_signature = get_detector_signature()
    diagnostics: dict[str, Any] = {
        "bbox": [round(v, 7) for v in request_bbox],
        "requested_zoom": int(zoom),
        "context_detector_model": detector_model,
        "detector_signature": detector_signature["signature"],
        "active_pool_models": detector_signature["weights"],
        "selection_scoped": bool(selection_polygon),
        "analysis_mask_applied": bool(selection_polygon),
        "analysis_mask_pixels": 0,
        "segmentation_mode": segmentation_mode,
        "detector_precision": detector_precision,
        "chunks_planned": 0,
        "chunks_scanned": 0,
        "chunks_with_tiles": 0,
        "tiles_attempted": 0,
        "tiles_loaded": 0,
        "raw_candidates": 0,
        "primary_pool_candidates": 0,
        "primary_solar_candidates": 0,
        "secondary_pool_candidates": 0,
        "pool_candidates_after_ensemble": 0,
        "solar_pool_overlap_candidates": 0,
        "rejected_candidates": 0,
        "negative_crops_saved": 0,
        "pool_elements": 0,
        "truncated": False,
        "reason": "not_started",
    }
    if _np is None or not (_HAS_CV2 or _Image is not None):
        diagnostics["reason"] = "missing_cv_dependencies"
        diagnostics["elapsed_ms"] = round((time.time() - started_at) * 1000)
        _set_last_detector_diagnostics(diagnostics)
        return []

    # Use the highest practical zoom for small pools, while backing off one
    # level for unusually broad bboxes so the scan covers the whole request.
    tile_zoom = _choose_tile_zoom(request_bbox, zoom, detector_precision)
    diagnostics["tile_zoom"] = tile_zoom
    diagnostics["planned_tiles"] = _planned_tile_count(request_bbox, tile_zoom)
    chunks = _iter_detection_bboxes(request_bbox, tile_zoom)
    diagnostics["chunks_planned"] = len(chunks)
    diagnostics["truncated"] = _planned_chunk_count(request_bbox, tile_zoom, _MAX_STITCH_TILE_SIDE) > _MAX_DETECTION_CHUNKS
    time_budget = _FAST_SCAN_TIME_BUDGET_SECONDS if detector_precision == "fast" else None

    fast_cursor_key = ""
    fast_cursor_offset = 0
    if detector_precision == "fast" and len(chunks) > 1:
        fast_cursor_key = ":".join(str(value) for value in [tile_zoom, *[round(v, 5) for v in request_bbox]])
        with _fast_scan_cursor_lock:
            fast_cursor_offset = _fast_scan_cursors.get(fast_cursor_key, 0) % len(chunks)
        if fast_cursor_offset:
            chunks = chunks[fast_cursor_offset:] + chunks[:fast_cursor_offset]
        diagnostics["fast_chunk_cursor_offset"] = fast_cursor_offset

    if not chunks:
        diagnostics["reason"] = "empty_bbox"
        diagnostics["elapsed_ms"] = round((time.time() - started_at) * 1000)
        _set_last_detector_diagnostics(diagnostics)
        return []

    elements: list[dict[str, Any]] = []
    element_id = 950_000_000  # synthetic id space for CV detections
    seen_centres: set[tuple[int, int]] = set()
    # Solar arrays detected by the same multi-class model pass (no extra
    # inference); published as power=generator elements alongside pools.
    solar_elements: list[dict[str, Any]] = []
    seen_solar_centres: set[tuple[int, int]] = set()
    south, west, north, east = request_bbox

    # Detector backend selection. A pool-trained YOLO model (if configured) is
    # the trusted visual detector for new/unmapped pools. The colour-CV water
    # segmentation is unreliable (solar panels, roofs, and shaded turf read as
    # water), so it is only used when no pool model exists AND the operator has
    # explicitly opted into the experimental fallback.
    pool_yolo_active = pool_yolo_available()
    run_cv_detector = _ENABLE_CV_POOL_FALLBACK and not pool_yolo_active
    diagnostics["pool_yolo_active"] = pool_yolo_active
    diagnostics["cv_fallback_active"] = run_cv_detector
    if _STRICT_WATER_SUPPLEMENT_MODE == "1":
        strict_water_on = True
    elif _STRICT_WATER_SUPPLEMENT_MODE in {"0", "false", "off"}:
        strict_water_on = False
    else:  # auto: skip the supplement when the ensemble's second model is active
        strict_water_on = not (_POOL_YOLO_ENSEMBLE and resolve_pool_yolo_weights_b())
    diagnostics["strict_water_supplement_active"] = bool(
        pool_yolo_active and strict_water_on and detector_precision in {"fast", "precise"}
    )
    if not pool_yolo_active and not run_cv_detector:
        diagnostics["reason"] = "no_pool_model_configured"
        diagnostics["elapsed_ms"] = round((time.time() - started_at) * 1000)
        diagnostics["note"] = (
            "No pool-trained YOLO weights found and colour-CV fallback is disabled; "
            "mapped pools are served from OpenStreetMap. Train weights via "
            "scripts/pool_yolo_pipeline.py to detect new/unmapped pools."
        )
        _set_last_detector_diagnostics(diagnostics)
        return []

    for chunk_index, chunk_bbox in enumerate(chunks, start=1):
        if time_budget is not None and time.time() - started_at >= time_budget:
            diagnostics["truncated"] = True
            diagnostics["reason"] = "time_budget_exhausted"
            break
        diagnostics["chunks_scanned"] += 1
        result = _stitch_tiles_with_stats(chunk_bbox, zoom=tile_zoom)
        if result is None:
            diagnostics["tiles_attempted"] += _planned_tile_count(chunk_bbox, tile_zoom)
            continue
        image, actual_bbox, tile_stats = result
        diagnostics["chunks_with_tiles"] += 1
        diagnostics["tiles_attempted"] += tile_stats.get("attempted_tiles", 0)
        diagnostics["tiles_loaded"] += tile_stats.get("loaded_tiles", 0)
        img_h, img_w = image.shape[:2]
        if selection_polygon:
            try:
                image, inside_pixels = _mask_image_to_analysis_polygon(
                    image,
                    actual_bbox,
                    selection_polygon,
                )
            except Exception as exc:
                diagnostics["analysis_mask_error"] = _safe_inference_error(exc)
                diagnostics["inference_failures"] = int(diagnostics.get("inference_failures", 0)) + 1
                continue
            diagnostics["analysis_mask_pixels"] = int(diagnostics.get("analysis_mask_pixels", 0)) + inside_pixels
            if inside_pixels <= 0:
                continue

        # Gather candidates for this chunk. Pool-trained YOLO (when configured)
        # is the trusted detector; colour-CV is only added as an opt-in fallback.
        chunk_detections: list[dict[str, Any]] = []
        if pool_yolo_active:
            yolo_conf = _POOL_YOLO_MIN_CONFIDENCE
            if detector_precision == "fast":
                yolo_conf = max(yolo_conf, 0.34)
            elif detector_precision == "auto":
                yolo_conf = max(yolo_conf, 0.32)
            chunk_detections.extend(
                _yolo_detect_pools(
                    image,
                    conf=yolo_conf,
                    runtime_diagnostics=diagnostics,
                )
            )
        if diagnostics["strict_water_supplement_active"]:
            strict_water = _detect_blue_pools_cv(
                image,
                min_area_px=12,
                # Allow large public/Olympic-pool blobs through; the strict colour
                # gates (and _is_large_clear_water_pool) still reject non-water.
                max_area_px=320000,
                detector_precision="fast",
            )
            for det in strict_water:
                det["detector"] = "strict_water_cv"
                det["confidence"] = max(float(det.get("confidence", 0.0) or 0.0), 0.90)
            chunk_detections.extend(strict_water)
        if run_cv_detector:
            # Keep tiny plunge/spa pools by allowing candidates down to a few
            # square metres after geospatial size conversion.
            chunk_detections.extend(_detect_blue_pools_cv(
                image,
                min_area_px=8,
                max_area_px=320000,
                detector_precision=detector_precision,
            ))
        diagnostics["raw_candidates"] += len(chunk_detections)
        if not chunk_detections:
            continue

        # Optional YOLO context validation. It is not the primary detector, but
        # when available it can slightly lift confidence near mapped structures.
        yolo_structures = []
        if _ENABLE_YOLO_CONTEXT and _HAS_YOLO:
            yolo_structures = _yolo_detect_structures(image, model_name=detector_model)

        for det in chunk_detections:
            is_yolo_det = det.get("detector") == "yolo"
            is_strict_water_det = det.get("detector") == "strict_water_cv"
            def reject_candidate(reason: str) -> None:
                diagnostics["rejected_candidates"] = int(diagnostics.get("rejected_candidates", 0)) + 1
                reasons = diagnostics.setdefault("reject_reasons", {})
                reasons[reason] = int(reasons.get(reason, 0)) + 1
                if _save_rejected_candidate_crop(
                    image,
                    det,
                    reason=reason,
                    chunk_index=chunk_index,
                    tile_zoom=tile_zoom,
                ):
                    diagnostics["negative_crops_saved"] = int(diagnostics.get("negative_crops_saved", 0)) + 1

            cx, cy = det["cx"], det["cy"]
            lat, lon = _pixel_to_latlon(cx, cy, actual_bbox[2], actual_bbox[1], img_w, img_h, actual_bbox)

            if lat < south or lat > north or lon < west or lon > east:
                continue
            if selection_polygon and not _point_in_analysis_polygon(lon, lat, selection_polygon):
                continue

            if det.get("detector") == "yolo_solar":
                # Solar arrays skip the pool gates entirely: no water requirement,
                # rooftop context is expected, and size spans a couple of panels
                # up to commercial arrays.
                solar_area_sqm = _estimate_pool_area_sqm(det["area_px"], actual_bbox, img_w, img_h)
                if not (2.0 <= solar_area_sqm <= 2500.0):
                    continue
                solar_key = (round(lat * 100000), round(lon * 100000))
                if solar_key in seen_solar_centres:
                    continue
                seen_solar_centres.add(solar_key)
                sx, sy, sw, sh = det["x"], det["y"], det["w"], det["h"]
                s_nw = _pixel_to_latlon(sx, sy, actual_bbox[2], actual_bbox[1], img_w, img_h, actual_bbox)
                s_ne = _pixel_to_latlon(sx + sw, sy, actual_bbox[2], actual_bbox[1], img_w, img_h, actual_bbox)
                s_se = _pixel_to_latlon(sx + sw, sy + sh, actual_bbox[2], actual_bbox[1], img_w, img_h, actual_bbox)
                s_sw = _pixel_to_latlon(sx, sy + sh, actual_bbox[2], actual_bbox[1], img_w, img_h, actual_bbox)
                element_id += 1
                weight_info = next(
                    (item for item in detector_signature["weights"] if item.get("role") == "primary"),
                    None,
                ) or {}
                solar_elements.append({
                    "type": "way",
                    "id": element_id,
                    "tags": {
                        "power": "generator",
                        "generator:source": "solar",
                        "generator:method": "photovoltaic",
                        "detector:source": "satellite-yolo-solar",
                        "detector:model": weight_info.get("name") or "pool-trained-yolo",
                        "detector:model_role": "primary",
                        "detector:weights_sha256": weight_info.get("sha256") or "unknown",
                        "detector:signature": detector_signature["signature"],
                        "detector:precision": detector_precision,
                        "detector:tile_zoom": str(tile_zoom),
                        "confidence:yolo": f"{det['confidence']:.2f}",
                        "solar:area_sqm": f"{solar_area_sqm:.1f}",
                    },
                    "geometry": [
                        {"lat": round(s_sw[0], 7), "lon": round(s_sw[1], 7)},
                        {"lat": round(s_se[0], 7), "lon": round(s_se[1], 7)},
                        {"lat": round(s_ne[0], 7), "lon": round(s_ne[1], 7)},
                        {"lat": round(s_nw[0], 7), "lon": round(s_nw[1], 7)},
                        {"lat": round(s_sw[0], 7), "lon": round(s_sw[1], 7)},
                    ],
                })
                continue

            if detector_precision == "fast":
                min_pool_area_sqm, max_pool_area_sqm, min_confidence = 8.0, 220.0, 0.88
            elif detector_precision == "auto":
                min_pool_area_sqm, max_pool_area_sqm, min_confidence = 7.0, 420.0, 0.76
            else:
                min_pool_area_sqm, max_pool_area_sqm, min_confidence = 3.0, 1800.0, 0.0
            area_sqm = _estimate_pool_area_sqm(det["area_px"], actual_bbox, img_w, img_h)
            # A genuinely large public/Olympic pool, confirmed by a strong clear-water
            # signature, is exempt from the per-mode size cap and the fast-mode
            # tile-edge rejection (big pools routinely fill / touch a chunk edge).
            big_pool = _is_large_clear_water_pool(det, area_sqm)
            if detector_precision == "fast" and not is_yolo_det:
                if float(det.get("aspect_ratio", 0.0)) > 6.2:
                    reject_candidate("fast_aspect_ratio")
                    continue
                if not big_pool and (
                    det["x"] <= 3 or det["y"] <= 3 or det["x"] + det["w"] >= img_w - 3 or det["y"] + det["h"] >= img_h - 3
                ):
                    reject_candidate("fast_tile_edge")
                    continue
                if float(det.get("water_fraction", 0.0)) < 0.62:
                    reject_candidate("fast_water_fraction")
                    continue
            # Large public/Olympic pools exceed the per-mode size cap; admit them only
            # when the strong clear-water signature confirms a real big pool.
            oversized_pool = area_sqm > max_pool_area_sqm and big_pool
            if area_sqm < min_pool_area_sqm or (area_sqm > max_pool_area_sqm and not oversized_pool):
                reject_candidate("size_range")
                continue
            if not is_yolo_det and float(det.get("confidence", 0.0)) < min_confidence:
                reject_candidate("confidence")
                continue
            # Colour/shape verifiers apply to colour-CV candidates. A
            # pool-trained YOLO box is trusted on model confidence only after a
            # water-pixel confirmation to drop lawn, roof, and shadow boxes.
            if is_yolo_det:
                if not _yolo_box_is_watery(det, detector_precision):
                    reject_candidate("yolo_not_watery")
                    continue
            elif is_strict_water_det:
                if not _passes_strict_water_supplement_candidate(det, area_sqm, detector_precision):
                    reject_candidate("strict_water_verifier")
                    continue
            else:
                if detector_precision == "precise" and not _passes_precise_pool_candidate(det, area_sqm):
                    reject_candidate("precise_verifier")
                    continue
                if detector_precision == "auto" and not _passes_auto_pool_candidate(det, area_sqm):
                    reject_candidate("auto_verifier")
                    continue

            centre_key = (round(lat * 100000), round(lon * 100000))
            if centre_key in seen_centres:
                continue
            seen_centres.add(centre_key)

            confidence = det["confidence"]
            if yolo_structures:
                min_dist = float("inf")
                for struct in yolo_structures:
                    dx = cx - struct["cx"]
                    dy = cy - struct["cy"]
                    min_dist = min(min_dist, math.hypot(dx, dy))
                if min_dist < 200:
                    confidence = min(0.99, confidence + 0.08)
                elif min_dist < 400:
                    confidence = min(0.99, confidence + 0.04)

            x, y, w, h = det["x"], det["y"], det["w"], det["h"]

            # True segmented outline where available: colour-CV detections carry their
            # own contour; YOLO boxes are refined against the water mask (or SAM2 when
            # configured). Falls back to the plain bounding box.
            ring_px: list[tuple[float, float]] | None = None
            if det.get("contour") is not None:
                ring_px = _contour_to_ring(det["contour"])
            elif is_yolo_det:
                ring_px = _refine_pool_outline_px(image, det)

            if ring_px and len(ring_px) >= 3:
                geometry = [
                    {
                        "lat": round(pll[0], 7),
                        "lon": round(pll[1], 7),
                    }
                    for pll in (
                        _pixel_to_latlon(px, py, actual_bbox[2], actual_bbox[1], img_w, img_h, actual_bbox)
                        for px, py in ring_px
                    )
                ]
                geometry.append(dict(geometry[0]))  # close the ring
            else:
                nw_lat, nw_lon = _pixel_to_latlon(x, y, actual_bbox[2], actual_bbox[1], img_w, img_h, actual_bbox)
                ne_lat, ne_lon = _pixel_to_latlon(x + w, y, actual_bbox[2], actual_bbox[1], img_w, img_h, actual_bbox)
                se_lat, se_lon = _pixel_to_latlon(x + w, y + h, actual_bbox[2], actual_bbox[1], img_w, img_h, actual_bbox)
                sw_lat, sw_lon = _pixel_to_latlon(x, y + h, actual_bbox[2], actual_bbox[1], img_w, img_h, actual_bbox)
                geometry = [
                    {"lat": round(sw_lat, 7), "lon": round(sw_lon, 7)},
                    {"lat": round(se_lat, 7), "lon": round(se_lon, 7)},
                    {"lat": round(ne_lat, 7), "lon": round(ne_lon, 7)},
                    {"lat": round(nw_lat, 7), "lon": round(nw_lon, 7)},
                    {"lat": round(sw_lat, 7), "lon": round(sw_lon, 7)},
                ]

            element_id += 1
            model_role = str(det.get("model_role") or "primary")
            weight_info = next(
                (item for item in detector_signature["weights"] if item.get("role") == model_role),
                None,
            ) or {}
            actual_model = (
                weight_info.get("name")
                if is_yolo_det
                else "strict-clear-water-v1"
                if is_strict_water_det
                else "satellite-pool-water-v2"
            )
            elements.append({
                "type": "way",
                "id": element_id,
                "tags": {
                    "leisure": "swimming_pool",
                    "detector:yolo": "yes" if is_yolo_det else ("validator" if yolo_structures else "no"),
                    "detector:model": actual_model,
                    "detector:model_role": model_role if is_yolo_det else "not-applicable",
                    "detector:weights_sha256": weight_info.get("sha256") or "not-applicable",
                    "detector:signature": detector_signature["signature"],
                    "detector:context_model": detector_model if yolo_structures else "disabled",
                    "detector:segmentation": segmentation_mode,
                    "detector:precision": detector_precision,
                    "detector:version": (
                        "pool-yolo-v1"
                        if is_yolo_det
                        else "strict-clear-water-v1"
                        if is_strict_water_det
                        else "satellite-pool-water-v2"
                    ),
                    "detector:source": "satellite-yolo" if is_yolo_det else "satellite-cv",
                    "detector:method": (
                        "pool-trained-yolo"
                        if is_yolo_det
                        else "strict-clear-water-cv"
                        if is_strict_water_det
                        else "satellite-visible-water-pool-v2" + ("+yolo-context-validator" if yolo_structures else "")
                    ),
                    "detector:tile_zoom": str(tile_zoom),
                    "detector:chunk": str(chunk_index),
                    "confidence:yolo": f"{confidence:.2f}",
                    "confidence:cv": f"{det['confidence']:.2f}",
                    "yolo:structure_count": str(len(yolo_structures)),
                    "pool:area_sqm": f"{area_sqm:.1f}",
                    "pool:circularity": f"{det['circularity']:.3f}",
                    "pool:aspect_ratio": f"{det.get('aspect_ratio', 0):.2f}",
                    "pool:extent": f"{det.get('extent', 0):.3f}",
                    "pool:water_fraction": f"{det.get('water_fraction', 0):.3f}",
                    "pool:value_mean": f"{det.get('value_mean', 0):.1f}",
                    "pool:saturation_mean": f"{det.get('saturation_mean', 0):.1f}",
                    "pool:green_minus_red": f"{det.get('green_minus_red', 0):.1f}",
                    "pool:blue_minus_red": f"{det.get('blue_minus_red', 0):.1f}",
                    "pool:context_roof_fraction": f"{det.get('context_roof_fraction', 0):.3f}",
                    "pool:context_dark_panel_fraction": f"{det.get('context_dark_panel_fraction', 0):.3f}",
                    "pool:context_vegetation_fraction": f"{det.get('context_vegetation_fraction', 0):.3f}",
                },
                "geometry": geometry,
            })

    if fast_cursor_key:
        with _fast_scan_cursor_lock:
            _fast_scan_cursors[fast_cursor_key] = (
                fast_cursor_offset + max(1, int(diagnostics.get("chunks_scanned", 0) or 1))
            ) % max(1, len(chunks))

    if len(elements) > 1:
        # Collapse the same pool when it straddles a chunk boundary and is
        # reported once per adjacent chip (applies to YOLO and colour-CV alike).
        elements = _deduplicate_detected_pool_elements(elements, min_distance_m=4.0)

    result_cap = None
    if detector_precision == "fast":
        result_cap = _MAX_FAST_POOL_ELEMENTS
    elif detector_precision == "auto":
        result_cap = _MAX_AUTO_POOL_ELEMENTS
    if result_cap is not None and len(elements) > result_cap:
        elements.sort(
            key=lambda element: float(element.get("tags", {}).get("confidence:yolo", "0") or 0),
            reverse=True,
        )
        elements = elements[:result_cap]
        diagnostics["truncated"] = True
        diagnostics["result_cap"] = result_cap

    diagnostics["pool_elements"] = len(elements)
    diagnostics["solar_elements"] = len(solar_elements)
    inference_failures = int(diagnostics.get("inference_failures", 0))
    inference_successes = int(diagnostics.get("inference_successes", 0))
    has_features = bool(elements or solar_elements)
    time_budget_exhausted = diagnostics.get("reason") == "time_budget_exhausted"
    if diagnostics["tiles_loaded"] <= 0:
        diagnostics["reason"] = diagnostics.get("reason") if diagnostics.get("reason") == "time_budget_exhausted" else "no_satellite_tiles_loaded"
    elif has_features:
        if elements and solar_elements:
            reason = "pools_and_solar_detected"
        elif solar_elements:
            reason = "solar_detected"
        else:
            reason = "pools_detected"
        if time_budget_exhausted:
            reason += "_partial_time_budget"
        if inference_failures:
            reason += "_partial_detector_error"
        diagnostics["reason"] = reason
    elif inference_failures and not inference_successes:
        diagnostics["reason"] = "detector_runtime_error"
        if time_budget_exhausted:
            diagnostics["reason"] += "_partial_time_budget"
    elif inference_failures:
        diagnostics["reason"] = "no_candidates_partial_detector_error"
    elif time_budget_exhausted:
        diagnostics["reason"] = "time_budget_exhausted_no_pool_candidates"
    else:
        diagnostics["reason"] = "no_pool_candidates_after_filter"
    if solar_elements:
        # Appended after the pool cap/dedupe so solar never displaces pools.
        elements = elements + solar_elements
    diagnostics["elapsed_ms"] = round((time.time() - started_at) * 1000)
    _set_last_detector_diagnostics(diagnostics)
    return elements


def get_available_models() -> dict[str, dict[str, str]]:
    """Return the dict of supported lightweight YOLO models."""
    return dict(SUPPORTED_MODELS)


def get_last_detector_diagnostics() -> dict[str, Any]:
    """Return telemetry for the most recent satellite detector run."""
    with _detector_diagnostics_lock:
        return dict(_last_detector_diagnostics)


def _module_available(module_name: str) -> bool:
    """Check whether a dependency is installed without importing heavy runtimes."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def get_detector_status() -> dict[str, Any]:
    """Return detector availability without loading Torch or model weights."""
    signature = get_detector_signature()
    opencv_available = _module_available("cv2") and _module_available("numpy")
    yolo_available = _module_available("ultralytics")
    pool_available = bool(yolo_available and resolve_pool_yolo_weights())
    runtime_status = get_pool_yolo_runtime_status()
    return {
        "opencv_available": opencv_available,
        "yolo_available": yolo_available,
        "context_yolo_available": bool(_ENABLE_YOLO_CONTEXT and yolo_available),
        "onnxruntime_available": _module_available("onnxruntime"),
        "sam_geo_available": _module_available("samgeo"),
        "mask_rcnn_available": _module_available("torchvision"),
        "runtime_import_initialized": _import_done,
        "pool_yolo_runtime_loaded": bool(_HAS_YOLO and _YOLO is not None),
        "pool_yolo_runtime": runtime_status,
        "pool_yolo_last_inference_success": runtime_status.get("last_success"),
        "pool_yolo_last_inference_error": runtime_status.get("last_error"),
        "default_model": DEFAULT_DETECTOR_MODEL,
        "supported_models": get_available_models(),
        "loaded_models": sorted(_loaded_models.keys()),
        "pool_yolo_available": pool_available,
        "pool_yolo_weights": resolve_pool_yolo_weights(),
        "pool_yolo_ensemble_available": bool(signature["ensemble_active"]),
        "active_pool_weights": signature["weights"],
        "detector_signature": signature["signature"],
        "detector_pipeline_version": signature["pipeline_version"],
        "detector_thresholds": signature["thresholds"],
        "cv_pool_fallback_enabled": _ENABLE_CV_POOL_FALLBACK,
        "primary_pool_detector": "pool_trained_yolo" if pool_available else (
            "satellite_water_shape_cv" if _ENABLE_CV_POOL_FALLBACK else "openstreetmap_mapped_pools_only"
        ),
        "secondary_validator": "yolo_property_context_validator" if _ENABLE_YOLO_CONTEXT else "disabled_by_default_for_speed",
        "optional_foundation_models": ["SAM-GEO", "Mask R-CNN"] if (_HAS_SAMGEO or _HAS_MASK_RCNN) else [],
        "last_scan": get_last_detector_diagnostics(),
        "pool_detection_note": (
            "Hybrid: OpenStreetMap mapped pools are always trusted. A pool-trained YOLO model "
            "(set QLDR_POOL_YOLO_WEIGHTS or drop pool_yolo.pt in the project root) detects new/unmapped "
            "pools. Generic COCO YOLO cannot detect pools. The colour-CV water detector is an opt-in "
            "experimental fallback (QLDR_ENABLE_CV_POOL_FALLBACK=1) and is prone to false positives."
        ),
    }


def is_available() -> bool:
    """Return True if CV dependencies are installed and detection can run."""
    _lazy_import()
    return _np is not None and (_HAS_CV2 or _Image is not None)
