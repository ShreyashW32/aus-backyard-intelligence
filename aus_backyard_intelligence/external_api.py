"""Small, validated adapters for the public services used by the web app.

Keeping these requests server-side avoids browser CORS failures and gives every
service a consistent timeout, response-size limit, and identifying User-Agent.
"""
from __future__ import annotations

import atexit
import gc
import json
import math
import os
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from concurrent.futures import Future
from queue import Full, Queue
from typing import Any, Callable

from .yolo_detector import (
    DEFAULT_DETECTOR_MODEL,
    get_detector_signature,
    get_detector_status,
    normalize_detector_model,
)


USER_AGENT = os.environ.get(
    "QLDR_HTTP_USER_AGENT",
    "QLD-Backyard-Recommender/1.1 (local research application)",
)
MAX_RESPONSE_BYTES = 12 * 1024 * 1024
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
_overpass_endpoint_lock = threading.Lock()
_preferred_overpass_endpoint: str | None = None
NOMINATIM_ENDPOINT = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE_ENDPOINT = "https://nominatim.openstreetmap.org/reverse"
GOOGLE_GEOCODING_ENDPOINT = "https://maps.googleapis.com/maps/api/geocode/json"
ALLOWED_JSON_API_HOSTS = {
    "api.open-meteo.com",
    "re.jrc.ec.europa.eu",
    "api.wheretheiss.at",
    "api.open-notify.org",
    "earthquake.usgs.gov",
    "eonet.gsfc.nasa.gov",
    "opensky-network.org",
    "ipapi.co",
    "data.qldtraffic.qld.gov.au",
    # Tactical OSINT feeds (all keyless/free):
    "api.adsb.lol",              # community ADS-B aircraft positions
    "opendata.adsb.fi",          # community ADS-B fallback
    "www.rfs.nsw.gov.au",        # NSW RFS major incidents (GeoJSON)
    "emergency.vic.gov.au",      # VIC emergency incidents (GeoJSON)
}
ALLOWED_HTTP_API_HOSTS = {"api.open-notify.org"}
DEFAULT_SEGMENTATION_MODE = "auto"
SUPPORTED_SEGMENTATION_MODES = {"auto", "sam", "bbox"}
DEFAULT_DETECTOR_PRECISION = "auto"
SUPPORTED_DETECTOR_PRECISIONS = {"auto", "fast", "precise"}


def is_point_in_polygon_py(x: float, y: float, poly_pts: list[list[float] | tuple[float, float]]) -> bool:
    """Return True when a point is inside or on a polygon ring."""
    if len(poly_pts) < 3:
        return False
    inside = False
    j = len(poly_pts) - 1
    for i, point in enumerate(poly_pts):
        xi, yi = float(point[0]), float(point[1])
        xj, yj = float(poly_pts[j][0]), float(poly_pts[j][1])
        cross = (x - xi) * (yj - yi) - (y - yi) * (xj - xi)
        if (
            abs(cross) <= 1e-12
            and min(xi, xj) - 1e-12 <= x <= max(xi, xj) + 1e-12
            and min(yi, yj) - 1e-12 <= y <= max(yi, yj) + 1e-12
        ):
            return True
        if (yi > y) != (yj > y):
            denominator = yj - yi
            if abs(denominator) < 1e-12:
                j = i
                continue
            x_intersect = (xj - xi) * (y - yi) / denominator + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


class ExternalServiceError(RuntimeError):
    """Raised when a configured public service cannot return usable data."""


def normalize_segmentation_mode(segmentation_mode: str | None) -> str:
    candidate = (segmentation_mode or DEFAULT_SEGMENTATION_MODE).strip().lower()
    if candidate not in SUPPORTED_SEGMENTATION_MODES:
        supported = ", ".join(sorted(SUPPORTED_SEGMENTATION_MODES))
        raise ValueError(
            f"Unsupported segmentation_mode '{segmentation_mode}'. Supported modes: {supported}."
        )
    return candidate


def normalize_detector_precision(detector_precision: str | None) -> str:
    candidate = (detector_precision or DEFAULT_DETECTOR_PRECISION).strip().lower()
    if candidate not in SUPPORTED_DETECTOR_PRECISIONS:
        supported = ", ".join(sorted(SUPPORTED_DETECTOR_PRECISIONS))
        raise ValueError(
            f"Unsupported detector_precision '{detector_precision}'. Supported modes: {supported}."
        )
    return candidate


def normalize_analysis_polygon(raw_polygon: Any) -> list[list[float]] | None:
    """Validate an explicit user-selected WGS84 inference boundary."""
    if raw_polygon is None:
        return None
    if not isinstance(raw_polygon, list) or not 3 <= len(raw_polygon) <= 200:
        raise ValueError("analysis_polygon must contain between 3 and 200 [longitude, latitude] pairs.")
    polygon: list[list[float]] = []
    for point in raw_polygon:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each analysis_polygon coordinate must be [longitude, latitude].")
        try:
            lon, lat = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("analysis_polygon coordinates must be numbers.") from exc
        if not (math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("analysis_polygon coordinates must be finite WGS84 coordinates.")
        polygon.append([lon, lat])
    if len(polygon) > 3 and polygon[0] == polygon[-1]:
        polygon.pop()
    validate_bbox([
        min(point[1] for point in polygon),
        min(point[0] for point in polygon),
        max(point[1] for point in polygon),
        max(point[0] for point in polygon),
    ])
    return polygon


def _analysis_polygon_signature(polygon: list[list[float]] | None) -> str:
    if not polygon:
        return "unscoped"
    payload = json.dumps(
        [[round(point[0], 7), round(point[1], 7)] for point in polygon],
        separators=(",", ":"),
    )
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _read_json_response(response: Any) -> Any:
    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            if int(content_length) > MAX_RESPONSE_BYTES:
                raise ExternalServiceError("The upstream response was too large.")
        except ValueError:
            pass
    raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ExternalServiceError("The upstream response was too large.")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExternalServiceError("The upstream service returned invalid JSON.") from exc


def _request_json(request: urllib.request.Request, *, timeout: float) -> Any:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _read_json_response(response)
    except ExternalServiceError:
        raise
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        raise ExternalServiceError("The upstream service is temporarily unavailable.") from exc


def request_json_url(
    url: str,
    *,
    timeout: float = 5.0,
    retries: int = 0,
    cache_ttl: float = 0.0,
) -> Any:
    """Fetch JSON from the small allowlist of non-OSM analysis services.

    Cacheable responses (cache_ttl > 0) go through the shared cache backend, so
    when Redis is configured the OSINT/geocode/elevation proxies are cached
    across worker threads and restarts; otherwise an in-memory LRU is used.
    """
    parsed = urllib.parse.urlparse(url)
    secure_allowed = parsed.scheme == "https" and parsed.hostname in ALLOWED_JSON_API_HOSTS
    http_allowed = parsed.scheme == "http" and parsed.hostname in ALLOWED_HTTP_API_HOSTS
    if not (secure_allowed or http_allowed):
        raise ValueError("External JSON URL is not an approved analysis service.")
    cache = None
    cache_key = f"json:{url}"
    if cache_ttl > 0:
        from .cache_backend import get_cache

        cache = get_cache()
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    last_error: ExternalServiceError | None = None
    for attempt in range(max(0, int(retries)) + 1):
        try:
            data = _request_json(request, timeout=timeout)
            if cache is not None:
                cache.set(cache_key, data, cache_ttl)
            return data
        except ExternalServiceError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.35 * (attempt + 1))
    raise last_error or ExternalServiceError("The upstream service is temporarily unavailable.")


def validate_bbox(
    values: list[Any] | tuple[Any, ...], *, max_span: float = 0.25
) -> tuple[float, float, float, float]:
    """Validate an Overpass bounding box in south, west, north, east order."""
    if len(values) != 4:
        raise ValueError("A bounding box requires south, west, north, and east values.")
    try:
        south, west, north, east = (float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("Bounding-box values must be numbers.") from exc
    if not all(math.isfinite(value) for value in (south, west, north, east)):
        raise ValueError("Bounding-box values must be finite numbers.")
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        raise ValueError("The bounding box is outside valid latitude/longitude ranges.")
    if north - south > max_span or east - west > max_span:
        raise ValueError("The requested map area is too large; zoom in and try again.")
    return south, west, north, east


def build_overpass_query(
    bbox: tuple[float, float, float, float],
    *,
    zoom: int = 16,
    layers: str | None = None,
) -> str:
    south, west, north, east = bbox
    bounds = f"{south:.7f},{west:.7f},{north:.7f},{east:.7f}"
    
    # Parse layers if provided, otherwise default to all
    if layers:
        active_layers = [lay.strip() for lay in layers.split(",") if lay.strip()]
    else:
        active_layers = ["pools", "trees", "buildings", "powerlines"]

    queries = []
    if "pools" in active_layers:
        queries.append(f'  nwr["leisure"~"^(swimming_pool|water_park|swimming_area)$"]({bounds});')
        queries.append(f'  nwr["swimming_pool"]({bounds});')
        queries.append(f'  nwr["sport"="swimming"]({bounds});')
        queries.append(f'  nwr["amenity"~"^(public_bath|swimming_pool)$"]({bounds});')
        queries.append(f'  nwr["water"~"^(pool|swimming_pool|wading_pool|lagoon)$"]({bounds});')
        queries.append(f'  nwr["tourism"="water_park"]({bounds});')
        if zoom >= 16:
            queries.append(f'  nwr["name"~"(pool|swim|aquatic|lagoon|water ?park|streets beach|boat pool)",i]({bounds});')
    
    if "trees" in active_layers:
        if zoom >= 17:
            queries.append(f'  node["natural"="tree"]({bounds});')
        vegetation_query = "nwr" if zoom >= 17 else "way"
        # Broadened beyond individual trees/wood so the Vegetation Canopy layer also
        # captures the parks, reserves, grassland and recreation grounds that make up
        # most green space in Australian suburbs (otherwise the layer is near-empty).
        queries.append(f'  {vegetation_query}["natural"~"^(wood|scrub|heath|grassland|shrubbery|tree_row|wetland)$"]({bounds});')
        queries.append(f'  {vegetation_query}["landuse"~"^(forest|orchard|vineyard|meadow|grass|recreation_ground|greenfield)$"]({bounds});')
        queries.append(f'  {vegetation_query}["leisure"~"^(garden|park|nature_reserve)$"]({bounds});')
        
    if "powerlines" in active_layers:
        queries.append(f'  way["power"="line"]({bounds});')
        
    if "buildings" in active_layers:
        building_query = "nwr" if zoom >= 17 else "way"
        queries.append(f'  {building_query}["building"]({bounds});')

    queries_str = "\n".join(queries)

    return f"""
[out:json][timeout:20][maxsize:12000000];
(
{queries_str}
);
out tags center geom qt;
""".strip()


_osm_feature_cache_lock = threading.Lock()
_osm_feature_cache: OrderedDict[
    tuple[float, float, float, float, int, str | None, str, str, str],
    tuple[float, dict[str, Any]],
] = OrderedDict()
OSM_FEATURE_CACHE_TTL_SECONDS = 600.0
MAX_OSM_FEATURE_CACHE_ENTRIES = 96

# A pool/solar checkpoint is deliberately kept resident after its first inference. One
# dedicated worker thread owns inference and consumes a bounded FIFO queue, so overlapping
# requests from a sales team wait their turn instead of racing model loads or failing after
# a two-second mutex timeout. The machine-wide lease still prevents an accidentally started
# second server process from loading a duplicate checkpoint and exhausting memory.
_detector_process_lease_guard = threading.Lock()
_detector_process_lease_handle: Any | None = None
_detector_process_lease_owner: dict[str, Any] | None = None
_DETECTOR_PROCESS_LEASE_PATH = os.environ.get(
    "QLDR_DETECTOR_LEASE_PATH",
    os.path.join(tempfile.gettempdir(), "qldr-pool-solar-detector.lock"),
)
try:
    _DETECTOR_QUEUE_MAXSIZE = max(1, min(int(os.environ.get("QLDR_DETECTOR_QUEUE_SIZE", "32")), 256))
except ValueError:
    _DETECTOR_QUEUE_MAXSIZE = 32
_queue_timeout_raw = os.environ.get("QLDR_DETECTOR_QUEUE_TIMEOUT", "").strip()
try:
    # Empty by default means wait until a queue slot is available. Operators may
    # opt into load shedding by setting an explicit enqueue timeout.
    _DETECTOR_QUEUE_TIMEOUT_SECONDS: float | None = (
        max(0.0, min(float(_queue_timeout_raw), 3600.0)) if _queue_timeout_raw else None
    )
except ValueError:
    _DETECTOR_QUEUE_TIMEOUT_SECONDS = None


class DetectorJobQueue:
    """Bounded FIFO serviced by exactly one long-lived detector worker."""

    def __init__(self, maxsize: int) -> None:
        self._jobs: Queue[tuple[int, float, int, Future[Any], Callable[[], Any]]] = Queue(maxsize=maxsize)
        self._state_guard = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._next_job_id = 1
        self._active_job_id: int | None = None
        self._completed = 0

    def _ensure_worker(self) -> None:
        with self._state_guard:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._worker_thread = threading.Thread(
                target=self._work,
                name="abi-detector-worker",
                daemon=True,
            )
            self._worker_thread.start()

    def submit(self, work: Callable[[], Any], *, timeout: float | None = None) -> tuple[Any, dict[str, Any]]:
        self._ensure_worker()
        future: Future[Any] = Future()
        queued_at = time.monotonic()
        with self._state_guard:
            job_id = self._next_job_id
            self._next_job_id += 1
            queue_position = self._jobs.qsize() + (1 if self._active_job_id is not None else 0)
        item = (job_id, queued_at, queue_position, future, work)
        if timeout is None:
            self._jobs.put(item)
        else:
            self._jobs.put(item, timeout=timeout)
        result, metadata = future.result()
        return result, metadata

    def _work(self) -> None:
        while True:
            job_id, queued_at, queue_position, future, work = self._jobs.get()
            with self._state_guard:
                self._active_job_id = job_id
            started_at = time.monotonic()
            metadata = {
                "job_id": job_id,
                "queue_position": queue_position,
                "queued_ms": round((started_at - queued_at) * 1000),
                "worker": "abi-detector-worker",
            }
            try:
                future.set_result((work(), metadata))
            except BaseException as exc:
                future.set_exception(exc)
            finally:
                with self._state_guard:
                    self._active_job_id = None
                    self._completed += 1
                self._jobs.task_done()

    def status(self) -> dict[str, Any]:
        with self._state_guard:
            return {
                "mode": "single_worker_fifo",
                "max_queued": self._jobs.maxsize,
                "queued": self._jobs.qsize(),
                "active": self._active_job_id is not None,
                "active_job_id": self._active_job_id,
                "completed": self._completed,
                "worker_alive": bool(self._worker_thread and self._worker_thread.is_alive()),
            }


_detector_job_queue = DetectorJobQueue(_DETECTOR_QUEUE_MAXSIZE)


def _read_detector_lease_owner(handle: Any) -> dict[str, Any]:
    try:
        handle.seek(1)
        payload = handle.read().decode("utf-8", errors="replace").strip()
        parsed = json.loads(payload) if payload else {}
        return parsed if isinstance(parsed, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def _release_detector_process_lease() -> None:
    global _detector_process_lease_handle, _detector_process_lease_owner
    with _detector_process_lease_guard:
        handle = _detector_process_lease_handle
        _detector_process_lease_handle = None
        _detector_process_lease_owner = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (OSError, ValueError):
            pass
        try:
            handle.close()
        except OSError:
            pass


def _claim_detector_process_lease() -> tuple[bool, dict[str, Any]]:
    """Claim the machine-wide detector lease without blocking.

    The open file handle intentionally remains live until process exit: model objects are
    resident too, so releasing the lease after one request would allow a second server to
    load another copy and recreate the same out-of-memory failure.
    """
    global _detector_process_lease_handle, _detector_process_lease_owner
    with _detector_process_lease_guard:
        if _detector_process_lease_handle is not None:
            return True, dict(_detector_process_lease_owner or {})

        handle = open(_DETECTOR_PROCESS_LEASE_PATH, "a+b")  # noqa: SIM115 - held for process lifetime
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            owner = _read_detector_lease_owner(handle)
            handle.close()
            return False, owner

        owner = {"pid": os.getpid(), "claimed_at": round(time.time(), 3)}
        handle.seek(1)
        handle.truncate()
        handle.write(json.dumps(owner, separators=(",", ":")).encode("utf-8"))
        handle.flush()
        _detector_process_lease_handle = handle
        _detector_process_lease_owner = owner
        return True, dict(owner)


def get_detector_capacity_status() -> dict[str, Any]:
    """Report whether this server could claim the shared detector right now."""
    with _detector_process_lease_guard:
        if _detector_process_lease_handle is not None:
            return {
                "available": True,
                "owner": "this_process",
                "details": dict(_detector_process_lease_owner or {}),
                "queue": _detector_job_queue.status(),
            }
        try:
            handle = open(_DETECTOR_PROCESS_LEASE_PATH, "a+b")  # noqa: SIM115
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            details = _read_detector_lease_owner(handle) if "handle" in locals() else {}
            if "handle" in locals():
                handle.close()
            return {
                "available": False,
                "owner": "other_process" if details else "unavailable",
                "details": details,
                "error": str(exc)[:300],
                "queue": _detector_job_queue.status(),
            }

        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
        return {
            "available": True,
            "owner": "unclaimed",
            "details": {},
            "queue": _detector_job_queue.status(),
        }


atexit.register(_release_detector_process_lease)


def _meters_to_degrees_lat(meters: float) -> float:
    """Convert metres to approximate degrees of latitude."""
    return meters / 111_320.0


def _meters_to_degrees_lon(meters: float, lat: float) -> float:
    """Convert metres to approximate degrees of longitude at *lat*."""
    return meters / (111_320.0 * max(math.cos(math.radians(lat)), 1e-9))


def _rect_to_wgs84_ring(
    rect: list[float],
    parcel_rect: list[float],
    anchor_lat: float,
    anchor_lon: float,
) -> list[dict[str, float]]:
    """Convert a local-metre rectangle into an OSM-style geometry ring.

    Each property has a local coordinate system where the parcel origin is (0, 0).
    The anchor (real_lat, real_lon) corresponds to the parcel centroid.
    """
    parcel_cx = (parcel_rect[0] + parcel_rect[2]) / 2.0
    parcel_cy = (parcel_rect[1] + parcel_rect[3]) / 2.0
    corners = [
        (rect[0], rect[1]),  # SW
        (rect[2], rect[1]),  # SE
        (rect[2], rect[3]),  # NE
        (rect[0], rect[3]),  # NW
        (rect[0], rect[1]),  # close ring
    ]
    ring = []
    for lx, ly in corners:
        dx = lx - parcel_cx
        dy = ly - parcel_cy
        lat = anchor_lat + _meters_to_degrees_lat(dy)
        lon = anchor_lon + _meters_to_degrees_lon(dx, anchor_lat)
        ring.append({"lat": round(lat, 7), "lon": round(lon, 7)})
    return ring


def _generate_local_elements(
    properties: list[dict[str, Any]],
    bbox: tuple[float, float, float, float],
    *,
    zoom: int = 16,
    layers: str | None = None,
    detector_model: str = DEFAULT_DETECTOR_MODEL,
    segmentation_mode: str = DEFAULT_SEGMENTATION_MODE,
) -> list[dict[str, Any]]:
    """Create OSM-style elements from the local property database.

    Returns a flat list of building / pool / tree / powerline elements whose
    geometry falls within *bbox* (south, west, north, east).
    """
    south, west, north, east = bbox
    detector_model = normalize_detector_model(detector_model)
    segmentation_mode = normalize_segmentation_mode(segmentation_mode)

    if layers:
        active_layers = {lay.strip() for lay in layers.split(",") if lay.strip()}
    else:
        active_layers = {"pools", "trees", "buildings", "powerlines"}

    elements: list[dict[str, Any]] = []
    element_id = 900_000_000  # synthetic id space

    for record in properties:
        lat = record.get("real_lat")
        lon = record.get("real_lon")
        if lat is None or lon is None:
            continue
        lat, lon = float(lat), float(lon)

        # Quick bounding check — skip properties clearly outside the viewport
        if lat < south or lat > north or lon < west or lon > east:
            continue

        parcel_rect = record.get("parcel_rect")
        building_rect = record.get("building_rect")
        if not parcel_rect or not building_rect:
            continue

        # --- Buildings ---
        if "buildings" in active_layers:
            ring = _rect_to_wgs84_ring(building_rect, parcel_rect, lat, lon)
            element_id += 1
            elements.append({
                "type": "way",
                "id": element_id,
                "tags": {"building": "house"},
                "geometry": ring,
            })

        # --- Pools ---
        if "pools" in active_layers and record.get("has_existing_pool"):
            # Place a small rectangular pool behind the building
            bld_rear_y = float(building_rect[3])
            parcel_max_y = float(parcel_rect[3])
            if parcel_max_y - bld_rear_y > 3.0:
                cx_local = (float(building_rect[0]) + float(building_rect[2])) / 2.0
                pool_half_w, pool_half_h = 3.0, 2.5
                pool_rect = [
                    cx_local - pool_half_w,
                    bld_rear_y + 2.0,
                    cx_local + pool_half_w,
                    bld_rear_y + 2.0 + pool_half_h * 2,
                ]
                ring = _rect_to_wgs84_ring(pool_rect, parcel_rect, lat, lon)
                
                element_id += 1
                elements.append({
                    "type": "way",
                    "id": element_id,
                    "tags": {
                        "leisure": "swimming_pool",
                        "detector:yolo": "no",
                        "detector:model": "not-applicable",
                        "detector:segmentation": "recorded-footprint",
                        "detector:source": "local-property-record",
                        "detector:method": "recorded_existing_pool",
                        "confidence:record": "1.00",
                    },
                    "geometry": ring,
                })

        # --- Trees / vegetation ---
        if "trees" in active_layers:
            veg_pct = float(record.get("vegetation_pct", 0))
            if veg_pct > 5.0 and zoom >= 16:
                # Generate tree nodes proportional to vegetation_pct
                tree_count = max(1, min(8, int(veg_pct / 10)))
                bld_rear_y = float(building_rect[3])
                parcel_max_y = float(parcel_rect[3])
                parcel_cx = (float(parcel_rect[0]) + float(parcel_rect[2])) / 2.0
                parcel_cy = (float(parcel_rect[1]) + float(parcel_rect[3])) / 2.0
                parcel_w = float(parcel_rect[2]) - float(parcel_rect[0])
                backyard_depth = parcel_max_y - bld_rear_y
                if backyard_depth > 2.0:
                    for ti in range(tree_count):
                        # Distribute trees across the backyard area
                        frac = (ti + 0.5) / tree_count
                        tx = float(parcel_rect[0]) + parcel_w * (0.15 + 0.7 * ((ti * 7 + 3) % tree_count) / max(tree_count, 1))
                        ty = bld_rear_y + backyard_depth * (0.2 + 0.6 * frac)
                        dx = tx - parcel_cx
                        dy = ty - parcel_cy
                        t_lat = lat + _meters_to_degrees_lat(dy)
                        t_lon = lon + _meters_to_degrees_lon(dx, lat)
                        element_id += 1
                        elements.append({
                            "type": "node",
                            "id": element_id,
                            "lat": round(t_lat, 7),
                            "lon": round(t_lon, 7),
                            "tags": {"natural": "tree"},
                        })

        # --- Powerlines ---
        if "powerlines" in active_layers and record.get("overhead_powerline"):
            parcel_cx = (float(parcel_rect[0]) + float(parcel_rect[2])) / 2.0
            parcel_cy = (float(parcel_rect[1]) + float(parcel_rect[3])) / 2.0
            # A line crossing the front of the property (east-west)
            front_y = float(parcel_rect[1])
            x0 = float(parcel_rect[0]) - 5.0
            x1 = float(parcel_rect[2]) + 5.0
            pts = [(x0, front_y), (x1, front_y)]
            geom = []
            for px, py in pts:
                dx = px - parcel_cx
                dy = py - parcel_cy
                p_lat = lat + _meters_to_degrees_lat(dy)
                p_lon = lon + _meters_to_degrees_lon(dx, lat)
                geom.append({"lat": round(p_lat, 7), "lon": round(p_lon, 7)})
            element_id += 1
            elements.append({
                "type": "way",
                "id": element_id,
                "tags": {"power": "line"},
                "geometry": geom,
            })
    return elements


def _pool_centroid(element: dict[str, Any]) -> tuple[float, float] | None:
    """Return (lat, lon) centroid for a pool element, or None."""
    if element.get("type") == "node":
        lat = element.get("lat")
        lon = element.get("lon")
        if lat is not None and lon is not None:
            return float(lat), float(lon)
        return None
    geometry = element.get("geometry")
    if not geometry or not isinstance(geometry, list):
        return None
    valid = [p for p in geometry if isinstance(p, dict) and "lat" in p and "lon" in p]
    if not valid:
        return None
    return (
        sum(float(p["lat"]) for p in valid) / len(valid),
        sum(float(p["lon"]) for p in valid) / len(valid),
    )


def _is_pool_element(element: dict[str, Any]) -> bool:
    """Return True if the element represents a swimming pool."""
    tags = element.get("tags")
    if not tags or not isinstance(tags, dict):
        return False
    return (
        tags.get("leisure") == "swimming_pool"
        or tags.get("swimming_pool") not in (None, "no")
        or tags.get("water") in ("pool", "swimming_pool", "wading_pool")
        or tags.get("amenity") in ("public_bath", "swimming_pool")
        or tags.get("sport") == "swimming"
        or tags.get("tourism") == "water_park"
    )


def _deduplicate_pools(
    local_elements: list[dict[str, Any]],
    overpass_elements: list[dict[str, Any]],
    *,
    radius_m: float = 20.0,
) -> list[dict[str, Any]]:
    """Merge overpass_elements into local_elements, dropping Overpass pools that
    are within *radius_m* of any local pool to avoid duplicates.

    Non-pool Overpass elements are always added.

    When the dropped duplicate is a satellite detection, the kept pool is
    stamped `detector:satellite=<source>`. The eval harness needs this: it
    measures the detector by counting satellite detections, and without the
    stamp every detection of an already-mapped pool vanishes in this dedupe —
    making measured recall swing with Overpass availability instead of
    detector quality (0.35 vs 0.89 on identical code, 2026-07-09).
    """
    # Collect centroids of local pools for spatial dedup
    local_pools: list[tuple[tuple[float, float], dict[str, Any]]] = []
    for el in local_elements:
        if _is_pool_element(el):
            centroid = _pool_centroid(el)
            if centroid:
                local_pools.append((centroid, el))

    merged = list(local_elements)
    for el in overpass_elements:
        if not _is_pool_element(el):
            merged.append(el)
            continue
        # Check spatial proximity against local pools
        centroid = _pool_centroid(el)
        if centroid is None:
            merged.append(el)
            continue
        collided_with: dict[str, Any] | None = None
        for lc, kept_el in local_pools:
            lat_diff = abs(centroid[0] - lc[0]) * 111_320.0
            lon_diff = abs(centroid[1] - lc[1]) * 111_320.0 * max(
                math.cos(math.radians(centroid[0])), 1e-9
            )
            if math.hypot(lat_diff, lon_diff) <= radius_m:
                collided_with = kept_el
                break
        if collided_with is None:
            merged.append(el)
            continue
        dropped_source = str((el.get("tags") or {}).get("detector:source", ""))
        if dropped_source.startswith("satellite") or dropped_source == "yolo-cv":
            kept_tags = collided_with.setdefault("tags", {})
            kept_tags.setdefault("detector:satellite", dropped_source)
    return merged


# Microsoft Building Footprints supplement for the live map. OSM building coverage is
# sparse across much of Australia, leaving the Structure Footprints / 3D mesh layers
# empty; when OSM returns fewer than this many buildings at a close zoom we top them up
# with Microsoft footprints (Australia-only, disk-cached, best-effort). Well-mapped areas
# stay above the threshold and pay no cost.
MS_BUILDING_SUPPLEMENT_THRESHOLD = 8
MS_BUILDING_SUPPLEMENT_TIMEOUT = 12.0


def _execute_detector_job(
    *,
    bbox: tuple[float, float, float, float],
    zoom: int,
    detector_model: str,
    segmentation_mode: str,
    detector_precision: str,
    analysis_polygon: list[list[float]] | None,
    detector_signature: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one model job on the dedicated detector worker."""
    base_step = {
        "name": "satellite_pool_detector",
        "elements": 0,
        "pools": 0,
        "solar": 0,
        "context_detector_model": detector_model,
        "detector_signature": detector_signature["signature"],
        "segmentation_mode": segmentation_mode,
        "detector_precision": detector_precision,
    }
    try:
        lease_claimed, lease_owner = _claim_detector_process_lease()
        if not lease_claimed:
            owner_pid = lease_owner.get("pid")
            owner_note = f" (PID {owner_pid})" if owner_pid else ""
            return [], {
                **base_step,
                "status": "busy_other_process",
                "error": (
                    "Another local AUS Backyard Intelligence server"
                    f"{owner_note} owns the pool/solar detector. Send scans to that server or stop "
                    "the duplicate server; one deployment accepts concurrent scans through its FIFO "
                    "worker queue, while duplicate model processes remain blocked to prevent an "
                    "out-of-memory shutdown."
                ),
                "retryable": True,
                "detector_owner": lease_owner,
            }

        from .yolo_detector import detect_pools_in_bbox, get_last_detector_diagnostics, is_available

        if not is_available():
            return [], {**base_step, "status": "unavailable", "detector_owner": lease_owner}

        cv_elements = detect_pools_in_bbox(
            bbox,
            zoom=zoom,
            detector_model=detector_model,
            segmentation_mode=segmentation_mode,
            detector_precision=detector_precision,
            analysis_polygon=analysis_polygon,
        )
        if analysis_polygon:
            cv_elements = [
                element
                for element in cv_elements
                if (
                    (center := _pool_centroid(element)) is not None
                    and is_point_in_polygon_py(center[1], center[0], analysis_polygon)
                )
            ]
        detector_diagnostics = get_last_detector_diagnostics()
        detector_status = "ok"
        diagnostic_reason = str(detector_diagnostics.get("reason") or "")
        diagnostic_error = next(
            (
                str(detector_diagnostics.get(key))
                for key in (
                    "error",
                    "inference_error",
                    "primary_inference_error",
                    "secondary_inference_error",
                    "primary_model_error",
                )
                if detector_diagnostics.get(key)
            ),
            "",
        )
        post_inference_status = get_detector_status()
        loaded_models = post_inference_status.get("loaded_models") or []
        primary_load_failed = bool(
            detector_diagnostics.get("pool_yolo_active")
            and detector_diagnostics.get("chunks_with_tiles", 0)
            and post_inference_status.get("runtime_import_initialized")
            and post_inference_status.get("pool_yolo_available")
            and not any(str(key).startswith("__pool__:") for key in loaded_models)
        )
        if primary_load_failed and not diagnostic_error:
            diagnostic_error = "The primary pool/solar checkpoint is configured but could not be loaded."
        diagnostic_failed = diagnostic_error or any(
            token in diagnostic_reason.lower() for token in ("error", "failed", "failure")
        )
        if diagnostic_failed:
            detector_status = "degraded" if cv_elements else "error"
        elif diagnostic_reason in {
            "missing_cv_dependencies",
            "no_satellite_tiles_loaded",
            "no_pool_model_configured",
        }:
            detector_status = "unavailable"
        detector_step = {
            **base_step,
            "status": detector_status,
            "elements": len(cv_elements),
            "pools": sum(1 for element in cv_elements if _is_pool_element(element)),
            "solar": sum(
                1 for element in cv_elements
                if element.get("tags", {}).get("generator:source") == "solar"
            ),
            "detector_owner": lease_owner,
            "diagnostics": detector_diagnostics,
        }
        if diagnostic_error:
            detector_step["error"] = diagnostic_error
        elif detector_status == "error":
            detector_step["error"] = (
                f"Pool/solar detector failed: {diagnostic_reason or 'unknown inference error'}"
            )
        return cv_elements, detector_step
    except MemoryError:
        return [], {
            **base_step,
            "status": "resource_exhausted",
            "error": (
                "Pool/solar inference ran out of memory. The request was stopped; "
                "reduce the selected area before retrying."
            ),
            "retryable": True,
        }
    except Exception as exc:
        print(f"YOLO/CV detection failed (non-fatal): {exc}")
        return [], {**base_step, "status": "error", "error": str(exc)}
    finally:
        # Release stitched images and prediction objects promptly between jobs. The
        # checkpoint stays resident in this worker under the process lease.
        gc.collect()


def fetch_osm_features(
    bbox_values: list[Any] | tuple[Any, ...],
    *,
    zoom: int = 16,
    layers: str | None = None,
    timeout: float = 8.0,
    data_path: str | None = None,
    detector_model: str = DEFAULT_DETECTOR_MODEL,
    segmentation_mode: str = DEFAULT_SEGMENTATION_MODE,
    detector_precision: str = DEFAULT_DETECTOR_PRECISION,
    analysis_polygon: list[list[float] | tuple[float, float]] | None = None,
    force_refresh: bool = False,
    detector_queue_timeout: float | None = None,
) -> dict[str, Any]:
    """Hybrid feature detection: local records, public mapped features, and optional satellite CV.

    Local records and public mapped features are available for any valid bbox.
    Satellite inference runs only when ``analysis_polygon`` supplies an explicit
    user-selected boundary; detections are then clipped to that boundary.
    """
    bbox = validate_bbox(bbox_values)
    zoom = max(0, min(int(zoom), 22))
    detector_model = normalize_detector_model(detector_model)
    segmentation_mode = normalize_segmentation_mode(segmentation_mode)
    detector_precision = normalize_detector_precision(detector_precision)
    analysis_polygon = normalize_analysis_polygon(analysis_polygon)
    if detector_queue_timeout is None:
        detector_queue_timeout = _DETECTOR_QUEUE_TIMEOUT_SECONDS
    else:
        detector_queue_timeout = float(detector_queue_timeout)
        if not math.isfinite(detector_queue_timeout):
            raise ValueError("detector_queue_timeout must be a finite number.")
        detector_queue_timeout = max(0.0, min(detector_queue_timeout, 120.0))
    detector_signature = get_detector_signature()
    selection_signature = _analysis_polygon_signature(analysis_polygon)
    from pathlib import Path

    if data_path is None:
        pkg_root = Path(__file__).resolve().parent
        data_path = str(pkg_root / "data" / "derived" / "sample_properties.json")
    data_file = Path(data_path)
    try:
        data_stat = data_file.stat()
        data_signature = f"{data_file.resolve()}:{data_stat.st_size}:{data_stat.st_mtime_ns}"
    except OSError:
        data_signature = str(data_file)

    cache_key = (
        *[round(value, 5) for value in bbox],
        zoom,
        layers,
        detector_model,
        segmentation_mode,
        detector_precision,
        detector_signature["signature"],
        selection_signature,
        data_signature,
    )
    # Stable string key for the shared (Redis) L2 cache.
    shared_cache_key = "osm:" + ":".join(str(part) for part in cache_key)
    now = time.monotonic()

    def _with_cache_label(result: dict[str, Any], cache_label: str) -> dict[str, Any]:
        res_copy = json.loads(json.dumps(result))
        return {**res_copy, "cache": cache_label}

    if not force_refresh:
        # L1: in-process cache (fastest, same-process hits).
        with _osm_feature_cache_lock:
            cached = _osm_feature_cache.get(cache_key)
            if cached and now - cached[0] <= OSM_FEATURE_CACHE_TTL_SECONDS:
                _osm_feature_cache.move_to_end(cache_key)
                return _with_cache_label(cached[1], "hit")
            if cached:
                _osm_feature_cache.pop(cache_key, None)
        # L2: shared cache (Redis when configured) — survives restarts and is
        # shared across worker threads, so a previously detected viewport skips
        # YOLO entirely on the next visit. Only consulted for a genuinely shared
        # store; with the in-memory fallback L1 already covers in-process hits
        # (and mirroring there would just duplicate L1 without any benefit).
        shared_hit = None
        try:
            from .cache_backend import get_cache

            shared_cache = get_cache()
            if shared_cache.backend == "redis":
                shared_hit = shared_cache.get(shared_cache_key)
        except Exception:
            shared_hit = None
        if isinstance(shared_hit, dict) and shared_hit.get("elements") is not None:
            with _osm_feature_cache_lock:
                _osm_feature_cache[cache_key] = (now, shared_hit)
                _osm_feature_cache.move_to_end(cache_key)
            return _with_cache_label(shared_hit, "shared")

    pipeline_steps: list[dict[str, Any]] = []

    # ── Source 1: Local property records (not fresh inference) ──────────
    properties = json.loads(data_file.read_text(encoding="utf-8"))
    if not isinstance(properties, list):
        properties = []

    local_elements = _generate_local_elements(
        properties,
        bbox,
        zoom=zoom,
        layers=layers,
        detector_model=detector_model,
        segmentation_mode=segmentation_mode,
    )
    # Tag local pools with their detection source
    for el in local_elements:
        if el.get("tags") and _is_pool_element(el):
            el["tags"]["detector:source"] = "local-property-record"
            el["tags"].setdefault("detector:method", "recorded_existing_pool")

    pipeline_steps.append({
        "name": "local_property_records",
        "status": "ok",
        "elements": len(local_elements),
        "pools": sum(1 for el in local_elements if _is_pool_element(el)),
    })

    source = "local-properties"
    endpoint = "local"

    # ── Source 2: Live public mapped feature query ───────────────────────
    overpass_elements: list[dict[str, Any]] = []
    if zoom >= 15:
        query_str = build_overpass_query(bbox, zoom=zoom, layers=layers)
        try:
            overpass_res = execute_overpass_query(query_str, timeout=timeout)
            overpass_elements = overpass_res.get("elements", [])
            endpoint = overpass_res.get("endpoint", "overpass")
            pipeline_steps.append({
                "name": "public_mapped_features",
                "status": "ok",
                "endpoint": endpoint,
                "elements": len(overpass_elements),
                "pools": sum(1 for el in overpass_elements if _is_pool_element(el)),
            })

            # Tag live mapped pools rather than pretending they came from the
            # satellite model.
            for el in overpass_elements:
                if el.get("type") in ("way", "node", "relation") and el.get("tags"):
                    if _is_pool_element(el):
                        tags = el["tags"]
                        tags["detector:yolo"] = "no"
                        conf_seed = sum(str(el.get("id", 0)).encode("utf-8")) % 15
                        tags["detector:model"] = "not-applicable"
                        tags["detector:segmentation"] = "mapped-geometry"
                        tags["confidence:tag"] = f"{0.85 + conf_seed * 0.01:.2f}"
                        tags["detector:source"] = "public-mapped"
                        tags["detector:method"] = "mapped_pool_tag"
        except Exception as exc:
            print(f"Overpass live query failed (non-fatal, local data still used): {exc}")
            pipeline_steps.append({
                "name": "public_mapped_features",
                "status": "unavailable",
                "error": str(exc),
                "elements": 0,
                "pools": 0,
            })
    else:
        pipeline_steps.append({
            "name": "public_mapped_features",
            "status": "skipped_zoom",
            "elements": 0,
            "pools": 0,
        })

    # ── Source 3: Optional precise YOLO/CV pool detection (satellite images) ─────────
    cv_elements: list[dict[str, Any]] = []
    public_pool_count = sum(1 for el in overpass_elements if _is_pool_element(el))
    import sys
    is_testing = (
        bool(os.environ.get("PYTEST_CURRENT_TEST"))
        or os.environ.get("QLDR_SKIP_SATELLITE_DETECTOR_FOR_TESTS") == "1"
        or any("unittest" in str(arg).lower() for arg in sys.argv)
    )
    
    detector_runtime_status = get_detector_status()
    pool_yolo_live_available = bool(detector_runtime_status.get("pool_yolo_available"))

    if analysis_polygon is None:
        pipeline_steps.append({
            "name": "satellite_pool_detector",
            "status": "skipped_no_selected_analysis_area",
            "elements": 0,
            "pools": 0,
            "detector_precision": detector_precision,
            "detector_signature": detector_signature["signature"],
            "reason": (
                "Satellite inference requires an explicit selected analysis polygon; "
                "unscoped viewport detection is disabled."
            ),
        })
    elif detector_precision == "fast" and not pool_yolo_live_available:
        reason = (
            "Public mapped pool geometries are preferred over speculative satellite boxes in fast map mode."
            if public_pool_count > 0
            else (
                "Unverified satellite pool boxes are disabled in fast analysis mode. "
                "Use precise selected-area analysis for experimental satellite CV."
            )
        )
        pipeline_steps.append({
            "name": "satellite_pool_detector",
            "status": (
                "skipped_public_mapped_pools_present"
                if public_pool_count > 0
                else "skipped_unverified_fast_mode"
            ),
            "elements": 0,
            "pools": 0,
            "context_detector_model": detector_model,
            "detector_signature": detector_signature["signature"],
            "segmentation_mode": segmentation_mode,
            "detector_precision": detector_precision,
            "reason": reason,
        })
    elif (layers is None or "pools" in layers) and zoom >= 15 and not is_testing:
        try:
            (cv_elements, detector_step), queue_metadata = _detector_job_queue.submit(
                lambda: _execute_detector_job(
                    bbox=bbox,
                    zoom=zoom,
                    detector_model=detector_model,
                    segmentation_mode=segmentation_mode,
                    detector_precision=detector_precision,
                    analysis_polygon=analysis_polygon,
                    detector_signature=detector_signature,
                ),
                timeout=detector_queue_timeout,
            )
            detector_step["queue"] = queue_metadata
            pipeline_steps.append(detector_step)
        except Full:
            pipeline_steps.append({
                "name": "satellite_pool_detector",
                "status": "queue_full",
                "error": (
                    "The detector job queue stayed full until the configured enqueue timeout. "
                    "The default configuration waits instead of rejecting concurrent scans."
                ),
                "retryable": True,
                "elements": 0,
                "pools": 0,
                "solar": 0,
                "context_detector_model": detector_model,
                "detector_signature": detector_signature["signature"],
                "segmentation_mode": segmentation_mode,
                "detector_precision": detector_precision,
                "queue": _detector_job_queue.status(),
            })
    else:
        pipeline_steps.append({
            "name": "satellite_pool_detector",
            "status": "skipped_testing" if is_testing else "skipped",
            "elements": 0,
            "pools": 0,
            "context_detector_model": detector_model,
            "detector_signature": detector_signature["signature"],
            "segmentation_mode": segmentation_mode,
            "detector_precision": detector_precision,
        })

    # ── Source 2b: Australia-wide building footprints (Microsoft Open Buildings) ──
    # When OSM returned few buildings at a close zoom, supplement so the Structure
    # Footprints / 3D mesh layers populate everywhere (not just well-mapped suburbs).
    ms_building_elements: list[dict[str, Any]] = []
    want_buildings = layers is None or "buildings" in (layers or "")
    osm_building_count = sum(1 for el in overpass_elements if el.get("tags", {}).get("building"))
    if (
        want_buildings
        and zoom >= 17
        and osm_building_count < MS_BUILDING_SUPPLEMENT_THRESHOLD
        and not is_testing
        and os.environ.get("QLDR_DISABLE_MS_BUILDINGS") != "1"
    ):
        try:
            from .ms_buildings import fetch_australia_buildings

            def _building_centroid(elem: dict[str, Any]) -> tuple[float, float] | None:
                geom = elem.get("geometry")
                if not isinstance(geom, list) or not geom:
                    return None
                lats = [float(p["lat"]) for p in geom if isinstance(p, dict) and "lat" in p]
                lons = [float(p["lon"]) for p in geom if isinstance(p, dict) and "lon" in p]
                if not lats or not lons:
                    return None
                return sum(lats) / len(lats), sum(lons) / len(lons)

            osm_building_centers = [
                center
                for el in overpass_elements
                if el.get("tags", {}).get("building") and (center := _building_centroid(el))
            ]
            for ms_el in fetch_australia_buildings(bbox, timeout=MS_BUILDING_SUPPLEMENT_TIMEOUT):
                center = _building_centroid(ms_el)
                if center is None:
                    continue
                # OSM is preferred (it carries addresses); skip a footprint that
                # duplicates an OSM building within ~11 m.
                if any(
                    abs(center[0] - oc[0]) <= 0.0001 and abs(center[1] - oc[1]) <= 0.0001
                    for oc in osm_building_centers
                ):
                    continue
                ms_building_elements.append(ms_el)
                osm_building_centers.append(center)
            pipeline_steps.append({
                "name": "australia_building_footprints",
                "status": "ok",
                "elements": len(ms_building_elements),
                "source": "microsoft_open_buildings",
            })
        except Exception as exc:  # best-effort national coverage
            pipeline_steps.append({
                "name": "australia_building_footprints",
                "status": "unavailable",
                "error": str(exc),
                "elements": 0,
            })

    # Collect building footprints to filter out blue roof false positives
    building_polygons = []
    for el in local_elements + overpass_elements + ms_building_elements:
        if el.get("type") == "way" and el.get("geometry") and el.get("tags", {}).get("building"):
            ring = [(float(pt["lon"]), float(pt["lat"])) for pt in el["geometry"]]
            building_polygons.append(ring)
            
    filtered_cv_elements = []
    for el in cv_elements:
        if _is_pool_element(el) and el.get("geometry"):
            pool_coords = [(float(pt["lon"]), float(pt["lat"])) for pt in el["geometry"]]
            p_cx = sum(pt[0] for pt in pool_coords) / len(pool_coords)
            p_cy = sum(pt[1] for pt in pool_coords) / len(pool_coords)
            
            inside_building = False
            for b_ring in building_polygons:
                if is_point_in_polygon_py(p_cx, p_cy, b_ring):
                    inside_building = True
                    break
            if inside_building:
                continue
        filtered_cv_elements.append(el)
    cv_elements = filtered_cv_elements

    # ── Merge & deduplicate ─────────────────────────────────────────────
    merged_elements = list(local_elements)

    if overpass_elements:
        merged_elements = _deduplicate_pools(merged_elements, overpass_elements)

    if cv_elements:
        merged_elements = _deduplicate_pools(merged_elements, cv_elements)

    # Microsoft footprints are buildings only (no pools), so append them directly.
    if ms_building_elements:
        merged_elements.extend(ms_building_elements)
        
    has_local = any(_is_pool_element(el) for el in local_elements)
    has_overpass = any(_is_pool_element(el) for el in overpass_elements)
    has_cv = any(_is_pool_element(el) for el in cv_elements)
    has_satellite_yolo = any(
        _is_pool_element(el) and el.get("tags", {}).get("detector:source") == "satellite-yolo"
        for el in cv_elements
    )
    
    if has_cv and (has_overpass or has_local):
        source = "hybrid"
    elif has_overpass and has_local:
        source = "hybrid"
    elif has_satellite_yolo:
        source = "satellite-yolo"
    elif has_cv:
        source = "satellite-cv"
    elif has_overpass:
        source = "hybrid"
    else:
        source = "local-properties"

    counts = {
        "pools": sum(1 for el in merged_elements if _is_pool_element(el)),
        "solar": sum(
            1
            for el in merged_elements
            if el.get("tags", {}).get("generator:source") == "solar"
        ),
        "buildings": sum(1 for el in merged_elements if el.get("tags", {}).get("building")),
        "trees": sum(
            1
            for el in merged_elements
            if el.get("tags", {}).get("natural") in {"tree", "wood", "scrub", "heath", "grassland", "shrubbery", "tree_row", "wetland"}
            or el.get("tags", {}).get("landuse") in {"forest", "orchard", "vineyard", "meadow", "grass", "recreation_ground", "greenfield"}
            or el.get("tags", {}).get("leisure") in {"garden", "park", "nature_reserve"}
        ),
    }
    satellite_step_status = next(
        (
            str(step.get("status") or "")
            for step in pipeline_steps
            if step.get("name") == "satellite_pool_detector"
        ),
        "",
    )
    detector_result_cacheable = satellite_step_status not in {
        "busy_this_process",
        "busy_other_process",
        "resource_exhausted",
        "error",
        "degraded",
        "unavailable",
    }
    result = {
        "elements": merged_elements,
        "source": source,
        "endpoint": endpoint,
        "counts": counts,
        "pipeline": {
            "version": "2026-07-13-selected-area-v1",
            "detection_scope": "selected-analysis-area-only",
            "selection_signature": selection_signature,
            "context_detector_model": detector_model,
            "detector_signature": detector_signature["signature"],
            "active_pool_weights": detector_signature["weights"],
            "segmentation_mode": segmentation_mode,
            "detector_precision": detector_precision,
            "cacheable": detector_result_cacheable,
            "steps": pipeline_steps,
            "detector_status": get_detector_status(),
        },
    }

    if not force_refresh and detector_result_cacheable:
        with _osm_feature_cache_lock:
            _osm_feature_cache[cache_key] = (now, result)
            _osm_feature_cache.move_to_end(cache_key)
            while len(_osm_feature_cache) > MAX_OSM_FEATURE_CACHE_ENTRIES:
                _osm_feature_cache.popitem(last=False)
        # Mirror into the shared (Redis) cache so other workers / later restarts
        # can serve this detection without re-running YOLO. Skipped for the
        # in-memory fallback, where L1 already holds the result.
        try:
            from .cache_backend import get_cache

            shared_cache = get_cache()
            if shared_cache.backend == "redis":
                shared_cache.set(shared_cache_key, result, OSM_FEATURE_CACHE_TTL_SECONDS)
        except Exception:
            pass

    return _with_cache_label(result, "bypass" if force_refresh else "miss")


def fetch_elevation_grid(
    bbox_values: list[Any] | tuple[Any, ...],
    *,
    size: int = 5,
    timeout: float = 6.0,
) -> dict[str, Any]:
    """Fetch a compact, cacheable elevation grid for real drainage contours."""
    south, west, north, east = validate_bbox(bbox_values)
    size = max(3, min(int(size), 8))
    latitudes = [south + (north - south) * index / (size - 1) for index in range(size)]
    longitudes = [west + (east - west) * index / (size - 1) for index in range(size)]
    sample_latitudes = [latitude for latitude in latitudes for _ in longitudes]
    sample_longitudes = [longitude for _ in latitudes for longitude in longitudes]
    lat_param = ",".join(f"{value:.6f}" for value in sample_latitudes)
    lon_param = ",".join(f"{value:.6f}" for value in sample_longitudes)
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_param}&longitude={lon_param}"
    payload = request_json_url(url, timeout=timeout, cache_ttl=300.0)
    elevations = payload.get("elevation") if isinstance(payload, dict) else None
    if not isinstance(elevations, list) or len(elevations) != size * size:
        raise ExternalServiceError("The elevation service returned an incomplete terrain grid.")
    try:
        rows = [
            [float(value) for value in elevations[row * size:(row + 1) * size]]
            for row in range(size)
        ]
    except (TypeError, ValueError) as exc:
        raise ExternalServiceError("The elevation service returned invalid terrain values.") from exc
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ExternalServiceError("The elevation service returned invalid terrain values.")
    return {
        "latitudes": latitudes,
        "longitudes": longitudes,
        "elevations": rows,
        "minimum_m": min(value for row in rows for value in row),
        "maximum_m": max(value for row in rows for value in row),
        "source": "open-meteo-elevation",
    }


def execute_overpass_query(query: str, *, timeout: float = 10.0) -> dict[str, Any]:
    """Execute validated application-owned Overpass QL with mirror failover."""
    global _preferred_overpass_endpoint
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Overpass query must not be empty.")
    payload = urllib.parse.urlencode({"data": query}).encode("utf-8")
    last_error: ExternalServiceError | None = None
    with _overpass_endpoint_lock:
        preferred = _preferred_overpass_endpoint
    endpoints = (
        (preferred,) + tuple(endpoint for endpoint in OVERPASS_ENDPOINTS if endpoint != preferred)
        if preferred
        else OVERPASS_ENDPOINTS
    )
    start_time = time.monotonic()
    num_endpoints = len(endpoints)
    for index, endpoint in enumerate(endpoints):
        elapsed = time.monotonic() - start_time
        remaining = timeout - elapsed
        if remaining <= 0.5:
            break
        
        remaining_endpoints = num_endpoints - index
        current_timeout = remaining if remaining_endpoints == 1 else min(remaining, max(2.0, remaining / remaining_endpoints))
        current_timeout = max(current_timeout, 1.0)
        
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )
        try:
            data = _request_json(request, timeout=current_timeout)
            elements = data.get("elements") if isinstance(data, dict) else None
            if not isinstance(elements, list):
                raise ExternalServiceError("Overpass returned an unexpected response.")
            with _overpass_endpoint_lock:
                _preferred_overpass_endpoint = endpoint
            return {"elements": elements, "endpoint": endpoint}
        except ExternalServiceError as exc:
            last_error = exc
    raise ExternalServiceError("OpenStreetMap data is temporarily unavailable.") from last_error


_nominatim_lock = threading.Lock()
_last_nominatim_request = 0.0
_geocode_cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
MAX_GEOCODE_CACHE_ENTRIES = 256


def geocode_places(
    query: str,
    *,
    limit: int = 5,
    timeout: float = 8.0,
    country_codes: str | None = None,
) -> list[dict[str, Any]]:
    """Search locations through Nominatim with caching and rate limiting.

    ``country_codes`` (comma-separated ISO codes, e.g. "au") restricts results to those
    countries — used to bias the area-search autocomplete to Australia.
    """
    normalized = " ".join(str(query).split())
    if len(normalized) < 2:
        raise ValueError("Enter at least two characters to search.")
    if len(normalized) > 160:
        raise ValueError("The search query is too long.")
    limit = max(1, min(int(limit), 5))
    country_codes = (country_codes or "").strip().lower()
    cache_key = f"{normalized.casefold()}:{limit}:{country_codes}"
    if cache_key in _geocode_cache:
        cached = _geocode_cache.pop(cache_key)
        _geocode_cache[cache_key] = cached
        return list(cached)

    query_params = {
        "q": normalized,
        "format": "jsonv2",
        "addressdetails": "1",
        "limit": str(limit),
    }
    if country_codes:
        query_params["countrycodes"] = country_codes
    params = urllib.parse.urlencode(query_params)
    request = urllib.request.Request(
        f"{NOMINATIM_ENDPOINT}?{params}",
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )

    global _last_nominatim_request
    with _nominatim_lock:
        delay = 1.0 - (time.monotonic() - _last_nominatim_request)
        if delay > 0:
            time.sleep(delay)
        data = _request_json(request, timeout=timeout)
        _last_nominatim_request = time.monotonic()

    if not isinstance(data, list):
        raise ExternalServiceError("The geocoding service returned an unexpected response.")
    results = []
    for item in data[:limit]:
        if not isinstance(item, dict) or "lat" not in item or "lon" not in item:
            continue
        results.append(
            {
                "display_name": str(item.get("display_name", "Unknown location")),
                "lat": str(item["lat"]),
                "lon": str(item["lon"]),
                "type": str(item.get("type", "")),
                "class": str(item.get("class") or item.get("category") or ""),
                "address": item.get("address") if isinstance(item.get("address"), dict) else {},
                "country": str(item.get("address", {}).get("country", "")) if isinstance(item.get("address"), dict) else "",
                "country_code": str(item.get("address", {}).get("country_code", "")).lower()
                if isinstance(item.get("address"), dict)
                else "",
                "boundingbox": item.get("boundingbox") if isinstance(item.get("boundingbox"), list) else None,
                "importance": item.get("importance"),
            }
        )
    _geocode_cache[cache_key] = list(results)
    while len(_geocode_cache) > MAX_GEOCODE_CACHE_ENTRIES:
        _geocode_cache.popitem(last=False)
    return results


_reverse_geocode_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
MAX_REVERSE_GEOCODE_CACHE_ENTRIES = 512


def _google_geocoding_api_key() -> str:
    """Return the configured server-side Google key without ever serialising it."""
    return (
        os.environ.get("ABI_GOOGLE_MAPS_API_KEY", "").strip()
        or os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    )


def google_geocoding_configured() -> bool:
    """Whether the optional official Google Geocoding adapter can be used."""
    return bool(_google_geocoding_api_key())


def _google_address_component(
    components: list[Any],
    component_type: str,
    *,
    short: bool = False,
) -> str:
    for component in components:
        if not isinstance(component, dict):
            continue
        types = component.get("types")
        if not isinstance(types, list) or component_type not in types:
            continue
        field = "short_name" if short else "long_name"
        return str(component.get(field, "")).strip()
    return ""


def _normalize_google_geocode_result(
    item: dict[str, Any],
    *,
    query_lat: float,
    query_lon: float,
) -> dict[str, Any]:
    raw_components = item.get("address_components")
    components = raw_components if isinstance(raw_components, list) else []
    country_code = _google_address_component(components, "country", short=True).upper()
    street_number = _google_address_component(components, "street_number")
    route = _google_address_component(components, "route")
    # A unit is populated only from Google's explicit subpremise component.  In
    # particular, do not try to parse one from formatted_address.
    unit = _google_address_component(components, "subpremise")
    locality = (
        _google_address_component(components, "locality")
        or _google_address_component(components, "postal_town")
        or _google_address_component(components, "sublocality_level_1")
        or _google_address_component(components, "administrative_area_level_2")
    )
    state = _google_address_component(components, "administrative_area_level_1")
    state_code = _google_address_component(
        components, "administrative_area_level_1", short=True
    )
    country = _google_address_component(components, "country")
    postcode = _google_address_component(components, "postal_code")

    raw_types = item.get("types")
    result_types = [str(value) for value in raw_types] if isinstance(raw_types, list) else []
    geometry = item.get("geometry") if isinstance(item.get("geometry"), dict) else {}
    location_type = str(geometry.get("location_type", "")).upper()
    raw_location = geometry.get("location") if isinstance(geometry.get("location"), dict) else {}
    location: dict[str, float] = {}
    distance_m: float | None = None
    try:
        result_lat = float(raw_location.get("lat"))
        result_lon = float(raw_location.get("lng"))
        if not (
            math.isfinite(result_lat)
            and math.isfinite(result_lon)
            and -90.0 <= result_lat <= 90.0
            and -180.0 <= result_lon <= 180.0
        ):
            raise ValueError("invalid Google result coordinate")
        location = {"lat": result_lat, "lng": result_lon}
        lat_1 = math.radians(query_lat)
        lat_2 = math.radians(result_lat)
        delta_lat = lat_2 - lat_1
        delta_lon = math.radians(result_lon - query_lon)
        haversine = (
            math.sin(delta_lat / 2.0) ** 2
            + math.cos(lat_1) * math.cos(lat_2) * math.sin(delta_lon / 2.0) ** 2
        )
        distance_m = round(
            2.0 * 6_371_008.8 * math.asin(min(1.0, math.sqrt(haversine))),
            2,
        )
    except (TypeError, ValueError):
        pass
    precision_by_location_type = {
        "ROOFTOP": "rooftop",
        "RANGE_INTERPOLATED": "interpolated",
        "GEOMETRIC_CENTER": "geometric_center",
        "APPROXIMATE": "approximate",
    }
    precision = precision_by_location_type.get(location_type, "unknown")
    supported_result = bool(
        {"subpremise", "premise", "street_address"}.intersection(result_types)
    )
    partial_match = bool(item.get("partial_match"))
    exact = bool(
        country_code == "AU"
        and street_number
        and route
        and supported_result
        and not partial_match
        and location_type == "ROOFTOP"
        and distance_m is not None
        and distance_m <= 35.0
    )
    return {
        "formatted_address": str(item.get("formatted_address", "")).strip(),
        "components": {
            "unit": unit,
            "street_number": street_number,
            "route": route,
            "locality": locality,
            "state": state,
            "state_code": state_code,
            "postcode": postcode,
            "country": country,
            "country_code": country_code,
        },
        "source": "google",
        "place_id": str(item.get("place_id", "")).strip(),
        "result_types": result_types,
        "partial_match": partial_match,
        "location_type": location_type,
        "location": location,
        "distance_m": distance_m,
        "precision": precision,
        "verified": exact,
        "exact": exact,
    }


def google_reverse_geocode(
    lat: float,
    lon: float,
    *,
    timeout: float = 8.0,
) -> dict[str, Any]:
    """Resolve a coordinate through Google's official Geocoding API v3.

    The adapter is optional and returns ``{}`` when no server-side key is
    configured, the coordinate is invalid, or Google cannot provide a usable
    result. API keys and upstream error details are deliberately never returned.
    """
    api_key = _google_geocoding_api_key()
    if not api_key:
        return {}
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return {}
    if not (
        math.isfinite(lat_f)
        and math.isfinite(lon_f)
        and -90.0 <= lat_f <= 90.0
        and -180.0 <= lon_f <= 180.0
    ):
        return {}

    params = urllib.parse.urlencode(
        {
            "latlng": f"{lat_f:.7f},{lon_f:.7f}",
            "key": api_key,
            "language": "en",
            "region": "au",
        }
    )
    request = urllib.request.Request(
        f"{GOOGLE_GEOCODING_ENDPOINT}?{params}",
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        data = _request_json(request, timeout=timeout)
    except Exception:
        # _request_json already redacts the URL. Keep this best-effort adapter
        # silent so a key can never escape through a returned exception string.
        return {}
    if not isinstance(data, dict) or data.get("status") != "OK":
        return {}
    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        return {}

    normalized = [
        _normalize_google_geocode_result(item, query_lat=lat_f, query_lon=lon_f)
        for item in raw_results
        if isinstance(item, dict)
    ]
    if not normalized:
        return {}

    # Google may return several granularities for a coordinate. Prefer a result
    # that satisfies our defensible exact-address contract, then an explicitly
    # unit-bearing result, then Google's original ordering.
    best = max(
        enumerate(normalized),
        key=lambda pair: (
            int(pair[1]["exact"]),
            int(bool(pair[1]["components"].get("unit"))),
            -pair[0],
        ),
    )[1]
    if not best["formatted_address"]:
        return {}

    return best


def reverse_geocode(lat: float, lon: float, *, zoom: int = 18, timeout: float = 8.0) -> dict[str, Any]:
    """Resolve a lat/lon to a street address via Nominatim reverse geocoding.

    Returns ``{"display_name": str, "address": {...}}`` (empty strings/dict on
    failure). Results are cached and rate-limited (Nominatim allows ~1 req/sec), so
    callers should reverse-geocode only their highest-value points, not every lead.
    """
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return {"display_name": "", "address": {}}
    if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
        return {"display_name": "", "address": {}}
    cache_key = f"{lat_f:.5f}:{lon_f:.5f}:{int(zoom)}"
    if cache_key in _reverse_geocode_cache:
        cached = _reverse_geocode_cache.pop(cache_key)
        _reverse_geocode_cache[cache_key] = cached
        return dict(cached)

    params = urllib.parse.urlencode(
        {
            "lat": f"{lat_f:.7f}",
            "lon": f"{lon_f:.7f}",
            "format": "jsonv2",
            "addressdetails": "1",
            "zoom": str(int(zoom)),
        }
    )
    request = urllib.request.Request(
        f"{NOMINATIM_REVERSE_ENDPOINT}?{params}",
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    global _last_nominatim_request
    try:
        with _nominatim_lock:
            delay = 1.0 - (time.monotonic() - _last_nominatim_request)
            if delay > 0:
                time.sleep(delay)
            data = _request_json(request, timeout=timeout)
            _last_nominatim_request = time.monotonic()
    except Exception:
        return {"display_name": "", "address": {}}

    if not isinstance(data, dict):
        return {"display_name": "", "address": {}}
    result = {
        "display_name": str(data.get("display_name", "")),
        "address": data.get("address") if isinstance(data.get("address"), dict) else {},
    }
    _reverse_geocode_cache[cache_key] = dict(result)
    while len(_reverse_geocode_cache) > MAX_REVERSE_GEOCODE_CACHE_ENTRIES:
        _reverse_geocode_cache.popitem(last=False)
    return result
