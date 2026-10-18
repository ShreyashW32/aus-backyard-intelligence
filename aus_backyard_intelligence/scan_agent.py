from __future__ import annotations

import math
import os
import threading
import time
import urllib.parse
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .external_api import (
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_SEGMENTATION_MODE,
    build_overpass_query,
    execute_overpass_query,
    fetch_osm_features,
    google_geocoding_configured,
    google_reverse_geocode,
    reverse_geocode,
    validate_bbox,
)
from .jurisdiction import POOL_COMPLIANCE, pool_compliance
from .ms_buildings import fetch_australia_buildings


MAX_SCAN_JOBS = 24
MAX_RETIRED_SCAN_TOMBSTONES = MAX_SCAN_JOBS * 4
MAX_JOB_LEADS = 80
# Cap on features returned to the client for map rendering. Pools are emitted first (see
# _feature_payload), so this must be high enough to carry ALL detected pools of a large
# scan — otherwise the map only ever shows the first ~1200, clustered in whichever blocks
# were scanned first, and the rest of the selection looks empty.
MAX_JOB_FEATURES = 8000
# YOLO checkpoints and stitched satellite images are memory intensive. A second active
# scan offers no throughput gain on a single local detector and can terminate the whole
# API process under memory pressure.
MAX_ACTIVE_SCAN_JOBS = 1
MAX_SCAN_ZOOM = 20
PRECISE_SCAN_MODEL = "yolov11s"

# Cross-pass refinement tuning. Each pass scans a slightly shifted window of the
# same area; these control how detections are clustered into one physical pool and
# which clustered detections survive as confirmed pools vs. rejected false positives.
POOL_MATCH_RADIUS_M = 7.0          # two detections within this distance are the same pool
POOL_KEEP_MIN_CONFIDENCE = 0.80    # a single strong detection is enough to confirm
POOL_REJECT_MIN_COVERAGE = 3       # only reject a 1-vote pool if this many passes covered it
# Accuracy gates for published satellite detections (mapped OSM pools are ground truth
# and exempt from all of these):
#  - HARD_FLOOR: below this the detection is junk regardless of votes.
#  - SOLO: a single sighting needs this much confidence to publish on its own.
#  - VOTED: with 2+ cross-pass sightings, this lower bar suffices — repeated agreement
#    at moderate confidence is stronger evidence than one lucky high score, and this is
#    what keeps recall up (real pools often score 0.55-0.75 on z20 tiles).
POOL_CONF_HARD_FLOOR = 0.50
POOL_CONF_SOLO_PUBLISH = 0.80
POOL_CONF_VOTED_PUBLISH = 0.55
POOL_CONF_PROBABLE_MIN = 0.65      # unresolved single sightings need at least this
# Physical plausibility for a residential/backyard pool footprint. Detections outside
# this range are rock pools, ocean patches, lakes, or tarps — reject them outright.
# (Large public aquatic centres are almost always OSM-mapped, which bypasses this.)
SATELLITE_POOL_MIN_AREA_SQM = 8.0
SATELLITE_POOL_MAX_AREA_SQM = 400.0

# Lead generation tuning (residential pool/landscaping prospecting).
LEAD_MIN_SCORE = 42.0              # publish threshold
LEAD_ON_PROPERTY_POOL_M = 22.0    # a pool this close counts as the property's own → not a lead
LEAD_NEIGHBOUR_POOL_M = 180.0     # pools within this (but not on-property) = pool-friendly area
LEAD_MIN_FOOTPRINT_SQM = 45.0     # smaller footprints are sheds/garages, not dwellings
LEAD_MAX_FOOTPRINT_SQM = 900.0    # larger footprints are commercial/industrial, not homes
LEAD_REVERSE_GEOCODE_LIMIT = 12   # keyless Nominatim fallback limit
LEAD_GOOGLE_GEOCODE_LIMIT = 24    # dashboard publishes the top 24; bound API latency/cost

# OSM building values that are dwellings we want to prospect, and ones to exclude.
RESIDENTIAL_BUILDING_VALUES = {
    "house", "detached", "semidetached_house", "semi", "terrace", "terraced_house",
    "residential", "bungalow", "dwelling", "villa", "duplex", "house;residential",
}
NON_RESIDENTIAL_BUILDING_VALUES = {
    "shed", "garage", "garages", "carport", "roof", "hut", "cabin", "container",
    "industrial", "commercial", "retail", "warehouse", "hangar", "farm_auxiliary",
    "service", "kiosk", "greenhouse", "barn", "stable", "office", "supermarket",
    "school", "hospital", "church", "civic", "public", "parking", "transportation",
}


def _now() -> float:
    return round(time.time(), 3)


def _polygon_scope_key(polygon: list[list[float]]) -> str:
    """Canonical selection identity, independent of ring start/direction."""
    points = tuple((round(float(point[0]), 7), round(float(point[1]), 7)) for point in polygon)
    if not points:
        return ""
    candidates: list[tuple[tuple[float, float], ...]] = []
    for sequence in (points, tuple(reversed(points))):
        candidates.extend(sequence[index:] + sequence[:index] for index in range(len(sequence)))
    canonical = min(candidates)
    return ";".join(f"{lon:.7f},{lat:.7f}" for lon, lat in canonical)


def _coerce_polygon(raw_polygon: Any) -> list[list[float]]:
    if not isinstance(raw_polygon, list) or len(raw_polygon) < 3:
        raise ValueError("polygon must contain at least three [longitude, latitude] pairs.")
    coords: list[list[float]] = []
    for point in raw_polygon:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each polygon coordinate must be [longitude, latitude].")
        lon = float(point[0])
        lat = float(point[1])
        if not (math.isfinite(lon) and math.isfinite(lat)):
            raise ValueError("Polygon coordinates must be finite numbers.")
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("Polygon coordinates are outside valid longitude/latitude ranges.")
        coords.append([lon, lat])
    if len(coords) > 3 and coords[0] == coords[-1]:
        coords.pop()
    return coords


def _bbox_from_request(payload: dict[str, Any]) -> tuple[float, float, float, float]:
    if payload.get("polygon") is not None:
        polygon = _coerce_polygon(payload["polygon"])
        south = min(point[1] for point in polygon)
        west = min(point[0] for point in polygon)
        north = max(point[1] for point in polygon)
        east = max(point[0] for point in polygon)
        return validate_bbox([south, west, north, east])
    bounds = payload.get("bounds")
    if isinstance(bounds, dict):
        return validate_bbox([bounds.get("south"), bounds.get("west"), bounds.get("north"), bounds.get("east")])
    if isinstance(bounds, (list, tuple)):
        return validate_bbox(bounds)
    raise ValueError("Provide either polygon or bounds for the scan agent.")


def _z20_tile_span_deg(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Approximate degrees covered by one max-zoom tile (lat, lon) at this latitude."""
    south, _west, north, _east = bbox
    lat_mid = (south + north) / 2.0
    lon_deg = 360.0 / (2 ** MAX_SCAN_ZOOM)
    lat_deg = lon_deg * max(math.cos(math.radians(lat_mid)), 1e-6)
    return lat_deg, lon_deg


def _grid_bboxes(
    bbox: tuple[float, float, float, float],
    nx: int,
    ny: int,
) -> list[tuple[float, float, float, float]]:
    """Split a bbox into an ``nx`` × ``ny`` grid of cells. A very large selection is
    subdivided into more cells than a small one, so each cell still fits inside the
    detector's per-call tile budget and gets full, untruncated attention."""
    south, west, north, east = bbox
    out: list[tuple[float, float, float, float]] = []
    for iy in range(max(1, ny)):
        for ix in range(max(1, nx)):
            s = south + (north - south) * iy / ny
            n = south + (north - south) * (iy + 1) / ny
            w = west + (east - west) * ix / nx
            e = west + (east - west) * (ix + 1) / nx
            try:
                out.append(validate_bbox([s, w, n, e]))
            except ValueError:
                continue
    return out


# The precise detector fully scans roughly this many max-zoom tiles per call before it
# truncates the rest (see _MAX_DETECTION_CHUNKS in yolo_detector). Keep each scan cell
# under this so every part of a large selection is actually reached instead of dropped.
SCAN_CELL_TILE_BUDGET = 1600
# Upper bound on coverage cells so a very large selection can't spawn an unbounded scan.
MAX_COVERAGE_CELLS = 60
# Each coverage block is grown by this fraction so neighbours overlap and a pool sitting
# on a block seam is fully imaged (and detected) in at least one block — a precision win.
CELL_OVERLAP_FRAC = 0.05
# Hard ceiling on detector calls per scan so a huge selection stays bounded in time.
# Coverage always wins the budget first (every block ≥1 sweep); leftover budget buys the
# requested refinement sweeps. A smaller selection therefore gets deeper refinement.
MAX_TOTAL_SCAN_WINDOWS = 160


def _coverage_grid_dims(tile_count: int) -> tuple[int, int]:
    """Grid (nx, ny) that tiles ``tile_count`` max-zoom tiles into cells each within the
    detector's per-call budget, so a large selection is covered without truncation. Small
    selections that already fit in one call stay a single window."""
    cells_needed = max(1, math.ceil(tile_count / SCAN_CELL_TILE_BUDGET))
    cells_needed = min(cells_needed, MAX_COVERAGE_CELLS)
    ny = max(1, math.ceil(math.sqrt(cells_needed)))
    nx = max(1, math.ceil(cells_needed / ny))
    return nx, ny


def _pad_bbox(bbox: tuple[float, float, float, float], frac: float) -> tuple[float, float, float, float]:
    """Grow a bbox outward by ``frac`` of its size (split across both edges)."""
    south, west, north, east = bbox
    dh = (north - south) * frac / 2.0
    dw = (east - west) * frac / 2.0
    try:
        return validate_bbox([south - dh, west - dw, north + dh, east + dw])
    except ValueError:
        return bbox


def _shift_bbox(bbox: tuple[float, float, float, float], dx: float, dy: float) -> tuple[float, float, float, float]:
    south, west, north, east = bbox
    try:
        return validate_bbox([south + dy, west + dx, north + dy, east + dx])
    except ValueError:
        return bbox


def _sweep_jitter(bbox: tuple[float, float, float, float], sweep_index: int) -> tuple[float, float]:
    """Per-sweep (dx, dy) offset. Sweep 0 is unshifted; later sweeps step through eight
    seam-busting directions (then shrink) so tile-alignment blind spots differ each sweep
    and a pool missed under one alignment is caught under another."""
    if sweep_index <= 0:
        return (0.0, 0.0)
    south, west, north, east = bbox
    lat_tile, lon_tile = _z20_tile_span_deg(bbox)
    # Seam-busting jitter ~0.6 of a tile, but never more than a third of the area so a
    # shifted window still overlaps the selection (and its pools) heavily.
    jx = min(lon_tile * 0.6, (east - west) / 3.0 or lon_tile * 0.6)
    jy = min(lat_tile * 0.6, (north - south) / 3.0 or lat_tile * 0.6)
    offsets = [
        (jx, jy), (-jx, -jy), (jx, -jy), (-jx, jy),  # diagonal shifts
        (jx, 0.0), (-jx, 0.0), (0.0, jy), (0.0, -jy),  # cardinal shifts
    ]
    dx, dy = offsets[(sweep_index - 1) % len(offsets)]
    scale = 1.0 / (1 + (sweep_index - 1) // len(offsets))  # tighter jitter each cycle
    return (dx * scale, dy * scale)


def _segments_intersect(
    p1: tuple[float, float], p2: tuple[float, float],
    p3: tuple[float, float], p4: tuple[float, float],
) -> bool:
    def orient(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _bbox_intersects_polygon(
    bbox: tuple[float, float, float, float],
    polygon: list[list[float]],
) -> bool:
    """True if the [south, west, north, east] cell overlaps the drawn polygon at all.
    Used to drop coverage cells that fall entirely outside a non-rectangular selection so
    the scan never wanders beyond the boundary the user actually drew."""
    if not polygon or len(polygon) < 3:
        return True
    south, west, north, east = bbox
    pminx = min(p[0] for p in polygon)
    pmaxx = max(p[0] for p in polygon)
    pminy = min(p[1] for p in polygon)
    pmaxy = max(p[1] for p in polygon)
    if east < pminx or west > pmaxx or north < pminy or south > pmaxy:
        return False  # bounding boxes don't even touch
    corners = [(west, south), (east, south), (east, north), (west, north)]  # (lon, lat)
    if any(_point_in_polygon(lon, lat, polygon) for lon, lat in corners):
        return True
    if _point_in_polygon((west + east) / 2.0, (south + north) / 2.0, polygon):
        return True
    if any(west <= p[0] <= east and south <= p[1] <= north for p in polygon):
        return True
    cell_edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    poly_edges = [(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))]
    for a, b in cell_edges:
        for c, d in poly_edges:
            if _segments_intersect(a, b, (c[0], c[1]), (d[0], d[1])):
                return True
    return False


@dataclass
class _ScanPlan:
    """The concrete list of detector windows for a scan, plus the two counts the UI needs
    to report progress honestly: ``blocks`` (spatial coverage tiles of the selection) and
    ``sweeps`` (refinement re-scans of each block for cross-pass voting)."""

    windows: list[tuple[float, float, float, float]]
    blocks: int
    sweeps: int
    requested_passes: int
    tile_count: int


def _build_scan_plan(
    bbox: tuple[float, float, float, float],
    passes: int,
    polygon: list[list[float]] | None = None,
) -> _ScanPlan:
    """Plan the detector windows for a selection.

    The detector truncates any single call at a fixed tile budget, so a large selection
    scanned as one bbox leaves most of its area (and pools) unreached. The area is
    therefore split into **coverage blocks** each small enough to scan in full, clipped to
    the drawn polygon so nothing outside the selection is scanned, and lightly overlapped
    so seam pools are still captured. Each block is then re-scanned for ``passes``
    **refinement sweeps** (with per-sweep jitter) so cross-pass voting can confirm real
    pools and reject flickering false positives.

    Coverage is guaranteed first: every block runs at least once. Only leftover budget
    (up to ``MAX_TOTAL_SCAN_WINDOWS`` total calls) buys extra sweeps, so a smaller / more
    tightly drawn selection automatically earns deeper refinement.
    """
    passes = max(1, int(passes))
    tile_count = _max_zoom_tile_count(bbox)
    nx, ny = _coverage_grid_dims(tile_count)
    if nx * ny > 1:
        cells = _grid_bboxes(bbox, nx, ny)
        if polygon:
            clipped = [cell for cell in cells if _bbox_intersects_polygon(cell, polygon)]
            if clipped:
                cells = clipped
        blocks = [_pad_bbox(cell, CELL_OVERLAP_FRAC) for cell in cells]
    else:
        blocks = [bbox]  # small selection fits one detector call — scan it whole
    n_blocks = max(1, len(blocks))

    # Coverage first, refinement with whatever budget is left.
    sweeps = passes
    if n_blocks * sweeps > MAX_TOTAL_SCAN_WINDOWS:
        sweeps = max(1, MAX_TOTAL_SCAN_WINDOWS // n_blocks)

    windows: list[tuple[float, float, float, float]] = []
    for sweep in range(sweeps):
        dx, dy = _sweep_jitter(bbox, sweep)
        for block in blocks:
            windows.append(_shift_bbox(block, dx, dy) if (dx or dy) else block)
    if not windows:
        windows = [bbox]
    return _ScanPlan(windows, n_blocks, sweeps, passes, tile_count)


def _scan_bboxes_for_passes(
    bbox: tuple[float, float, float, float],
    passes: int,
    polygon: list[list[float]] | None = None,
) -> list[tuple[float, float, float, float]]:
    """Thin wrapper returning just the ordered scan windows (see ``_build_scan_plan``)."""
    return _build_scan_plan(bbox, passes, polygon).windows


def _lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    lon = max(min(lon, 180.0), -180.0)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def _max_zoom_tile_count(bbox: tuple[float, float, float, float], zoom: int = MAX_SCAN_ZOOM) -> int:
    south, west, north, east = bbox
    x_west, y_south = _lat_lon_to_tile(south, west, zoom)
    x_east, y_north = _lat_lon_to_tile(north, east, zoom)
    x_min, x_max = sorted((x_west, x_east))
    y_min, y_max = sorted((y_north, y_south))
    return max(1, x_max - x_min + 1) * max(1, y_max - y_min + 1)


def _feature_key(element: dict[str, Any], fallback_index: int) -> str:
    return f"{element.get('type', 'feature')}:{element.get('id', fallback_index)}"


def _is_pool(element: dict[str, Any]) -> bool:
    tags = element.get("tags") if isinstance(element, dict) else None
    if not isinstance(tags, dict):
        return False
    return (
        tags.get("leisure") == "swimming_pool"
        or tags.get("swimming_pool") not in (None, "no")
        or tags.get("water") in {"pool", "swimming_pool", "wading_pool"}
        or tags.get("amenity") in {"public_bath", "swimming_pool"}
        or tags.get("sport") == "swimming"
    )


def _is_solar(element: dict[str, Any]) -> bool:
    tags = element.get("tags") if isinstance(element, dict) else None
    if not isinstance(tags, dict):
        return False
    return (
        tags.get("generator:source") == "solar"
        or str(tags.get("detector:source", "")).startswith("satellite-yolo-solar")
    )


def _solar_feature_key(element: dict[str, Any], fallback_index: int) -> str:
    """Stable-enough key across shifted passes so one array is not published repeatedly."""
    center = _centroid(element)
    if center is None:
        return _feature_key(element, fallback_index)
    return f"solar:{center[0]:.5f}:{center[1]:.5f}"


def _is_satellite_pool(element: dict[str, Any]) -> bool:
    """True for a pool detected from satellite imagery (YOLO/CV) rather than an
    OpenStreetMap-mapped pool. Mapped pools are trusted ground truth and are never
    rejected as false positives; satellite detections must earn their place by
    cross-pass agreement or detector confidence."""
    tags = element.get("tags") if isinstance(element, dict) else None
    if not isinstance(tags, dict):
        return False
    source = str(tags.get("detector:source", ""))
    return source.startswith("satellite") or source in {"yolo-cv"}


def _pool_detection_confidence(element: dict[str, Any]) -> float:
    """Per-detection confidence in [0, 1]. Mapped pools score 1.0; satellite pools
    use the higher of the YOLO / colour-CV confidence the detector recorded."""
    if not _is_satellite_pool(element):
        return 1.0
    tags = element.get("tags") if isinstance(element, dict) else {}
    if not isinstance(tags, dict):
        return 0.0
    best = 0.0
    for key in ("confidence:yolo", "confidence:cv"):
        try:
            best = max(best, float(tags.get(key, 0) or 0))
        except (TypeError, ValueError):
            continue
    return best


def _is_building(element: dict[str, Any]) -> bool:
    tags = element.get("tags") if isinstance(element, dict) else None
    return isinstance(tags, dict) and bool(tags.get("building")) and tags.get("building") not in {"no", "demolished"}


def _polygon_area_sqm(element: dict[str, Any]) -> float:
    """Footprint area of a building polygon in square metres (shoelace in a local
    equirectangular projection — accurate enough for footprint sizing)."""
    geometry = element.get("geometry")
    if not isinstance(geometry, list) or len(geometry) < 3:
        return 0.0
    pts = [
        (float(p["lat"]), float(p["lon"]))
        for p in geometry
        if isinstance(p, dict) and "lat" in p and "lon" in p
    ]
    if len(pts) < 3:
        return 0.0
    lat0 = sum(p[0] for p in pts) / len(pts)
    m_per_lat = 111_320.0
    m_per_lon = 111_320.0 * max(math.cos(math.radians(lat0)), 1e-9)
    xy = [(lon * m_per_lon, lat * m_per_lat) for lat, lon in pts]
    area2 = 0.0
    for i in range(len(xy)):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % len(xy)]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) / 2.0


def _satellite_pool_area_sqm(element: dict[str, Any]) -> float:
    """Return the detector's physical box area, falling back to output geometry.

    Pool YOLO records ``pool:area_sqm`` before its optional water-mask outline
    refinement.  The refined ring is presentation geometry and may expand into
    the detector's padded crop or contract to one visible water component, so
    recomputing the Leads plausibility gate from that ring can move an otherwise
    valid detection across the 8--400 m² limits.  The detector tag is internal,
    finite, and uses the same pixel-to-ground conversion as the upstream gate.
    Older/mapped elements without the tag retain the geometry fallback.
    """
    tags = element.get("tags") if isinstance(element, dict) else None
    if isinstance(tags, dict):
        try:
            detector_area = float(tags.get("pool:area_sqm", 0) or 0)
        except (TypeError, ValueError):
            detector_area = 0.0
        if math.isfinite(detector_area) and detector_area > 0:
            return detector_area
    return _polygon_area_sqm(element)


def _is_residential_building(element: dict[str, Any]) -> bool:
    """True for a dwelling worth prospecting (a house with a backyard), filtering out
    sheds, garages, and commercial/industrial structures."""
    if not _is_building(element):
        return False
    tags = element.get("tags") or {}
    value = str(tags.get("building", "")).strip().lower()
    if value in NON_RESIDENTIAL_BUILDING_VALUES:
        return False
    # Explicit non-residential function tags rule it out (unless tagged residential).
    if value not in RESIDENTIAL_BUILDING_VALUES and any(
        tags.get(k) for k in ("shop", "amenity", "office", "tourism", "craft", "industrial", "healthcare")
    ):
        return False
    if value in RESIDENTIAL_BUILDING_VALUES:
        return True
    # building=yes / unknown: accept only if the footprint is house-sized.
    if value in {"yes", "", "construction", "house;yes"}:
        area = _polygon_area_sqm(element)
        return LEAD_MIN_FOOTPRINT_SQM <= area <= LEAD_MAX_FOOTPRINT_SQM
    return False


def _is_tree_or_greenery(element: dict[str, Any]) -> bool:
    tags = element.get("tags") if isinstance(element, dict) else None
    if not isinstance(tags, dict):
        return False
    return (
        tags.get("natural") in {"tree", "wood", "scrub", "heath"}
        or tags.get("landuse") in {"forest", "orchard", "vineyard"}
        or tags.get("leisure") == "garden"
    )


def _is_powerline(element: dict[str, Any]) -> bool:
    tags = element.get("tags") if isinstance(element, dict) else None
    return isinstance(tags, dict) and tags.get("power") == "line"


def _detector_info_from_result(result: dict[str, Any]) -> dict[str, Any]:
    pipeline = result.get("pipeline") if isinstance(result, dict) else {}
    if not isinstance(pipeline, dict):
        pipeline = {}
    detector_status = pipeline.get("detector_status") if isinstance(pipeline.get("detector_status"), dict) else {}
    satellite_step = {}
    steps = pipeline.get("steps")
    if isinstance(steps, list):
        satellite_step = next(
            (step for step in steps if isinstance(step, dict) and step.get("name") == "satellite_pool_detector"),
            {},
        )
    active_weights = detector_status.get("active_pool_weights") or pipeline.get("active_pool_weights") or []
    active_model_names = [
        str(item.get("name"))
        for item in active_weights
        if isinstance(item, dict) and item.get("name")
    ]
    return {
        "engine": detector_status.get("primary_pool_detector") or "satellite_water_shape_cv",
        "model": " + ".join(active_model_names) or "mapped-data-only",
        "detector_signature": detector_status.get("detector_signature") or pipeline.get("detector_signature"),
        "context_model": pipeline.get("context_detector_model") or satellite_step.get("context_detector_model"),
        "precision": pipeline.get("detector_precision") or satellite_step.get("detector_precision"),
        "validator": detector_status.get("secondary_validator") or "none",
        "status": satellite_step.get("status") or "unknown",
        "error": satellite_step.get("error"),
        "retryable": bool(satellite_step.get("retryable")),
        "diagnostics": satellite_step.get("diagnostics") if isinstance(satellite_step.get("diagnostics"), dict) else {},
        "note": detector_status.get("pool_detection_note")
        or "Manual scans currently use satellite water/shape CV unless a trained pool YOLO model is configured.",
    }


def _address_label(element: dict[str, Any], center: tuple[float, float]) -> str:
    tags = element.get("tags") if isinstance(element, dict) else {}
    if not isinstance(tags, dict):
        tags = {}
    # addr:flats may describe every unit in a block; it does not identify which
    # dwelling the footprint represents. Only addr:unit is a selected unit.
    unit = tags.get("addr:unit")
    house_number = tags.get("addr:housenumber")
    street = tags.get("addr:street")
    suburb = tags.get("addr:suburb") or tags.get("addr:city")
    if house_number and street:
        label = f"{house_number} {street}"
        if unit:
            label = f"Unit {unit}, {label}"
        if suburb:
            label = f"{label}, {suburb}"
        return str(label)
    if street:
        return f"Property on {street}"
    return "Address lookup pending"


def _osm_address_components(element: dict[str, Any]) -> dict[str, str]:
    """Normalize explicit OSM building address tags without inventing a unit."""
    tags = element.get("tags") if isinstance(element, dict) else {}
    if not isinstance(tags, dict):
        tags = {}

    def value(*keys: str) -> str:
        for key in keys:
            candidate = str(tags.get(key) or "").strip()
            if candidate:
                return candidate
        return ""

    return {
        "unit": value("addr:unit"),
        "available_units": value("addr:flats"),
        "house_number": value("addr:housenumber"),
        "road": value("addr:street"),
        "suburb": value("addr:suburb", "addr:city", "addr:place"),
        "state": value("addr:state"),
        "postcode": value("addr:postcode"),
        "country_code": value("addr:country").lower(),
    }


def _address_details(
    address: dict[str, Any],
    *,
    source: str,
    precision: str,
    verified: bool,
    formatted: str = "",
    status: str | None = None,
    source_id: str = "",
    place_id: str = "",
    checked_at: float | None = None,
) -> dict[str, Any]:
    """Public address contract used by the API and dashboard.

    ``unit`` is copied only from an explicit source component. It is never derived
    from a footprint, lot number, or nearby address.
    """
    addr = address if isinstance(address, dict) else {}
    unit = str(addr.get("unit") or addr.get("subpremise") or addr.get("flat") or "").strip()
    street_number = str(addr.get("street_number") or addr.get("house_number") or "").strip()
    route = str(addr.get("route") or addr.get("road") or "").strip()
    locality = str(
        addr.get("locality")
        or addr.get("suburb")
        or addr.get("neighbourhood")
        or addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or ""
    ).strip()
    details = {
        "formatted": str(formatted or _format_street_address(addr)).strip(),
        "unit": unit,
        "street_number": street_number,
        "route": route,
        "locality": locality,
        "state": str(addr.get("state") or "").strip(),
        "postcode": str(addr.get("postcode") or addr.get("postal_code") or "").strip(),
        "country_code": str(addr.get("country_code") or "").strip().lower(),
        "source": source,
        "source_id": source_id,
        "precision": precision,
        "status": status or ("verified" if verified else "approximate"),
        "verified": bool(verified),
        "unit_identified": bool(unit),
        "checked_at": checked_at,
    }
    available_units = str(addr.get("available_units") or "").strip()
    if available_units:
        details["available_units"] = available_units
    if place_id:
        details["place_id"] = place_id
    return details


def _pending_address_details() -> dict[str, Any]:
    return _address_details(
        {},
        source="pending",
        precision="pending",
        verified=False,
        status="pending",
    )


# State land-titles registries — the lawful first stop for owner/title lookups in AU.
_AU_LAND_TITLES = {
    "new south wales": "https://www.nswlrs.com.au/Title-Searching",
    "queensland": "https://www.titlesqld.com.au/",
    "victoria": "https://www.landata.vic.gov.au/",
    "south australia": "https://www.sailis.sa.gov.au/",
    "western australia": "https://www0.landgate.wa.gov.au/",
    "tasmania": "https://www.thelist.tas.gov.au/",
    "australian capital territory": "https://www.accesscanberra.act.gov.au/s/article/land-titles-tab-overview",
    "northern territory": "https://nt.gov.au/property/land-titles",
}
_AU_LAND_TITLES.update(
    {
        "nsw": _AU_LAND_TITLES["new south wales"],
        "qld": _AU_LAND_TITLES["queensland"],
        "vic": _AU_LAND_TITLES["victoria"],
        "sa": _AU_LAND_TITLES["south australia"],
        "wa": _AU_LAND_TITLES["western australia"],
        "tas": _AU_LAND_TITLES["tasmania"],
        "act": _AU_LAND_TITLES["australian capital territory"],
        "nt": _AU_LAND_TITLES["northern territory"],
    }
)


def _extract_osm_contact(element: dict[str, Any]) -> dict[str, str]:
    """Pull any contact details the owner has *published* in OpenStreetMap (mostly
    businesses). These are public; residential homes almost never carry them."""
    tags = element.get("tags") if isinstance(element, dict) else {}
    if not isinstance(tags, dict):
        tags = {}

    def first(*keys: str) -> str:
        for key in keys:
            value = tags.get(key)
            if value:
                return str(value).strip()
        return ""

    website = first("website", "contact:website", "url", "contact:url")
    # OSM tag values are community-edited; only pass through web URLs so a
    # crafted javascript:/data: value can never become a clickable link.
    if website and not website.lower().startswith(("http://", "https://")):
        website = ""

    return {
        "name": first("name", "operator", "brand"),
        "phone": first("phone", "contact:phone", "contact:mobile", "mobile"),
        "website": website,
        "email": first("email", "contact:email"),
    }


def _format_street_address(address: dict[str, Any]) -> str:
    """Build a human street address while preserving an explicit unit."""
    if not isinstance(address, dict):
        return ""
    unit = str(address.get("unit") or address.get("subpremise") or address.get("flat") or "").strip()
    house = str(address.get("street_number") or address.get("house_number") or "").strip()
    road = str(address.get("route") or address.get("road") or "").strip()
    suburb = str(
        address.get("locality")
        or address.get("suburb")
        or address.get("neighbourhood")
        or address.get("city")
        or address.get("town")
        or address.get("village")
        or ""
    ).strip()
    state = str(address.get("state", "")).strip()
    postcode = str(address.get("postcode") or address.get("postal_code") or "").strip()
    line = " ".join(part for part in (house, road) if part).strip()
    if unit and line:
        line = f"Unit {unit}, {line}"
    elif unit:
        line = f"Unit {unit}"
    tail = ", ".join(part for part in (suburb, state, postcode) if part)
    if line and tail:
        return f"{line}, {tail}"
    return line or tail


def _lookup_links(
    title: str,
    address: dict[str, Any],
    center: tuple[float, float],
    contact: dict[str, str],
) -> dict[str, str]:
    """Targeted, lawful lookup links for following a lead up via public records."""
    lat, lon = center
    quote = urllib.parse.quote_plus
    addr = address if isinstance(address, dict) else {}
    state = str(addr.get("state", "")).strip().lower()
    council = str(
        addr.get("municipality")
        or addr.get("locality")
        or addr.get("city")
        or addr.get("town")
        or addr.get("suburb")
        or ""
    ).strip()
    land_titles = _AU_LAND_TITLES.get(
        state, "https://www.google.com/search?q=" + quote(f"{state} land titles registry title search".strip())
    )
    place_id = str(addr.get("place_id") or "").strip()
    maps_query = title if place_id else f"{lat:.7f},{lon:.7f}"
    google_maps = "https://www.google.com/maps/search/?api=1&query=" + quote(maps_query)
    if place_id:
        google_maps += "&query_place_id=" + quote(place_id)
    links = {
        "google_maps": google_maps,
        "street_view": f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat:.7f},{lon:.7f}",
        "land_titles": land_titles,
    }
    # Listing/profile searches are only meaningful once the source supplies a
    # numbered street address. Coordinate placeholders must never become URLs
    # that look like verified property records.
    house = str(addr.get("street_number") or addr.get("house_number") or "").strip()
    road = str(addr.get("route") or addr.get("road") or "").strip()
    if house and road:
        slug = quote(title.replace(",", "").replace(" ", "-").lower())
        links["realestate"] = "https://www.realestate.com.au/property/" + slug
        links["domain"] = "https://www.domain.com.au/property-profile/" + slug
        links["planning"] = "https://www.google.com/search?q=" + quote(
            f"{council} council development application {title}".strip()
        )
    # Jurisdiction-correct pool compliance link (QLD/NSW: public register; others:
    # the responsible regulator). Registration rules are state law, not national.
    compliance = pool_compliance(lat, lon)
    if compliance:
        links["pool_register"] = compliance["register_url"]
    if contact.get("name"):
        links["business_lookup"] = "https://www.google.com/search?q=" + quote(f"{contact['name']} {title} contact".strip())
        links["abn_lookup"] = "https://abr.business.gov.au/Search/ResultsActive?SearchText=" + quote(contact["name"])
    return links


def _centroid(element: dict[str, Any]) -> tuple[float, float] | None:
    if element.get("type") == "node":
        lat = element.get("lat")
        lon = element.get("lon")
        if lat is not None and lon is not None:
            return float(lat), float(lon)
    geometry = element.get("geometry")
    if isinstance(geometry, list):
        points = [
            (float(point["lat"]), float(point["lon"]))
            for point in geometry
            if isinstance(point, dict) and "lat" in point and "lon" in point
        ]
        if points:
            return (
                sum(point[0] for point in points) / len(points),
                sum(point[1] for point in points) / len(points),
            )
    return None


def _point_in_polygon(lon: float, lat: float, polygon: list[list[float]]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    j = len(polygon) - 1
    for i, point in enumerate(polygon):
        xi, yi = float(point[0]), float(point[1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if (yi > lat) != (yj > lat):
            denominator = yj - yi
            if abs(denominator) < 1e-12:
                j = i
                continue
            x_intersect = (xj - xi) * (lat - yi) / denominator + xi
            if lon < x_intersect:
                inside = not inside
        j = i
    return inside


def _feature_intersects_polygon(element: dict[str, Any], polygon: list[list[float]] | None) -> bool:
    if polygon is None:
        return True
    if element.get("type") == "node":
        lat = element.get("lat")
        lon = element.get("lon")
        if lat is None or lon is None:
            return False
        return _point_in_polygon(float(lon), float(lat), polygon)
    geometry = element.get("geometry")
    if isinstance(geometry, list):
        points = [
            (float(point["lon"]), float(point["lat"]))
            for point in geometry
            if isinstance(point, dict) and "lat" in point and "lon" in point
        ]
        if not points:
            return False
        if any(_point_in_polygon(lon, lat, polygon) for lon, lat in points):
            return True
        center = _centroid(element)
        return bool(center and _point_in_polygon(center[1], center[0], polygon))
    center = _centroid(element)
    return bool(center and _point_in_polygon(center[1], center[0], polygon))


def _distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat_mid = math.radians((a[0] + b[0]) / 2.0)
    dy = (a[0] - b[0]) * 111_320.0
    dx = (a[1] - b[1]) * 111_320.0 * max(math.cos(lat_mid), 1e-9)
    return math.hypot(dx, dy)


def _lead_score(
    center: tuple[float, float],
    area_sqm: float,
    neighbour_pools: int,
    trees: list[tuple[float, float]],
    powerlines: list[tuple[float, float]],
) -> tuple[float, dict[str, Any]]:
    """Score a residential property (with no pool of its own) as a pool/landscaping
    prospect. Higher = stronger lead. Assumes the caller already excluded properties
    whose own footprint has a pool."""
    score = 50.0
    # Footprint size: a house-sized lot has room for a pool/deck; tiny or huge less so.
    if 70.0 <= area_sqm <= 400.0:
        score += 18.0
    elif 45.0 <= area_sqm < 70.0 or 400.0 < area_sqm <= 700.0:
        score += 8.0
    elif area_sqm > 700.0:
        score += 2.0
    # Pools in the neighbourhood (but not on this lot) signal local appetite/feasibility.
    score += min(20.0, neighbour_pools * 5.0)
    # Some greenery nearby means a real backyard; very dense greenery means less room.
    nearby_trees = sum(1 for tree in trees if _distance_m(center, tree) <= 40.0)
    if nearby_trees == 0:
        score += 2.0
    elif nearby_trees <= 3:
        score += 6.0
    else:
        score -= min(10.0, (nearby_trees - 3) * 2.0)
    # Overhead powerlines reduce pool feasibility/safety.
    nearby_power = sum(1 for power in powerlines if _distance_m(center, power) <= 35.0)
    score -= min(18.0, nearby_power * 9.0)
    return round(max(0.0, min(100.0, score)), 1), {
        "footprint_sqm": round(area_sqm, 1),
        "neighbour_pools": neighbour_pools,
        "nearby_trees": nearby_trees,
        "nearby_powerlines": nearby_power,
    }


class _PoolAggregator:
    """Accumulates pool detections across passes, clustering ones that refer to the
    same physical pool, tallying how often each was detected (votes) vs. how often a
    pass could have seen it (coverage), and finally classifying each cluster as
    mapped / confirmed / probable / rejected. This is what turns repeated passes into
    refinement instead of repetition: real pools accrue votes and survive; flickering
    false positives are covered repeatedly, rarely detected, and get rejected."""

    def __init__(self, match_radius_m: float = POOL_MATCH_RADIUS_M) -> None:
        self.match_radius_m = match_radius_m
        self.clusters: list[dict[str, Any]] = []

    def add(self, element: dict[str, Any]) -> None:
        center = _centroid(element)
        if center is None:
            return
        mapped = not _is_satellite_pool(element)
        conf = _pool_detection_confidence(element)
        for cluster in self.clusters:
            if _distance_m(center, cluster["center"]) <= self.match_radius_m:
                cluster["votes"] += 1
                cluster["max_conf"] = max(cluster["max_conf"], conf)
                if mapped:
                    cluster["mapped"] = True
                # Keep the most trustworthy representative geometry/tags.
                if (mapped and not cluster["rep_mapped"]) or (
                    not cluster["rep_mapped"] and conf >= cluster["rep_conf"]
                ):
                    cluster["rep"] = element
                    cluster["rep_conf"] = conf
                    cluster["rep_mapped"] = mapped
                # Running-mean the centre so it stabilises as votes accumulate.
                votes = cluster["votes"]
                cluster["center"] = (
                    (cluster["center"][0] * (votes - 1) + center[0]) / votes,
                    (cluster["center"][1] * (votes - 1) + center[1]) / votes,
                )
                return
        self.clusters.append(
            {
                "center": center,
                "votes": 1,
                "coverage": 0,
                "max_conf": conf,
                "mapped": mapped,
                "rep": element,
                "rep_conf": conf,
                "rep_mapped": mapped,
                # Pin the first detection's identity so the kept pool keeps a stable
                # type/id across polls even when the representative geometry is later
                # swapped for a higher-confidence detection (no marker flicker).
                "stable_type": element.get("type", "way"),
                "stable_id": element.get("id"),
            }
        )

    def mark_coverage(self, window: tuple[float, float, float, float]) -> None:
        """Record that a pass over ``window`` was an opportunity to detect any pool
        whose centre falls inside it."""
        south, west, north, east = window
        for cluster in self.clusters:
            lat, lon = cluster["center"]
            if south <= lat <= north and west <= lon <= east:
                cluster["coverage"] += 1

    @staticmethod
    def _verdict(cluster: dict[str, Any]) -> str:
        if cluster["mapped"]:
            return "mapped"
        # Junk floor: never scored even moderately across any pass.
        if cluster["max_conf"] < POOL_CONF_HARD_FLOOR:
            return "rejected"
        # One strong sighting is enough.
        if cluster["max_conf"] >= POOL_CONF_SOLO_PUBLISH:
            return "confirmed"
        # Repeated cross-pass agreement at moderate confidence: real pools on z20 tiles
        # often score 0.55-0.75, and independent re-detection is the strongest signal we
        # have — this is what keeps recall high without letting flicker through.
        if cluster["votes"] >= 2 and cluster["max_conf"] >= POOL_CONF_VOTED_PUBLISH:
            return "confirmed"
        # A single sighting that several passes covered but never reproduced is noise.
        if cluster["votes"] == 1 and cluster["coverage"] >= POOL_REJECT_MIN_COVERAGE:
            return "rejected"
        # Unresolved single sighting: keep only if reasonably confident.
        return "probable" if cluster["max_conf"] >= POOL_CONF_PROBABLE_MIN else "rejected"

    def kept_pool_elements(self) -> tuple[list[dict[str, Any]], int]:
        """Return ``(kept_pool_elements, rejected_count)`` where each kept element is
        tagged with the cross-pass evidence used to keep it."""
        kept: list[dict[str, Any]] = []
        rejected = 0
        for cluster in self.clusters:
            verdict = self._verdict(cluster)
            if verdict == "rejected":
                rejected += 1
                continue
            element = dict(cluster["rep"])
            # Keep the cluster's pinned identity regardless of which detection won.
            element["type"] = cluster["stable_type"]
            if cluster["stable_id"] is not None:
                element["id"] = cluster["stable_id"]
            tags = dict(element.get("tags") or {})
            coverage = max(1, cluster["coverage"])
            tags["scan:passes_seen"] = str(cluster["votes"])
            tags["scan:coverage"] = str(cluster["coverage"])
            tags["scan:agreement"] = f"{cluster['votes'] / coverage:.2f}"
            tags["scan:confidence"] = f"{cluster['max_conf']:.2f}"
            tags["scan:verdict"] = verdict
            element["tags"] = tags
            kept.append(element)
        return kept, rejected


class AreaScanAgent:
    """Small in-process scan agent for user-selected areas.

    The agent runs 3-40 detector passes across the selected bbox at maximum
    supported satellite zoom, caches detected pools and infrastructure, and
    publishes practical lead summaries for dashboards.
    """

    def __init__(self, *, data_path: str) -> None:
        self.data_path = data_path
        self._lock = threading.Lock()
        self._scan_execution_lock = threading.Lock()
        self._jobs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cancel_tokens: dict[str, threading.Event] = {}
        self._retired_scan_ids: OrderedDict[str, float] = OrderedDict()

    def start_scan(
        self,
        payload: dict[str, Any],
        *,
        passes: int = 12,
        source: str = "manual",
        label: str | None = None,
    ) -> dict[str, Any]:
        passes = max(3, min(40, int(passes)))
        if payload.get("polygon") is None:
            raise ValueError("Area scans require an explicit selected polygon; viewport-only scans are disabled.")
        bbox = _bbox_from_request(payload)
        polygon = _coerce_polygon(payload["polygon"])
        scope_key = _polygon_scope_key(polygon)
        replace_active = payload.get("replace_active") is True
        scan_id = f"scan-{uuid.uuid4().hex[:12]}"
        cancel_token = threading.Event()
        job = {
            "scan_id": scan_id,
            "status": "queued",
            "source": source,
            "label": label or ("Selected area" if polygon else "Viewport area"),
            "passes_requested": passes,
            "passes_completed": 0,
            "created_at": _now(),
            "updated_at": _now(),
            "bbox": {"south": bbox[0], "west": bbox[1], "north": bbox[2], "east": bbox[3]},
            "polygon": polygon,
            "scope_key": scope_key,
            "counts": {"pools": 0, "solar": 0, "buildings": 0, "trees": 0, "powerlines": 0},
            "leads": [],
            "feature_elements": [],
            "detector": {"engine": "queued", "note": "Scan has not started."},
            "errors": [],
            "summary": "Scan queued.",
        }
        with self._lock:
            # Admission and insertion must be one critical section. The previous split
            # check allowed two simultaneous requests to both observe zero active jobs,
            # then start two memory-heavy detector threads.
            active_jobs = [
                existing
                for existing in self._jobs.values()
                if existing.get("status") in {"queued", "running"}
            ]
            matching_jobs = [job for job in active_jobs if job.get("scope_key") == scope_key]
            if matching_jobs:
                newest = matching_jobs[-1]
                newest["summary"] = (
                    "This selected area is already being scanned. Reusing its active "
                    "pool/solar scan."
                )
                newest["updated_at"] = _now()
                public = self._public_job(newest)
                public["reused_active_scan"] = True
                return public
            if active_jobs and not replace_active:
                raise ValueError(
                    "A different selected-area scan is already active. "
                    "Set replace_active to true to cancel it before starting this area."
                )
            superseded_scan_ids: list[str] = []
            for existing in active_jobs:
                superseded_scan_ids.append(str(existing["scan_id"]))
                self._cancel_job_locked(
                    existing,
                    reason="Superseded by a different selected analysis area.",
                    superseded_by=scan_id,
                )
            if superseded_scan_ids:
                job["superseded_scan_ids"] = superseded_scan_ids
            self._cancel_tokens[scan_id] = cancel_token
            self._jobs[scan_id] = job
            self._jobs.move_to_end(scan_id)
            while len(self._jobs) > MAX_SCAN_JOBS:
                evicted_scan_id, _evicted_job = self._jobs.popitem(last=False)
                self._retire_scan_locked(evicted_scan_id)
        thread = threading.Thread(
            target=self._run_job_safely,
            args=(scan_id, bbox, passes, polygon, cancel_token),
            daemon=True,
            name=f"area-scan-{scan_id[-6:]}",
        )
        thread.start()
        return self.get_scan(scan_id) or job

    def list_scans(self, *, limit: int = 8) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(reversed(self._jobs.values()))[: max(1, min(25, int(limit)))]
            return [self._public_job(job, include_features=False) for job in jobs]

    def get_scan(self, scan_id: str, *, include_features: bool = True) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(scan_id)
            return self._public_job(job, include_features=include_features) if job else None

    def _cancel_job_locked(
        self,
        job: dict[str, Any],
        *,
        reason: str,
        superseded_by: str | None = None,
    ) -> None:
        self._retire_scan_locked(str(job["scan_id"]))
        now = _now()
        job.update({
            "status": "cancelled",
            "cancel_requested": True,
            "cancel_reason": reason[:300],
            "cancelled_at": now,
            "finished_at": now,
            "updated_at": now,
            "counts": {key: 0 for key in job.get("counts", {})},
            "leads": [],
            "feature_elements": [],
            "summary": f"Area scan cancelled: {reason}"[:500],
            "cancellation_accepted": True,
        })
        if superseded_by:
            job["superseded_by"] = superseded_by

    def _retire_scan_locked(self, scan_id: str) -> None:
        """Signal a worker even when its bounded public-history record is evicted."""
        token = self._cancel_tokens.get(scan_id)
        if token is not None:
            token.set()
        self._retired_scan_ids[scan_id] = _now()
        self._retired_scan_ids.move_to_end(scan_id)
        while len(self._retired_scan_ids) > MAX_RETIRED_SCAN_TOMBSTONES:
            self._retired_scan_ids.popitem(last=False)

    def cancel_scan(self, scan_id: str, *, reason: str = "Cancelled by the user.") -> dict[str, Any] | None:
        """Idempotently cancel active work and retire any unpublished area results."""
        clean_reason = " ".join(str(reason).split())[:300] or "Cancelled by the user."
        with self._lock:
            job = self._jobs.get(scan_id)
            if job is None:
                return None
            if job.get("status") in {"queued", "running"}:
                self._cancel_job_locked(job, reason=clean_reason)
            elif job.get("status") == "cancelled":
                job["cancellation_accepted"] = True
            else:
                # Completed/failed jobs are already terminal. Preserve their honest
                # outcome while making repeat cancellation requests harmless.
                public = self._public_job(job)
                public["cancellation_accepted"] = False
                public["already_terminal"] = True
                return public
            return self._public_job(job)

    def aggregate_leads(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            leads: list[dict[str, Any]] = []
            for job in reversed(self._jobs.values()):
                for lead in job.get("leads", []):
                    public_lead = self._public_lead(lead)
                    public_lead.update({"scan_id": job["scan_id"], "scan_label": job["label"]})
                    leads.append(public_lead)
            leads.sort(key=lambda lead: (-float(lead.get("score", 0)), str(lead.get("id", ""))))
            return leads[: max(1, min(100, int(limit)))]

    @staticmethod
    def _public_lead(lead: dict[str, Any]) -> dict[str, Any]:
        """Strip internal scan flags from every API representation of a lead."""
        return {
            key: value
            for key, value in dict(lead).items()
            if not str(key).startswith("_")
        }

    def _update_job(self, scan_id: str, **updates: Any) -> None:
        with self._lock:
            job = self._jobs.get(scan_id)
            if not job:
                return
            if job.get("status") == "cancelled" and updates.get("status") != "cancelled":
                return
            job.update(updates)
            job["updated_at"] = _now()

    def _public_job(self, job: dict[str, Any], *, include_features: bool = True) -> dict[str, Any]:
        public = {
            **job,
            "leads": [self._public_lead(lead) for lead in job.get("leads", [])[:MAX_JOB_LEADS]],
            "errors": list(job.get("errors", []))[-8:],
        }
        if include_features:
            public["feature_elements"] = list(job.get("feature_elements", []))[:MAX_JOB_FEATURES]
        else:
            public.pop("feature_elements", None)
        return public

    def _feature_payload(
        self,
        elements: list[dict[str, Any]],
        polygon: list[list[float]] | None = None,
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for element in elements:
            if not _feature_intersects_polygon(element, polygon):
                continue
            if (
                _is_pool(element)
                or _is_solar(element)
                or _is_building(element)
                or _is_tree_or_greenery(element)
                or _is_powerline(element)
            ):
                payload.append(element)
                if len(payload) >= MAX_JOB_FEATURES:
                    break
        return payload

    def _run_job_safely(
        self,
        scan_id: str,
        bbox: tuple[float, float, float, float],
        passes: int,
        polygon: list[list[float]] | None,
        cancel_token: threading.Event | None = None,
    ) -> None:
        """Never leave a crashed background thread looking permanently active."""
        try:
            with self._scan_execution_lock:
                if self._is_cancelled(scan_id, cancel_token):
                    return
                self._run_job(scan_id, bbox, passes, polygon)
            with self._lock:
                job = self._jobs.get(scan_id)
                status = str(job.get("status", "")) if job else ""
                previous_errors = list(job.get("errors", [])) if job else []
            if status in {"queued", "running"}:
                previous_errors.append("Scan worker exited before recording a terminal result.")
                self._update_job(
                    scan_id,
                    status="failed",
                    errors=previous_errors,
                    summary="Area scan failed before a terminal result was recorded.",
                    finished_at=_now(),
                )
        except BaseException as exc:  # background failures must remain pollable
            message = str(exc).strip() or exc.__class__.__name__
            with self._lock:
                job = self._jobs.get(scan_id)
                previous_errors = list(job.get("errors", [])) if job else []
            previous_errors.append(f"Scan stopped: {message}"[:500])
            if isinstance(exc, MemoryError):
                summary = (
                    "Area scan stopped because detector memory was exhausted. "
                    "The server is still available; select a smaller area and retry."
                )
            else:
                summary = f"Area scan failed: {message}"[:500]
            self._update_job(
                scan_id,
                status="failed",
                errors=previous_errors,
                summary=summary,
                finished_at=_now(),
            )
        finally:
            if cancel_token is not None:
                with self._lock:
                    if self._cancel_tokens.get(scan_id) is cancel_token:
                        self._cancel_tokens.pop(scan_id, None)

    def _is_cancelled(self, scan_id: str, cancel_token: threading.Event | None = None) -> bool:
        if cancel_token is not None and cancel_token.is_set():
            return True
        with self._lock:
            job = self._jobs.get(scan_id)
            if scan_id in self._retired_scan_ids or job is None:
                return True
            return bool(job.get("cancel_requested") or job.get("status") == "cancelled")

    def _run_job(
        self,
        scan_id: str,
        bbox: tuple[float, float, float, float],
        passes: int,
        polygon: list[list[float]] | None,
    ) -> None:
        if self._is_cancelled(scan_id):
            return
        tile_count = _max_zoom_tile_count(bbox)
        self._update_job(
            scan_id,
            status="running",
            summary=f"Running detector over {tile_count} max-zoom Google satellite tile{'' if tile_count == 1 else 's'}.",
        )
        # Split the selection into coverage blocks (clipped to the drawn polygon so we
        # never scan beyond it) and re-scan each block over the requested refinement
        # sweeps. Progress is reported as block × sweep, and if a very large selection
        # forced fewer sweeps than requested we say so, so the user's input is honoured
        # or transparently explained rather than silently turned into a bigger number.
        plan = _build_scan_plan(bbox, passes, polygon)
        scan_bboxes = plan.windows
        total_passes = len(scan_bboxes)
        n_blocks = plan.blocks
        n_sweeps = plan.sweeps
        # Report progress against the real work: total detector calls = blocks × sweeps.
        # (passes_requested stays the user's input so the UI can show "12 sweeps requested".)
        self._update_job(
            scan_id,
            passes_total=total_passes,
            blocks=n_blocks,
            sweeps=n_sweeps,
        )
        sweeps_note = (
            ""
            if n_sweeps >= plan.requested_passes
            else (
                f" Auto-tuned from {plan.requested_passes} to {n_sweeps} refinement sweep"
                f"{'' if n_sweeps == 1 else 's'} so all {n_blocks} coverage blocks fit the "
                f"scan budget — draw a smaller area for deeper per-spot refinement."
            )
        )
        # Pools are aggregated spatially (so the same physical pool detected in
        # several passes counts once and accrues confidence); everything else is a
        # plain union keyed by stable OSM id.
        aggregator = _PoolAggregator()
        non_pool: OrderedDict[str, dict[str, Any]] = OrderedDict()
        errors: list[str] = []
        detector_info: dict[str, Any] = {
            "engine": "resolving_selected_area_detector",
            "precision": "precise",
            "model": "pending detector signature",
            "validator": "pending",
            "status": "pending",
            "note": "The first selected-area pass will report the exact active pool checkpoint(s) and signature.",
        }

        def snapshot() -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
            kept_pools, rejected = aggregator.kept_pool_elements()
            elements = kept_pools + list(non_pool.values())
            return elements, rejected, kept_pools

        # Fetch mapped structures (buildings/trees/powerlines) ONCE with a generous
        # timeout. These don't change between passes, and starving them ~2.6s per pass
        # is why scans used to return 0 buildings → 0 leads. A background job can wait.
        try:
            structures = execute_overpass_query(
                build_overpass_query(bbox, zoom=18, layers="buildings,trees,powerlines"),
                timeout=28.0,
            )
            for fallback_index, element in enumerate(structures.get("elements", [])):
                if not isinstance(element, dict):
                    continue
                if not _feature_intersects_polygon(element, polygon):
                    continue
                if not _is_pool(element):
                    non_pool[_feature_key(element, fallback_index)] = element
        except Exception as exc:  # public Overpass mirrors are best-effort
            errors.append(f"OSM structures: {exc}")
        if self._is_cancelled(scan_id):
            return

        # Supplement sparse OSM building coverage with Australia-wide Microsoft
        # footprints (so leads work everywhere, not just well-mapped suburbs). OSM is
        # preferred where it exists (it carries real addresses), so any Microsoft
        # footprint coinciding with an OSM building is skipped.
        if os.environ.get("QLDR_DISABLE_MS_BUILDINGS") != "1":
            try:
                osm_building_centers = [
                    center
                    for element in non_pool.values()
                    if _is_building(element) and (center := _centroid(element))
                ]
                ms_added = 0
                for ms_element in fetch_australia_buildings(bbox, timeout=30.0):
                    if not _feature_intersects_polygon(ms_element, polygon):
                        continue
                    center = _centroid(ms_element)
                    if center is None:
                        continue
                    if any(_distance_m(center, oc) <= 9.0 for oc in osm_building_centers):
                        continue
                    non_pool[_feature_key(ms_element, 0)] = ms_element
                    osm_building_centers.append(center)
                    ms_added += 1
                if ms_added:
                    self._update_job(
                        scan_id,
                        summary=f"Loaded {ms_added} Australia-wide building footprints to widen lead coverage.",
                    )
            except Exception as exc:  # best-effort national coverage
                errors.append(f"AU buildings: {exc}")
        if self._is_cancelled(scan_id):
            return

        # Building footprints (OSM + Microsoft) collected above, used to drop satellite
        # pool detections that land on a roof. A real pool is never inside a building, so
        # this strips rooftop false positives without touching the detector or affecting
        # genuine-pool recall. (The per-pass pool query is pools-only, so the detector's
        # own building filter has nothing to work with — this supplies it here.)
        building_footprints: list[tuple[tuple[float, float], list[list[float]]]] = []
        for element in non_pool.values():
            if not _is_building(element):
                continue
            geometry = element.get("geometry")
            if not isinstance(geometry, list) or len(geometry) < 3:
                continue
            ring = [
                [float(point["lon"]), float(point["lat"])]
                for point in geometry
                if isinstance(point, dict) and "lat" in point and "lon" in point
            ]
            center = _centroid(element)
            if len(ring) >= 3 and center is not None:
                building_footprints.append((center, ring))

        def _detection_building_context(pool_center: tuple[float, float]) -> tuple[bool, bool]:
            """``(on_roof, has_building_nearby)`` for a detection centre. A genuine
            backyard pool sits NEXT TO a house but not ON its roof, so a detection that
            is on a roof — or has no building anywhere near it (open water / sand /
            boats) — is a false positive."""
            plat, plon = pool_center
            on_roof = False
            near_building = False
            for (blat, blon), ring in building_footprints:
                dlat = abs(blat - plat)
                dlon = abs(blon - plon)
                if dlat <= 0.00075 and dlon <= 0.00085:  # within ~85 m → a building is nearby
                    near_building = True
                    if dlat <= 0.0006 and dlon <= 0.0006 and _point_in_polygon(plon, plat, ring):
                        on_roof = True
                        break
            return on_roof, near_building

        for index, scan_bbox in enumerate(scan_bboxes, start=1):
            if self._is_cancelled(scan_id):
                return
            elements, rejected, kept_pools = snapshot()
            counts, leads = self._summarize_features(elements, polygon=polygon)
            before_pools = len(kept_pools)
            self._update_job(
                scan_id,
                passes_completed=index - 1,
                counts=counts,
                leads=leads,
                feature_elements=self._feature_payload(elements, polygon),
                detector=detector_info,
                errors=errors,
                summary=(
                    (
                        f"Scanning block {(index - 1) % n_blocks + 1}/{n_blocks} of the selected "
                        f"area · refinement sweep {(index - 1) // n_blocks + 1}/{n_sweeps} "
                        f"({tile_count} max-zoom tiles total) — voting on detections.{sweeps_note}"
                    )
                    if n_blocks > 1
                    else (
                        f"Refinement sweep {index}/{n_sweeps}: re-scanning the selected area "
                        f"({tile_count} max-zoom tile{'' if tile_count == 1 else 's'}) and "
                        f"voting on detections."
                    )
                ),
            )
            try:
                result = fetch_osm_features(
                    scan_bbox,
                    zoom=MAX_SCAN_ZOOM,
                    layers="pools",
                    timeout=8.0,
                    data_path=self.data_path,
                    detector_model=PRECISE_SCAN_MODEL or DEFAULT_DETECTOR_MODEL,
                    segmentation_mode=DEFAULT_SEGMENTATION_MODE,
                    detector_precision="precise",
                    analysis_polygon=polygon,
                    force_refresh=True,
                    detector_queue_timeout=30.0,
                )
                # Cancellation may arrive while satellite inference is in flight. Do
                # not ingest or publish that obsolete area's returned detections.
                if self._is_cancelled(scan_id):
                    return
                detector_info = _detector_info_from_result(result)
                if detector_info.get("status") == "degraded" and detector_info.get("error"):
                    warning = f"Detector degraded: {detector_info['error']}"[:500]
                    if warning not in errors:
                        errors.append(warning)
                if detector_info.get("status") in {
                    "busy_this_process",
                    "busy_other_process",
                    "resource_exhausted",
                    "error",
                }:
                    detector_error = str(
                        detector_info.get("error")
                        or "The selected-area pool/solar detector could not complete this pass."
                    )
                    errors.append(f"Detector: {detector_error}"[:500])
                    elements, rejected, _kept_pools = snapshot()
                    counts, leads = self._summarize_features(elements, polygon=polygon)
                    self._update_job(
                        scan_id,
                        status="failed",
                        passes_completed=index - 1,
                        counts=counts,
                        leads=leads,
                        feature_elements=self._feature_payload(elements, polygon),
                        detector=detector_info,
                        errors=errors,
                        summary=f"Area scan stopped: {detector_error}"[:500],
                        finished_at=_now(),
                    )
                    return
                for fallback_index, element in enumerate(result.get("elements", [])):
                    if not isinstance(element, dict):
                        continue
                    if not _feature_intersects_polygon(element, polygon):
                        continue
                    if _is_pool(element):
                        if _is_satellite_pool(element):
                            # Physical plausibility: a backyard pool is 8-400 m². Bigger
                            # blobs are ocean patches / rock pools / lakes; smaller are
                            # noise. Mapped OSM pools are exempt (ground truth).
                            # Use the detector's pre-refinement box area for the
                            # physical plausibility gate.  The published outline is
                            # a padded water-mask refinement and is not a stable size
                            # measurement (it can be materially larger or smaller).
                            det_area = _satellite_pool_area_sqm(element)
                            if det_area > 0 and not (
                                SATELLITE_POOL_MIN_AREA_SQM <= det_area <= SATELLITE_POOL_MAX_AREA_SQM
                            ):
                                continue
                            # A satellite detection must look like a real backyard pool —
                            # NEAR a building but not ON its roof. This drops rooftop
                            # false positives and isolated detections on open water /
                            # sand / boats. Only applied when we actually have building
                            # footprints to judge against.
                            if building_footprints:
                                center = _centroid(element)
                                if center is not None:
                                    on_roof, near_building = _detection_building_context(center)
                                    if on_roof or not near_building:
                                        continue
                        aggregator.add(element)
                    elif _is_solar(element):
                        non_pool[_solar_feature_key(element, fallback_index)] = element
                    else:
                        non_pool[_feature_key(element, fallback_index)] = element
            except Exception as exc:  # public mirrors are best-effort enrichment
                errors.append(f"Pass {index}: {exc}")
            if self._is_cancelled(scan_id):
                return
            # Every pass over this window was an opportunity to (re)detect any pool
            # whose centre lies within it — record that for agreement scoring.
            aggregator.mark_coverage(scan_bbox)

            elements, rejected, kept_pools = snapshot()
            counts, leads = self._summarize_features(elements, polygon=polygon)
            new_pools = max(0, len(kept_pools) - before_pools)
            self._update_job(
                scan_id,
                passes_completed=index,
                counts=counts,
                leads=leads,
                feature_elements=self._feature_payload(elements, polygon),
                detector=detector_info,
                errors=errors,
                summary=self._summary_text(
                    counts, leads, index, total_passes, new_pools=new_pools, rejected=rejected
                ),
            )

        status = "completed_with_warnings" if errors else "completed"
        if self._is_cancelled(scan_id):
            return
        elements, rejected, _kept_pools = snapshot()
        counts, leads = self._summarize_features(elements, polygon=polygon)
        # Auto-pipeline: harvest scan-verified detections as fine-tuning
        # candidates. Best-effort — data collection must never fail a scan.
        try:
            from .evaluation import record_training_candidates

            harvested = record_training_candidates(elements, source=f"scan:{scan_id}")
            if harvested.get("pool") or harvested.get("solar"):
                counts = dict(counts)
                counts["training_candidates_added"] = harvested["pool"] + harvested["solar"]
        except Exception:
            pass
        # Resolve real street addresses + targeted lookup links for the best leads
        # (done once at the end so polling stays fast and Nominatim isn't hammered).
        if self._is_cancelled(scan_id):
            return
        leads = self._enrich_lead_addresses(leads)
        if self._is_cancelled(scan_id):
            return
        self._update_job(
            scan_id,
            status=status,
            passes_completed=len(scan_bboxes),
            counts=counts,
            leads=leads,
            feature_elements=self._feature_payload(elements, polygon),
            detector=detector_info,
            errors=errors,
            summary=self._summary_text(counts, leads, len(scan_bboxes), total_passes, rejected=rejected),
            finished_at=_now(),
        )

    def _summarize_features(
        self,
        features: list[dict[str, Any]],
        *,
        polygon: list[list[float]] | None = None,
    ) -> tuple[dict[str, int], list[dict[str, Any]]]:
        elements = [element for element in features if _feature_intersects_polygon(element, polygon)]
        pools = [element for element in elements if _is_pool(element)]
        solar = [element for element in elements if _is_solar(element)]
        buildings = [element for element in elements if _is_building(element)]
        trees = [element for element in elements if _is_tree_or_greenery(element)]
        powerlines = [element for element in elements if _is_powerline(element)]
        pool_centers = [center for element in pools if (center := _centroid(element))]
        tree_centers = [center for element in trees if (center := _centroid(element))]
        power_centers = [center for element in powerlines if (center := _centroid(element))]

        residential = [building for building in buildings if _is_residential_building(building)]
        leads: list[dict[str, Any]] = []
        seen_lead_bins: set[str] = set()
        for building in residential[:240]:
            center = _centroid(building)
            if center is None:
                continue
            # A pool on the property's own footprint disqualifies it as a new-pool lead;
            # pools elsewhere in the neighbourhood are a positive signal instead.
            on_property = any(_distance_m(center, pc) <= LEAD_ON_PROPERTY_POOL_M for pc in pool_centers)
            if on_property:
                continue
            neighbour_pools = sum(
                1 for pc in pool_centers if _distance_m(center, pc) <= LEAD_NEIGHBOUR_POOL_M
            )
            area_sqm = _polygon_area_sqm(building)
            score, evidence = _lead_score(center, area_sqm, neighbour_pools, tree_centers, power_centers)
            if score < LEAD_MIN_SCORE:
                continue
            lead_bin_key = f"{round(center[0], 4)}:{round(center[1], 4)}"
            if lead_bin_key in seen_lead_bins:
                continue
            seen_lead_bins.add(lead_bin_key)
            tags = building.get("tags") or {}
            precise_osm_addr = bool(tags.get("addr:housenumber") and tags.get("addr:street"))
            osm_address = _osm_address_components(building)
            title = _address_label(building, center)
            if precise_osm_addr:
                precision = "unit" if osm_address.get("unit") else "premise"
                address_details = _address_details(
                    osm_address,
                    source="openstreetmap",
                    source_id=f"{building.get('type', 'feature')}/{building.get('id', '')}",
                    precision=precision,
                    verified=True,
                    formatted=title,
                )
            elif osm_address.get("road"):
                address_details = _address_details(
                    osm_address,
                    source="openstreetmap",
                    source_id=f"{building.get('type', 'feature')}/{building.get('id', '')}",
                    precision="street",
                    verified=False,
                    formatted=title,
                )
            else:
                address_details = _pending_address_details()
            osm_contact = _extract_osm_contact(building)
            has_published = bool(osm_contact["phone"] or osm_contact["email"] or osm_contact["website"])
            contact = {
                "status": "published_osm" if has_published else "lookup_required",
                "name": osm_contact["name"],
                "phone": osm_contact["phone"],
                "email": osm_contact["email"],
                "website": osm_contact["website"],
                "note": (
                    "Contact details published by the owner in OpenStreetMap."
                    if has_published
                    else "No published contact in map data (typical for homes). Use the lawful lookup "
                    "links — land-titles search, council planning portal, and listing sites — to find "
                    "the owner/occupier before outreach."
                ),
            }
            leads.append(
                {
                    "id": _feature_key(building, len(leads)),
                    "title": title,
                    "score": score,
                    "lat": round(center[0], 7),
                    "lon": round(center[1], 7),
                    "pitch": (
                        "Residential lot with no pool on its own footprint. "
                        f"~{round(area_sqm)} m² footprint"
                        + (f", {neighbour_pools} pool(s) nearby" if neighbour_pools else "")
                        + " — a pool, deck, fencing, or landscaping prospect."
                    ),
                    "evidence": evidence,
                    "contact": contact,
                    "address": address_details.get("formatted", "") if precise_osm_addr else "",
                    "address_details": address_details,
                    "links": _lookup_links(title, osm_address, center, osm_contact),
                    "_precise_osm_addr": precise_osm_addr,
                }
            )

        leads.sort(key=lambda lead: (-float(lead["score"]), str(lead["id"])))
        counts = {
            "pools": len(pools),
            "solar": len(solar),
            "buildings": len(buildings),
            "residential": len(residential),
            "trees": len(trees),
            "powerlines": len(powerlines),
        }
        return counts, leads[:MAX_JOB_LEADS]

    def _enrich_lead_addresses(self, leads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Resolve lead addresses with provenance and an explicit exactness gate.

        Explicit per-building OSM addresses are preserved. When configured, Google's
        official Geocoding API is used for the dashboard's highest-ranked leads; without a key the
        rate-limited Nominatim fallback covers only the highest-scoring leads and is
        labelled approximate. A unit is accepted only from an explicit source field.
        """
        from .cadastre import lookup_parcel

        # Every lead carries its jurisdiction and that state's compliance rules —
        # registration/inspection is state law (compliance matrix in jurisdiction.py).
        for lead in leads:
            compliance = pool_compliance(lead.get("lat", 0.0), lead.get("lon", 0.0))
            if compliance:
                lead["state"] = compliance["state"]
                lead["compliance"] = {
                    "register_kind": compliance["register_kind"],
                    "register_label": compliance["register_label"],
                    "note": compliance["note"],
                }

        google_enabled = google_geocoding_configured()
        lookup_limit = min(len(leads), LEAD_GOOGLE_GEOCODE_LIMIT) if google_enabled else min(
            len(leads), LEAD_REVERSE_GEOCODE_LIMIT
        )
        for index, lead in enumerate(leads[:lookup_limit]):
            # Legal parcel (lot/plan) attribution from open QLD/NSW cadastre — makes the
            # lead lot-exact instead of "nearest building". Best-effort, cached.
            try:
                parcel = lookup_parcel(lead["lat"], lead["lon"])
            except Exception:
                parcel = None
            if parcel:
                lead["parcel"] = parcel
                # Cadastre state is authoritative near the Murray/Tweed borders where
                # the bbox guess can be a few km off — re-stamp the jurisdiction.
                parcel_state = str(parcel.get("state") or "").lower()
                rules = POOL_COMPLIANCE.get(parcel_state)
                if rules and parcel_state != str(lead.get("state", "")).lower():
                    lead["state"] = parcel_state.upper()
                    lead["compliance"] = {
                        key: rules[key] for key in ("register_kind", "register_label", "note")
                    }
                    lead.setdefault("links", {})["pool_register"] = rules["register_url"]
            details = lead.get("address_details") if isinstance(lead.get("address_details"), dict) else {}
            address: dict[str, Any] = {}
            place_id = ""
            if lead.get("_precise_osm_addr"):
                # The OSM address is attached to this exact building footprint.
                address = {
                    "unit": details.get("unit"),
                    "street_number": details.get("street_number"),
                    "route": details.get("route"),
                    "locality": details.get("locality"),
                    "state": details.get("state"),
                    "postcode": details.get("postcode"),
                    "country_code": details.get("country_code"),
                }
            else:
                google_result: dict[str, Any] = {}
                if google_enabled:
                    try:
                        google_result = google_reverse_geocode(lead["lat"], lead["lon"])
                    except Exception:
                        google_result = {}
                if google_result:
                    components = google_result.get("components")
                    address = components if isinstance(components, dict) else {}
                    # Trust unit text only when Google supplies the structured
                    # subpremise component. If it is absent, rebuild from the
                    # components so an apparent unit in free-form text cannot leak
                    # into the dashboard as a selected dwelling.
                    formatted = (
                        str(google_result.get("formatted_address") or "").strip()
                        if address.get("unit")
                        else _format_street_address(address)
                    )
                    verified = bool(google_result.get("verified") or google_result.get("exact"))
                    place_id = str(google_result.get("place_id") or "").strip()
                    details = _address_details(
                        address,
                        source="google",
                        source_id=place_id,
                        place_id=place_id,
                        precision=str(google_result.get("precision") or "unknown"),
                        verified=verified,
                        formatted=formatted,
                        status="verified" if verified else "approximate",
                        checked_at=_now(),
                    )
                    details["result_types"] = list(google_result.get("result_types") or [])
                    details["partial_match"] = bool(google_result.get("partial_match"))
                    details["location_type"] = str(google_result.get("location_type") or "")
                    details["location"] = dict(google_result.get("location") or {})
                    details["distance_m"] = google_result.get("distance_m")
                if not details.get("formatted") and index < LEAD_REVERSE_GEOCODE_LIMIT:
                    try:
                        resolved = reverse_geocode(lead["lat"], lead["lon"])
                    except Exception:
                        resolved = {}
                    fallback_address = resolved.get("address") if isinstance(resolved, dict) else {}
                    if isinstance(fallback_address, dict) and fallback_address:
                        address = fallback_address
                        formatted = _format_street_address(address)
                        if formatted:
                            details = _address_details(
                                address,
                                source="openstreetmap_nominatim",
                                precision="nearest",
                                verified=False,
                                formatted=formatted,
                                status="approximate",
                                checked_at=_now(),
                            )

                if details.get("formatted"):
                    lead["title"] = str(details["formatted"])
                    lead["address"] = str(details["formatted"])
                else:
                    details = _address_details(
                        {},
                        source="unavailable",
                        precision="unknown",
                        verified=False,
                        status="unavailable",
                        checked_at=_now(),
                    )
                    lead["title"] = "Address not verified"
                    lead["address"] = ""
                lead["address_details"] = details

            # Retain the legacy component field for integrations while the
            # structured address_details contract is adopted by all clients.
            lead["address_components"] = {
                key: value
                for key, value in {
                    "unit": details.get("unit"),
                    "house_number": details.get("street_number"),
                    "road": details.get("route"),
                    "suburb": details.get("locality"),
                    "state": details.get("state"),
                    "postcode": details.get("postcode"),
                }.items()
                if value
            }
            contact_name = lead.get("contact", {}).get("name", "")
            pool_register = (lead.get("links") or {}).get("pool_register")
            link_address = {**address, "place_id": place_id}
            lead["links"] = _lookup_links(
                lead["title"],
                link_address,
                (lead["lat"], lead["lon"]),
                {"name": contact_name},
            )
            if not details.get("verified"):
                for address_link in ("realestate", "domain", "planning"):
                    lead["links"].pop(address_link, None)
            if pool_register:  # links rebuild must not drop the QLD compliance link
                lead["links"]["pool_register"] = pool_register

        # Once enrichment is finished, unresolved leads are unavailable rather
        # than indefinitely "pending" (the keyless fallback is deliberately capped).
        for lead in leads[lookup_limit:]:
            current = lead.get("address_details") if isinstance(lead.get("address_details"), dict) else {}
            if current.get("status") == "pending":
                lead["address_details"] = _address_details(
                    {},
                    source="unavailable",
                    precision="unknown",
                    verified=False,
                    status="unavailable",
                    checked_at=_now(),
                )
                lead["title"] = "Address not verified"
                lead["address"] = ""

        deduped: list[dict[str, Any]] = []
        seen_titles: set[str] = set()
        for lead in leads:
            lead.pop("_precise_osm_addr", None)  # internal flag, never expose
            details = lead.get("address_details") if isinstance(lead.get("address_details"), dict) else {}
            formatted = " ".join(str(details.get("formatted") or "").split()).strip().lower()
            key = formatted if details.get("verified") and formatted else str(lead.get("id", ""))
            if key and key in seen_titles:
                continue
            seen_titles.add(key)
            deduped.append(lead)
        return deduped

    @staticmethod
    def _summary_text(
        counts: dict[str, int],
        leads: list[dict[str, Any]],
        completed: int,
        requested: int,
        *,
        new_pools: int | None = None,
        rejected: int | None = None,
    ) -> str:
        new_pool_text = (
            f" Found {new_pools} new pool cache entr{'y' if new_pools == 1 else 'ies'} this pass."
            if new_pools is not None
            else ""
        )
        rejected_text = (
            f" Rejected {rejected} false-positive candidate{'' if rejected == 1 else 's'} on cross-pass review."
            if rejected
            else ""
        )
        return (
            f"Ran {completed}/{requested} detection passes across the selected max-zoom tiles. "
            f"Cached {counts.get('pools', 0)} pools, {counts.get('solar', 0)} solar arrays, "
            f"{counts.get('buildings', 0)} structures, {counts.get('trees', 0)} green features, "
            f"and {counts.get('powerlines', 0)} powerline segments. "
            f"Published {len(leads)} target lead summaries."
            f"{new_pool_text}{rejected_text}"
        )
