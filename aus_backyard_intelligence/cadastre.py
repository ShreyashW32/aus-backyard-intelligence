"""Cadastral parcel (lot/plan) lookup from open state government services.

Attributes a point to its legal land parcel so leads can reference the exact lot/plan
instead of "nearest building". Sources (both verified public ArcGIS REST services):

* QLD — DCDB via the Land Parcel Property Framework MapServer (layer 4, nightly refresh).
* NSW — SIX Maps NSW_Cadastre MapServer (layer 9 "Lot").

Point-in-polygon query, WGS84 in, attributes out. Results are memoised (rounded ~11 m)
because neighbouring leads often share a parcel. Lookups are best-effort: any network
or schema hiccup returns ``None`` rather than breaking lead enrichment.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.parse
import urllib.request
from typing import Any

QLD_PARCEL_URL = (
    "https://spatial-gis.information.qld.gov.au/arcgis/rest/services/"
    "PlanningCadastre/LandParcelPropertyFramework/MapServer/4/query"
)
NSW_LOT_URL = (
    "https://maps.six.nsw.gov.au/arcgis/rest/services/public/NSW_Cadastre/MapServer/9/query"
)

# Rough state extents used only to pick which service to try first.
_QLD_BBOX = (-29.20, 137.90, -8.90, 153.60)   # south, west, north, east
_NSW_BBOX = (-37.60, 140.90, -28.10, 153.70)

_cache: dict[tuple[float, float], dict[str, Any] | None] = {}
_cache_lock = threading.Lock()
_MAX_CACHE = 4096


def _point_query(url: str, lat: float, lon: float, out_fields: str, timeout: float) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "geometry": f"{lon:.7f},{lat:.7f}",
            "geometryType": "esriGeometryPoint",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": out_fields,
            "returnGeometry": "false",
            "f": "json",
        }
    )
    request = urllib.request.Request(f"{url}?{params}", headers={"User-Agent": "qld-backyard-recommender"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8", "replace"))
    features = payload.get("features")
    return [f.get("attributes", {}) for f in features] if isinstance(features, list) else []


def _lookup_qld(lat: float, lon: float, timeout: float) -> dict[str, Any] | None:
    rows = _point_query(QLD_PARCEL_URL, lat, lon, "lot,plan,lotplan", timeout)
    for row in rows:
        lotplan = str(row.get("lotplan") or "").strip()
        if lotplan:
            return {
                "state": "QLD",
                "lotplan": lotplan,
                "lot": str(row.get("lot") or ""),
                "plan": str(row.get("plan") or ""),
                "display": f"Lot {row.get('lot')} on {row.get('plan')}",
                "source": "QLD DCDB (Land Parcel Property Framework)",
            }
    return None


def _lookup_nsw(lat: float, lon: float, timeout: float) -> dict[str, Any] | None:
    rows = _point_query(NSW_LOT_URL, lat, lon, "lotnumber,sectionnumber,planlabel", timeout)
    for row in rows:
        lot = str(row.get("lotnumber") or "").strip()
        plan = str(row.get("planlabel") or "").strip()
        if lot and plan:
            section = str(row.get("sectionnumber") or "").strip()
            lotplan = f"{lot}/{section or '-'}/{plan}"
            return {
                "state": "NSW",
                "lotplan": lotplan,
                "lot": lot,
                "plan": plan,
                "display": f"Lot {lot}{f' Sec {section}' if section else ''} {plan}",
                "source": "NSW SIX Maps cadastre",
            }
    return None


def _in_bbox(lat: float, lon: float, bbox: tuple[float, float, float, float]) -> bool:
    south, west, north, east = bbox
    return south <= lat <= north and west <= lon <= east


def lookup_parcel(lat: float, lon: float, *, timeout: float = 6.0) -> dict[str, Any] | None:
    """Return ``{state, lotplan, lot, plan, display, source}`` for the parcel containing
    the point, or ``None``. Disabled under pytest (network) via PYTEST_CURRENT_TEST."""
    if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("QLDR_ALLOW_CADASTRE_IN_TESTS"):
        return None
    key = (round(float(lat), 4), round(float(lon), 4))
    with _cache_lock:
        if key in _cache:
            return _cache[key]

    order = []
    if _in_bbox(lat, lon, _QLD_BBOX):
        order.append(_lookup_qld)
    if _in_bbox(lat, lon, _NSW_BBOX):
        order.append(_lookup_nsw)
    if not order:
        order = [_lookup_qld, _lookup_nsw]

    result: dict[str, Any] | None = None
    for lookup in order:
        try:
            result = lookup(float(lat), float(lon), timeout)
        except Exception:
            result = None
        if result:
            break

    with _cache_lock:
        if len(_cache) >= _MAX_CACHE:
            _cache.clear()
        _cache[key] = result
    return result
