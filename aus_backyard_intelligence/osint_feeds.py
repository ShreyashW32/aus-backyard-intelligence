"""Tactical OSINT feed adapters for the viewer's intelligence fusion layer.

Every upstream here is keyless and free (community ADS-B, state emergency
GeoJSON feeds, USGS, OpenStreetMap). Each adapter degrades to an empty result
with a warning instead of failing the whole intelligence layer when one feed
is down. Requests are proxied server-side (browser CORS + shared caching), and
reuse the validated HTTP helpers in external_api.
"""
from __future__ import annotations

import math
import time
from typing import Any

from .external_api import (
    ExternalServiceError,
    execute_overpass_query,
    request_json_url,
    validate_bbox,
)

AUSTRALIA_OSINT_BBOX = (-50.0, 105.0, -5.0, 165.0)  # generous AU + surrounds
NSW_RFS_FEED = "https://www.rfs.nsw.gov.au/feeds/majorIncidents.json"
VIC_EMERGENCY_FEED = "https://emergency.vic.gov.au/public/osom-geojson.json"
USGS_QUAKE_FEED = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_week.geojson"
OPENSKY_STATES_ENDPOINT = "https://opensky-network.org/api/states/all"
ADSB_LOL_ENDPOINT = "https://api.adsb.lol/v2/lat/{lat:.4f}/lon/{lon:.4f}/dist/{dist:.0f}"


def _osint_bbox(values: list[Any] | tuple[Any, ...] | None) -> tuple[float, float, float, float]:
    """Loose bbox validation for continent-scale OSINT queries (flights need a
    far larger span than the Overpass detector bbox allows)."""
    if not values or all(value is None for value in values):
        return AUSTRALIA_OSINT_BBOX
    try:
        south, west, north, east = (float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("Bounding-box values must be numbers.") from exc
    if not all(math.isfinite(value) for value in (south, west, north, east)):
        raise ValueError("Bounding-box values must be finite numbers.")
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        raise ValueError("The bounding box is outside valid latitude/longitude ranges.")
    return south, west, north, east


def _knots_to_ms(value: Any) -> float | None:
    try:
        return round(float(value) * 0.514444, 1)
    except (TypeError, ValueError):
        return None


def _feet_to_m(value: Any) -> float | None:
    if value == "ground":
        return 0.0
    try:
        return round(float(value) * 0.3048, 1)
    except (TypeError, ValueError):
        return None


def _flights_from_adsb_lol(bbox: tuple[float, float, float, float]) -> list[dict[str, Any]]:
    south, west, north, east = bbox
    lat = (south + north) / 2
    lon = (west + east) / 2
    # Radius (nm) that covers the bbox corner, capped at the API's 250 nm limit.
    half_diag_km = 111.32 * math.hypot(
        (north - south) / 2, (east - west) / 2 * math.cos(math.radians(lat))
    )
    dist_nm = min(250.0, max(20.0, half_diag_km / 1.852))
    url = ADSB_LOL_ENDPOINT.format(lat=lat, lon=lon, dist=dist_nm)
    data = request_json_url(url, timeout=8.0, cache_ttl=12.0)
    aircraft = data.get("ac") if isinstance(data, dict) else None
    items: list[dict[str, Any]] = []
    for entry in aircraft or []:
        if not isinstance(entry, dict):
            continue
        ac_lat, ac_lon = entry.get("lat"), entry.get("lon")
        if not isinstance(ac_lat, (int, float)) or not isinstance(ac_lon, (int, float)):
            continue
        if not (south <= ac_lat <= north and west <= ac_lon <= east):
            continue
        items.append({
            "icao24": str(entry.get("hex", "")).strip(),
            "callsign": str(entry.get("flight", "")).strip(),
            "origin_country": "",
            "lat": round(float(ac_lat), 5),
            "lon": round(float(ac_lon), 5),
            "altitude_m": _feet_to_m(entry.get("alt_baro")),
            "velocity_ms": _knots_to_ms(entry.get("gs")),
            "track_deg": entry.get("track") if isinstance(entry.get("track"), (int, float)) else None,
            "vertical_rate_ms": _feet_to_m(entry.get("baro_rate")),
            "squawk": str(entry.get("squawk", "")).strip(),
            "aircraft_type": str(entry.get("t", "")).strip(),
            "on_ground": entry.get("alt_baro") == "ground",
        })
    return items


def _flights_from_opensky(bbox: tuple[float, float, float, float]) -> list[dict[str, Any]]:
    south, west, north, east = bbox
    url = (
        f"{OPENSKY_STATES_ENDPOINT}?lamin={south:.3f}&lomin={west:.3f}"
        f"&lamax={north:.3f}&lomax={east:.3f}"
    )
    data = request_json_url(url, timeout=10.0, cache_ttl=15.0)
    states = data.get("states") if isinstance(data, dict) else None
    items: list[dict[str, Any]] = []
    for row in states or []:
        if not isinstance(row, (list, tuple)) or len(row) < 12:
            continue
        lat, lon = row[6], row[5]
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            continue
        items.append({
            "icao24": str(row[0] or "").strip(),
            "callsign": str(row[1] or "").strip(),
            "origin_country": str(row[2] or "").strip(),
            "lat": round(float(lat), 5),
            "lon": round(float(lon), 5),
            "altitude_m": round(float(row[7]), 1) if isinstance(row[7], (int, float)) else None,
            "velocity_ms": round(float(row[9]), 1) if isinstance(row[9], (int, float)) else None,
            "track_deg": row[10] if isinstance(row[10], (int, float)) else None,
            "vertical_rate_ms": row[11] if isinstance(row[11], (int, float)) else None,
            "squawk": str(row[14] or "").strip() if len(row) > 14 else "",
            "aircraft_type": "",
            "on_ground": bool(row[8]),
        })
    return items


def fetch_osint_flights(
    bbox: list[Any] | tuple[Any, ...] | None = None, *, limit: int = 1500
) -> dict[str, Any]:
    """Live aircraft state vectors from community ADS-B (adsb.lol) with an
    OpenSky anonymous fallback. Both are keyless."""
    box = _osint_bbox(bbox)
    limit = max(1, min(int(limit or 1500), 5000))
    warnings: list[str] = []
    items: list[dict[str, Any]] = []
    source = ""
    try:
        items = _flights_from_adsb_lol(box)
        source = "adsb.lol"
    except (ExternalServiceError, ValueError) as exc:
        warnings.append(f"adsb.lol unavailable: {exc}")
    if not items:
        try:
            items = _flights_from_opensky(box)
            source = "opensky-network"
        except (ExternalServiceError, ValueError) as exc:
            warnings.append(f"OpenSky unavailable: {exc}")
    return {
        "items": items[:limit],
        "count": min(len(items), limit),
        "source": source,
        "warnings": warnings,
        "fetched_at": time.time(),
    }


def _explode_geometry_collection(feature: dict[str, Any]) -> list[dict[str, Any]]:
    """MapLibre layer filters cannot address GeometryCollection children, so
    split them into sibling features that share the parent's properties."""
    geometry = feature.get("geometry") or {}
    if geometry.get("type") != "GeometryCollection":
        return [feature]
    exploded = []
    for child in geometry.get("geometries") or []:
        if isinstance(child, dict) and child.get("type"):
            exploded.append({
                "type": "Feature",
                "properties": dict(feature.get("properties") or {}),
                "geometry": child,
            })
    return exploded or [feature]


def _emergency_feed_features() -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Fetch + normalize the state emergency feeds. Returns (features, sources,
    warnings); each feature's properties carry title/category/status/jurisdiction
    and a 'kind' of 'fire' or 'other'."""
    features: list[dict[str, Any]] = []
    sources: list[str] = []
    warnings: list[str] = []

    def classify(text: str) -> str:
        lowered = text.lower()
        return "fire" if "fire" in lowered or "burn" in lowered else "other"

    try:
        nsw = request_json_url(NSW_RFS_FEED, timeout=10.0, cache_ttl=120.0)
        for feature in (nsw.get("features") or []) if isinstance(nsw, dict) else []:
            props = feature.get("properties") or {}
            category = str(props.get("category", ""))
            normalized = {
                "title": str(props.get("title", "")),
                "category": category,
                "status": str(props.get("status", "")),
                "description": str(props.get("description", ""))[:600],
                "updated": str(props.get("pubDate", props.get("updated", ""))),
                "jurisdiction": "nsw",
                "source": "NSW RFS",
                # Human-facing page for the incident (falls back to the RFS
                # "Fires Near Me" map when the feed omits a per-incident link).
                "link": str(props.get("link") or "https://www.rfs.nsw.gov.au/fire-information/fires-near-me"),
                "kind": classify(category or str(props.get("title", ""))),
            }
            features.extend(_explode_geometry_collection(
                {"type": "Feature", "properties": normalized, "geometry": feature.get("geometry")}
            ))
        sources.append("NSW RFS major incidents")
    except (ExternalServiceError, ValueError) as exc:
        warnings.append(f"NSW RFS feed unavailable: {exc}")

    try:
        vic = request_json_url(VIC_EMERGENCY_FEED, timeout=10.0, cache_ttl=120.0)
        for feature in (vic.get("features") or []) if isinstance(vic, dict) else []:
            props = feature.get("properties") or {}
            category = f"{props.get('category1', '')} {props.get('category2', '')}".strip()
            feed_type = str(props.get("feedType", ""))
            normalized = {
                "title": str(props.get("sourceTitle", props.get("name", "")) or category or "Incident"),
                "category": category or feed_type,
                "status": str(props.get("status", "")),
                "description": str(props.get("webBody", ""))[:600],
                "updated": str(props.get("updated", props.get("created", ""))),
                "jurisdiction": "vic",
                "source": "Emergency Management VIC",
                # Human-facing page for the incident (falls back to the VIC
                # emergency map when the feed omits a per-incident URL).
                "link": str(props.get("url") or "https://emergency.vic.gov.au/respond/"),
                "kind": "fire" if feed_type == "burn-area" else classify(category),
            }
            features.extend(_explode_geometry_collection(
                {"type": "Feature", "properties": normalized, "geometry": feature.get("geometry")}
            ))
        sources.append("Emergency Management VIC")
    except (ExternalServiceError, ValueError) as exc:
        warnings.append(f"VIC emergency feed unavailable: {exc}")

    return features, sources, warnings


def fetch_osint_wildfires() -> dict[str, Any]:
    """Active fire incidents/perimeters from the state emergency feeds."""
    features, sources, warnings = _emergency_feed_features()
    fire_features = [f for f in features if (f.get("properties") or {}).get("kind") == "fire"]
    return {
        "type": "FeatureCollection",
        "features": fire_features,
        "count": len(fire_features),
        "sources": sources,
        "warnings": warnings,
    }


def fetch_osint_incidents() -> dict[str, Any]:
    """Non-fire emergency incidents (flood, rescue, hazmat, tree down, ...)."""
    features, sources, warnings = _emergency_feed_features()
    other = [f for f in features if (f.get("properties") or {}).get("kind") != "fire"]
    return {
        "type": "FeatureCollection",
        "features": other,
        "count": len(other),
        "sources": sources,
        "warnings": warnings,
    }


def fetch_osint_cameras(
    bbox: list[Any] | tuple[Any, ...], *, limit: int = 400, timeout: float = 12.0
) -> dict[str, Any]:
    """Mapped surveillance/CCTV nodes from OpenStreetMap via Overpass."""
    south, west, north, east = validate_bbox(bbox, max_span=0.35)
    limit = max(1, min(int(limit or 400), 1000))
    query = f"""
[out:json][timeout:{int(timeout)}];
node["man_made"="surveillance"]({south:.6f},{west:.6f},{north:.6f},{east:.6f});
out body {limit};
""".strip()
    data = execute_overpass_query(query, timeout=timeout)
    items: list[dict[str, Any]] = []
    for element in data.get("elements", []):
        lat, lon = element.get("lat"), element.get("lon")
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            continue
        tags = element.get("tags") or {}
        items.append({
            "id": element.get("id"),
            "lat": round(float(lat), 6),
            "lon": round(float(lon), 6),
            "surveillance": str(tags.get("surveillance", "")),
            "camera_type": str(tags.get("surveillance:type", tags.get("camera:type", ""))),
            "operator": str(tags.get("operator", "")),
            "name": str(tags.get("name", tags.get("description", ""))),
            "zone": str(tags.get("surveillance:zone", "")),
        })
    return {
        "items": items,
        "count": len(items),
        "source": "openstreetmap-overpass",
    }


def fetch_osint_earthquakes(*, region: str = "au") -> dict[str, Any]:
    """Recent M2.5+ earthquakes from the USGS week feed, defaulting to the
    Australian region."""
    data = request_json_url(USGS_QUAKE_FEED, timeout=10.0, cache_ttl=300.0)
    south, west, north, east = AUSTRALIA_OSINT_BBOX
    items: list[dict[str, Any]] = []
    for feature in (data.get("features") or []) if isinstance(data, dict) else []:
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates") or []
        if len(coords) < 2:
            continue
        lon, lat = coords[0], coords[1]
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            continue
        if region != "global" and not (south <= lat <= north and west <= lon <= east):
            continue
        props = feature.get("properties") or {}
        items.append({
            "id": str(feature.get("id", "")),
            "lat": round(float(lat), 5),
            "lon": round(float(lon), 5),
            "depth_km": round(float(coords[2]), 1) if len(coords) > 2 and isinstance(coords[2], (int, float)) else None,
            "mag": props.get("mag"),
            "place": str(props.get("place", "")),
            "time_ms": props.get("time"),
            "url": str(props.get("url", "")),
        })
    items.sort(key=lambda item: item.get("time_ms") or 0, reverse=True)
    return {
        "items": items,
        "count": len(items),
        "source": "usgs",
        "region": region,
    }
