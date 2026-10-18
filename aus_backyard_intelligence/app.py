from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import math
import os
import re
import sys
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus
from wsgiref.simple_server import WSGIServer, make_server

from .eligibility import enrich_property_record
from .external_api import google_geocoding_configured
from .qgis import qgis_status
from .scan_agent import AreaScanAgent

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PACKAGE_ROOT / "data" / "derived" / "sample_properties.json"
DEFAULT_STATIC_INDEX = PACKAGE_ROOT / "static" / "index.html"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_JSON_BODY_BYTES = 1_000_000
# Dev/eval + feedback APIs (label store, benchmark runner, training export) are
# operator tooling — set ABI_DEV_API=0 to disable them in a public deployment.
DEV_API_ENABLED = os.environ.get("QLDR_DEV_API", "1") != "0"

AUSTRALIAN_STATE_ALIASES = {
    "act": "Australian Capital Territory",
    "nsw": "New South Wales",
    "nt": "Northern Territory",
    "qld": "Queensland",
    "sa": "South Australia",
    "tas": "Tasmania",
    "vic": "Victoria",
    "wa": "Western Australia",
}


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    """Allow slow public-data calls without blocking health/static requests."""

    daemon_threads = True

MODULE_ALIASES = {
    "pool": "pool_3d",
    "pool_2d": "pool_2d",
    "pool_3d": "pool_3d",
    "fence": "fence",
    "deck": "deck",
}

MODULE_TITLES = {
    "pool_2d": "Pool Suitability 2D",
    "pool_3d": "Pool Suitability 3D",
    "fence": "Fence Opportunity",
    "deck": "Deck and Patio Opportunity",
}

DATA_NOTE = (
    "Fixture-backed demo dataset. Rankings exclude sites that are not verified as "
    "private residential pool or fence candidates. Replace the sample JSON with derived "
    "open-data features when you connect the production pipeline."
)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def scale(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp((value - low) / (high - low), 0.0, 1.0)


def rect_width(rect: list[float]) -> float:
    return max(0.0, float(rect[2]) - float(rect[0]))


def rect_depth(rect: list[float]) -> float:
    return max(0.0, float(rect[3]) - float(rect[1]))


def rect_area(rect: list[float] | None) -> float:
    if rect is None:
        return 0.0
    return rect_width(rect) * rect_depth(rect)


def rear_backyard_rect(parcel_rect: list[float], building_rect: list[float]) -> list[float] | None:
    rear_y = max(float(building_rect[3]), float(parcel_rect[1]))
    if rear_y >= float(parcel_rect[3]):
        return None
    return [float(parcel_rect[0]), rear_y, float(parcel_rect[2]), float(parcel_rect[3])]


def side_clearances(parcel_rect: list[float], building_rect: list[float]) -> tuple[float, float]:
    left = max(0.0, float(building_rect[0]) - float(parcel_rect[0]))
    right = max(0.0, float(parcel_rect[2]) - float(building_rect[2]))
    return left, right


def round1(value: float) -> float:
    return round(value, 1)


def _cache_backend_name() -> str:
    """Report the active cache backend ('redis' or 'memory') for /api/health.
    Never raises — cache availability must not break the health probe."""
    try:
        from .cache_backend import cache_backend_name

        return cache_backend_name()
    except Exception:
        return "memory"


def parse_bounded_int(value: Any, *, name: str, default: int, lower: int, upper: int) -> int:
    candidate = default if value in (None, "") else value
    try:
        parsed = int(float(candidate))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not lower <= parsed <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}.")
    return parsed


def australian_place_query_variants(place_name: str) -> list[str]:
    """Return suburb/city-friendly Australian geocoder queries for user input."""
    cleaned = re.sub(r"\baustralia\b", "", place_name, flags=re.IGNORECASE)
    cleaned = re.sub(r"[\s,]+", " ", cleaned).strip(" ,")
    variants: list[str] = []
    parts = cleaned.split()
    if len(parts) >= 2:
        state_key = parts[-1].rstrip(".").casefold()
        state_name = AUSTRALIAN_STATE_ALIASES.get(state_key)
        if state_name:
            area = " ".join(parts[:-1]).strip(" ,")
            if area:
                variants.extend(
                    [
                        f"{area}, {state_name}, Australia",
                        f"{area}, Australia",
                        f"{area}, {parts[-1].upper()}, Australia",
                    ]
                )
    variants.append(place_name if "australia" in place_name.lower() else f"{place_name}, Australia")
    if cleaned:
        variants.append(f"{cleaned}, Australia")

    unique: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        key = " ".join(variant.casefold().split())
        if key not in seen:
            unique.append(variant)
            seen.add(key)
    return unique


def load_properties(data_path: str | Path | None = None) -> list[dict[str, Any]]:
    source_path = Path(data_path or DEFAULT_DATA_PATH)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Sample property file must contain a JSON list.")
    normalized = []
    for record in payload:
        candidate = dict(record)
        # Older generated OSM fixtures used hash-based pool flags. They have no
        # evidence source and must not exclude otherwise valid homes.
        if (
            str(candidate.get("id", "")).startswith("OSM-")
            and candidate.get("has_existing_pool")
            and not candidate.get("existing_pool_source")
        ):
            candidate["has_existing_pool"] = False
            candidate["exclude_from_pool_candidates"] = False
            candidate["pool_detection_status"] = "legacy_unverified_flag_ignored"
        normalized.append(candidate)
    enriched = [enrich_property_record(record) for record in normalized]
    return [record for record in enriched if record["pool_fence_candidate"]]


def derive_features(record: dict[str, Any]) -> dict[str, Any]:
    parcel_rect = record["parcel_rect"]
    building_rect = record["building_rect"]
    backyard_rect = rear_backyard_rect(parcel_rect, building_rect)
    land_area = rect_area(parcel_rect)
    building_area = rect_area(building_rect)
    backyard_area = rect_area(backyard_rect)
    backyard_width = rect_width(backyard_rect) if backyard_rect else 0.0
    backyard_depth = rect_depth(backyard_rect) if backyard_rect else 0.0
    left_clearance, right_clearance = side_clearances(parcel_rect, building_rect)
    elevations = record["elevation_corners_m"]
    elevation_values = [float(elevations[key]) for key in ("sw", "se", "nw", "ne")]
    front_avg = (float(elevations["sw"]) + float(elevations["se"])) / 2.0
    rear_avg = (float(elevations["nw"]) + float(elevations["ne"])) / 2.0
    west_avg = (float(elevations["sw"]) + float(elevations["nw"])) / 2.0
    east_avg = (float(elevations["se"]) + float(elevations["ne"])) / 2.0
    parcel_width = max(rect_width(parcel_rect), 1.0)
    parcel_depth = max(rect_depth(parcel_rect), 1.0)
    slope_y_pct = abs(rear_avg - front_avg) / parcel_depth * 100.0
    slope_x_pct = abs(east_avg - west_avg) / parcel_width * 100.0
    slope_pct = max(slope_x_pct, slope_y_pct)
    relief_m = max(elevation_values) - min(elevation_values)
    buildable_rear_zone_sqm = max(0.0, backyard_area - 24.0)
    return {
        "land_area_sqm": round1(land_area),
        "building_area_sqm": round1(building_area),
        "building_coverage_ratio": round(building_area / land_area, 3) if land_area else 0.0,
        "backyard_area_sqm": round1(backyard_area),
        "backyard_width_m": round1(backyard_width),
        "backyard_depth_m": round1(backyard_depth),
        "buildable_rear_zone_sqm": round1(buildable_rear_zone_sqm),
        "left_side_clearance_m": round1(left_clearance),
        "right_side_clearance_m": round1(right_clearance),
        "max_side_clearance_m": round1(max(left_clearance, right_clearance)),
        "slope_pct": round1(slope_pct),
        "slope_x_pct": round1(slope_x_pct),
        "slope_y_pct": round1(slope_y_pct),
        "relief_m": round1(relief_m),
        "elevation_min_m": round1(min(elevation_values)),
        "elevation_max_m": round1(max(elevation_values)),
        "solar_access_pct": round1(float(record["solar_access_pct"])),
        "view_quality": int(record["view_quality"]),
        "vegetation_pct": round1(float(record["vegetation_pct"])),
        "road_frontage_m": round1(float(record["road_frontage_m"])),
    }


def score_pool_2d(features: dict[str, Any], record: dict[str, Any]) -> float:
    if record.get("has_existing_pool") or record.get("exclude_from_pool_candidates"):
        return 0.0
    
    is_small = features["backyard_area_sqm"] < 120.0 and features["buildable_rear_zone_sqm"] >= 15.0
    if is_small:
        area_score = scale(features["backyard_area_sqm"], 20.0, 120.0) * 38.0
        depth_score = scale(features["backyard_depth_m"], 3.0, 8.0) * 12.0
    else:
        area_score = scale(features["backyard_area_sqm"], 120.0, 420.0) * 38.0
        depth_score = scale(features["backyard_depth_m"], 8.0, 18.0) * 12.0

    access_score = scale(features["max_side_clearance_m"], 1.2, 3.2) * 16.0
    coverage_score = (1.0 - scale(features["building_coverage_ratio"], 0.28, 0.62)) * 12.0
    solar_score = scale(record["solar_access_pct"], 45.0, 90.0) * 10.0
    vegetation_penalty = scale(record["vegetation_pct"], 10.0, 70.0) * 8.0
    constraint_penalty = (
        float(record["flood_risk"]) * 9.0
        + float(bool(record["heritage_overlay"])) * 10.0
        + float(bool(record["overhead_powerline"])) * 6.0
    )
    score = 22.0 + area_score + depth_score + access_score + coverage_score + solar_score
    upper_bound = 75.0 if is_small else 100.0
    return round1(clamp(score - vegetation_penalty - constraint_penalty, 0.0, upper_bound))


def score_pool_3d(features: dict[str, Any], record: dict[str, Any], pool_2d_score: float) -> float:
    if record.get("has_existing_pool") or record.get("exclude_from_pool_candidates"):
        return 0.0
    slope_penalty = scale(features["slope_pct"], 3.0, 16.0) * 20.0
    relief_penalty = scale(features["relief_m"], 0.7, 5.5) * 10.0
    drainage_bonus = 6.0 if 1.0 <= features["slope_pct"] <= 6.0 else 0.0
    view_bonus = scale(record["view_quality"], 4.0, 10.0) * 3.0
    score = pool_2d_score - slope_penalty - relief_penalty + drainage_bonus + view_bonus
    
    is_small = features["backyard_area_sqm"] < 120.0 and features["buildable_rear_zone_sqm"] >= 15.0
    upper_bound = 75.0 if is_small else 100.0
    return round1(clamp(score, 0.0, upper_bound))


def score_fence(features: dict[str, Any], record: dict[str, Any]) -> float:
    privacy_exposure = scale(record["road_frontage_m"], 14.0, 28.0) * 16.0
    corner_bonus = 10.0 if record["corner_lot"] else 0.0
    boundary_length_score = scale(features["backyard_depth_m"], 6.0, 18.0) * 16.0
    low_screening_bonus = scale(100.0 - record["vegetation_pct"], 25.0, 85.0) * 18.0
    access_score = scale(features["max_side_clearance_m"], 1.0, 3.0) * 10.0
    constraint_penalty = float(bool(record["heritage_overlay"])) * 12.0 + float(record["flood_risk"]) * 6.0
    score = 25.0 + privacy_exposure + corner_bonus + boundary_length_score + low_screening_bonus + access_score
    return round1(clamp(score - constraint_penalty, 0.0, 100.0))


def score_deck(features: dict[str, Any], record: dict[str, Any]) -> float:
    area_score = scale(features["backyard_area_sqm"], 90.0, 260.0) * 22.0
    solar_score = scale(record["solar_access_pct"], 50.0, 90.0) * 18.0
    view_score = scale(record["view_quality"], 3.0, 10.0) * 16.0
    terrain_bonus = 10.0 if 1.0 <= features["slope_pct"] <= 8.0 else 4.0
    terrain_bonus += scale(features["relief_m"], 0.8, 4.0) * 8.0
    vegetation_penalty = scale(record["vegetation_pct"], 20.0, 70.0) * 8.0
    constraint_penalty = float(record["flood_risk"]) * 8.0 + float(bool(record["heritage_overlay"])) * 6.0
    score = 24.0 + area_score + solar_score + view_score + terrain_bonus
    return round1(clamp(score - vegetation_penalty - constraint_penalty, 0.0, 100.0))


def build_scores(record: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    pool_2d = score_pool_2d(features, record)
    pool_3d = score_pool_3d(features, record, pool_2d)
    is_small = features["backyard_area_sqm"] < 120.0 and features["buildable_rear_zone_sqm"] >= 15.0
    pool_type = "small" if is_small else "standard"
    return {
        "pool_2d": pool_2d,
        "pool_3d": pool_3d,
        "fence": score_fence(features, record),
        "deck": score_deck(features, record),
        "pool_type": pool_type,
    }


def constraint_labels(record: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    if record["flood_risk"]:
        labels.append("flood sensitivity")
    if record["heritage_overlay"]:
        labels.append("heritage planning friction")
    if record["overhead_powerline"]:
        labels.append("overhead service conflict")
    return labels


def explain_pool_2d(features: dict[str, Any], record: dict[str, Any]) -> str:
    if record.get("has_existing_pool") or record.get("exclude_from_pool_candidates"):
        return "Property already has an existing swimming pool."
    pieces = []
    is_small = features["backyard_area_sqm"] < 120.0 and features["buildable_rear_zone_sqm"] >= 15.0
    if is_small:
        pieces.append(f"Backyard of {features['backyard_area_sqm']} sqm is a great candidate for a compact plunge or splash pool (using {features['buildable_rear_zone_sqm']} sqm buildable rear zone).")
    else:
        if features["backyard_area_sqm"] >= 320.0:
            pieces.append(f"Large rear yard of {features['backyard_area_sqm']} sqm gives strong pool fit.")
        elif features["backyard_area_sqm"] >= 220.0:
            pieces.append(f"Usable rear yard of {features['backyard_area_sqm']} sqm supports a medium pool concept.")
        else:
            pieces.append(f"Tighter rear yard of {features['backyard_area_sqm']} sqm limits pool layout flexibility.")
    if features["max_side_clearance_m"] >= 2.4:
        pieces.append(f"Side access of {features['max_side_clearance_m']} m is friendly for construction access.")
    elif features["max_side_clearance_m"] >= 1.5:
        pieces.append(f"Side access of {features['max_side_clearance_m']} m is workable but not generous.")
    else:
        pieces.append(f"Side access of {features['max_side_clearance_m']} m is a likely logistics pinch point.")
    if record["vegetation_pct"] >= 45.0:
        pieces.append("Tree cover is meaningful, so clearing or design avoidance may be needed.")
    labels = constraint_labels(record)
    pieces.append("Constraints are low." if not labels else f"Watch {', '.join(labels)}.")
    return " ".join(pieces)


def explain_pool_3d(features: dict[str, Any], record: dict[str, Any]) -> str:
    if record.get("has_existing_pool") or record.get("exclude_from_pool_candidates"):
        return "Property already has an existing swimming pool."
    pieces = [explain_pool_2d(features, record)]
    if features["slope_pct"] < 4.0:
        pieces.append(f"Gentle slope of {features['slope_pct']}% keeps earthworks manageable.")
    elif features["slope_pct"] < 8.0:
        pieces.append(f"Moderate slope of {features['slope_pct']}% should be buildable with some retaining.")
    else:
        pieces.append(f"Steeper slope of {features['slope_pct']}% will push retaining and excavation costs up.")
    pieces.append(f"Relief is {features['relief_m']} m across the parcel.")
    return " ".join(pieces)


def explain_fence(features: dict[str, Any], record: dict[str, Any]) -> str:
    pieces = []
    if record["corner_lot"]:
        pieces.append("Corner lot geometry creates extra privacy and enclosure upside.")
    else:
        pieces.append("Standard lot form keeps fence delivery straightforward.")
    if record["road_frontage_m"] >= 22.0:
        pieces.append(f"Road frontage of {record['road_frontage_m']} m gives a long visible boundary to improve.")
    if record["vegetation_pct"] <= 20.0:
        pieces.append("Low natural screening increases the value of a privacy fence.")
    else:
        pieces.append("Existing vegetation already contributes some privacy.")
    if record["heritage_overlay"]:
        pieces.append("Heritage controls may slow approvals for street-facing treatments.")
    return " ".join(pieces)


def explain_deck(features: dict[str, Any], record: dict[str, Any]) -> str:
    pieces = []
    if record["view_quality"] >= 8:
        pieces.append("This parcel has strong elevated outlook potential for a deck or patio.")
    elif record["view_quality"] >= 6:
        pieces.append("There is reasonable outlook value for an entertainment deck.")
    else:
        pieces.append("Deck value here leans more on outdoor living space than views.")
    if 1.0 <= features["slope_pct"] <= 8.0:
        pieces.append(f"A {features['slope_pct']}% slope can help create stepped entertaining zones.")
    else:
        pieces.append(f"Slope of {features['slope_pct']}% is not a major deck differentiator.")
    pieces.append(f"Solar access is {record['solar_access_pct']}%, which supports outdoor usability.")
    return " ".join(pieces)


def build_explanations(features: dict[str, Any], record: dict[str, Any]) -> dict[str, str]:
    return {
        "pool_2d": explain_pool_2d(features, record),
        "pool_3d": explain_pool_3d(features, record),
        "fence": explain_fence(features, record),
        "deck": explain_deck(features, record),
    }


def build_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    features = derive_features(record)
    scores = build_scores(record, features)
    explanations = build_explanations(features, record)
    return {
        "property_id": record["id"],
        "address": record["address"],
        "suburb": record["suburb"],
        "lga": record["lga"],
        "property": record,
        "derived_features": features,
        "scores": scores,
        "explanations": explanations,
        "headline_score": scores["pool_3d"],
        "candidate_label": record["candidate_label"],
        "eligibility_confidence": record["eligibility_confidence"],
        "eligibility_reason": record["eligibility_reason"],
    }


def normalize_module(module_name: str | None) -> str:
    candidate = (module_name or "pool_3d").strip().lower()
    if candidate not in MODULE_ALIASES:
        raise ValueError(f"Unsupported module '{module_name}'.")
    return MODULE_ALIASES[candidate]


def sort_rankings(items: list[dict[str, Any]], module_name: str) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: (-item["scores"][module_name], item["property_id"]))


def snapshot_to_ranking_row(snapshot: dict[str, Any], module_name: str, rank: int) -> dict[str, Any]:
    features = snapshot["derived_features"]
    record = snapshot["property"]
    lat = record.get("real_lat", record.get("anchor_lat"))
    lon = record.get("real_lon", record.get("anchor_lon"))
    map_query = f"{lat},{lon}" if lat is not None and lon is not None else snapshot["address"]
    encoded_query = quote_plus(str(map_query))
    return {
        "rank": rank,
        "property_id": snapshot["property_id"],
        "address": snapshot["address"],
        "suburb": snapshot["suburb"],
        "lga": snapshot["lga"],
        "module": module_name,
        "module_title": MODULE_TITLES[module_name],
        "module_score": snapshot["scores"][module_name],
        "scores": snapshot["scores"],
        "derived_features": features,
        "explanation": snapshot["explanations"][module_name],
        "candidate_label": snapshot["candidate_label"],
        "eligibility_confidence": snapshot["eligibility_confidence"],
        "property_classification": record["property_classification"],
        "ownership_class": record["ownership_class"],
        "contact_available": record.get("contact_available", "no"),
        "owner": {
            "name": record.get("owner_name", ""),
            "phone": record.get("owner_phone", ""),
            "email": record.get("owner_email", ""),
            "available": record.get("contact_available", "no"),
        },
        "location": {
            "lat": lat,
            "lon": lon,
        },
        "links": {
            "google_maps": f"https://www.google.com/maps/search/?api=1&query={encoded_query}",
            "street_view": f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}" if lat is not None and lon is not None else "",
            "viewer": f"/viewer?module={quote_plus(module_name)}&lga={quote_plus(snapshot['lga'])}",
            "property_api": f"/api/property/{snapshot['property_id']}",
        },
    }


class BackyardDemoApp:
    def __init__(self, data_path: str | Path | None = None, static_index: str | Path | None = None) -> None:
        self.data_path = Path(data_path or DEFAULT_DATA_PATH)
        self.static_index = Path(static_index or DEFAULT_STATIC_INDEX)
        self._snapshots = self._load_snapshots()
        self.scan_agent = AreaScanAgent(data_path=str(self.data_path))

    def _load_snapshots(self) -> list[dict[str, Any]]:
        return [build_snapshot(record) for record in load_properties(self.data_path)]

    def refresh(self) -> None:
        self._snapshots = self._load_snapshots()
        self.scan_agent.data_path = str(self.data_path)

    @property
    def snapshots(self) -> list[dict[str, Any]]:
        return list(self._snapshots)

    def rankings(self, module_name: str, lga: str | None = None, top_k: int = 5) -> list[dict[str, Any]]:
        normalized_module = normalize_module(module_name)
        candidates = self.snapshots
        if normalized_module in {"pool_2d", "pool_3d"}:
            candidates = [
                item
                for item in candidates
                if not item["property"].get("has_existing_pool")
                and not item["property"].get("exclude_from_pool_candidates")
            ]
        if lga:
            candidates = [item for item in candidates if item["lga"].lower() == lga.lower()]
        ordered = sort_rankings(candidates, normalized_module)
        limited = ordered[: max(1, top_k)]
        return [
            snapshot_to_ranking_row(snapshot, normalized_module, rank=index)
            for index, snapshot in enumerate(limited, start=1)
        ]

    def get_property(self, property_id: str) -> dict[str, Any] | None:
        for snapshot in self._snapshots:
            if snapshot["property_id"].lower() == property_id.lower():
                return snapshot
        return None

    def list_lgas(self) -> list[str]:
        return sorted({snapshot["lga"] for snapshot in self._snapshots})

    def build_rankings_payload(self, query: dict[str, list[str]]) -> dict[str, Any]:
        requested_module = query.get("module", ["pool_3d"])[0]
        lga = query.get("lga", [None])[0]
        top_k = parse_bounded_int(
            query.get("top_k", ["5"])[0], name="top_k", default=5, lower=1, upper=100
        )
        module_name = normalize_module(requested_module)
        items = self.rankings(module_name, lga=lga, top_k=top_k)
        return {
            "metadata": {
                "module": module_name,
                "module_title": MODULE_TITLES[module_name],
                "lga": lga,
                "top_k": top_k,
                "returned": len(items),
                "available_lgas": self.list_lgas(),
                "candidate_policy": "private_residential_only",
                "data_note": DATA_NOTE,
            },
            "items": items,
        }

    def build_rankings_csv(self, query: dict[str, list[str]]) -> str:
        payload = self.build_rankings_payload(query)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "rank",
                "property_id",
                "address",
                "suburb",
                "lga",
                "module",
                "module_score",
                "pool_2d",
                "pool_3d",
                "fence",
                "deck",
                "candidate_label",
                "ownership_class",
                "property_classification",
                "backyard_area_sqm",
                "backyard_depth_m",
                "max_side_clearance_m",
                "slope_pct",
                "relief_m",
                "explanation",
            ]
        )
        for item in payload["items"]:
            features = item["derived_features"]
            writer.writerow(
                [
                    item["rank"],
                    item["property_id"],
                    item["address"],
                    item["suburb"],
                    item["lga"],
                    item["module"],
                    item["module_score"],
                    item["scores"]["pool_2d"],
                    item["scores"]["pool_3d"],
                    item["scores"]["fence"],
                    item["scores"]["deck"],
                    item["candidate_label"],
                    item["ownership_class"],
                    item["property_classification"],
                    features["backyard_area_sqm"],
                    features["backyard_depth_m"],
                    features["max_side_clearance_m"],
                    features["slope_pct"],
                    features["relief_m"],
                    item["explanation"],
                ]
            )
        return output.getvalue()

    def _json(self, start_response: Any, status: str, payload: dict[str, Any]) -> list[bytes]:
        def _default(obj):
            try:
                import numpy as np
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
            except ImportError:
                pass
            return str(obj)
        body = json.dumps(payload, indent=2, default=_default).encode("utf-8")
        headers = [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            ("X-Content-Type-Options", "nosniff"),
        ]
        start_response(status, headers)
        return [body]

    def _text(
        self,
        start_response: Any,
        status: str,
        text: str,
        content_type: str = "text/plain; charset=utf-8",
        extra_headers: list[tuple[str, str]] | None = None,
    ) -> list[bytes]:
        body = text.encode("utf-8")
        headers = [
            ("Content-Type", content_type),
            ("Content-Length", str(len(body))),
            ("X-Content-Type-Options", "nosniff"),
        ]
        if extra_headers:
            headers.extend(extra_headers)
        start_response(status, headers)
        return [body]

    def _read_json_body(self, environ: dict[str, Any]) -> dict[str, Any]:
        length = parse_bounded_int(
            environ.get("CONTENT_LENGTH"),
            name="Content-Length",
            default=0,
            lower=0,
            upper=MAX_JSON_BODY_BYTES,
        )
        raw = environ["wsgi.input"].read(length) if length > 0 else b""
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Request body must be valid UTF-8 JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def __call__(self, environ: dict[str, Any], start_response: Any) -> list[bytes]:
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = environ.get("PATH_INFO", "/") or "/"
        query = parse_qs(environ.get("QUERY_STRING", ""), keep_blank_values=True)

        try:
            # Secure static asset serving router
            if method == "GET" and (path.startswith("/static/") or path.endswith((".css", ".js", ".png", ".jpg", ".jpeg", ".svg", ".json"))):
                clean_path = path.lstrip("/")
                if not clean_path.startswith("static/"):
                    clean_path = "static/" + clean_path
                safe_path = (PACKAGE_ROOT / clean_path).resolve()
                static_root = (PACKAGE_ROOT / "static").resolve()
                # is_relative_to (not a string prefix) so a sibling dir whose name
                # merely starts with "static" can never be served.
                if safe_path.is_relative_to(static_root) and safe_path.is_file():
                    ext = safe_path.suffix.lower()
                    mime_types = {
                        ".css": "text/css; charset=utf-8",
                        ".js": "application/javascript; charset=utf-8",
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".svg": "image/svg+xml",
                        ".json": "application/json; charset=utf-8",
                    }
                    content_type = mime_types.get(ext, "application/octet-stream")
                    body = safe_path.read_bytes()
                    # Always revalidate static assets so browsers never run stale
                    # cached JS/CSS (a recurring source of "blank map after a fix").
                    headers = [
                        ("Content-Type", content_type),
                        ("Content-Length", str(len(body))),
                        ("Cache-Control", "no-cache, no-store, must-revalidate"),
                        ("X-Content-Type-Options", "nosniff"),
                    ]
                    start_response("200 OK", headers)
                    return [body]

            if method == "GET" and path in {"/", "/index.html", "/viewer", "/viewer.html"}:
                viewer_path = PACKAGE_ROOT / "static" / "viewer.html"
                if not viewer_path.exists():
                    return self._text(start_response, "404 Not Found", "viewer.html not found.")
                return self._text(
                    start_response,
                    "200 OK",
                    viewer_path.read_text(encoding="utf-8"),
                    content_type="text/html; charset=utf-8",
                )
            if method == "GET" and path in {"/dashboard", "/dashboard.html"}:
                return self._text(
                    start_response,
                    "200 OK",
                    self.static_index.read_text(encoding="utf-8"),
                    content_type="text/html; charset=utf-8",
                )
            if method == "GET" and path.startswith("/3d"):
                model_path = PACKAGE_ROOT / "static" / "model3d.html"
                if not model_path.exists():
                    return self._text(start_response, "404 Not Found", "model3d.html not found.")
                return self._text(
                    start_response,
                    "200 OK",
                    model_path.read_text(encoding="utf-8"),
                    content_type="text/html; charset=utf-8",
                )
            if method == "GET" and path in {"/api/viewer/map"}:
                module_name = normalize_module(query.get("module", ["pool_3d"])[0])
                lga = query.get("lga", [None])[0]
                top_k = parse_bounded_int(
                    query.get("top_k", ["15"])[0], name="top_k", default=15, lower=1, upper=100
                )
                
                from .map_export import get_map_geojson
                geojson = get_map_geojson(data_path=self.data_path, module_name=module_name, lga=lga, top_k=top_k)
                return self._json(start_response, "200 OK", geojson)
            if method == "GET" and path.startswith("/api/viewer/property/"):
                property_id = path.rsplit("/", 1)[-1]
                from .map_export import get_property_geojson
                geojson = get_property_geojson(property_id=property_id, data_path=self.data_path)
                if not geojson:
                    return self._json(start_response, "404 Not Found", {"error": "Property geometries not found."})
                return self._json(start_response, "200 OK", geojson)
            if method == "POST" and path == "/api/analyze/area":
                body = self._read_json_body(environ)
                if body.get("polygon") is None and body.get("bounds") is None:
                    raise ValueError("Provide a polygon with at least three coordinate pairs.")
                from .map_export import analyze_area
                result = analyze_area(body)
                return self._json(start_response, "200 OK", result)
            if method == "GET" and path in ("/api/osm/features", "/api/local/features"):
                from .external_api import (
                    DEFAULT_DETECTOR_MODEL,
                    DEFAULT_DETECTOR_PRECISION,
                    DEFAULT_SEGMENTATION_MODE,
                    fetch_osm_features,
                )

                bbox = [
                    query.get("south", [None])[0],
                    query.get("west", [None])[0],
                    query.get("north", [None])[0],
                    query.get("east", [None])[0],
                ]
                zoom = parse_bounded_int(
                    query.get("zoom", ["16"])[0], name="zoom", default=16, lower=0, upper=22
                )
                layers = query.get("layers", [None])[0]
                detector_model = query.get("detector_model", [DEFAULT_DETECTOR_MODEL])[0]
                segmentation_mode = query.get("segmentation_mode", [DEFAULT_SEGMENTATION_MODE])[0]
                detector_precision = query.get("detector_precision", ["fast" if path == "/api/local/features" else DEFAULT_DETECTOR_PRECISION])[0]
                raw_analysis_polygon = query.get("analysis_polygon", [None])[0]
                analysis_polygon = json.loads(raw_analysis_polygon) if raw_analysis_polygon else None
                force_refresh = str(query.get("force_refresh", ["0"])[0]).strip().lower() in {"1", "true", "yes", "on"}
                
                target_timeout = 10.0 if layers == "pools" else 14.0
                kwargs: dict[str, Any] = {
                    "zoom": zoom,
                    "timeout": target_timeout,
                    "detector_model": detector_model,
                    "segmentation_mode": segmentation_mode,
                    "detector_precision": detector_precision,
                    "analysis_polygon": analysis_polygon,
                    "force_refresh": force_refresh,
                }
                if layers is not None:
                    kwargs["layers"] = layers

                return self._json(
                    start_response,
                    "200 OK",
                    fetch_osm_features(bbox, data_path=str(self.data_path), **kwargs),
                )
            if method == "GET" and path == "/api/terrain/elevation-grid":
                from .external_api import ExternalServiceError, fetch_elevation_grid

                bbox = [
                    query.get("south", [None])[0],
                    query.get("west", [None])[0],
                    query.get("north", [None])[0],
                    query.get("east", [None])[0],
                ]
                size = parse_bounded_int(
                    query.get("size", ["5"])[0], name="size", default=5, lower=3, upper=8
                )
                try:
                    return self._json(start_response, "200 OK", fetch_elevation_grid(bbox, size=size))
                except ExternalServiceError as exc:
                    return self._json(start_response, "503 Service Unavailable", {"error": str(exc)})
            if method == "GET" and path == "/api/detection/status":
                from .external_api import get_detector_capacity_status
                from .yolo_detector import get_detector_status

                return self._json(
                    start_response,
                    "200 OK",
                    {
                        **get_detector_status(),
                        "detector_process_capacity": get_detector_capacity_status(),
                    },
                )
            if method == "GET" and path == "/api/detection/pipeline":
                from .external_api import (
                    DEFAULT_DETECTOR_MODEL,
                    DEFAULT_DETECTOR_PRECISION,
                    DEFAULT_SEGMENTATION_MODE,
                    fetch_osm_features,
                )

                bbox = [
                    query.get("south", [None])[0],
                    query.get("west", [None])[0],
                    query.get("north", [None])[0],
                    query.get("east", [None])[0],
                ]
                zoom = parse_bounded_int(
                    query.get("zoom", ["16"])[0], name="zoom", default=16, lower=0, upper=22
                )
                layers = query.get("layers", ["pools,buildings,trees"])[0]
                detector_model = query.get("detector_model", [DEFAULT_DETECTOR_MODEL])[0]
                segmentation_mode = query.get("segmentation_mode", [DEFAULT_SEGMENTATION_MODE])[0]
                detector_precision = query.get("detector_precision", [DEFAULT_DETECTOR_PRECISION])[0]
                raw_analysis_polygon = query.get("analysis_polygon", [None])[0]
                analysis_polygon = json.loads(raw_analysis_polygon) if raw_analysis_polygon else None
                return self._json(
                    start_response,
                    "200 OK",
                    fetch_osm_features(
                        bbox,
                        data_path=str(self.data_path),
                        zoom=zoom,
                        layers=layers,
                        timeout=12.0,
                        detector_model=detector_model,
                        segmentation_mode=segmentation_mode,
                        detector_precision=detector_precision,
                        analysis_polygon=analysis_polygon,
                    ),
                )
            if method == "POST" and path == "/api/agent/scan-area":
                body = self._read_json_body(environ)
                passes = parse_bounded_int(
                    body.get("passes"),
                    name="passes",
                    default=12,
                    lower=3,
                    upper=40,
                )
                source = str(body.get("source") or "manual")[:40]
                label = body.get("label")
                if label is not None:
                    label = str(label)[:120]
                result = self.scan_agent.start_scan(body, passes=passes, source=source, label=label)
                return self._json(start_response, "202 Accepted", result)
            if method == "GET" and path == "/api/agent/scans":
                limit = parse_bounded_int(
                    query.get("limit", ["8"])[0],
                    name="limit",
                    default=8,
                    lower=1,
                    upper=25,
                )
                return self._json(start_response, "200 OK", {"items": self.scan_agent.list_scans(limit=limit)})
            if method == "POST" and path.startswith("/api/agent/scans/") and path.endswith("/cancel"):
                parts = path.strip("/").split("/")
                if len(parts) != 5 or parts[:3] != ["api", "agent", "scans"] or parts[-1] != "cancel":
                    return self._json(start_response, "404 Not Found", {"error": "Scan job not found."})
                scan_id = parts[3]
                body = self._read_json_body(environ)
                reason = str(body.get("reason") or "Cancelled by the user.")[:300]
                result = self.scan_agent.cancel_scan(scan_id, reason=reason)
                if result is None:
                    return self._json(start_response, "404 Not Found", {"error": "Scan job not found."})
                return self._json(start_response, "200 OK", result)
            if method == "GET" and path.startswith("/api/agent/scans/"):
                scan_id = path.rsplit("/", 1)[-1]
                result = self.scan_agent.get_scan(scan_id)
                if result is None:
                    return self._json(start_response, "404 Not Found", {"error": "Scan job not found."})
                return self._json(start_response, "200 OK", result)
            if method == "GET" and path == "/api/agent/leads":
                limit = parse_bounded_int(
                    query.get("limit", ["50"])[0],
                    name="limit",
                    default=50,
                    lower=1,
                    upper=100,
                )
                return self._json(
                    start_response,
                    "200 OK",
                    {
                        "items": self.scan_agent.aggregate_leads(limit=limit),
                        "address_lookup": {
                            "google_configured": google_geocoding_configured(),
                            "exact_provider": "google" if google_geocoding_configured() else None,
                            "fallback_provider": "openstreetmap_nominatim",
                            "unit_policy": "explicit_source_only",
                        },
                    },
                )

            # ---- Evaluation harness + detection feedback (dev mode) ----
            if (path.startswith("/api/eval") or path.startswith("/api/feedback")) and not DEV_API_ENABLED:
                return self._json(
                    start_response,
                    "403 Forbidden",
                    {"error": "Dev/eval APIs are disabled on this deployment (ABI_DEV_API=0)."},
                )
            if method == "GET" and path == "/api/eval/labels":
                from . import evaluation

                labels = evaluation.load_labels()
                return self._json(
                    start_response,
                    "200 OK",
                    {"items": labels[:1200], "counts": evaluation.label_counts(labels)},
                )
            if method == "POST" and path == "/api/eval/bootstrap":
                from . import evaluation
                from .external_api import validate_bbox

                body = self._read_json_body(environ)
                bounds = body.get("bounds")
                if isinstance(bounds, dict):
                    bbox = validate_bbox(
                        [bounds.get("south"), bounds.get("west"), bounds.get("north"), bounds.get("east")]
                    )
                else:
                    bbox = validate_bbox(bounds)
                area = str(body.get("area") or "")[:80]
                return self._json(start_response, "200 OK", evaluation.bootstrap_labels(bbox, area=area))
            if method == "POST" and path == "/api/eval/holdout-labels":
                from . import evaluation

                body = self._read_json_body(environ)
                return self._json(
                    start_response,
                    "200 OK",
                    evaluation.import_holdout_labels(body.get("records")),
                )
            if method == "POST" and path == "/api/eval/run":
                from . import evaluation

                body = self._read_json_body(environ)
                max_cells = parse_bounded_int(
                    body.get("max_cells"), name="max_cells", default=12, lower=1, upper=40
                )
                report = evaluation.run_evaluation(
                    data_path=str(self.data_path),
                    detector_model=str(body.get("detector_model") or "yolov11s"),
                    max_cells=max_cells,
                    build_tag=str(body.get("build") or ""),
                    evaluation_split=str(body.get("split") or "development"),
                )
                status = "200 OK" if "error" not in report else "400 Bad Request"
                return self._json(start_response, status, report)
            if method == "GET" and path == "/api/eval/report":
                from . import evaluation

                report = evaluation.load_report()
                if report is None:
                    return self._json(start_response, "404 Not Found", {"error": "No evaluation report yet."})
                return self._json(start_response, "200 OK", report)
            if method == "POST" and path == "/api/feedback/pool":
                from . import evaluation

                body = self._read_json_body(environ)
                entry = evaluation.add_feedback(body)
                return self._json(start_response, "200 OK", {"saved": entry, "stats": evaluation.feedback_stats()})
            if method == "GET" and path == "/api/feedback/stats":
                from . import evaluation

                return self._json(start_response, "200 OK", evaluation.feedback_stats())
            if method == "GET" and path == "/api/feedback/export":
                from . import evaluation

                payload = evaluation.export_feedback().encode("utf-8")
                start_response(
                    "200 OK",
                    [
                        ("Content-Type", "application/jsonl; charset=utf-8"),
                        ("Content-Disposition", "attachment; filename=pool_feedback.jsonl"),
                        ("Content-Length", str(len(payload))),
                    ],
                )
                return [payload]
            if method == "GET" and path == "/api/eval/pipeline":
                from . import evaluation

                return self._json(start_response, "200 OK", evaluation.pipeline_status())
            if method == "GET" and path == "/api/eval/training-export":
                from . import evaluation

                payload = evaluation.export_training_candidates().encode("utf-8")
                start_response(
                    "200 OK",
                    [
                        ("Content-Type", "application/jsonl; charset=utf-8"),
                        ("Content-Disposition", "attachment; filename=training_candidates.jsonl"),
                        ("Content-Length", str(len(payload))),
                    ],
                )
                return [payload]
            if method == "POST" and path == "/api/agent/scan-place":
                from .external_api import geocode_places, validate_bbox

                body = self._read_json_body(environ)
                place_name = " ".join(str(body.get("area") or body.get("place") or "").split())
                if len(place_name) < 2:
                    raise ValueError("area must contain at least two characters.")
                passes = parse_bounded_int(
                    body.get("passes"),
                    name="passes",
                    default=12,
                    lower=3,
                    upper=40,
                )
                radius_km = body.get("radius_km", 3.0)
                try:
                    radius_km = float(radius_km)
                except (TypeError, ValueError) as exc:
                    raise ValueError("radius_km must be a number.") from exc
                if not math.isfinite(radius_km) or not 0.5 <= radius_km <= 12.0:
                    raise ValueError("radius_km must be between 0.5 and 12.")

                area_types = {
                    "city",
                    "town",
                    "suburb",
                    "village",
                    "municipality",
                    "locality",
                    "administrative",
                    "county",
                    "state",
                }

                def _is_australian_place(place: dict[str, Any]) -> bool:
                    return place.get("country_code") == "au" or "australia" in str(place.get("display_name", "")).lower()

                def _is_area_place(place: dict[str, Any]) -> bool:
                    place_type = str(place.get("type", "")).lower()
                    place_class = str(place.get("class", "")).lower()
                    return place_type in area_types or place_class in {"boundary", "place"}

                places: list[dict[str, Any]] = []
                for search_text in australian_place_query_variants(place_name):
                    query_places = geocode_places(search_text, limit=5)
                    places.extend(query_places)
                    if any(_is_australian_place(candidate) and _is_area_place(candidate) for candidate in query_places):
                        break

                australian_places = [place for place in places if _is_australian_place(place)]

                def _place_rank(place: dict[str, Any]) -> tuple[int, float, float]:
                    bbox_area = 0.0
                    raw_bbox = place.get("boundingbox")
                    if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
                        try:
                            south, north, west, east = (float(raw_bbox[index]) for index in range(4))
                            bbox_area = max(0.0, north - south) * max(0.0, east - west)
                        except (TypeError, ValueError):
                            bbox_area = 0.0
                    is_area = int(_is_area_place(place))
                    importance = float(place.get("importance") or 0.0)
                    return is_area, bbox_area, importance

                place = max(australian_places, key=_place_rank) if australian_places else None
                if place is None:
                    return self._json(
                        start_response,
                        "404 Not Found",
                        {"error": f"No Australian place found for '{place_name}'."},
                    )

                lat = float(place["lat"])
                lon = float(place["lon"])
                lat_delta = radius_km / 111.32
                lon_delta = radius_km / (111.32 * max(math.cos(math.radians(lat)), 1e-9))
                bbox_values = [lat - lat_delta, lon - lon_delta, lat + lat_delta, lon + lon_delta]
                raw_bbox = place.get("boundingbox")
                if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
                    try:
                        south, north, west, east = (float(raw_bbox[index]) for index in range(4))
                        if 0.003 <= north - south <= 0.18 and 0.003 <= east - west <= 0.18:
                            bbox_values = [south, west, north, east]
                    except (TypeError, ValueError):
                        pass
                bbox = validate_bbox(bbox_values)
                scan = self.scan_agent.start_scan(
                    {
                        "polygon": [
                            [bbox[1], bbox[0]],
                            [bbox[3], bbox[0]],
                            [bbox[3], bbox[2]],
                            [bbox[1], bbox[2]],
                        ],
                        "area": place_name,
                        "replace_active": body.get("replace_active") is True,
                    },
                    passes=passes,
                    source="dashboard-place",
                    label=str(place.get("display_name") or place_name)[:120],
                )
                return self._json(start_response, "202 Accepted", {**scan, "place": place})
            # ---- Tactical OSINT fusion endpoints (keyless public feeds, proxied
            # server-side for CORS + shared caching; see osint_feeds.py) ----
            if method == "GET" and path == "/api/osint/flights":
                from .external_api import ExternalServiceError
                from .osint_feeds import fetch_osint_flights

                bbox = [
                    query.get("south", [None])[0],
                    query.get("west", [None])[0],
                    query.get("north", [None])[0],
                    query.get("east", [None])[0],
                ]
                limit = parse_bounded_int(
                    query.get("limit", ["1500"])[0], name="limit", default=1500, lower=1, upper=5000
                )
                try:
                    return self._json(start_response, "200 OK", fetch_osint_flights(bbox, limit=limit))
                except ExternalServiceError as exc:
                    return self._json(start_response, "503 Service Unavailable", {"error": str(exc), "items": []})
            if method == "GET" and path == "/api/osint/wildfires":
                from .external_api import ExternalServiceError
                from .osint_feeds import fetch_osint_wildfires

                try:
                    return self._json(start_response, "200 OK", fetch_osint_wildfires())
                except ExternalServiceError as exc:
                    return self._json(start_response, "503 Service Unavailable", {"error": str(exc), "features": []})
            if method == "GET" and path == "/api/osint/incidents":
                from .external_api import ExternalServiceError
                from .osint_feeds import fetch_osint_incidents

                try:
                    return self._json(start_response, "200 OK", fetch_osint_incidents())
                except ExternalServiceError as exc:
                    return self._json(start_response, "503 Service Unavailable", {"error": str(exc), "features": []})
            if method == "GET" and path == "/api/osint/cameras":
                from .external_api import ExternalServiceError
                from .osint_feeds import fetch_osint_cameras

                bbox = [
                    query.get("south", [None])[0],
                    query.get("west", [None])[0],
                    query.get("north", [None])[0],
                    query.get("east", [None])[0],
                ]
                limit = parse_bounded_int(
                    query.get("limit", ["400"])[0], name="limit", default=400, lower=1, upper=1000
                )
                try:
                    return self._json(start_response, "200 OK", fetch_osint_cameras(bbox, limit=limit))
                except ExternalServiceError as exc:
                    return self._json(start_response, "503 Service Unavailable", {"error": str(exc), "items": []})
            if method == "GET" and path == "/api/osint/earthquakes":
                from .external_api import ExternalServiceError
                from .osint_feeds import fetch_osint_earthquakes

                region = query.get("region", ["au"])[0]
                try:
                    return self._json(
                        start_response, "200 OK",
                        fetch_osint_earthquakes(region="global" if region == "global" else "au"),
                    )
                except ExternalServiceError as exc:
                    return self._json(start_response, "503 Service Unavailable", {"error": str(exc), "items": []})
            if method == "GET" and path == "/api/geocode":
                from .external_api import ExternalServiceError, geocode_places

                search_query = query.get("q", [""])[0]
                # Bias the area-search autocomplete to Australia (this is an AU app);
                # pass ?cc= (empty) for a worldwide search.
                country_codes = query.get("cc", ["au"])[0]
                try:
                    return self._json(
                        start_response,
                        "200 OK",
                        {
                            "items": geocode_places(search_query, country_codes=country_codes),
                            "source": "nominatim",
                        },
                    )
                except ExternalServiceError as exc:
                    return self._json(start_response, "503 Service Unavailable", {"error": str(exc)})
            if method == "GET" and path in {"/health", "/api/health"}:
                return self._json(
                    start_response,
                    "200 OK",
                    {
                        "status": "ok",
                        "property_count": len(self._snapshots),
                        "lgas": self.list_lgas(),
                        "modules": MODULE_TITLES,
                        "qgis": qgis_status(probe_version=False),
                        "cache_backend": _cache_backend_name(),
                        "address_lookup": {
                            "google_configured": google_geocoding_configured(),
                            "unit_policy": "explicit_source_only",
                        },
                        "data_note": DATA_NOTE,
                    },
                )
            if method == "GET" and path in {"/rankings", "/api/rankings"}:
                return self._json(start_response, "200 OK", self.build_rankings_payload(query))
            if method == "GET" and path in {"/rankings.csv", "/api/rankings.csv"}:
                module_name = normalize_module(query.get("module", ["pool_3d"])[0])
                csv_payload = self.build_rankings_csv(query)
                filename = f"{module_name}_rankings.csv"
                return self._text(
                    start_response,
                    "200 OK",
                    csv_payload,
                    content_type="text/csv; charset=utf-8",
                    extra_headers=[("Content-Disposition", f'attachment; filename="{filename}"')],
                )
            if method == "POST" and path == "/api/export/qgis/area":
                analysis = self._read_json_body(environ)
                if not isinstance(analysis, dict) or analysis.get("polygon") is None:
                    raise ValueError("Provide an analyzed area (with a polygon) to export.")
                geometries = analysis.get("geometries") or {}
                element_total = sum(
                    len(geometries.get(key, []) or []) for key in ("buildings", "pools", "trees")
                )
                if element_total > 10_000:
                    raise ValueError("Too many features to export in one GeoPackage.")
                try:
                    out_dir = PACKAGE_ROOT.parent / "artifacts" / "qgis"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / "area_analysis_export.gpkg"
                    from .map_export import export_area_geopackage

                    export_area_geopackage(analysis, out_path)
                    file_bytes = out_path.read_bytes()
                    headers = [
                        ("Content-Type", "application/geopackage+sqlite3"),
                        ("Content-Length", str(len(file_bytes))),
                        ("Content-Disposition", 'attachment; filename="area_analysis_export.gpkg"'),
                        ("Cache-Control", "no-cache"),
                    ]
                    start_response("200 OK", headers)
                    return [file_bytes]
                except ImportError as exc:
                    return self._json(
                        start_response,
                        "500 Internal Server Error",
                        {"error": f"QGIS export requires geopandas and shapely: {exc}"},
                    )
                except ValueError:
                    raise
                except Exception as exc:
                    return self._json(
                        start_response,
                        "500 Internal Server Error",
                        {"error": f"Area QGIS export failed: {exc}"},
                    )
            if method == "GET" and path in {"/api/export/qgis", "/export/qgis"}:
                module_name = normalize_module(query.get("module", ["pool_3d"])[0])
                lga = query.get("lga", [None])[0]
                top_k = parse_bounded_int(
                    query.get("top_k", ["15"])[0], name="top_k", default=15, lower=1, upper=100
                )
                
                try:
                    # Write to a temporary file in artifacts/qgis
                    out_dir = PACKAGE_ROOT.parent / "artifacts" / "qgis"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{module_name}_qgis_export.gpkg"
                    
                    from .map_export import export_ranked_map
                    export_ranked_map(
                        output_path=out_path,
                        data_path=self.data_path,
                        module_name=module_name,
                        lga=lga,
                        top_k=top_k
                    )
                    
                    file_bytes = out_path.read_bytes()
                    filename = f"{module_name}_qgis_export.gpkg"
                    
                    headers = [
                        ("Content-Type", "application/geopackage+sqlite3"),
                        ("Content-Length", str(len(file_bytes))),
                        ("Content-Disposition", f'attachment; filename="{filename}"'),
                        ("Cache-Control", "no-cache")
                    ]
                    start_response("200 OK", headers)
                    return [file_bytes]
                except ImportError as exc:
                    return self._json(
                        start_response,
                        "500 Internal Server Error",
                        {"error": f"QGIS export requires geopandas and fiona: {exc}"},
                    )
                except Exception as exc:
                    return self._json(
                        start_response,
                        "500 Internal Server Error",
                        {"error": f"QGIS export failed: {exc}"},
                    )
            if method == "GET" and (path.startswith("/property/") or path.startswith("/api/property/")):
                property_id = path.rsplit("/", 1)[-1]
                snapshot = self.get_property(property_id)
                if snapshot is None:
                    return self._json(start_response, "404 Not Found", {"error": "Property not found."})
                return self._json(
                    start_response,
                    "200 OK",
                    {
                        "property": snapshot["property"],
                        "derived_features": snapshot["derived_features"],
                        "scores": snapshot["scores"],
                        "explanations": snapshot["explanations"],
                        "data_note": DATA_NOTE,
                    },
                )
            if method == "POST" and path in {"/refresh-area", "/api/refresh-area"}:
                payload = self._read_json_body(environ)
                requested_lga = payload.get("lga")
                self.refresh()
                recomputed = self._snapshots
                if requested_lga:
                    recomputed = [item for item in recomputed if item["lga"].lower() == str(requested_lga).lower()]
                if requested_lga and not recomputed:
                    return self._json(
                        start_response,
                        "404 Not Found",
                        {"error": f"No properties found for LGA '{requested_lga}'."},
                    )
                return self._json(
                    start_response,
                    "200 OK",
                    {
                        "status": "accepted",
                        "lga": requested_lga,
                        "properties_recomputed": len(recomputed),
                        "qgis_hook": qgis_status(probe_version=False),
                        "data_note": DATA_NOTE,
                    },
                )
            return self._json(start_response, "404 Not Found", {"error": "Route not found."})
        except ValueError as exc:
            return self._json(start_response, "400 Bad Request", {"error": str(exc)})
        except MemoryError:
            gc.collect()
            return self._json(
                start_response,
                "503 Service Unavailable",
                {
                    "error": (
                        "The request exceeded available detector memory. The server is still "
                        "running; select a smaller analysis area and retry."
                    ),
                    "code": "resource_exhausted",
                },
            )
        except Exception as exc:  # pragma: no cover - surfaced in tests if it breaks
            return self._json(start_response, "500 Internal Server Error", {"error": str(exc)})


def create_app(data_path: str | Path | None = None) -> BackyardDemoApp:
    return BackyardDemoApp(data_path=data_path)


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, data_path: str | Path | None = None) -> None:
    app = create_app(data_path=data_path)
    with make_server(host, port, app, server_class=ThreadingWSGIServer) as httpd:
        if sys.stdout is not None:
            try:
                print(f"Serving AUS Backyard Intelligence at http://{host}:{port}", flush=True)
            except (OSError, ValueError):
                pass
        httpd.serve_forever()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the AUS Backyard Intelligence demo server.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host for the local demo server.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port for the local demo server.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to a JSON file containing property fixtures.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print a quick snapshot summary instead of starting the server.",
    )
    args = parser.parse_args(argv)
    app = create_app(data_path=args.data)
    if args.check:
        preview = {
            "property_count": len(app.snapshots),
            "lgas": app.list_lgas(),
            "qgis": qgis_status(),
            "top_pool_3d": app.rankings("pool_3d", top_k=3),
        }
        print(json.dumps(preview, indent=2))
        return
    serve(host=args.host, port=args.port, data_path=args.data)


if __name__ == "__main__":
    main()
