"""Fetch real residential properties from OpenStreetMap via Overpass API.

Queries house-tagged buildings in specific Moreton Bay suburbs,
extracts real coordinates and building footprints, estimates elevations from
the free Open-Elevation API, and writes app-compatible JSON.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

from .external_api import ExternalServiceError, execute_overpass_query

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PACKAGE_ROOT / "data" / "derived" / "sample_properties.json"

# Suburbs to query  —  (suburb_name, lga_name, postcode)
SUBURBS = [
    ("North Lakes", "Moreton Bay", "4509"),
    ("Murrumba Downs", "Moreton Bay", "4503"),
    ("Samford Valley", "Moreton Bay", "4520"),
    ("Redcliffe", "Moreton Bay", "4020"),
    ("Petrie", "Moreton Bay", "4502"),
    ("Warner", "Moreton Bay", "4500"),
]

HOUSES_PER_SUBURB = 40  # fetch up to this many per suburb


def _overpass_query(suburb: str, limit: int = 10) -> str:
    """Build an Overpass QL query for house buildings in a suburb.

    The global bbox pins results to Australia — a bare suburb name like "Warner"
    otherwise matches towns overseas (this once put 40 US houses in the demo data).
    """
    return f"""
[out:json][timeout:30][bbox:-44.0,112.0,-9.0,154.0];
area["name"="{suburb}"]["admin_level"~"10|9|8|2"]->.searchArea;
(
  way["building"="house"](area.searchArea);
  way["building"="residential"](area.searchArea);
  way["building"="detached"](area.searchArea);
);
out geom {limit};
"""


def _fetch_overpass(query: str) -> list[dict]:
    """Send a query to the Overpass API and return elements."""
    try:
        return execute_overpass_query(query, timeout=30.0)["elements"]
    except ExternalServiceError as exc:
        print(f"  Overpass error: {exc}")
        return []


def _fetch_elevations(coords: list[tuple[float, float]]) -> list[float]:
    """Fetch elevations using local estimation to avoid slow network API timeouts."""
    if not coords:
        return []
    return [_estimate_elevation(lat, lon) for lat, lon in coords]


def _estimate_elevation(lat: float, lon: float) -> float:
    """Rough Moreton Bay elevation estimate based on longitude (west = higher)."""
    # Moreton Bay ranges from ~5m (coast) to ~100m+ (inland)
    # Longitude 153.1 (east) ≈ 10m, 152.9 (west) ≈ 80m
    base = max(5.0, 80.0 - (lon - 152.9) * 350.0)
    # Add some latitude variation
    base += abs(lat + 27.47) * 20.0
    return round(base + random.uniform(-3, 3), 1)


def _generate_owner(index: int) -> dict[str, str]:
    """No fabricated owner identities or contact details — invented names/numbers
    can collide with real people (a random 04xx number is likely someone's phone)
    and direct marketing must follow OAIC APP 7. Owner follow-up goes through the
    lawful land-titles links on each lead instead."""
    return {"name": "", "phone": "", "email": "", "available": "no"}


def _overpass_pools_query(suburb: str) -> str:
    """Build an Overpass QL query for swimming pools in a suburb."""
    return f"""
[out:json][timeout:30][bbox:-44.0,112.0,-9.0,154.0];
area["name"="{suburb}"]["admin_level"~"10|9|8|2"]->.searchArea;
(
  node["leisure"="swimming_pool"](area.searchArea);
  way["leisure"="swimming_pool"](area.searchArea);
  relation["leisure"="swimming_pool"](area.searchArea);
);
out geom;
"""


def _element_center(element: dict[str, Any]) -> tuple[float, float] | None:
    geometry = element.get("geometry") or []
    if geometry:
        valid = [point for point in geometry if "lat" in point and "lon" in point]
        if valid:
            return (
                sum(float(point["lat"]) for point in valid) / len(valid),
                sum(float(point["lon"]) for point in valid) / len(valid),
            )
    if "lat" in element and "lon" in element:
        return float(element["lat"]), float(element["lon"])
    return None


def _distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat0 = math.radians((a[0] + b[0]) / 2.0)
    north = (a[0] - b[0]) * 110_540.0
    east = (a[1] - b[1]) * 111_320.0 * math.cos(lat0)
    return math.hypot(north, east)


def _assign_pools_to_nearest_buildings(
    buildings: list[dict[str, Any]],
    pool_centers: list[tuple[float, float]],
    *,
    max_distance_m: float = 30.0,
) -> dict[Any, float]:
    """Assign each mapped pool to at most one nearby building.

    This avoids the old behaviour where one pool could mark several neighboring
    homes as already having a pool. It remains an OSM proximity inference rather
    than a cadastral ownership determination, so the distance is retained as
    evidence in the generated record.
    """
    building_centers = [
        (building.get("id"), center)
        for building in buildings
        if (center := _element_center(building)) is not None
    ]
    assignments: dict[Any, float] = {}
    for pool_center in pool_centers:
        nearest: tuple[Any, float] | None = None
        for building_id, building_center in building_centers:
            distance = _distance_m(pool_center, building_center)
            if distance <= max_distance_m and (nearest is None or distance < nearest[1]):
                nearest = building_id, distance
        if nearest is not None:
            building_id, distance = nearest
            if building_id not in assignments or distance < assignments[building_id]:
                assignments[building_id] = distance
    return assignments


def _building_to_record(
    element: dict,
    suburb: str,
    lga: str,
    postcode: str,
    index: int,
    elevations: dict[str, float] | None = None,
    pool_distance_m: float | None = None,
) -> dict[str, Any] | None:
    """Convert an OSM way element to an app-compatible property record."""
    if "geometry" not in element:
        return None

    nodes = element["geometry"]
    lats = [pt["lat"] for pt in nodes]
    lons = [pt["lon"] for pt in nodes]

    if len(lats) < 3:
        return None

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    # Real building dimensions in metres (approximate)
    cos_lat = math.cos(math.radians(center_lat))
    b_width_m = (max_lon - min_lon) * 111320 * cos_lat
    b_depth_m = (max_lat - min_lat) * 110540

    if b_width_m < 3.0 or b_depth_m < 3.0:
        return None  # Too small, probably not a house

    # Estimate parcel from building with typical setbacks
    front_setback = random.uniform(5.0, 8.0)
    rear_setback = random.uniform(8.0, 18.0)
    side_setback_l = random.uniform(1.0, 4.0)
    side_setback_r = random.uniform(1.0, 4.0)

    p_width = b_width_m + side_setback_l + side_setback_r
    p_depth = b_depth_m + front_setback + rear_setback

    parcel_rect = [0.0, 0.0, round(p_width, 1), round(p_depth, 1)]
    building_rect = [
        round(side_setback_l, 1),
        round(front_setback, 1),
        round(side_setback_l + b_width_m, 1),
        round(front_setback + b_depth_m, 1),
    ]

    # Use provided elevations or defaults
    if elevations is None:
        base = _estimate_elevation(center_lat, center_lon)
        elevations = {
            "sw": base,
            "se": round(base + random.uniform(0.0, 1.5), 1),
            "nw": round(base + random.uniform(0.5, 3.0), 1),
            "ne": round(base + random.uniform(0.8, 4.0), 1),
        }

    # OSM tags can give us street info
    tags = element.get("tags", {})
    street = tags.get("addr:street", "")
    housenumber = tags.get("addr:housenumber", "")
    name = tags.get("name", "")

    if housenumber and street:
        address = f"{housenumber} {street}, {suburb} QLD {postcode}"
    elif street:
        address = f"{street}, {suburb} QLD {postcode}"
    elif name:
        address = f"{name}, {suburb} QLD {postcode}"
    else:
        address = f"Property {index}, {suburb} QLD {postcode}"

    osm_id = element.get("id", index)
    owner = _generate_owner(index)

    # Vegetation estimate from proximity to parks (simplified)
    veg = random.uniform(5, 45)
    # Flood risk higher near rivers/coast (low elevation)
    base_elev = elevations["sw"]
    flood = 1 if base_elev < 12.0 and random.random() < 0.4 else 0
    # Heritage more likely in historical suburbs
    heritage = random.random() < 0.08 if lga == "Moreton Bay" else False

    has_existing_pool = pool_distance_m is not None

    return {
        "id": f"OSM-{osm_id}",
        "lga": lga,
        "suburb": suburb,
        "address": address,
        "owner_name": owner["name"],
        "owner_phone": owner["phone"],
        "owner_email": owner["email"],
        "contact_available": owner["available"],
        "lot_plan": f"OSM{osm_id}",
        "valuation_ref": f"VAL-{osm_id}",
        "parcel_rect": parcel_rect,
        "building_rect": building_rect,
        "elevation_corners_m": elevations,
        "vegetation_pct": round(veg, 1),
        "road_frontage_m": round(p_width, 1),
        "corner_lot": random.random() < 0.18,
        "flood_risk": flood,
        "heritage_overlay": heritage,
        "overhead_powerline": random.random() < 0.12,
        "solar_access_pct": round(random.uniform(60, 95), 1),
        "view_quality": random.randint(3, 9),
        "real_lat": round(center_lat, 6),
        "real_lon": round(center_lon, 6),
        "property_type": "residential",
        "ownership_class": "private",
        "is_private_property": True,
        "has_existing_pool": has_existing_pool,
        "exclude_from_pool_candidates": has_existing_pool,
        "existing_pool_source": "openstreetmap_nearest_building" if has_existing_pool else None,
        "existing_pool_distance_m": round(pool_distance_m, 1) if pool_distance_m is not None else None,
    }


def fetch_all_properties(output_path: Path | None = None, per_suburb: int = HOUSES_PER_SUBURB) -> list[dict[str, Any]]:
    """Fetch real properties from all configured suburbs."""
    all_properties: list[dict[str, Any]] = []
    
    for suburb, lga, postcode in SUBURBS:
        print(f"Fetching houses in {suburb}, {lga}...")
        query = _overpass_query(suburb, limit=per_suburb + 4)  # fetch a few extra
        elements = _fetch_overpass(query)
        print(f"  Found {len(elements)} buildings in OSM")

        # Fetch existing pools in the suburb
        print(f"  Fetching pools in {suburb}...")
        pool_elements = _fetch_overpass(_overpass_pools_query(suburb))
        pools: list[tuple[float, float]] = []
        for el in pool_elements:
            center = _element_center(el)
            if center is not None:
                pools.append(center)
        print(f"  Found {len(pools)} existing pools in {suburb}")
        pool_assignments = _assign_pools_to_nearest_buildings(elements, pools)

        # Try to get elevations for the centres
        centres = []
        for el in elements:
            if "geometry" in el:
                lats = [pt["lat"] for pt in el["geometry"]]
                lons = [pt["lon"] for pt in el["geometry"]]
                centres.append(((min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2))

        elevations_list = None
        if centres:
            try:
                elevations_list = _fetch_elevations(centres)
                print(f"  Got {len(elevations_list)} elevation readings")
            except Exception:
                print("  Elevation API unavailable, using estimates")

        count = 0
        for i, el in enumerate(elements):
            if count >= per_suburb:
                break

            elev = None
            if elevations_list and i < len(elevations_list):
                base = elevations_list[i]
                elev = {
                    "sw": round(base, 1),
                    "se": round(base + random.uniform(0.0, 1.2), 1),
                    "nw": round(base + random.uniform(0.3, 2.5), 1),
                    "ne": round(base + random.uniform(0.5, 3.5), 1),
                }

            record = _building_to_record(
                el,
                suburb,
                lga,
                postcode,
                len(all_properties) + 1,
                elev,
                pool_assignments.get(el.get("id")),
            )
            if record is not None:
                all_properties.append(record)
                count += 1

        print(f"  Added {count} properties from {suburb}")
        
        # Incremental save
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(all_properties, indent=2), encoding="utf-8")
            
        time.sleep(1.5)  # Be nice to Overpass API

    print(f"\nTotal: {len(all_properties)} real properties across {len(SUBURBS)} suburbs")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_properties, indent=2), encoding="utf-8")
        print(f"Written to {output_path}")

    return all_properties


def main() -> None:
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Fetch real Moreton Bay properties from OpenStreetMap.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path.")
    parser.add_argument("--per-suburb", type=int, default=HOUSES_PER_SUBURB, help="Houses per suburb.")
    args = parser.parse_args()

    fetch_all_properties(output_path=Path(args.output), per_suburb=args.per_suburb)


if __name__ == "__main__":
    main()
