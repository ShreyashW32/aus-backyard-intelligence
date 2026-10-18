from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

from .app import BackyardDemoApp, rear_backyard_rect
from .external_api import ExternalServiceError, request_json_url
from .qgis import open_in_qgis

DEFAULT_EXPORT_CRS = "EPSG:7856"

SAMPLE_SUBURB_ANCHORS = {
    ("Moreton Bay", "North Lakes"): (153.0250, -27.2480),
    ("Moreton Bay", "Murrumba Downs"): (153.0140, -27.2730),
    ("Moreton Bay", "Samford Valley"): (152.8900, -27.3720),
    ("Moreton Bay", "Redcliffe"): (153.1060, -27.2290),
    ("Moreton Bay", "Petrie"): (152.9830, -27.2690),
}



def _snapshot_for_row(app: BackyardDemoApp, row: dict[str, Any]) -> dict[str, Any]:
    snapshot = app.get_property(row["property_id"])
    if snapshot is None:
        raise KeyError(f"Property id {row['property_id']} not found in app data.")
    return snapshot


def _load_real_geometry(record: dict[str, Any], key: str):
    from shapely import wkt

    geometry_wkt = record.get(key)
    if not geometry_wkt:
        return None
    return wkt.loads(geometry_wkt)


def _sample_anchor(record: dict[str, Any]) -> tuple[float, float]:
    from pyproj import Transformer

    lon = record.get("anchor_lon", record.get("real_lon"))
    lat = record.get("anchor_lat", record.get("real_lat"))
    if lon is not None and lat is not None:
        transformer = Transformer.from_crs("EPSG:4326", DEFAULT_EXPORT_CRS, always_xy=True)
        x, y = transformer.transform(float(lon), float(lat))
        return x, y
    key = (record["lga"], record["suburb"])
    lon, lat = SAMPLE_SUBURB_ANCHORS.get(key, (153.0251, -27.4698))
    transformer = Transformer.from_crs("EPSG:4326", DEFAULT_EXPORT_CRS, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y


def _synthetic_geometry(record: dict[str, Any], key: str):
    from shapely.geometry import box

    parcel_rect = record["parcel_rect"]
    building_rect = record["building_rect"]
    backyard_rect = rear_backyard_rect(parcel_rect, building_rect)
    rect_by_geometry_key = {
        "parcel_wkt": parcel_rect,
        "building_wkt": building_rect,
        "backyard_wkt": backyard_rect,
    }
    rect = rect_by_geometry_key[key]
    if rect is None:
        return None
    minx, miny, maxx, maxy = rect
    width = maxx - minx
    depth = maxy - miny
    anchor_x, anchor_y = _sample_anchor(record)
    
    # Place exactly at the anchor without random offsets
    base_minx = anchor_x - ((record["parcel_rect"][2] - record["parcel_rect"][0]) / 2.0)
    base_miny = anchor_y - ((record["parcel_rect"][3] - record["parcel_rect"][1]) / 2.0)
    return box(base_minx + minx, base_miny + miny, base_minx + minx + width, base_miny + miny + depth)


def _geometry_for(record: dict[str, Any], key: str):
    return _load_real_geometry(record, key) or _synthetic_geometry(record, key)


def _geometry_crs(record: dict[str, Any]) -> str:
    return record.get("source_crs") or DEFAULT_EXPORT_CRS


def _feature_row(snapshot: dict[str, Any], module_name: str, rank: int) -> dict[str, Any]:
    record = snapshot["property"]
    features = snapshot["derived_features"]
    building_height_m = round(max(5.0, min(11.0, 5.5 + (features["building_area_sqm"] / 120.0))), 1)
    score_height_m = round(max(1.0, snapshot["scores"][module_name] / 12.0), 1)
    if record.get("parcel_wkt"):
        geometry_source = "real"
    elif record.get("anchor_lon") is not None or record.get("real_lon") is not None:
        geometry_source = "geocoded-anchor"
    else:
        geometry_source = "suburb-anchor"
    return {
        "property_id": snapshot["property_id"],
        "owner_name": record.get("owner_name", ""),
        "owner_phone": record.get("owner_phone", ""),
        "owner_email": record.get("owner_email", ""),
        "contact_available": record.get("contact_available", "no"),
        "has_existing_pool": bool(record.get("has_existing_pool")),
        "existing_pool_source": record.get("existing_pool_source"),
        "rank": rank,
        "module": module_name,
        "module_score": snapshot["scores"][module_name],
        "pool_2d": snapshot["scores"]["pool_2d"],
        "pool_3d": snapshot["scores"]["pool_3d"],
        "fence": snapshot["scores"]["fence"],
        "deck": snapshot["scores"]["deck"],
        "address": snapshot["address"],
        "suburb": snapshot["suburb"],
        "lga": snapshot["lga"],
        "land_area_sqm": features["land_area_sqm"],
        "building_area_sqm": features["building_area_sqm"],
        "backyard_area_sqm": features["backyard_area_sqm"],
        "backyard_depth_m": features["backyard_depth_m"],
        "max_side_clearance_m": features["max_side_clearance_m"],
        "slope_pct": features["slope_pct"],
        "relief_m": features["relief_m"],
        "solar_access_pct": features["solar_access_pct"],
        "vegetation_pct": features["vegetation_pct"],
        "road_frontage_m": features["road_frontage_m"],
        "base_elev_m": features["elevation_min_m"],
        "building_height_m": building_height_m,
        "score_height_m": score_height_m,
        "ownership_class": record["ownership_class"],
        "property_classification": record["property_classification"],
        "residential_type": record["residential_type"],
        "private_property": bool(record["is_private_property"]),
        "candidate_label": record["candidate_label"],
        "eligibility_confidence": record["eligibility_confidence"],
        "eligibility_reason": record["eligibility_reason"],
        "geometry_source": geometry_source,
        "explanation": snapshot["explanations"][module_name],
    }


def ranked_layers(
    *,
    data_path: str | Path | None = None,
    module_name: str = "pool_3d",
    lga: str | None = None,
    top_k: int = 5,
    property_ids: list[str] | None = None,
) -> dict[str, Any]:
    import geopandas as gpd

    app = BackyardDemoApp(data_path=data_path)
    if property_ids:
        snapshots = []
        for property_id in property_ids:
            snapshot = app.get_property(property_id)
            if snapshot is not None:
                snapshots.append(snapshot)
        ordered = sorted(snapshots, key=lambda item: (-item["scores"][module_name], item["property_id"]))
        selected = list(enumerate(ordered, start=1))
    else:
        ranking_rows = app.rankings(module_name, lga=lga, top_k=top_k)
        selected = [(row["rank"], _snapshot_for_row(app, row)) for row in ranking_rows]

    parcel_rows = []
    building_rows = []
    backyard_rows = []
    parcel_geometries = []
    building_geometries = []
    backyard_geometries = []
    crs = None

    for rank, snapshot in selected:
        record = snapshot["property"]
        crs = crs or _geometry_crs(record)
        feature_row = _feature_row(snapshot, module_name, rank)
        parcel_geometry = _geometry_for(record, "parcel_wkt")
        building_geometry = _geometry_for(record, "building_wkt")
        backyard_geometry = _geometry_for(record, "backyard_wkt")
        parcel_rows.append(feature_row)
        parcel_geometries.append(parcel_geometry)
        building_rows.append(feature_row)
        building_geometries.append(building_geometry)
        if backyard_geometry is not None:
            backyard_rows.append(feature_row)
            backyard_geometries.append(backyard_geometry)

    parcel_gdf = gpd.GeoDataFrame(parcel_rows, geometry=parcel_geometries, crs=crs or DEFAULT_EXPORT_CRS)
    building_gdf = gpd.GeoDataFrame(building_rows, geometry=building_geometries, crs=crs or DEFAULT_EXPORT_CRS)
    backyard_gdf = gpd.GeoDataFrame(backyard_rows, geometry=backyard_geometries, crs=crs or DEFAULT_EXPORT_CRS)
    return {
        "ranked_parcels": parcel_gdf,
        "ranked_buildings": building_gdf,
        "ranked_backyards": backyard_gdf,
    }


def export_ranked_map(
    output_path: str | Path,
    *,
    data_path: str | Path | None = None,
    module_name: str = "pool_3d",
    lga: str | None = None,
    top_k: int = 5,
    property_ids: list[str] | None = None,
) -> dict[str, Any]:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    layers = ranked_layers(
        data_path=data_path,
        module_name=module_name,
        lga=lga,
        top_k=top_k,
        property_ids=property_ids,
    )
    for layer_name, frame in layers.items():
        frame.to_file(destination, layer=layer_name, driver="GPKG")
    return {
        "output_path": str(destination),
        "layers": {layer_name: len(frame) for layer_name, frame in layers.items()},
        "module": module_name,
        "lga": lga,
        "top_k": top_k,
        "property_ids": property_ids or [],
    }


def export_area_geopackage(analysis: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Write a drawn Area-Analysis result to a multi-layer GeoPackage.

    Unlike :func:`export_ranked_map` (which exports the demo's ranked properties), this
    exports the *user's selected area*: the polygon carrying the full suitability
    assessment as attributes, plus the buildings, pools, and trees detected inside it.
    Geometry is WGS84 (EPSG:4326) — the native lat/lng the viewer works in, and valid
    Australia-wide (the projected export CRS is zone-specific).
    """
    import geopandas as gpd
    from shapely.geometry import LineString, Point, Polygon

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    polygon = analysis.get("polygon")
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError("The analysis is missing a valid polygon to export.")
    ring = [(float(c[0]), float(c[1])) for c in polygon if isinstance(c, (list, tuple)) and len(c) >= 2]
    if len(ring) < 3:
        raise ValueError("The analysis polygon is malformed.")

    scores = analysis.get("scores") or {}
    features = analysis.get("features") or {}
    profile = analysis.get("site_profile") or {}
    shape = analysis.get("shape") or {}

    # Layer 1: the selected area, with the whole assessment as queryable attributes.
    area_attributes = {
        "area_sqm": analysis.get("area_sqm"),
        "perimeter_m": analysis.get("perimeter_m"),
        "site_kind": profile.get("kind"),
        "site_label": profile.get("label"),
        "is_residential": profile.get("is_residential"),
        "has_existing_pool": analysis.get("has_existing_pool"),
        "selected_area_class": analysis.get("selected_area_class"),
        "score_pool_3d": scores.get("pool_3d"),
        "score_pool_2d": scores.get("pool_2d"),
        "score_fence": scores.get("fence"),
        "score_deck": scores.get("deck"),
        "score_landscaping": scores.get("landscaping"),
        "backyard_sqm": features.get("backyard_area_sqm"),
        "slope_pct": features.get("slope_pct"),
        "solar_access_pct": features.get("solar_access_pct"),
        "vegetation_pct": features.get("vegetation_pct"),
        "flood_risk_pct": features.get("flood_risk_pct"),
        "side_clearance_m": features.get("max_side_clearance_m"),
        "compactness": shape.get("compactness"),
    }
    layers: dict[str, Any] = {
        "analysis_area": gpd.GeoDataFrame(
            [area_attributes], geometry=[Polygon(ring)], crs="EPSG:4326"
        )
    }

    def _osm_element_to_shapely(element: dict[str, Any]):
        geometry = element.get("geometry")
        if element.get("type") == "node" or not isinstance(geometry, list):
            lat, lon = element.get("lat"), element.get("lon")
            return Point(float(lon), float(lat)) if lat is not None and lon is not None else None
        points = [
            (float(p["lon"]), float(p["lat"]))
            for p in geometry
            if isinstance(p, dict) and "lat" in p and "lon" in p
        ]
        if len(points) >= 3:
            return Polygon(points)
        if len(points) == 2:
            return LineString(points)
        if len(points) == 1:
            return Point(points[0])
        return None

    geometries = analysis.get("geometries") or {}
    for layer_name in ("buildings", "pools", "trees"):
        rows: list[dict[str, Any]] = []
        geoms: list[Any] = []
        for element in geometries.get(layer_name, []) or []:
            if not isinstance(element, dict):
                continue
            shapely_geom = _osm_element_to_shapely(element)
            if shapely_geom is None:
                continue
            tags = element.get("tags") or {}
            rows.append(
                {
                    "osm_id": element.get("id"),
                    "kind": layer_name[:-1],
                    "source": tags.get("detector:source", ""),
                    "name": tags.get("name", ""),
                }
            )
            geoms.append(shapely_geom)
        if rows:
            layers[layer_name] = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")

    for layer_name, frame in layers.items():
        frame.to_file(destination, layer=layer_name, driver="GPKG")

    return {
        "output_path": str(destination),
        "layers": {name: len(frame) for name, frame in layers.items()},
        "area_sqm": analysis.get("area_sqm"),
        "site_kind": profile.get("kind"),
    }


def get_map_geojson(
    *,
    data_path: str | Path | None = None,
    module_name: str = "pool_3d",
    lga: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    from shapely.geometry import mapping as shapely_mapping

    layers = ranked_layers(data_path=data_path, module_name=module_name, lga=lga, top_k=top_k)
    parcels_gdf = layers["ranked_parcels"]
    buildings_gdf = layers["ranked_buildings"]
    backyards_gdf = layers["ranked_backyards"]

    if len(parcels_gdf) == 0:
        return {
            "parcels": {"type": "FeatureCollection", "features": []},
            "buildings": {"type": "FeatureCollection", "features": []},
            "backyards": {"type": "FeatureCollection", "features": []},
        }

    wgs84_parcels = parcels_gdf.to_crs("EPSG:4326")
    wgs84_buildings = buildings_gdf.to_crs("EPSG:4326")
    wgs84_backyards = backyards_gdf.to_crs("EPSG:4326")

    def _build_features(gdf, layer_type):
        features = []
        for _idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            centroid = geom.centroid
            props = {
                "property_id": row["property_id"],
                "address": row["address"],
                "suburb": row["suburb"],
                "lga": row["lga"],
                "score": row["module_score"],
                "rank": row["rank"],
                "layer": layer_type,
                "centroid": [centroid.x, centroid.y],
                "backyard_area_sqm": row.get("backyard_area_sqm", 0),
                "owner_name": row.get("owner_name", ""),
                "contact_available": row.get("contact_available", "no"),
            }
            features.append({
                "type": "Feature",
                "geometry": shapely_mapping(geom),
                "properties": props,
            })
        return {"type": "FeatureCollection", "features": features}

    return {
        "parcels": _build_features(wgs84_parcels, "parcel"),
        "buildings": _build_features(wgs84_buildings, "building"),
        "backyards": _build_features(wgs84_backyards, "backyard"),
    }


def get_property_geojson(property_id: str, data_path: str | Path | None = None) -> dict[str, Any] | None:
    layers = ranked_layers(data_path=data_path, property_ids=[property_id])
    
    parcels_gdf = layers["ranked_parcels"]
    buildings_gdf = layers["ranked_buildings"]
    backyards_gdf = layers["ranked_backyards"]
    
    if len(parcels_gdf) == 0:
        return None
        
    wgs84_parcels = parcels_gdf.to_crs("EPSG:4326")
    wgs84_buildings = buildings_gdf.to_crs("EPSG:4326")
    wgs84_backyards = backyards_gdf.to_crs("EPSG:4326")
    
    parcel_row = wgs84_parcels.iloc[0]
    building_row = wgs84_buildings.iloc[0] if len(wgs84_buildings) > 0 else None
    backyard_row = wgs84_backyards.iloc[0] if len(wgs84_backyards) > 0 else None
    
    from shapely.geometry import mapping
    def geometry_to_dict(geom):
        if geom is None or geom.is_empty:
            return None
        return mapping(geom)

    geometry_source = parcel_row.get("geometry_source", "unknown")
    geometry_notes = {
        "real": "Source parcel/building geometry; roof form and facade remain procedural.",
        "geocoded-anchor": "Illustrative parcel/building geometry anchored to the selected property location.",
        "suburb-anchor": "Illustrative parcel/building geometry positioned at a suburb-level anchor.",
    }
        
    return {
        "property_id": parcel_row["property_id"],
        "address": parcel_row.get("address", ""),
        "suburb": parcel_row.get("suburb", ""),
        "lga": parcel_row.get("lga", ""),
        "score": parcel_row["module_score"],
        "has_existing_pool": bool(parcel_row.get("has_existing_pool", False)),
        "existing_pool_source": parcel_row.get("existing_pool_source"),
        "scores": {
            "pool_2d": parcel_row.get("pool_2d", 0),
            "pool_3d": parcel_row.get("pool_3d", 0),
            "fence": parcel_row.get("fence", 0),
            "deck": parcel_row.get("deck", 0),
        },
        "explanation": parcel_row.get("explanation", ""),
        "owner": {
            "name": parcel_row.get("owner_name"),
            "phone": parcel_row.get("owner_phone"),
            "email": parcel_row.get("owner_email"),
            "available": parcel_row.get("contact_available"),
        },
        "elevations": {
            "base_elev_m": parcel_row["base_elev_m"],
            "building_height_m": parcel_row["building_height_m"],
            "building_height_source": "derived estimate",
        },
        "features": {
            "land_area_sqm": parcel_row.get("land_area_sqm", 0),
            "building_area_sqm": parcel_row.get("building_area_sqm", 0),
            "slope_pct": parcel_row["slope_pct"],
            "relief_m": parcel_row["relief_m"],
            "backyard_area_sqm": parcel_row.get("backyard_area_sqm", 0),
            "backyard_depth_m": parcel_row.get("backyard_depth_m", 0),
            "max_side_clearance_m": parcel_row.get("max_side_clearance_m", 0),
            "solar_access_pct": parcel_row.get("solar_access_pct", 0),
            "vegetation_pct": parcel_row.get("vegetation_pct", 0),
            "has_existing_pool": bool(parcel_row.get("has_existing_pool", False)),
        },
        "geometry_provenance": {
            "geometry_source": geometry_source,
            "scope": "selected-property-only",
            "height_source": "derived estimate",
            "roof_source": "procedural massing",
            "note": geometry_notes.get(
                geometry_source,
                "Property geometry source is not recorded; treat the reconstruction as indicative.",
            ),
        },
        "geometries": {
            "parcel_geom": geometry_to_dict(parcel_row.geometry),
            "building_geom": geometry_to_dict(building_row.geometry) if building_row is not None else None,
            "backyard_geom": geometry_to_dict(backyard_row.geometry) if backyard_row is not None else None,
        }
    }


@lru_cache(maxsize=512)
def get_elevation_corners_with_source(
    lat: float, lon: float, w_m: float, h_m: float
) -> tuple[dict[str, float], str]:
    # Earth radius in meters
    R = 6378137
    # Approximate offset coordinates for sw, se, nw, ne
    dlat = (h_m / 2.0) / R * (180.0 / math.pi)
    dlon = (w_m / 2.0) / (R * math.cos(math.radians(lat))) * (180.0 / math.pi)
    
    corners = {
        "sw": (lat - dlat, lon - dlon),
        "se": (lat - dlat, lon + dlon),
        "nw": (lat + dlat, lon - dlon),
        "ne": (lat + dlat, lon + dlon)
    }
    
    # Query Open-Meteo Elevation API in a single batch lookup
    lats = [c[0] for c in corners.values()]
    lons = [c[1] for c in corners.values()]
    lat_str = ",".join(f"{latitude:.6f}" for latitude in lats)
    lon_str = ",".join(f"{longitude:.6f}" for longitude in lons)
    
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_str}&longitude={lon_str}"
        res_data = request_json_url(url, timeout=4.0)
        elevations = res_data.get("elevation", []) if isinstance(res_data, dict) else []
        if not isinstance(elevations, list) or len(elevations) < 4:
            raise ExternalServiceError("The elevation service returned incomplete data.")
        values = [float(value) for value in elevations[:4]]
        if not all(math.isfinite(value) for value in values):
            raise ExternalServiceError("The elevation service returned invalid data.")
        keys = ["sw", "se", "nw", "ne"]
        return {keys[i]: round(values[i], 2) for i in range(4)}, "open_meteo"
    except (ExternalServiceError, TypeError, ValueError):
        fallback = 20.0
        return {"sw": fallback, "se": fallback, "nw": fallback, "ne": fallback}, "fallback_flat_estimate"


def get_elevation_corners(lat: float, lon: float, w_m: float, h_m: float) -> dict[str, float]:
    return get_elevation_corners_with_source(lat, lon, w_m, h_m)[0]


@lru_cache(maxsize=512)
def get_solar_access_with_source(lat: float, lon: float) -> tuple[float, str]:
    try:
        # PVGIS 5.2 API
        url = f"https://re.jrc.ec.europa.eu/api/v5_2/PVcalc?lat={lat}&lon={lon}&peakpower=1&loss=14&outputformat=json"
        data = request_json_url(url, timeout=4.0)
        ey = data.get("outputs", {}).get("totals", {}).get("fixed", {}).get("E_y")
        if ey is None or not math.isfinite(float(ey)):
            raise ExternalServiceError("The solar service returned incomplete data.")
        pct = 45.0 + (max(0.0, min(1000.0, float(ey) - 800.0)) / 1000.0) * 50.0
        return round(pct, 1), "pvgis"
    except (ExternalServiceError, TypeError, ValueError):
        abs_lat = abs(lat)
        pct = 90.0 - (abs_lat / 90.0) * 35.0
        return round(pct, 1), "latitude_fallback_estimate"


def get_solar_access(lat: float, lon: float) -> float:
    return get_solar_access_with_source(lat, lon)[0]


def is_point_in_polygon_py(x: float, y: float, poly_pts: list[list[float] | tuple[float, float]]) -> bool:
    """Ray casting containment check (PNPOLY algorithm)."""
    inside = False
    n = len(poly_pts)
    if n < 3:
        return False
    # Copy to avoid modifying the input, and convert to tuples
    pts = [(float(pt[0]), float(pt[1])) for pt in poly_pts]
    # Handle closed polygons where the first and last point are identical
    if len(pts) > 3 and pts[0] == pts[-1]:
        pts = pts[:-1]
        n = len(pts)
        
    p1x, p1y = pts[0]
    for i in range(n + 1):
        p2x, p2y = pts[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _polygon_area_degrees(coords: list[list[float]] | list[tuple[float, float]]) -> float:
    area = 0.0
    for index, point in enumerate(coords):
        next_point = coords[(index + 1) % len(coords)]
        area += float(point[0]) * float(next_point[1])
        area -= float(next_point[0]) * float(point[1])
    return abs(area) / 2.0


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> int:
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(value) < 1e-12:
        return 0
    return 1 if value > 0 else 2


def _point_on_segment(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
    return (
        _orientation(a, b, c) == 0
        and min(a[0], c[0]) - 1e-12 <= b[0] <= max(a[0], c[0]) + 1e-12
        and min(a[1], c[1]) - 1e-12 <= b[1] <= max(a[1], c[1]) + 1e-12
    )


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and _point_on_segment(a, c, b))
        or (o2 == 0 and _point_on_segment(a, d, b))
        or (o3 == 0 and _point_on_segment(c, a, d))
        or (o4 == 0 and _point_on_segment(c, b, d))
    )


def _closed_edges(points: list[tuple[float, float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [(points[index], points[(index + 1) % len(points)]) for index in range(len(points))]


def _open_edges(points: list[tuple[float, float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [(points[index], points[index + 1]) for index in range(len(points) - 1)]


def _polygon_has_self_intersection(coords: list[list[float]]) -> bool:
    points = [(float(lon), float(lat)) for lon, lat in coords]
    edges = _closed_edges(points)
    for left_index, (left_a, left_b) in enumerate(edges):
        for right_index in range(left_index + 1, len(edges)):
            if abs(left_index - right_index) == 1:
                continue
            if left_index == 0 and right_index == len(edges) - 1:
                continue
            right_a, right_b = edges[right_index]
            if _segments_intersect(left_a, left_b, right_a, right_b):
                return True
    return False


def _polygon_covers_point(polygon: list[list[float]] | list[tuple[float, float]], point: tuple[float, float]) -> bool:
    ring = [(float(lon), float(lat)) for lon, lat in polygon]
    if any(_point_on_segment(a, point, b) for a, b in _closed_edges(ring)):
        return True
    return is_point_in_polygon_py(point[0], point[1], ring)


def _geometry_intersects_polygon(points: list[tuple[float, float]], polygon: list[list[float]]) -> bool:
    if not points:
        return False
    polygon_points = [(float(lon), float(lat)) for lon, lat in polygon]
    if any(_polygon_covers_point(polygon_points, point) for point in points):
        return True
    if len(points) >= 3 and any(_polygon_covers_point(points, point) for point in polygon_points):
        return True
    feature_edges = _closed_edges(points) if len(points) >= 3 else _open_edges(points)
    polygon_edges = _closed_edges(polygon_points)
    return any(
        _segments_intersect(feature_a, feature_b, polygon_a, polygon_b)
        for feature_a, feature_b in feature_edges
        for polygon_a, polygon_b in polygon_edges
    )


def validate_analysis_polygon(raw_polygon: Any) -> list[list[float]]:
    """Validate and normalize a user-drawn WGS84 polygon."""
    if not isinstance(raw_polygon, list) or not 3 <= len(raw_polygon) <= 200:
        raise ValueError("polygon must contain between 3 and 200 coordinate pairs.")
    coords: list[list[float]] = []
    for coordinate in raw_polygon:
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
            raise ValueError("Each polygon coordinate must be [longitude, latitude].")
        try:
            lon, lat = float(coordinate[0]), float(coordinate[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("Polygon coordinates must be numbers.") from exc
        if not (math.isfinite(lon) and math.isfinite(lat)):
            raise ValueError("Polygon coordinates must be finite numbers.")
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("Polygon coordinates are outside valid longitude/latitude ranges.")
        coords.append([lon, lat])
    if len(coords) > 3 and coords[0] == coords[-1]:
        coords.pop()
    if _polygon_area_degrees(coords) == 0 or _polygon_has_self_intersection(coords):
        raise ValueError("polygon must be a valid, non-self-intersecting area.")
    min_lon = min(point[0] for point in coords)
    max_lon = max(point[0] for point in coords)
    min_lat = min(point[1] for point in coords)
    max_lat = max(point[1] for point in coords)
    if max_lon - min_lon > 0.25 or max_lat - min_lat > 0.25:
        raise ValueError("The selected area is too large; zoom in and draw a smaller polygon.")
    return coords


def _coerce_lon_lat(lon_value: Any, lat_value: Any) -> tuple[float, float] | None:
    try:
        lon, lat = float(lon_value), float(lat_value)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lon) and math.isfinite(lat)):
        return None
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        return None
    return lon, lat


def _feature_geometry_points(feature: dict[str, Any]) -> list[tuple[float, float]]:
    geometry = feature.get("geometry") or []
    if not isinstance(geometry, list):
        return []
    points: list[tuple[float, float]] = []
    for coordinate in geometry:
        if isinstance(coordinate, dict):
            point = _coerce_lon_lat(coordinate.get("lon"), coordinate.get("lat"))
        elif isinstance(coordinate, (list, tuple)) and len(coordinate) == 2:
            point = _coerce_lon_lat(coordinate[0], coordinate[1])
        else:
            return []
        if point is None:
            return []
        points.append(point)
    return points


def osm_feature_intersects_polygon(feature: dict[str, Any], polygon: list[list[float]] | Any) -> bool:
    """Return whether an OSM node/line/polygon overlaps the selected parcel."""
    if not isinstance(feature, dict):
        return False
    if hasattr(polygon, "exterior") and hasattr(polygon.exterior, "coords"):
        polygon = list(polygon.exterior.coords)
    if feature.get("type") == "node":
        point = _coerce_lon_lat(feature.get("lon"), feature.get("lat"))
        if point is None:
            return False
        return _polygon_covers_point(polygon, point)
    points = _feature_geometry_points(feature)
    return _geometry_intersects_polygon(points, polygon)


# Building tag vocabularies used to read the geography of a selected area.
_RESIDENTIAL_BUILDING_TAGS = {
    "house", "detached", "residential", "semidetached_house", "terrace",
    "bungalow", "villa", "duplex", "apartments", "dwelling", "cabin", "house;yes",
}
_NON_RESIDENTIAL_BUILDING_TAGS = {
    "commercial", "industrial", "retail", "warehouse", "civic", "public",
    "school", "hospital", "stadium", "sports_centre", "hotel", "supermarket",
    "government", "church", "roof", "hangar", "grandstand", "pavilion",
}
_PUBLIC_LEISURE_TAGS = {
    "park", "sports_centre", "stadium", "recreation_ground", "pitch",
    "water_park", "nature_reserve", "playground", "marina",
}


def _feature_area_sqm(feature: dict[str, Any]) -> float:
    """Footprint area of an OSM way/polygon feature in square metres (shoelace in a
    local equirectangular projection — accurate enough for footprint sizing)."""
    points = _feature_geometry_points(feature)  # [(lon, lat), ...]
    if len(points) < 3:
        return 0.0
    lat0 = sum(p[1] for p in points) / len(points)
    m_per_lat = 111_320.0
    m_per_lon = 111_320.0 * max(math.cos(math.radians(lat0)), 1e-9)
    xy = [(lon * m_per_lon, lat * m_per_lat) for lon, lat in points]
    area2 = 0.0
    for i in range(len(xy)):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % len(xy)]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) / 2.0


def classify_site_geography(
    *,
    area_sqm: float,
    buildings: list[dict[str, Any]],
    pools: list[dict[str, Any]],
    vegetation_pct: float,
    has_existing_pool: bool,
    selected_area_class: str,
) -> dict[str, Any]:
    """Read what a selected area actually *is* from its geography, so the suitability
    report can reason at the right scale instead of always assuming a suburban
    backyard. Returns a profile with a ``kind``, a human label, an ``is_residential``
    flag (drives whether residential setback/frontage framing applies), and a summary.

    Signals used: selection size, real building footprints (residential-scale vs
    large/commercial), pool size/quantity and public tags, and vegetation cover. The
    coverage here is computed from the *actual* footprints supplied — not the
    estimated fallback used elsewhere — so an unbuilt park is not mistaken for a lot.
    """
    n_buildings = len(buildings)
    residential_buildings = 0
    non_residential_buildings = 0
    largest_building = 0.0
    real_building_area = 0.0
    for b in buildings:
        tags = b.get("tags") if isinstance(b, dict) else {}
        tags = tags if isinstance(tags, dict) else {}
        value = str(tags.get("building", "")).strip().lower()
        footprint = _feature_area_sqm(b)
        real_building_area += footprint
        largest_building = max(largest_building, footprint)
        commercial_tag = any(tags.get(k) for k in ("shop", "office", "industrial", "tourism", "craft"))
        if value in _NON_RESIDENTIAL_BUILDING_TAGS or footprint > 1200.0 or commercial_tag:
            non_residential_buildings += 1
        elif value in _RESIDENTIAL_BUILDING_TAGS or 35.0 <= footprint <= 900.0:
            residential_buildings += 1

    real_coverage = real_building_area / area_sqm if area_sqm > 0 else 0.0

    def _tag(feature: dict[str, Any], key: str) -> str:
        tags = feature.get("tags") if isinstance(feature, dict) else {}
        return str((tags or {}).get(key, "")).strip().lower()

    public_signal = any(
        _tag(p, "access") == "public" or (p.get("tags") or {}).get("operator") or _tag(p, "leisure") == "water_park"
        for p in pools
    ) or any(
        _tag(b, "leisure") in _PUBLIC_LEISURE_TAGS or _tag(b, "amenity") in {"community_centre", "swimming_pool"}
        for b in buildings
    )
    big_pool = selected_area_class == "selected_large_pool_footprint" or any(
        _feature_area_sqm(p) >= 200.0 for p in pools
    )

    if has_existing_pool and (big_pool or public_signal) and real_coverage < 0.25:
        kind = "aquatic_venue"
    elif non_residential_buildings >= 1 and area_sqm >= 1500.0 and residential_buildings == 0:
        kind = "built_precinct"
    elif residential_buildings >= 1 and non_residential_buildings == 0 and area_sqm <= 3000.0:
        kind = "residential"
    elif n_buildings == 0 and real_coverage < 0.08 and area_sqm >= 1500.0:
        kind = "open_space"
    elif residential_buildings >= 1:
        kind = "residential"
    elif non_residential_buildings >= 1 or largest_building > 1200.0:
        kind = "built_precinct"
    else:
        kind = "parcel"

    is_residential = kind in {"residential", "parcel"}
    open_pct = round(max(0.0, 100.0 - real_coverage * 100.0), 1)

    if kind == "open_space" and vegetation_pct < 18.0:
        label = "Large open / cleared site"
    else:
        label = {
            "aquatic_venue": "Existing aquatic venue / pool precinct",
            "open_space": "Open recreation land",
            "residential": "Residential lot",
            "built_precinct": "Built-up / commercial precinct",
            "parcel": "Mixed / unclassified parcel",
        }[kind]

    area_txt = f"{round(area_sqm):,} m²"
    feature_bits: list[str] = []
    if n_buildings:
        feature_bits.append(f"{n_buildings} building{'' if n_buildings == 1 else 's'}")
    if pools:
        feature_bits.append(f"{len(pools)} pool{'' if len(pools) == 1 else 's'}")
    feature_bits.append(f"{round(real_coverage * 100)}% built coverage")
    feature_bits.append(f"{round(vegetation_pct)}% vegetation")
    detected = ", ".join(feature_bits)

    if kind == "aquatic_venue":
        summary = (
            f"This {area_txt} selection reads as an existing aquatic venue / pool precinct "
            f"({detected}). Suitability is framed at facility scale, not as a residential "
            "backyard build, so residential setback and frontage metrics below are indicative only."
        )
    elif kind == "open_space":
        summary = (
            f"This {area_txt} selection reads as {label.lower()} ({detected}). A pool here would be a "
            "community or public facility rather than a backyard build, so residential side-setback "
            "rules do not apply; scoring reflects open area, terrain, and solar access."
        )
    elif kind == "built_precinct":
        summary = (
            f"This {area_txt} selection reads as a built-up / commercial precinct ({detected}). Open "
            "ground for a new pool is limited; assess at site-plan scale rather than as a single backyard."
        )
    elif kind == "residential":
        summary = (
            f"This {area_txt} selection reads as a residential lot ({detected}). Suitability below is "
            "assessed for a private backyard build with standard residential setbacks."
        )
    else:
        summary = (
            f"This {area_txt} selection ({detected}) could not be confidently classified; it is assessed "
            "as a generic parcel, so treat residential-specific metrics as approximate."
        )

    return {
        "kind": kind,
        "label": label,
        "is_residential": is_residential,
        "summary": summary,
        "building_count": n_buildings,
        "residential_building_count": residential_buildings,
        "non_residential_building_count": non_residential_buildings,
        "pool_count": len(pools),
        "real_coverage_pct": round(real_coverage * 100.0, 1),
        "open_pct": open_pct,
    }


def analyze_area(body: dict[str, Any]) -> dict[str, Any]:
    """Analyze a user-drawn polygon or pre-defined property for suitability.
    
    Accepts:
      - polygon: [[lng,lat], ...] list of coordinates
      - features: dict of verified metrics passed from the frontend (optional)
      - osm_features: dict of raw OSM structures within the parcel (optional)
      - bounds: {south,north,west,east} rectangle (legacy)
    """
    import hashlib

    R = 6378137  # earth radius meters

    def _ll_to_m(lng, lat, lng0, lat0):
        """Convert lng/lat to local meters."""
        x = math.radians(lng - lng0) * R * math.cos(math.radians(lat0))
        y = math.radians(lat - lat0) * R
        return x, y

    # Parse polygon or bounds
    polygon = body.get("polygon")  # [[lng,lat], ...]
    if polygon is not None:
        coords = validate_analysis_polygon(polygon)
    elif body.get("bounds") is not None:
        # Legacy bounds mode
        b = body.get("bounds", {})
        if not isinstance(b, dict):
            raise ValueError("bounds must be a JSON object.")
        try:
            s = float(b["south"])
            n = float(b["north"])
            w = float(b["west"])
            e = float(b["east"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("bounds south, north, west, and east must be valid numbers.") from exc
        if not all(math.isfinite(value) for value in (s, n, w, e)):
            raise ValueError("bounds south, north, west, and east must be valid numbers.")
        if not (-90 <= s <= 90 and -90 <= n <= 90 and -180 <= w <= 180 and -180 <= e <= 180):
            raise ValueError("bounds are outside valid longitude/latitude ranges.")
        if s >= n or w >= e:
            raise ValueError("bounds must have south < north and west < east.")
        coords = validate_analysis_polygon([[w, s], [e, s], [e, n], [w, n]])
    else:
        raise ValueError("Provide a polygon with at least three coordinate pairs.")

    # Centroid
    cx = sum(c[0] for c in coords) / len(coords)
    cy = sum(c[1] for c in coords) / len(coords)

    selection_polygon = coords

    # Convert to local meters
    pts = [_ll_to_m(c[0], c[1], cx, cy) for c in coords]

    # Shoelace area
    n_pts = len(pts)
    area_sqm = 0.0
    for i in range(n_pts):
        j = (i + 1) % n_pts
        area_sqm += pts[i][0] * pts[j][1]
        area_sqm -= pts[j][0] * pts[i][1]
    area_sqm = abs(area_sqm) / 2.0

    # Perimeter
    perimeter = 0.0
    for i in range(n_pts):
        j = (i + 1) % n_pts
        dx = pts[j][0] - pts[i][0]
        dy = pts[j][1] - pts[i][1]
        perimeter += math.sqrt(dx*dx + dy*dy)

    # Bounding box dimensions
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)
    bbox_area = bbox_w * bbox_h if bbox_w > 0 and bbox_h > 0 else 1

    # Shape metrics
    compactness = (4 * math.pi * area_sqm) / (perimeter * perimeter) if perimeter > 0 else 0
    fill_ratio = area_sqm / bbox_area if bbox_area > 0 else 0
    aspect_ratio = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 0.1)

    # Longest/shortest edges as spatial metrics
    edge_lengths = []
    for i in range(n_pts):
        j = (i + 1) % n_pts
        edge_lengths.append(math.sqrt((pts[j][0]-pts[i][0])**2 + (pts[j][1]-pts[i][1])**2))
    shortest_edge = min(edge_lengths) if edge_lengths else 0
    longest_edge = max(edge_lengths) if edge_lengths else 0
    edge_aspect_ratio = longest_edge / max(shortest_edge, 0.1)

    # Location seed for deterministic simulated terrain properties when real data is unavailable
    seed_str = f"{cx:.8f}{cy:.8f}{area_sqm:.2f}"
    seed = hashlib.md5(seed_str.encode()).hexdigest()
    sv = [int(seed[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]

    # Read verified inputs from frontend or fallback to realistic terrain models
    frontend_features = body.get("features", {})
    osm_features = body.get("local_features") or body.get("osm_features") or {}
    if not isinstance(frontend_features, dict) or not isinstance(osm_features, dict):
        raise ValueError("features and osm_features must be JSON objects.")

    def _optional_number(name: str, lower: float, upper: float) -> float | None:
        value = frontend_features.get(name)
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"features.{name} must be a number.") from exc
        if not math.isfinite(parsed) or not lower <= parsed <= upper:
            raise ValueError(f"features.{name} must be between {lower} and {upper}.")
        return parsed

    def _optional_bool(name: str, default: bool = False) -> bool:
        value = frontend_features.get(name)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"features.{name} must be a boolean.")

    # Dynamic live queries with robust deterministic fallbacks
    slope_pct = _optional_number("slope_pct", 0.0, 200.0)
    relief_m = _optional_number("relief_m", 0.0, 1000.0)
    solar_access_pct = _optional_number("solar_access_pct", 0.0, 100.0)
    needs_elevation = slope_pct is None or relief_m is None
    needs_solar = solar_access_pct is None
    elevation_source = "provided"
    solar_source = "provided"

    elevation_result = None
    solar_result = None
    if needs_elevation and needs_solar:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="site-analysis") as executor:
            elevation_future = executor.submit(get_elevation_corners_with_source, cy, cx, bbox_w, bbox_h)
            solar_future = executor.submit(get_solar_access_with_source, cy, cx)
            elevation_result = elevation_future.result()
            solar_result = solar_future.result()
    elif needs_elevation:
        elevation_result = get_elevation_corners_with_source(cy, cx, bbox_w, bbox_h)
    elif needs_solar:
        solar_result = get_solar_access_with_source(cy, cx)

    if elevation_result is not None:
        elev_corners, elevation_source = elevation_result
        elevation_values = [elev_corners[key] for key in ("sw", "se", "nw", "ne")]
        front_avg = (elev_corners["sw"] + elev_corners["se"]) / 2.0
        rear_avg = (elev_corners["nw"] + elev_corners["ne"]) / 2.0
        west_avg = (elev_corners["sw"] + elev_corners["nw"]) / 2.0
        east_avg = (elev_corners["se"] + elev_corners["ne"]) / 2.0
        parcel_width = max(bbox_w, 1.0)
        parcel_depth = max(bbox_h, 1.0)
        slope_y_pct = abs(rear_avg - front_avg) / parcel_depth * 100.0
        slope_x_pct = abs(east_avg - west_avg) / parcel_width * 100.0
        calc_slope = max(slope_x_pct, slope_y_pct)
        calc_relief = max(elevation_values) - min(elevation_values)
        
        if slope_pct is None:
            slope_pct = round(calc_slope, 1)
        if relief_m is None:
            relief_m = round(calc_relief, 1)

    if solar_result is not None:
        solar_access_pct, solar_source = solar_result

    provided_vegetation_pct = _optional_number("vegetation_pct", 0.0, 100.0)
    vegetation_pct = provided_vegetation_pct if provided_vegetation_pct is not None else round(5 + sv[3] * 40, 1)
    provided_drainage = _optional_number("soil_drainage_pct", 0.0, 100.0)
    soil_drainage = provided_drainage if provided_drainage is not None else round(45 + sv[4] * 50, 1)
    provided_flood_risk = _optional_number("flood_risk_pct", 0.0, 100.0)
    flood_risk = provided_flood_risk if provided_flood_risk is not None else round(sv[5] * 20, 1)
    provided_orientation = _optional_number("orientation_deg", 0.0, 360.0)
    orientation_deg = provided_orientation if provided_orientation is not None else round(sv[6] * 360, 1)
    overhead_powerline = _optional_bool("overhead_powerline")
    # Read exact OSM geometries or custom properties
    buildings_list = osm_features.get("buildings", [])
    pools_list = osm_features.get("pools", [])
    trees_list = osm_features.get("trees", [])
    if not all(isinstance(items, list) for items in (buildings_list, pools_list, trees_list)):
        raise ValueError("osm_features buildings, pools, and trees must be arrays.")
    if sum(len(items) for items in (buildings_list, pools_list, trees_list)) > 10_000:
        raise ValueError("Too many OSM features were supplied for one analysis request.")

    buildings_list = [
        feature for feature in buildings_list if osm_feature_intersects_polygon(feature, selection_polygon)
    ]
    pools_list = [
        feature for feature in pools_list if osm_feature_intersects_polygon(feature, selection_polygon)
    ]
    trees_list = [
        feature for feature in trees_list if osm_feature_intersects_polygon(feature, selection_polygon)
    ]

    # Nearby pool evidence is optional context supplied by a scoped scan. It is
    # deliberately kept separate from ``pools_list``: a pool outside the drawn
    # area can support neighbourhood feasibility, but must never be treated as an
    # existing pool inside the selected area.
    context_features = body.get("context_features") or {}
    if not isinstance(context_features, dict):
        raise ValueError("context_features must be a JSON object.")
    nearby_pools = context_features.get("nearby_pools") or []
    if not isinstance(nearby_pools, list):
        raise ValueError("context_features.nearby_pools must be an array.")
    nearby_pool_count = min(20, len(nearby_pools))

    # Calculate real structure metrics from structures list
    building_area_sqm = _optional_number("building_area_sqm", 0.0, 10_000_000.0)
    if building_area_sqm is None:
        if buildings_list:
            # Sum up building area from geometry list if present
            building_area_sqm = 0.0
            for b in buildings_list:
                b_geom = _feature_geometry_points(b)
                if len(b_geom) >= 3:
                    # Shoelace inside local relative coordinates
                    b_pts = [_ll_to_m(lon, lat, cx, cy) for lon, lat in b_geom]
                    n_b = len(b_pts)
                    b_area = 0.0
                    for k in range(n_b):
                        next_index = (k + 1) % n_b
                        b_area += b_pts[k][0] * b_pts[next_index][1]
                        b_area -= b_pts[next_index][0] * b_pts[k][1]
                    building_area_sqm += abs(b_area) / 2.0
            building_area_sqm = round(building_area_sqm, 1)
        else:
            estimated_building_pct = 25 + sv[7] * 18
            building_area_sqm = round(area_sqm * (estimated_building_pct / 100), 1)

    # Area Analysis is a suitability workflow, not an existing-pool detector.
    # Keep any supplied pool geometries available for exports/overlays, but do
    # not let them—or polygon shape heuristics—suppress new-pool scoring.
    has_existing_pool = False
    selected_area_class = "site_area"
    pool_detection_reason = "Existing-pool detection is disabled for area suitability analysis."
    
    # Calculate real vegetation canopy metrics
    tree_count = len(trees_list)
    if tree_count > 0 and provided_vegetation_pct is None:
        canopy_area = 0.0
        for t in trees_list:
            t_geom = _feature_geometry_points(t)
            if t.get("tags", {}).get("natural") == "wood" and len(t_geom) >= 3:
                # Calculate actual wood area using Shoelace
                t_pts = [_ll_to_m(lon, lat, cx, cy) for lon, lat in t_geom]
                n_t = len(t_pts)
                t_area = 0.0
                for k in range(n_t):
                    next_index = (k + 1) % n_t
                    t_area += t_pts[k][0] * t_pts[next_index][1]
                    t_area -= t_pts[next_index][0] * t_pts[k][1]
                canopy_area += abs(t_area) / 2.0
            else:
                # Individual tree canopy approximation: ~25.0 sqm area
                canopy_area += 25.0
        vegetation_pct = round(min(85.0, (canopy_area / area_sqm) * 100.0), 1)

    # Precise Zoning & Backyard calculations
    site_coverage_ratio = round(building_area_sqm / area_sqm, 3) if area_sqm > 0 else 0.0
    backyard_area = round(max(0.0, area_sqm - building_area_sqm - (30.0 if has_existing_pool else 0.0)), 1)
    depth_m = round(max(1.0, math.sqrt(backyard_area) * 0.9), 1)
    frontage_m = round(bbox_w if bbox_w < bbox_h else bbox_h, 1)
    buildable_rear_zone_sqm = round(max(0.0, backyard_area - 24.0), 1)
    
    # Custom side clearance assessment based on shortest edge proxy
    side_clearance = _optional_number("max_side_clearance_m", 0.0, 1000.0)
    if side_clearance is None:
        side_clearance = round(max(0.8, shortest_edge * 0.35), 1)

    # ===== AREA GEOGRAPHY =====
    # Read what this area actually is (residential lot vs park vs aquatic venue vs
    # built precinct) so the scoring and narrative reason at the right scale rather
    # than always assuming a suburban backyard.
    site_profile = classify_site_geography(
        area_sqm=area_sqm,
        buildings=buildings_list,
        pools=[],
        vegetation_pct=vegetation_pct,
        has_existing_pool=has_existing_pool,
        selected_area_class=selected_area_class,
    )
    is_residential_site = bool(site_profile["is_residential"])

    # ===== AI SUITABILITY SCORING (ENTERPRISE GRADE) =====
    
    # 1. Pool Suitability
    if has_existing_pool:
        pool_3d = 0.0
        pool_2d = 0.0
        pool_type = "standard"
    else:
        is_small = backyard_area < 120.0 and buildable_rear_zone_sqm >= 15.0
        pool_type = "small" if is_small else "standard"
        
        # Base scores using real spatial metrics
        if is_small:
            area_score = min(100.0, ((backyard_area - 20.0) / 100.0) * 100.0) if backyard_area >= 20.0 else 0.0
            depth_score = min(100.0, ((depth_m - 3.0) / 5.0) * 100.0) if depth_m >= 3.0 else 0.0
        else:
            area_score = min(100.0, (backyard_area / 240.0) * 100.0)
            depth_score = min(100.0, (depth_m / 11.0) * 100.0)
            
        access_score = min(100.0, (side_clearance / 3.0) * 100.0)

        # Residential side-setback (1.2m) only constrains a private backyard build. On
        # open recreation land, public/aquatic precincts, or large non-residential sites
        # there is no neighbouring lot boundary to clear, so the harsh penalty would be a
        # false negative — treat access as compliant and score on area/terrain/solar.
        if not is_residential_site:
            access_score = max(access_score, 80.0)
        elif side_clearance < 1.2:
            access_score *= 0.25

        # Penalize excessive building coverage ratio (Moreton Bay zoning: max 50% site coverage)
        coverage_penalty = max(0.0, (site_coverage_ratio - 0.50) * 80.0)
        
        # Terrain factor: slope raises excavation/retaining cost, but a sloped
        # residential lot is not automatically unusable. The former exponential
        # curve reduced a 13.9% site to zero terrain credit, which made the 3D
        # rating contradict the strong area/access score visible in the report.
        slope_score = max(30.0, 100.0 - (slope_pct * 4.0))
        
        # Environmental overlays
        veg_penalty = max(0.0, (vegetation_pct - 25.0) * 1.2)
        # ``flood_risk_pct`` is a bounded site-risk index in this workflow, not a
        # literal probability. Keep the caution meaningful without turning a
        # roughly 20-point index into a 32-point suitability penalty.
        flood_penalty = min(15.0, max(0.0, flood_risk * 0.6))
        solar_bonus = max(0.0, (solar_access_pct - 65.0) * 0.4)
        powerline_penalty = 25.0 if overhead_powerline else 0.0
        nearby_pool_bonus = min(6.0, nearby_pool_count * 2.0)
        
        upper_bound = 75.0 if is_small else 100.0
        pool_3d = round(min(upper_bound, max(0.0,
            area_score * 0.26 +
            slope_score * 0.22 +
            depth_score * 0.16 +
            access_score * 0.16 +
            solar_bonus * 0.10 +
            (soil_drainage * 0.10) -
            coverage_penalty -
            veg_penalty -
            flood_penalty -
            powerline_penalty +
            nearby_pool_bonus
        )), 1)
        
        pool_2d = round(min(upper_bound, max(0.0,
            area_score * 0.34 +
            depth_score * 0.24 +
            access_score * 0.22 +
            (fill_ratio * 10.0) +
            ((100.0 - flood_risk) * 0.10) -
            coverage_penalty -
            veg_penalty * 0.5 -
            powerline_penalty
        )), 1)

    # 2. Fencing Suitability
    fence_perim_score = min(100.0, (perimeter / 75.0) * 100.0)
    fence_access_score = min(100.0, (side_clearance / 2.0) * 100.0)
    fence_slope_penalty = max(0.0, slope_pct * 3.5)
    # High vegetation canopy can act as screening, reducing the absolute utility of a fence
    fence_veg_factor = max(0.0, 100.0 - vegetation_pct)
    
    fence = round(min(100.0, max(0.0,
        fence_perim_score * 0.35 +
        fence_access_score * 0.20 +
        (compactness * 15.0) +
        (fence_veg_factor * 0.18) +
        ((100.0 - fence_slope_penalty) * 0.12)
    )), 1)

    # 3. Deck / Patio Suitability
    deck_area_score = min(100.0, (backyard_area / 140.0) * 100.0)
    deck_solar_score = solar_access_pct
    
    # Moderate slope (1-6%) allows stepped decks which look incredible
    if 1.0 <= slope_pct <= 6.0:
        deck_slope_score = 95.0
    else:
        deck_slope_score = max(0.0, 100.0 - slope_pct * 6.5)
        
    deck_aspect_score = 100.0 if 45.0 <= orientation_deg <= 315.0 else 65.0
    deck_veg_penalty = max(0.0, (vegetation_pct - 35.0) * 1.0)
    
    deck = round(min(100.0, max(0.0,
        deck_area_score * 0.22 +
        deck_solar_score * 0.24 +
        deck_slope_score * 0.24 +
        deck_aspect_score * 0.15 +
        ((100.0 - deck_veg_penalty) * 0.10) +
        (compactness * 5.0)
    )), 1)

    # 4. Landscaping Suitability
    landscaping = round(min(100.0, max(0.0,
        min(100.0, (backyard_area / 180.0) * 100.0) * 0.30 +
        vegetation_pct * 0.25 +
        solar_access_pct * 0.20 +
        soil_drainage * 0.15 +
        (100.0 - slope_pct * 4.0) * 0.10
    )), 1)

    # ===== EXPLAINABLE AI REPORT GENERATION =====
    def _pool_report(v, area, slope, access, p_type):
        if has_existing_pool:
            if selected_area_class == "selected_large_pool_footprint":
                return (
                    f"Existing pool present (verified). This {round(area_sqm)} sqm polygon has the footprint of a "
                    "large outdoor or public aquatic basin, so the site already supports a pool — a new-build "
                    "suitability score does not apply. The actionable opportunities here are resurfacing/renovation, "
                    "compliant pool-safety fencing, decking and shade structures, plant-room/equipment upgrades, and "
                    "surrounding landscaping. Select an adjacent pool-free backyard to score a brand-new pool build."
                )
            if selected_area_class == "selected_pool_footprint":
                return (
                    f"Existing pool present. The selected {round(area_sqm)} sqm polygon is already pool-shaped, so the "
                    "site clearly supports a pool and a new-build score is not applicable to this footprint. "
                    "Relevant work is renovation, glass safety fencing, decking, and landscaping. Select the "
                    "surrounding backyard or parcel to assess expansion or new-build opportunities."
                )
            return (
                "Existing swimming pool confirmed inside this area — the site already supports an in-ground pool, so "
                "this is not a new-build pool lead. The relevant opportunities are renovation/resurfacing, compliant "
                "pool-safety fencing, heating and equipment upgrades, decking, and landscaping around the pool. "
                "Select an adjacent pool-free backyard to score a brand-new pool build."
            )
        if not is_residential_site:
            kind = site_profile["kind"]
            if kind == "open_space":
                return (
                    f"Open recreation land ({v}/100, precinct scale). This is open public/community ground rather than "
                    f"a residential backyard, so a pool here would be a community or public aquatic facility. The score "
                    f"reflects {area} sqm of open area, a {slope}% grade, and solar access — residential side-setback "
                    "rules are not applied. Select an individual dwelling lot to score a private backyard build."
                )
            if kind == "built_precinct":
                return (
                    f"Built-up / commercial precinct ({v}/100). Substantial building coverage leaves limited open ground "
                    "for a new pool; evaluate at site-plan scale rather than as a single backyard, factoring in access "
                    "for plant and excavation across the site."
                )
            return (
                f"Large / non-residential parcel ({v}/100). Open area of {area} sqm on a {slope}% grade; assess this as a "
                "site plan, since residential backyard setback and frontage metrics are only an approximation at this scale."
            )
        if p_type == "small":
            return f"Compact pool candidate ({v}/75). The compact backyard of {area} sqm is well-suited for a plunge or splash pool. Setback clearance of {access}m and buildable zone are compliant for small shell installs."
        if v >= 80:
            return f"Excellent suitability ({v}/100). The generous backyard area of {area} sqm coupled with a gentle slope of {slope}% offers a premium canvas. The side clearance of {access}m safely exceeds the 1.2m regulatory requirement, supporting heavy excavation machinery access with minimal site preparation."
        if v >= 65:
            return f"Good suitability ({v}/100). A workable {area} sqm open backyard with side clearance of {access}m. Slope of {slope}% is manageable, though retaining walls may be needed for leveling."
        if v >= 45:
            return f"Moderate suitability ({v}/100). The {area} sqm backyard or the slope of {slope}% creates constraints. Consider a specialized plunge pool or highly custom deck-integrated design to optimize build footprint."
        return f"Challenging terrain suitability ({v}/100). Significant constraints due to tight side clearance ({access}m), steep grade ({slope}%), or high site coverage. Spa installations or small-scale custom structural shells are the only practical candidate designs."

    def _fence_report(v, perim, veg):
        if v >= 80:
            return f"Excellent fence utility ({v}/100). Long parcel perimeter ({perim}m) with limited natural foliage ({veg}%), creating an ideal opportunity for high-security boundary or decorative Colorbond fencing."
        return f"Workable fence suitability ({v}/100). A {perim}m boundary. Dense vegetation coverage ({veg}%) contributes natural privacy, reducing the immediate utility of a structural fence overlay."

    def _deck_report(v, solar, slope):
        if v >= 80:
            return f"High-fidelity entertaining platform candidate ({v}/100). Excellent solar access ({solar}%) and perfect terrain slope ({slope}%) support custom tiered timber decking and outdoor patios directly connected to the main structure."
        return f"Moderate deck suitability ({v}/100). Shaded layout ({100-solar}% blocked) or uneven terrain requires custom engineered post structures and stepped platforms."

    # ── Synthesized bottom-line recommendation ──────────────────────────
    # Rank the candidate improvements and call out the single best opportunity, so the
    # report opens with an actionable verdict instead of leaving the reader to compare
    # five scores. An existing pool reframes the advice toward upgrades around it.
    opportunities = [
        ("a new swimming pool", -1.0 if has_existing_pool else pool_3d),
        ("boundary fencing", fence),
        ("an entertainment deck / patio", deck),
        ("landscaping & gardens", landscaping),
    ]
    best_use, best_score = max(opportunities, key=lambda item: item[1])
    if has_existing_pool:
        bottom_line = (
            "This site already has a pool, so the value is in upgrading around it — "
            f"decking ({deck}/100), safety fencing ({fence}/100), and landscaping ({landscaping}/100)."
        )
    elif best_score >= 80:
        bottom_line = f"Strong candidate for {best_use} ({round(best_score)}/100) — the standout opportunity on this site."
    elif best_score >= 60:
        bottom_line = f"Best suited to {best_use} ({round(best_score)}/100); the other improvements are secondary here."
    elif best_score >= 40:
        bottom_line = f"A constrained site — {best_use} ({round(best_score)}/100) is the most workable improvement, and only with custom design."
    else:
        bottom_line = f"Heavily constrained site; even {best_use} ({round(best_score)}/100) needs specialised design — treat as low priority."

    used_fallback = "fallback" in elevation_source or "fallback" in solar_source
    confidence_line = (
        "Confidence: MODERATE — some terrain/solar figures use modelled fallbacks (live services were unavailable)."
        if used_fallback
        else "Confidence: HIGH — elevation and solar figures come from live external data."
    )

    if landscaping >= 75:
        landscaping_note = "Excellent canvas for structured native and ornamental gardens."
    elif landscaping >= 50:
        landscaping_note = "Workable for hardy native plantings and low-water garden beds."
    else:
        landscaping_note = "Limited growing conditions; raised beds or container planting suit best."

    context_line = (
        f"Neighbourhood context: {nearby_pool_count} mapped/detected pool"
        f"{'s' if nearby_pool_count != 1 else ''} nearby support feasibility, but do not mark this selected area as occupied."
        if nearby_pool_count
        else "Neighbourhood pool context was not available; the rating is based on the selected area's own geometry and site metrics."
    )

    explanation_parts = [
        "PROPERTY INTELLIGENCE ANALYSIS & LAND SUITABILITY SUMMARY",
        "----------------------------------------------------------------",
        f"Parcel Area: {round(area_sqm)} sqm · Backyard Space: {backyard_area} sqm · Site Coverage: {round(site_coverage_ratio * 100, 1)}%",
        f"Slope Gradient: {slope_pct}% · Elevation Relief: {relief_m}m · Compass Heading: {orientation_deg}° ({'North-East facing' if 45 <= orientation_deg <= 135 else 'North-West facing' if 135 < orientation_deg <= 225 else 'South facing'})",
        f"Soil Drainage: {soil_drainage}% (Good) · Flood Exposure: {flood_risk}% (Low)" if flood_risk < 10 else f"Soil Drainage: {soil_drainage}% · Flood Exposure: {flood_risk}% (Caution: flood overlay intersects)",
        "Overhead Utility: CLEAR (No overhead lines intersect backyard)" if not overhead_powerline else "Overhead Utility: DANGER (Overhead service conflict / powerline detected in backyard)",
        "",
        f"AREA GEOGRAPHY — {site_profile['label']}",
        f"  {site_profile['summary']}",
        f"  {context_line}",
        "",
        "BOTTOM LINE",
        f"  {bottom_line}",
        f"  {confidence_line}",
        "",
        "SWIMMING POOL SUITABILITY",
        f"  {_pool_report(pool_3d, backyard_area, slope_pct, side_clearance, pool_type)}",
        "",
        "BOUNDARY FENCING SUITABILITY",
        f"  {_fence_report(fence, round(perimeter), round(vegetation_pct))}",
        "",
        "ENTERTAINMENT DECK & PATIO SUITABILITY",
        f"  {_deck_report(deck, round(solar_access_pct), slope_pct)}",
        "",
        "LANDSCAPING & CANOPY RATING",
        f"  Score: {landscaping}/100. Tree Count: {tree_count} detected. Vegetation density: {vegetation_pct}%. {landscaping_note}",
    ]

    # Geometries package format to unify 3D loader pipeline
    geometries = {
        "parcel_geom": {
            "type": "Polygon",
            "coordinates": [coords + [coords[0]]]  # closed loop
        },
        "buildings": buildings_list,
        "pools": pools_list,
        "trees": trees_list
    }

    return {
        "area_sqm": round(area_sqm, 1),
        "perimeter_m": round(perimeter, 1),
        "polygon": coords,
        "centroid": [cx, cy],
        "has_existing_pool": has_existing_pool,
        "selected_area_class": selected_area_class,
        "pool_detection_reason": pool_detection_reason,
        "site_profile": site_profile,
        "data_sources": {
            "elevation": elevation_source,
            "solar": solar_source,
            "map_features": "openstreetmap" if osm_features else "none",
            "pool_detection": selected_area_class,
        },
        "data_warnings": [
            warning
            for warning, active in (
                ("Elevation service unavailable; a flat fallback estimate was used.", "fallback" in elevation_source),
                ("Solar service unavailable; a latitude-based fallback estimate was used.", "fallback" in solar_source),
            )
            if active
        ],
        "shape": {
            "compactness": round(compactness, 3),
            "fill_ratio": round(fill_ratio, 3),
            "aspect_ratio": round(aspect_ratio, 2),
            "n_vertices": len(coords),
        },
        "scores": {
            "pool_3d": pool_3d,
            "pool_2d": pool_2d,
            "fence": fence,
            "deck": deck,
            "landscaping": landscaping,
            "pool_type": pool_type,
        },
        "features": {
            "total_area_sqm": round(area_sqm, 1),
            "backyard_area_sqm": backyard_area,
            "backyard_depth_m": depth_m,
            "frontage_m": frontage_m,
            "max_side_clearance_m": side_clearance,
            "slope_pct": slope_pct,
            "relief_m": relief_m,
            "solar_access_pct": solar_access_pct,
            "vegetation_pct": vegetation_pct,
            "soil_drainage_pct": soil_drainage,
            "flood_risk_pct": flood_risk,
            "nearby_pool_count": nearby_pool_count,
            "orientation_deg": orientation_deg,
            "perimeter_m": round(perimeter, 1),
            "buildable_rear_zone_sqm": buildable_rear_zone_sqm,
            "pool_type": pool_type,
            "has_existing_pool": has_existing_pool,
            "selected_area_class": selected_area_class,
            "pool_detection_reason": pool_detection_reason,
            "site_profile_kind": site_profile["kind"],
            "site_profile_label": site_profile["label"],
        },
        "geometries": geometries,
        "explanation": "\n".join(explanation_parts),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export ranked parcels, buildings, and backyard candidates to a QGIS-friendly GeoPackage.")
    parser.add_argument("--data", help="Optional path to the app data JSON. Defaults to the sample properties file.")
    parser.add_argument("--module", default="pool_3d", help="Ranking module to export.")
    parser.add_argument("--lga", help="Optional LGA filter.")
    parser.add_argument("--top-k", type=int, default=5, help="How many top-ranked properties to export.")
    parser.add_argument("--property-id", action="append", dest="property_ids", help="Specific property id to export. Repeat to export multiple.")
    parser.add_argument(
        "--output",
        default=str(Path("artifacts") / "qgis" / "ranked_selection.gpkg"),
        help="Output GeoPackage path.",
    )
    parser.add_argument("--open-qgis", action="store_true", help="Open the exported GeoPackage in QGIS Desktop.")
    args = parser.parse_args(argv)

    result = export_ranked_map(
        output_path=args.output,
        data_path=args.data,
        module_name=args.module,
        lga=args.lga,
        top_k=args.top_k,
        property_ids=args.property_ids,
    )
    if args.open_qgis:
        result["qgis"] = open_in_qgis(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
