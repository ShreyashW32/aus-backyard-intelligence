from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import rasterio
from pyproj import CRS, Transformer
from shapely.geometry import LineString, box
from shapely.ops import unary_union

from ..app import clamp, derive_features, rear_backyard_rect
from ..eligibility import enrich_property_record

DEFAULT_TARGET_EPSG = 7856


@dataclass
class BuildResult:
    records: list[dict[str, Any]]
    skipped_without_buildings: int
    skipped_invalid_geometry: int
    skipped_non_private: int


def _read_layer(path: str | Path, target_epsg: int = DEFAULT_TARGET_EPSG) -> gpd.GeoDataFrame:
    frame = gpd.read_file(path)
    if frame.empty:
        return frame
    if frame.crs is None:
        frame = frame.set_crs(epsg=target_epsg)
    if CRS.from_user_input(frame.crs).to_epsg() != target_epsg:
        frame = frame.to_crs(epsg=target_epsg)
    return frame


def _optional_read(path: str | Path | None, target_epsg: int = DEFAULT_TARGET_EPSG) -> gpd.GeoDataFrame | None:
    if path is None:
        return None
    return _read_layer(path, target_epsg=target_epsg)


def _coerce_string(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _estimate_frontage(geometry) -> float:
    if geometry is None or geometry.is_empty:
        return 0.0
    rectangle = geometry.minimum_rotated_rectangle
    coords = list(rectangle.exterior.coords)
    if len(coords) < 4:
        minx, miny, maxx, maxy = geometry.bounds
        return round(min(maxx - minx, maxy - miny), 1)
    edges = []
    for start, end in zip(coords, coords[1:], strict=False):
        length = LineString([start, end]).length
        if length > 0:
            edges.append(length)
    if not edges:
        minx, miny, maxx, maxy = geometry.bounds
        return round(min(maxx - minx, maxy - miny), 1)
    return round(min(edges), 1)


def _sample_dem_corners(dem_path: str | Path, bounds: tuple[float, float, float, float], bounds_crs: Any) -> dict[str, float]:
    with rasterio.open(dem_path) as src:
        xmin, ymin, xmax, ymax = bounds
        eps_x = max((xmax - xmin) * 1e-6, 0.01)
        eps_y = max((ymax - ymin) * 1e-6, 0.01)
        points = {
            "sw": (xmin + eps_x, ymin + eps_y),
            "se": (xmax - eps_x, ymin + eps_y),
            "nw": (xmin + eps_x, ymax - eps_y),
            "ne": (xmax - eps_x, ymax - eps_y),
        }
        transformer = None
        if src.crs and bounds_crs and CRS.from_user_input(src.crs) != CRS.from_user_input(bounds_crs):
            transformer = Transformer.from_crs(bounds_crs, src.crs, always_xy=True)
        sample_points = []
        labels = []
        for label, point in points.items():
            x, y = point
            if transformer:
                x, y = transformer.transform(x, y)
            sample_points.append((x, y))
            labels.append(label)
        sampled = list(src.sample(sample_points))
        nodata = src.nodata
        results: dict[str, float] = {}
        fallback = 0.0
        for label, values in zip(labels, sampled, strict=False):
            value = float(values[0])
            if nodata is not None and value == nodata:
                value = fallback
            results[label] = round(value, 1)
            fallback = value
        return results


def _cover_percentage(parcel_geom, overlay: gpd.GeoDataFrame | None) -> float:
    if overlay is None or overlay.empty:
        return 0.0
    candidates = overlay.iloc[list(overlay.sindex.intersection(parcel_geom.bounds))]
    if candidates.empty:
        return 0.0
    clipped = [geom.intersection(parcel_geom) for geom in candidates.geometry if geom.intersects(parcel_geom)]
    clipped = [geom for geom in clipped if not geom.is_empty]
    if not clipped:
        return 0.0
    area = unary_union(clipped).area
    return round(clamp(area / parcel_geom.area * 100.0, 0.0, 100.0), 1)


def _intersects_any(parcel_geom, overlay: gpd.GeoDataFrame | None) -> bool:
    if overlay is None or overlay.empty:
        return False
    candidates = overlay.iloc[list(overlay.sindex.intersection(parcel_geom.bounds))]
    if candidates.empty:
        return False
    return bool(candidates.intersects(parcel_geom).any())


def _candidate_buildings(parcel_geom, buildings: gpd.GeoDataFrame) -> list[Any]:
    candidates = buildings.iloc[list(buildings.sindex.intersection(parcel_geom.bounds))]
    if candidates.empty:
        return []
    clipped = []
    for geom in candidates.geometry:
        if geom is None or geom.is_empty or not geom.intersects(parcel_geom):
            continue
        clipped_geom = geom.intersection(parcel_geom)
        if clipped_geom.is_empty or clipped_geom.area <= 0:
            continue
        clipped.append(clipped_geom)
    return clipped


def _base_record(
    property_id: str,
    lga: str,
    suburb: str,
    address: str,
    parcel_geom,
    building_geom,
    elevations: dict[str, float],
    vegetation_pct: float,
    road_frontage_m: float,
    corner_lot: bool,
    flood_risk: int,
    heritage_overlay: bool,
    overhead_powerline: bool,
    source_crs: str,
    land_use: str = "",
    zoning: str = "",
    property_type: str = "",
    ownership_class: str = "",
    owner_name: str = "",
    tenure: str = "",
    has_existing_pool: bool = False,
    existing_pool_source: str | None = None,
) -> dict[str, Any]:
    parcel_bounds = [round(value, 1) for value in parcel_geom.bounds]
    building_bounds = [round(value, 1) for value in building_geom.bounds]
    backyard_bounds = rear_backyard_rect(parcel_bounds, building_bounds)
    backyard_geom = None
    if backyard_bounds is not None:
        backyard_geom = box(*backyard_bounds)
    record = {
        "id": property_id,
        "lga": lga,
        "suburb": suburb,
        "address": address,
        "parcel_rect": parcel_bounds,
        "building_rect": building_bounds,
        "elevation_corners_m": elevations,
        "vegetation_pct": round(vegetation_pct, 1),
        "road_frontage_m": round(road_frontage_m, 1),
        "corner_lot": corner_lot,
        "flood_risk": int(flood_risk),
        "heritage_overlay": bool(heritage_overlay),
        "overhead_powerline": bool(overhead_powerline),
        "has_existing_pool": bool(has_existing_pool),
        "exclude_from_pool_candidates": bool(has_existing_pool),
        "existing_pool_source": existing_pool_source if has_existing_pool else None,
        "solar_access_pct": 70.0,
        "view_quality": 5,
        "land_use": land_use,
        "zoning": zoning,
        "property_type": property_type,
        "ownership_class": ownership_class,
        "owner_name": owner_name,
        "tenure": tenure,
        "parcel_wkt": parcel_geom.wkt,
        "building_wkt": building_geom.wkt,
        "backyard_wkt": backyard_geom.wkt if backyard_geom is not None else None,
        "source_crs": source_crs,
    }
    features = derive_features(record)
    record["solar_access_pct"] = round(
        clamp(87.0 - (record["vegetation_pct"] * 0.32) - max(0.0, features["slope_pct"] - 6.0) * 1.4, 38.0, 92.0),
        1,
    )
    record["view_quality"] = int(round(clamp(4.0 + features["relief_m"] * 1.2 + features["elevation_max_m"] / 30.0, 3.0, 9.0)))
    return record


def build_property_records(
    parcels_path: str | Path,
    buildings_path: str | Path,
    dem_path: str | Path,
    *,
    output_path: str | Path | None = None,
    lga_name: str | None = None,
    limit: int | None = None,
    id_field: str | None = None,
    lga_field: str | None = None,
    suburb_field: str | None = None,
    address_field: str | None = None,
    land_use_field: str | None = None,
    zoning_field: str | None = None,
    property_type_field: str | None = None,
    ownership_field: str | None = None,
    owner_name_field: str | None = None,
    tenure_field: str | None = None,
    corner_lot_field: str | None = None,
    road_frontage_field: str | None = None,
    flood_field: str | None = None,
    heritage_field: str | None = None,
    overhead_powerline_field: str | None = None,
    pools_field: str | None = None,
    vegetation_path: str | Path | None = None,
    flood_path: str | Path | None = None,
    heritage_path: str | Path | None = None,
    powerline_path: str | Path | None = None,
    pools_path: str | Path | None = None,
    exclude_non_private: bool = True,
    target_epsg: int = DEFAULT_TARGET_EPSG,
) -> BuildResult:
    parcels = _read_layer(parcels_path, target_epsg=target_epsg)
    buildings = _read_layer(buildings_path, target_epsg=target_epsg)
    vegetation = _optional_read(vegetation_path, target_epsg=target_epsg)
    flood = _optional_read(flood_path, target_epsg=target_epsg)
    heritage = _optional_read(heritage_path, target_epsg=target_epsg)
    powerline = _optional_read(powerline_path, target_epsg=target_epsg)
    pools = _optional_read(pools_path, target_epsg=target_epsg)

    records: list[dict[str, Any]] = []
    skipped_without_buildings = 0
    skipped_invalid_geometry = 0
    skipped_non_private = 0

    for index, row in parcels.iterrows():
        if limit is not None and len(records) >= limit:
            break
        parcel_geom = row.geometry
        if parcel_geom is None or parcel_geom.is_empty or parcel_geom.area <= 0:
            skipped_invalid_geometry += 1
            continue
        building_parts = _candidate_buildings(parcel_geom, buildings)
        if not building_parts:
            skipped_without_buildings += 1
            continue
        building_geom = unary_union(building_parts)
        if building_geom.is_empty or building_geom.area <= 0:
            skipped_without_buildings += 1
            continue
        property_id = _coerce_string(row.get(id_field) if id_field else None, f"PROP-{index + 1:05d}")
        lga = _coerce_string(row.get(lga_field) if lga_field else None, lga_name or "Unknown LGA")
        suburb = _coerce_string(row.get(suburb_field) if suburb_field else None, "Unknown suburb")
        address = _coerce_string(row.get(address_field) if address_field else None, property_id)
        land_use = _coerce_string(row.get(land_use_field) if land_use_field else None, "")
        zoning = _coerce_string(row.get(zoning_field) if zoning_field else None, "")
        property_type = _coerce_string(row.get(property_type_field) if property_type_field else None, "")
        ownership_class = _coerce_string(row.get(ownership_field) if ownership_field else None, "")
        owner_name = _coerce_string(row.get(owner_name_field) if owner_name_field else None, "")
        tenure = _coerce_string(row.get(tenure_field) if tenure_field else None, "")
        corner_lot = _coerce_bool(row.get(corner_lot_field)) if corner_lot_field else False
        road_frontage_m = float(row.get(road_frontage_field)) if road_frontage_field and row.get(road_frontage_field) is not None else _estimate_frontage(parcel_geom)
        flood_risk = int(_coerce_bool(row.get(flood_field))) if flood_field else int(_intersects_any(parcel_geom, flood))
        heritage_overlay = _coerce_bool(row.get(heritage_field)) if heritage_field else _intersects_any(parcel_geom, heritage)
        overhead_powerline = _coerce_bool(row.get(overhead_powerline_field)) if overhead_powerline_field else _intersects_any(parcel_geom, powerline)
        has_existing_pool = _coerce_bool(row.get(pools_field)) if pools_field else _intersects_any(parcel_geom, pools)
        existing_pool_source = None
        if has_existing_pool:
            existing_pool_source = "parcel_attribute" if pools_field else "configured_pool_layer"
        vegetation_pct = _cover_percentage(parcel_geom, vegetation)
        elevations = _sample_dem_corners(dem_path, parcel_geom.bounds, parcels.crs)
        record = enrich_property_record(
            _base_record(
                property_id=property_id,
                lga=lga,
                suburb=suburb,
                address=address,
                parcel_geom=parcel_geom,
                building_geom=building_geom,
                elevations=elevations,
                vegetation_pct=vegetation_pct,
                road_frontage_m=road_frontage_m,
                corner_lot=corner_lot,
                flood_risk=flood_risk,
                heritage_overlay=heritage_overlay,
                overhead_powerline=overhead_powerline,
                source_crs=str(parcels.crs) if parcels.crs else f"EPSG:{target_epsg}",
                land_use=land_use,
                zoning=zoning,
                property_type=property_type,
                ownership_class=ownership_class,
                owner_name=owner_name,
                tenure=tenure,
                has_existing_pool=has_existing_pool,
                existing_pool_source=existing_pool_source,
            )
        )
        if exclude_non_private and not record["pool_fence_candidate"]:
            skipped_non_private += 1
            continue
        records.append(record)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(records, indent=2), encoding="utf-8")

    return BuildResult(
        records=records,
        skipped_without_buildings=skipped_without_buildings,
        skipped_invalid_geometry=skipped_invalid_geometry,
        skipped_non_private=skipped_non_private,
    )


def _summary(result: BuildResult, output_path: str | Path | None) -> dict[str, Any]:
    return {
        "records_written": len(result.records),
        "skipped_without_buildings": result.skipped_without_buildings,
        "skipped_invalid_geometry": result.skipped_invalid_geometry,
        "skipped_non_private": result.skipped_non_private,
        "output_path": str(output_path) if output_path else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build app-compatible property records from parcel, building, and DEM layers.")
    parser.add_argument("--parcels", required=True, help="Path to the parcel polygon layer.")
    parser.add_argument("--buildings", required=True, help="Path to the building polygon layer.")
    parser.add_argument("--dem", required=True, help="Path to a local DEM raster.")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "data" / "derived" / "property_features_v1.json"),
        help="Output JSON file path for the app-compatible property records.",
    )
    parser.add_argument("--lga-name", help="Fallback LGA name if the parcel layer does not carry one.")
    parser.add_argument("--limit", type=int, help="Optional max number of parcels to process.")
    parser.add_argument("--id-field", help="Parcel field containing a stable property or parcel id.")
    parser.add_argument("--lga-field", help="Parcel field containing the LGA name.")
    parser.add_argument("--suburb-field", help="Parcel field containing the suburb name.")
    parser.add_argument("--address-field", help="Parcel field containing the address label.")
    parser.add_argument("--land-use-field", help="Parcel field describing land use or site use.")
    parser.add_argument("--zoning-field", help="Parcel field describing zoning or planning zone.")
    parser.add_argument("--property-type-field", help="Parcel field describing the property or dwelling type.")
    parser.add_argument("--ownership-field", help="Parcel field describing ownership or tenure class.")
    parser.add_argument("--owner-name-field", help="Parcel field with owner or agency name.")
    parser.add_argument("--tenure-field", help="Parcel field describing tenure.")
    parser.add_argument("--corner-lot-field", help="Parcel field containing a corner-lot indicator.")
    parser.add_argument("--road-frontage-field", help="Parcel field containing road frontage metres.")
    parser.add_argument("--flood-field", help="Parcel field containing a flood indicator.")
    parser.add_argument("--heritage-field", help="Parcel field containing a heritage indicator.")
    parser.add_argument("--overhead-powerline-field", help="Parcel field containing a powerline indicator.")
    parser.add_argument("--pools-field", help="Parcel field containing an existing pool indicator.")
    parser.add_argument("--vegetation", help="Optional vegetation polygon layer used to estimate vegetation percentage.")
    parser.add_argument("--flood", help="Optional flood polygon layer used to derive a flood flag.")
    parser.add_argument("--heritage", help="Optional heritage polygon layer used to derive a heritage flag.")
    parser.add_argument("--powerlines", help="Optional powerline layer used to derive an overhead service flag.")
    parser.add_argument("--pools", help="Optional existing pools layer used to derive a has_existing_pool flag.")
    parser.add_argument(
        "--include-public",
        action="store_true",
        help="Keep parcels that are not verified as private residential sites. By default they are excluded.",
    )
    parser.add_argument("--target-epsg", type=int, default=DEFAULT_TARGET_EPSG, help="Projected CRS used for processing.")
    args = parser.parse_args(argv)

    result = build_property_records(
        parcels_path=args.parcels,
        buildings_path=args.buildings,
        dem_path=args.dem,
        output_path=args.output,
        lga_name=args.lga_name,
        limit=args.limit,
        id_field=args.id_field,
        lga_field=args.lga_field,
        suburb_field=args.suburb_field,
        address_field=args.address_field,
        land_use_field=args.land_use_field,
        zoning_field=args.zoning_field,
        property_type_field=args.property_type_field,
        ownership_field=args.ownership_field,
        owner_name_field=args.owner_name_field,
        tenure_field=args.tenure_field,
        corner_lot_field=args.corner_lot_field,
        road_frontage_field=args.road_frontage_field,
        flood_field=args.flood_field,
        heritage_field=args.heritage_field,
        overhead_powerline_field=args.overhead_powerline_field,
        pools_field=args.pools_field,
        vegetation_path=args.vegetation,
        flood_path=args.flood,
        heritage_path=args.heritage,
        powerline_path=args.powerlines,
        pools_path=args.pools,
        exclude_non_private=not args.include_public,
        target_epsg=args.target_epsg,
    )
    print(json.dumps(_summary(result, args.output), indent=2))


if __name__ == "__main__":
    main()
