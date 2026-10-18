"""Bootstrap a YOLO swimming-pool dataset from OpenStreetMap-mapped pools.

Generic COCO YOLO models cannot detect pools (no such class), so the hybrid
detector needs *pool-trained* weights. The pools already mapped in OSM are free,
high-quality labels: this script downloads a satellite chip around each mapped
pool, writes a YOLO bounding-box label from the pool geometry, and (optionally)
adds empty-label hard negatives so the model learns "no pool here" on roofs,
solar panels, and turf.

Example
-------
    python scripts/build_pool_dataset.py \
        --bbox -27.30 152.95 -27.20 153.10 \
        --out data/pool_dataset --zoom 20 --negatives 200

Then train with the existing pipeline::

    python scripts/pool_yolo_pipeline.py train \
        --data data/pool_dataset/dataset.yaml --weights yolo11s.pt

The trained weights (runs/.../best.pt) are picked up automatically by the app,
or point QLDR_POOL_YOLO_WEIGHTS at them.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
from pathlib import Path

# Reuse the app's tile + overpass machinery so geometry/zoom math matches the
# live detector exactly.
from aus_backyard_intelligence import yolo_detector as yd
from aus_backyard_intelligence.external_api import execute_overpass_query


# Class ids MUST match the shipped checkpoint (models/pool_yolo.pt: v8l-P2,
# names {0: 'solarpanel', 1: 'pool'}). Fine-tuning with a different class
# layout would reset the head and silently drop solar detection.
CLASS_SOLAR = 0
CLASS_POOL = 1


def _pool_elements(bbox: tuple[float, float, float, float], timeout: float) -> list[dict]:
    south, west, north, east = bbox
    query = (
        "[out:json][timeout:60];("
        f'way["leisure"="swimming_pool"]({south},{west},{north},{east});'
        f'relation["leisure"="swimming_pool"]({south},{west},{north},{east});'
        ");out geom;"
    )
    return execute_overpass_query(query, timeout=timeout).get("elements", [])


def _solar_elements(bbox: tuple[float, float, float, float], timeout: float) -> list[dict]:
    south, west, north, east = bbox
    query = (
        "[out:json][timeout:60];("
        f'way["power"="generator"]["generator:source"="solar"]({south},{west},{north},{east});'
        ");out geom;"
    )
    try:
        return execute_overpass_query(query, timeout=timeout).get("elements", [])
    except Exception:
        return []  # solar labels are a bonus; a failed query must not kill the build


def _polygon_latlon(element: dict) -> list[tuple[float, float]]:
    geometry = element.get("geometry") or []
    return [(float(p["lat"]), float(p["lon"])) for p in geometry if "lat" in p and "lon" in p]


def _chip_bbox(lat: float, lon: float, half_span_deg: float, jitter: float) -> tuple[float, float, float, float]:
    # Random jitter so pools are not always dead-centre (reduces positional bias).
    dlat = (random.random() - 0.5) * 2 * jitter
    dlon = (random.random() - 0.5) * 2 * jitter
    return (
        lat - half_span_deg + dlat,
        lon - half_span_deg + dlon,
        lat + half_span_deg + dlat,
        lon + half_span_deg + dlon,
    )


def _yolo_label_for_polygon(
    polygon: list[tuple[float, float]],
    actual_bbox: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    class_id: int = CLASS_POOL,
) -> str | None:
    south, west, north, east = actual_bbox
    if north <= south or east <= west:
        return None
    xs = [(lon - west) / (east - west) * img_w for _lat, lon in polygon]
    ys = [(north - lat) / (north - south) * img_h for lat, _lon in polygon]
    x0, x1 = max(0.0, min(xs)), min(float(img_w), max(xs))
    y0, y1 = max(0.0, min(ys)), min(float(img_h), max(ys))
    bw, bh = x1 - x0, y1 - y0
    if bw < 4 or bh < 4:
        return None  # feature fell off the chip or is degenerate
    cx = (x0 + x1) / 2.0 / img_w
    cy = (y0 + y1) / 2.0 / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {bw / img_w:.6f} {bh / img_h:.6f}"


def _polygon_intersects_bbox(
    polygon: list[tuple[float, float]],
    bbox: tuple[float, float, float, float],
) -> bool:
    """Return whether a feature's bounds touch the stitched image bounds.

    This intentionally uses an inclusive bounding-box intersection. A false
    positive here merely discards a proposed negative chip; a false negative
    would silently teach the model that a real pool or solar array is
    background.
    """
    if not polygon:
        return False
    south, west, north, east = bbox
    if north < south or east < west:
        return False
    feature_south = min(lat for lat, _lon in polygon)
    feature_north = max(lat for lat, _lon in polygon)
    feature_west = min(lon for _lat, lon in polygon)
    feature_east = max(lon for _lat, lon in polygon)
    return not (
        feature_north < south
        or feature_south > north
        or feature_east < west
        or feature_west > east
    )


def _known_feature_intersects_bbox(
    features: list[tuple[list[tuple[float, float]], int]],
    bbox: tuple[float, float, float, float],
) -> bool:
    """Guard negative mining against every known pool and solar feature."""
    return any(_polygon_intersects_bbox(polygon, bbox) for polygon, _class_id in features)


def _labels_for_visible_features(
    features: list[tuple[list[tuple[float, float]], int]],
    actual_bbox: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> list[str]:
    """Build labels for all sufficiently visible known features in a chip."""
    labels: list[str] = []
    for polygon, class_id in features:
        label = _yolo_label_for_polygon(polygon, actual_bbox, img_w, img_h, class_id)
        if label is not None:
            labels.append(label)
    return labels


def _spatial_split_for_bbox(
    actual_bbox: tuple[float, float, float, float],
    *,
    val_frac: float,
    seed: int,
    group_span_deg: float,
) -> str:
    """Assign a stitched chip to a stable geographic train/validation group.

    Python's built-in ``hash`` is process-randomised, so use SHA-256 over the
    coarse spatial cell. Identical stitched bboxes, and nearby bboxes whose
    centres occupy the same cell, can never be randomly split across train and
    validation.
    """
    if val_frac <= 0:
        return "train"
    if val_frac >= 1:
        return "val"
    if not math.isfinite(group_span_deg) or group_span_deg <= 0:
        raise ValueError("group_span_deg must be a positive finite number")
    south, west, north, east = actual_bbox
    center_lat = (south + north) / 2.0
    center_lon = (west + east) / 2.0
    group_lat = math.floor(center_lat / group_span_deg)
    group_lon = math.floor(center_lon / group_span_deg)
    digest = hashlib.sha256(f"{int(seed)}:{group_lat}:{group_lon}".encode("ascii")).digest()
    sample = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "val" if sample < val_frac else "train"


def _save(img, label_text: str, stem: str, images_dir: Path, labels_dir: Path) -> bool:
    yd._lazy_import()
    image_path = images_dir / f"{stem}.jpg"
    if yd._HAS_CV2 and yd._cv2 is not None:
        yd._cv2.imwrite(str(image_path), img)
    elif yd._Image is not None:
        yd._Image.fromarray(img[:, :, ::-1]).save(image_path, quality=90)
    else:
        return False
    (labels_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")
    return True


def build(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    yd._lazy_import()
    if yd._np is None or not (yd._HAS_CV2 or yd._Image is not None):
        raise SystemExit("NumPy and OpenCV/Pillow are required to build the dataset.")

    bbox = tuple(args.bbox)
    out = Path(args.out)
    splits = {"train": (out / "images/train", out / "labels/train"),
              "val": (out / "images/val", out / "labels/val")}
    for img_dir, lbl_dir in splits.values():
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    pools = _pool_elements(bbox, args.timeout)
    solar = _solar_elements(bbox, args.timeout)
    print(f"OSM pools in bbox: {len(pools)}; solar generators: {len(solar)}")
    half_span = args.chip_m / 111_320.0  # metres -> degrees (lat); good enough for chips
    jitter = half_span * 0.4
    split_group_m = float(getattr(args, "split_group_m", 250.0))
    if not math.isfinite(split_group_m) or split_group_m <= 0:
        raise ValueError("split_group_m must be a positive finite number")
    split_group_deg = split_group_m / 111_320.0

    features: list[tuple[list[tuple[float, float]], int]] = []
    for el in pools:
        poly = _polygon_latlon(el)
        if len(poly) >= 3:
            features.append((poly, CLASS_POOL))
    for el in solar:
        poly = _polygon_latlon(el)
        if len(poly) >= 3:
            features.append((poly, CLASS_SOLAR))
    centres = [
        (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))
        for poly, _cid in features
    ]

    positives = 0
    for idx, (poly, _cid) in enumerate(features):
        lat, lon = centres[idx]
        chip = _chip_bbox(lat, lon, half_span, jitter)
        res = yd._stitch_tiles_with_stats(chip, zoom=args.zoom)
        if not res:
            continue
        img, actual_bbox, _ = res
        h, w = img.shape[:2]
        # Label EVERY feature visible in this chip, not just the anchor — an
        # unlabeled pool in frame is treated as background by the trainer and
        # teaches the model to suppress real pools (dense suburbs cluster them).
        lines = _labels_for_visible_features(features, actual_bbox, w, h)
        if not lines:
            continue
        split = _spatial_split_for_bbox(
            actual_bbox,
            val_frac=args.val_frac,
            seed=args.seed,
            group_span_deg=split_group_deg,
        )
        if _save(img, "\n".join(lines), f"chip_{idx:05d}", *splits[split]):
            positives += 1
    print(f"positive chips written: {positives}")

    negatives = 0
    south, west, north, east = bbox
    attempts = 0
    while negatives < args.negatives and attempts < args.negatives * 25:
        attempts += 1
        lat = south + random.random() * (north - south)
        lon = west + random.random() * (east - west)
        # Cheap pre-download screen. The stitched tile bounds are larger than
        # the requested chip, so a second authoritative intersection check is
        # required after the imagery provider snaps the request to its tile grid.
        if any(yd._latlon_distance_m(lat, lon, plat, plon) < args.chip_m * 1.5 for plat, plon in centres):
            continue
        res = yd._stitch_tiles_with_stats(_chip_bbox(lat, lon, half_span, 0.0), zoom=args.zoom)
        if not res:
            continue
        img, actual_bbox, _ = res
        # Never create an empty label when a known pool or solar footprint is
        # anywhere in the actual stitched image. Even a tiny/edge intersection
        # is grounds to discard the proposed negative rather than poison it.
        if _known_feature_intersects_bbox(features, actual_bbox):
            continue
        split = _spatial_split_for_bbox(
            actual_bbox,
            val_frac=args.val_frac,
            seed=args.seed,
            group_span_deg=split_group_deg,
        )
        if _save(img, "", f"neg_{negatives:05d}", *splits[split]):
            negatives += 1
    print(f"negative chips written: {negatives}")

    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: solarpanel\n  1: pool\n",
        encoding="utf-8",
    )
    print(f"dataset.yaml -> {yaml_path}")
    print("Next: python scripts/pool_yolo_pipeline.py train "
          f"--data {yaml_path} --weights models/pool_yolo.pt")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a YOLO pool dataset from OSM-mapped pools.")
    p.add_argument("--bbox", type=float, nargs=4, metavar=("S", "W", "N", "E"), required=True,
                   help="Region to harvest pools from: south west north east (WGS84).")
    p.add_argument("--out", default="data/pool_dataset", help="Output dataset directory.")
    p.add_argument("--zoom", type=int, default=20, help="Satellite tile zoom (19-20).")
    p.add_argument("--chip-m", type=float, default=45.0, help="Half-size of each chip in metres.")
    p.add_argument("--negatives", type=int, default=150, help="Number of pool-free hard negatives.")
    p.add_argument("--val-frac", type=float, default=0.15, help="Validation split fraction.")
    p.add_argument(
        "--split-group-m",
        type=float,
        default=250.0,
        help="Geographic grouping width for deterministic train/validation assignment.",
    )
    p.add_argument("--timeout", type=float, default=60.0, help="Overpass timeout seconds.")
    p.add_argument("--seed", type=int, default=7, help="RNG seed for reproducibility.")
    p.set_defaults(func=build)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
