"""Ground-truth evaluation harness and detection-feedback store.

Turns "screenshot vibes" into measured precision/recall:

* **Labels** — point labels persisted in ``eval_data/labels.json``. OSM bootstrap and
  product feedback are development data only. Publishable holdout labels must be
  independently hand-labelled and explicitly imported with ``split=holdout``.

* **Evaluation** — runs the satellite detector over the labelled areas and matches
  detections to labels by distance. Mapped OSM pools returned by the pipeline are
  EXCLUDED from matching: the question is whether the *detector* finds the pool, not
  whether OSM already knew about it. Metrics reported:
    - ``recall``            detected pool labels / all pool labels covered
    - ``fp_rate``           violated no_pool labels / all no_pool labels covered
    - ``precision_proxy``   TP / (TP + FP)  (a proxy: only labelled ground is judged)
  Reports include source/city composition, Wilson confidence intervals, per-city
  metrics, and a marketing-eligibility gate. They persist to
  ``eval_data/eval_report.json`` (+ append-only history).

* **Feedback** — every ✓/✗ from the map UI is appended to ``eval_data/feedback.jsonl``
  and mirrored into the label store, so the eval set grows with real usage and the
  JSONL doubles as a fine-tuning dataset for the pool YOLO model later.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from .external_api import build_overpass_query, execute_overpass_query, fetch_osm_features

PACKAGE_ROOT = Path(__file__).resolve().parent
EVAL_DIR = Path(os.environ.get("QLDR_EVAL_DIR", str(PACKAGE_ROOT.parent / "eval_data")))
LABELS_PATH = EVAL_DIR / "labels.json"
REPORT_PATH = EVAL_DIR / "eval_report.json"
HISTORY_PATH = EVAL_DIR / "eval_history.jsonl"
FEEDBACK_PATH = EVAL_DIR / "feedback.jsonl"
# Auto-harvested fine-tuning candidates (pseudo-labels). Deliberately a SEPARATE
# file from labels.json: eval ground truth stays OSM + human feedback only, so
# the model can never grade its own homework.
TRAINING_PATH = EVAL_DIR / "training_candidates.jsonl"

# Matching distances (metres).
POOL_MATCH_M = 15.0          # a detection this close to a pool label counts as found
NO_POOL_VIOLATION_M = 20.0   # a detection this close to a no_pool label counts as FP
LABEL_DEDUPE_M = 8.0         # labels closer than this are considered the same spot
NEGATIVE_CLEARANCE_M = 40.0  # a building must be this far from any mapped pool to be a negative

# Bootstrap size guards.
MAX_POSITIVES_PER_RUN = 400
MAX_NEGATIVES_PER_RUN = 500

# Evaluation batching: labels are grouped into grid cells that are each scanned with one
# detector call, so runtime is bounded and every label's ground actually gets scanned.
EVAL_CELL_DEG = 0.004        # ~440 m cells
EVAL_CELL_PAD_DEG = 0.0006   # ~65 m pad so edge labels are fully covered
DEFAULT_MAX_EVAL_CELLS = 12

# A national headline requires independent holdout ground truth across multiple
# metros. These are deliberately minimum evidence gates, not accuracy targets.
MIN_HOLDOUT_METROS = 3
MIN_HOLDOUT_LABELS_PER_METRO = 150
MIN_SOLAR_POSITIVES = 100
MIN_SOLAR_NEGATIVES = 100
DEVELOPMENT_AREAS = {"pitch-bench"}
VALID_LABELS = {"pool", "no_pool", "solar", "no_solar"}
VALID_SPLITS = {"development", "holdout"}

_lock = threading.Lock()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _label_split(label: dict[str, Any]) -> str:
    """Legacy labels predate explicit splits and are development data."""
    return "holdout" if label.get("split") == "holdout" else "development"


def _count_by(labels: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(label.get(key) or "(blank)") for label in labels)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _wilson_interval(successes: int, total: int) -> dict[str, Any] | None:
    """95% Wilson score interval for a binomial rate."""
    if total <= 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + (z * z / total)
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) / total) + z * z / (4 * total * total)) / denominator
    return {
        "low": round(max(0.0, centre - margin), 3),
        "high": round(min(1.0, centre + margin), 3),
        "confidence": 0.95,
        "method": "Wilson score",
        "n": total,
    }


def dataset_profile(labels: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(labels)
    osm_count = sum(1 for label in labels if label.get("source") == "osm-bootstrap")
    return {
        "total": total,
        "by_source": _count_by(labels, "source"),
        "by_area": _count_by(labels, "area"),
        "by_label": _count_by(labels, "label"),
        "by_split": dict(Counter(_label_split(label) for label in labels)),
        "osm_bootstrap_share": round(osm_count / total, 3) if total else 0.0,
    }


def _marketing_audit(
    labels: list[dict[str, Any]],
    *,
    evaluation_split: str,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    if evaluation_split != "holdout":
        blockers.append("This is a development-set run, not an untouched holdout evaluation.")
    if any(label.get("source") == "osm-bootstrap" for label in labels):
        blockers.append("OSM-bootstrap labels measure pools already represented in OSM, not recall against reality.")
    if any(str(label.get("area") or "").strip().lower() in DEVELOPMENT_AREAS for label in labels):
        blockers.append("The evaluated subset includes pitch-bench, which is excluded from marketing metrics.")
    if any(not str(label.get("area") or "").strip() for label in labels):
        blockers.append("Some evaluated labels have no city/area provenance.")
    if any(_label_split(label) != "holdout" for label in labels):
        blockers.append("Every marketing label must carry split=holdout.")
    if any(label.get("source") != "human-holdout" for label in labels):
        blockers.append("Every marketing label must be independently hand-labelled (source=human-holdout).")

    metro_label_counts: dict[str, Counter[str]] = {}
    for label in labels:
        area = str(label.get("area") or "").strip()
        label_kind = str(label.get("label") or "")
        if not area or area.lower() in DEVELOPMENT_AREAS:
            continue
        metro_label_counts.setdefault(area, Counter())[label_kind] += 1
    qualified_metros = sorted(
        area
        for area, counts in metro_label_counts.items()
        if counts["pool"] >= MIN_HOLDOUT_LABELS_PER_METRO
        and counts["no_pool"] >= MIN_HOLDOUT_LABELS_PER_METRO
    )
    if len(qualified_metros) < MIN_HOLDOUT_METROS:
        blockers.append(
            f"Need at least {MIN_HOLDOUT_METROS} holdout metros with "
            f"{MIN_HOLDOUT_LABELS_PER_METRO}+ pool and "
            f"{MIN_HOLDOUT_LABELS_PER_METRO}+ no-pool labels each; found {len(qualified_metros)}."
        )
    if errors:
        blockers.append("One or more labelled cells failed during evaluation.")

    solar_positives = sum(1 for label in labels if label.get("label") == "solar")
    solar_negatives = sum(1 for label in labels if label.get("label") == "no_solar")
    solar_blockers: list[str] = []
    if solar_positives < MIN_SOLAR_POSITIVES:
        solar_blockers.append(f"Need {MIN_SOLAR_POSITIVES}+ hand-labelled solar positives; found {solar_positives}.")
    if solar_negatives < MIN_SOLAR_NEGATIVES:
        solar_blockers.append(f"Need {MIN_SOLAR_NEGATIVES}+ hand-labelled no-solar negatives; found {solar_negatives}.")
    if len(qualified_metros) < MIN_HOLDOUT_METROS:
        solar_blockers.append(
            f"Solar validation must also cover at least {MIN_HOLDOUT_METROS} qualified metros."
        )
    solar_shortfalls = []
    for area in qualified_metros:
        counts = metro_label_counts[area]
        if counts["solar"] < MIN_SOLAR_POSITIVES or counts["no_solar"] < MIN_SOLAR_NEGATIVES:
            solar_shortfalls.append(
                f"{area} ({counts['solar']} solar, {counts['no_solar']} no-solar)"
            )
    if solar_shortfalls:
        solar_blockers.append(
            "Every qualified metro needs "
            f"{MIN_SOLAR_POSITIVES}+ hand-labelled solar positives and "
            f"{MIN_SOLAR_NEGATIVES}+ hand-labelled no-solar negatives; shortfalls: "
            + "; ".join(solar_shortfalls)
            + "."
        )

    return {
        "pool_national_claim_eligible": not blockers,
        "solar_claim_eligible": not blockers and not solar_blockers,
        "claim_scope": "national_holdout" if not blockers else "development_only",
        "qualified_metros": qualified_metros,
        "blockers": blockers,
        "solar_blockers": solar_blockers,
    }


def _distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat_mid = math.radians((a[0] + b[0]) / 2.0)
    dy = (a[0] - b[0]) * 111_320.0
    dx = (a[1] - b[1]) * 111_320.0 * max(math.cos(lat_mid), 1e-9)
    return math.hypot(dx, dy)


def _centroid(element: dict[str, Any]) -> tuple[float, float] | None:
    if element.get("type") == "node" and element.get("lat") is not None:
        return float(element["lat"]), float(element["lon"])
    geometry = element.get("geometry")
    if isinstance(geometry, list):
        pts = [(float(p["lat"]), float(p["lon"])) for p in geometry if isinstance(p, dict) and "lat" in p]
        if pts:
            return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    return None


def _is_mapped_pool(element: dict[str, Any]) -> bool:
    tags = element.get("tags") or {}
    if str(tags.get("detector:source", "")).startswith("satellite"):
        return False
    return tags.get("leisure") == "swimming_pool" or tags.get("water") in {"pool", "swimming_pool"}


def _is_solar_detection(element: dict[str, Any]) -> bool:
    tags = element.get("tags") or {}
    return tags.get("detector:source") == "satellite-yolo-solar"


def _is_satellite_detection(element: dict[str, Any]) -> bool:
    """Satellite POOL detection. Solar arrays share the satellite- prefix but
    must never count for/against pool labels — keep the types separate."""
    if _is_solar_detection(element):
        return False
    tags = element.get("tags") or {}
    source = str(tags.get("detector:source", ""))
    if source.startswith("satellite") or source == "yolo-cv":
        return True
    # Mapped pool the detector independently found: the merge drops the
    # satellite box (UI dedupe) but stamps the kept pool with detector:satellite.
    # Counting the stamp keeps measured recall independent of Overpass health —
    # without it, every detection of an already-mapped pool disappears from the
    # merged output whenever Overpass is up, and the metric measures Overpass
    # availability instead of the detector.
    return bool(tags.get("detector:satellite"))


# ---------------------------------------------------------------------------
# Label store
# ---------------------------------------------------------------------------

def load_labels() -> list[dict[str, Any]]:
    try:
        raw = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, list) else []
    except (OSError, ValueError):
        return []


def _save_labels(labels: list[dict[str, Any]]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(json.dumps(labels, indent=1), encoding="utf-8")


def upsert_label(
    labels: list[dict[str, Any]],
    *,
    lat: float,
    lon: float,
    label: str,
    source: str,
    area: str = "",
    split: str = "development",
) -> bool:
    """Add a label unless one already exists at the same spot.

    Independently hand-labelled holdout data has the highest precedence and cannot be
    silently changed by in-product feedback. Feedback can still correct OSM bootstrap.
    """
    if label not in VALID_LABELS:
        raise ValueError(f"label must be one of {sorted(VALID_LABELS)}")
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}")
    if split == "holdout" and source != "human-holdout":
        raise ValueError("holdout labels must use source=human-holdout")
    priority = {"osm-bootstrap": 0, "feedback": 1, "human-holdout": 2}
    for existing in labels:
        if _distance_m((lat, lon), (existing["lat"], existing["lon"])) <= LABEL_DEDUPE_M:
            existing_priority = priority.get(str(existing.get("source")), 0)
            incoming_priority = priority.get(source, 0)
            if incoming_priority > existing_priority:
                existing.update({
                    "label": label,
                    "source": source,
                    "area": area,
                    "split": split,
                    "updated_at": _now_iso(),
                })
                return True
            return False
    labels.append(
        {
            "id": f"{source}-{uuid.uuid4().hex[:10]}",
            "lat": round(float(lat), 7),
            "lon": round(float(lon), 7),
            "label": label,
            "source": source,
            "area": area,
            "split": split,
            "created_at": _now_iso(),
        }
    )
    return True


def label_counts(labels: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(labels),
        "pool": sum(1 for label in labels if label.get("label") == "pool"),
        "no_pool": sum(1 for label in labels if label.get("label") == "no_pool"),
        "solar": sum(1 for label in labels if label.get("label") == "solar"),
        "no_solar": sum(1 for label in labels if label.get("label") == "no_solar"),
        "from_feedback": sum(1 for label in labels if label.get("source") == "feedback"),
        "holdout": sum(1 for label in labels if _label_split(label) == "holdout"),
    }


def import_holdout_labels(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Import independently hand-labelled, city-attributed holdout points."""
    if not isinstance(records, list) or not 1 <= len(records) <= 5000:
        raise ValueError("records must contain between 1 and 5000 holdout labels")
    cleaned: list[tuple[float, float, str, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"records[{index}] must be an object")
        try:
            lat = float(record.get("lat"))
            lon = float(record.get("lon"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"records[{index}] has invalid lat/lon") from exc
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"records[{index}] has out-of-range lat/lon")
        label = str(record.get("label") or "").strip().lower()
        if label not in VALID_LABELS:
            raise ValueError(f"records[{index}].label must be one of {sorted(VALID_LABELS)}")
        area = str(record.get("area") or "").strip()[:80]
        if not area:
            raise ValueError(f"records[{index}].area is required for holdout provenance")
        if area.lower() in DEVELOPMENT_AREAS:
            raise ValueError(f"records[{index}].area is reserved for development benchmarking")
        cleaned.append((lat, lon, label, area))

    with _lock:
        labels = load_labels()
        added = 0
        for lat, lon, label, area in cleaned:
            if upsert_label(
                labels,
                lat=lat,
                lon=lon,
                label=label,
                source="human-holdout",
                area=area,
                split="holdout",
            ):
                added += 1
        _save_labels(labels)
    return {"added": added, "submitted": len(cleaned), "counts": label_counts(labels)}


# ---------------------------------------------------------------------------
# Bootstrap from OSM
# ---------------------------------------------------------------------------

def bootstrap_labels(
    bbox: tuple[float, float, float, float],
    *,
    area: str = "",
    timeout: float = 28.0,
) -> dict[str, Any]:
    """Seed the label set for a bbox: OSM-mapped pools become ``pool`` positives and
    residential buildings well clear of any mapped pool become ``no_pool`` negatives."""
    from .scan_agent import _is_residential_building  # local import avoids cycle

    result = execute_overpass_query(
        build_overpass_query(bbox, zoom=18, layers="pools,buildings"),
        timeout=timeout,
    )
    elements = [e for e in result.get("elements", []) if isinstance(e, dict)]

    pool_centers: list[tuple[float, float]] = []
    for element in elements:
        if _is_mapped_pool(element):
            center = _centroid(element)
            if center:
                pool_centers.append(center)

    with _lock:
        labels = load_labels()
        added_pos = added_neg = 0
        for center in pool_centers[:MAX_POSITIVES_PER_RUN]:
            if upsert_label(labels, lat=center[0], lon=center[1], label="pool", source="osm-bootstrap", area=area):
                added_pos += 1
        for element in elements:
            if added_neg >= MAX_NEGATIVES_PER_RUN or added_neg >= 2 * max(1, len(pool_centers)):
                break
            if not _is_residential_building(element):
                continue
            center = _centroid(element)
            if center is None:
                continue
            if any(_distance_m(center, pc) <= NEGATIVE_CLEARANCE_M for pc in pool_centers):
                continue
            if upsert_label(labels, lat=center[0], lon=center[1], label="no_pool", source="osm-bootstrap", area=area):
                added_neg += 1
        # OSM-mapped rooftop solar generators give free ground truth for the
        # solar detector too. Runs LAST so a solar label can never block a
        # pool/no_pool label at the same spot (one label per ~8 m).
        added_solar = 0
        try:
            south, west, north, east = bbox
            solar_query = (
                '[out:json][timeout:20];('
                f'way["power"="generator"]["generator:source"="solar"]({south},{west},{north},{east});'
                f'node["power"="generator"]["generator:source"="solar"]({south},{west},{north},{east});'
                ');out geom;'
            )
            solar_result = execute_overpass_query(solar_query, timeout=timeout)
            for element in solar_result.get("elements", []):
                if not isinstance(element, dict):
                    continue
                center = _centroid(element)
                if center and upsert_label(
                    labels, lat=center[0], lon=center[1], label="solar", source="osm-bootstrap", area=area
                ):
                    added_solar += 1
        except Exception:
            pass
        _save_labels(labels)
        counts = label_counts(labels)

    return {
        "added_pool_labels": added_pos,
        "added_no_pool_labels": added_neg,
        "added_solar_labels": added_solar,
        "note": (
            "Negatives are OSM-derived and noisy (an unmapped pool looks like a negative). "
            "Use the map's dev-mode ✓/✗ feedback to correct them; feedback labels override."
        ),
        "counts": counts,
    }


# ---------------------------------------------------------------------------
# Automatic training-data pipeline
# ---------------------------------------------------------------------------

def record_training_candidates(elements: list[dict[str, Any]], *, source: str) -> dict[str, int]:
    """Auto-harvest high-trust detections as fine-tuning candidates.

    Only scan-VERIFIED pools (confirmed verdict, seen in 2+ passes) and
    confident solar arrays qualify — one-shot realtime boxes are too noisy for
    pseudo-labels. Appends to training_candidates.jsonl with an 8 m dedupe.
    """
    added = {"pool": 0, "solar": 0}
    if not elements:
        return added
    with _lock:
        existing: list[tuple[float, float]] = []
        if TRAINING_PATH.exists():
            for line in TRAINING_PATH.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    existing.append((float(row["lat"]), float(row["lon"])))
                except (ValueError, KeyError, TypeError):
                    continue
        out_lines: list[str] = []
        for element in elements:
            if not isinstance(element, dict):
                continue
            tags = element.get("tags") or {}
            centre = _centroid(element)
            if centre is None:
                continue
            kind = None
            conf = 0.0
            if tags.get("leisure") == "swimming_pool" and str(tags.get("detector:source", "")).startswith("satellite"):
                try:
                    seen = int(str(tags.get("scan:passes_seen", "0")) or 0)
                except ValueError:
                    seen = 0
                try:
                    conf = float(tags.get("scan:confidence") or tags.get("confidence:yolo") or 0.0)
                except (TypeError, ValueError):
                    conf = 0.0
                if tags.get("scan:verdict") == "confirmed" and seen >= 2 and conf >= 0.6:
                    kind = "pool"
            elif tags.get("detector:source") == "satellite-yolo-solar":
                try:
                    conf = float(tags.get("confidence:yolo") or 0.0)
                except (TypeError, ValueError):
                    conf = 0.0
                if conf >= 0.7:
                    kind = "solar"
            if kind is None:
                continue
            if any(_distance_m(centre, spot) <= LABEL_DEDUPE_M for spot in existing):
                continue
            existing.append(centre)
            out_lines.append(json.dumps({
                "ts": _now_iso(),
                "kind": kind,
                "lat": round(centre[0], 7),
                "lon": round(centre[1], 7),
                "confidence": round(conf, 2),
                "source": source,
            }))
            added[kind] += 1
        if out_lines:
            EVAL_DIR.mkdir(parents=True, exist_ok=True)
            with TRAINING_PATH.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(out_lines) + "\n")
    return added


def export_training_candidates() -> str:
    with _lock:
        if not TRAINING_PATH.exists():
            return ""
        return TRAINING_PATH.read_text(encoding="utf-8")


def pipeline_status() -> dict[str, Any]:
    """One-glance dataset health for the dev-mode auto-pipeline line."""
    labels = load_labels()
    training = {"pool": 0, "solar": 0, "total": 0}
    with _lock:
        if TRAINING_PATH.exists():
            for line in TRAINING_PATH.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                training["total"] += 1
                kind = str(row.get("kind", ""))
                if kind in training:
                    training[kind] += 1
    stats = feedback_stats()
    return {
        "labels": {
            "total": len(labels),
            "pool": sum(1 for label in labels if label.get("label") == "pool"),
            "no_pool": sum(1 for label in labels if label.get("label") == "no_pool"),
            "solar": sum(1 for label in labels if label.get("label") == "solar"),
            "no_solar": sum(1 for label in labels if label.get("label") == "no_solar"),
            "from_feedback": sum(1 for label in labels if label.get("source") == "feedback"),
            "holdout": sum(1 for label in labels if _label_split(label) == "holdout"),
        },
        "dataset_profile": dataset_profile(labels),
        "marketing_audit": _marketing_audit(labels, evaluation_split="development"),
        "training_candidates": training,
        "feedback_total": stats.get("total", 0),
    }


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def _label_cells(labels: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    # floor, NOT int(): int() truncates toward zero, which for southern-
    # hemisphere (negative) latitudes shifted every label one cell north —
    # the evaluation then scanned bboxes that did not contain their own
    # labels and reported recall against the wrong ground.
    cells: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for label in labels:
        key = (
            math.floor(label["lat"] / EVAL_CELL_DEG),
            math.floor(label["lon"] / EVAL_CELL_DEG),
        )
        cells.setdefault(key, []).append(label)
    return cells


def _score_labels(
    labels: list[dict[str, Any]],
    detections: list[tuple[float, float]],
    solar_detections: list[tuple[float, float]],
) -> dict[str, Any]:
    pool_labels = [label for label in labels if label.get("label") == "pool"]
    no_pool_labels = [label for label in labels if label.get("label") == "no_pool"]
    solar_labels = [label for label in labels if label.get("label") == "solar"]
    no_solar_labels = [label for label in labels if label.get("label") == "no_solar"]

    tp = sum(
        1 for label in pool_labels
        if any(_distance_m((label["lat"], label["lon"]), detection) <= POOL_MATCH_M for detection in detections)
    )
    fn = len(pool_labels) - tp
    fp = sum(
        1 for label in no_pool_labels
        if any(
            _distance_m((label["lat"], label["lon"]), detection) <= NO_POOL_VIOLATION_M
            for detection in detections
        )
    )
    tn = len(no_pool_labels) - fp
    solar_tp = sum(
        1 for label in solar_labels
        if any(
            _distance_m((label["lat"], label["lon"]), detection) <= POOL_MATCH_M
            for detection in solar_detections
        )
    )
    solar_fn = len(solar_labels) - solar_tp
    solar_fp = sum(
        1 for label in no_solar_labels
        if any(
            _distance_m((label["lat"], label["lon"]), detection) <= NO_POOL_VIOLATION_M
            for detection in solar_detections
        )
    )
    solar_tn = len(no_solar_labels) - solar_fp

    recall = tp / len(pool_labels) if pool_labels else None
    fp_rate = fp / len(no_pool_labels) if no_pool_labels else None
    precision_proxy = tp / (tp + fp) if (tp + fp) else None
    f1_proxy = (
        2 * precision_proxy * recall / (precision_proxy + recall)
        if precision_proxy and recall
        else None
    )
    solar_recall = solar_tp / len(solar_labels) if solar_labels else None
    solar_fp_rate = solar_fp / len(no_solar_labels) if no_solar_labels else None
    solar_precision_proxy = solar_tp / (solar_tp + solar_fp) if (solar_tp + solar_fp) else None
    return {
        "labels": len(labels),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": round(recall, 3) if recall is not None else None,
        "recall_ci_95": _wilson_interval(tp, len(pool_labels)),
        "fp_rate": round(fp_rate, 3) if fp_rate is not None else None,
        "fp_rate_ci_95": _wilson_interval(fp, len(no_pool_labels)),
        "precision_proxy": round(precision_proxy, 3) if precision_proxy is not None else None,
        "f1_proxy": round(f1_proxy, 3) if f1_proxy is not None else None,
        "solar_labels": len(solar_labels),
        "solar_tp": solar_tp,
        "solar_fn": solar_fn,
        "solar_recall": round(solar_recall, 3) if solar_recall is not None else None,
        "solar_recall_ci_95": _wilson_interval(solar_tp, len(solar_labels)),
        "no_solar_labels": len(no_solar_labels),
        "solar_fp": solar_fp,
        "solar_tn": solar_tn,
        "solar_fp_rate": round(solar_fp_rate, 3) if solar_fp_rate is not None else None,
        "solar_fp_rate_ci_95": _wilson_interval(solar_fp, len(no_solar_labels)),
        "solar_precision_proxy": (
            round(solar_precision_proxy, 3) if solar_precision_proxy is not None else None
        ),
    }


def run_evaluation(
    *,
    data_path: str,
    detector_model: str = "yolov11s",
    max_cells: int = DEFAULT_MAX_EVAL_CELLS,
    build_tag: str = "",
    evaluation_split: str = "development",
) -> dict[str, Any]:
    """Scan every labelled cell (up to ``max_cells``, densest first) with the precise
    detector and score detections against the labels. Returns and persists the report."""
    if evaluation_split not in VALID_SPLITS:
        raise ValueError(f"evaluation_split must be one of {sorted(VALID_SPLITS)}")
    all_labels = load_labels()
    if not all_labels:
        return {"error": "No labels yet. Import independent holdout labels or bootstrap development labels."}
    labels = [label for label in all_labels if _label_split(label) == evaluation_split]
    if not labels:
        return {
            "error": f"No {evaluation_split} labels are available.",
            "dataset_profile": dataset_profile(all_labels),
            "marketing_audit": _marketing_audit([], evaluation_split=evaluation_split),
        }

    cells = _label_cells(labels)
    ordered = sorted(cells.items(), key=lambda kv: -len(kv[1]))[: max(1, int(max_cells))]

    detections: list[tuple[float, float]] = []
    solar_detections: list[tuple[float, float]] = []
    covered_labels: list[dict[str, Any]] = []
    cells_scanned = 0
    errors: list[str] = []
    for (cy, cx), cell_labels in ordered:
        south = cy * EVAL_CELL_DEG - EVAL_CELL_PAD_DEG
        west = cx * EVAL_CELL_DEG - EVAL_CELL_PAD_DEG
        north = (cy + 1) * EVAL_CELL_DEG + EVAL_CELL_PAD_DEG
        east = (cx + 1) * EVAL_CELL_DEG + EVAL_CELL_PAD_DEG
        try:
            result = fetch_osm_features(
                (south, west, north, east),
                zoom=20,
                layers="pools",
                timeout=8.0,
                data_path=data_path,
                detector_model=detector_model,
                detector_precision="precise",
                analysis_polygon=[
                    [west, south],
                    [east, south],
                    [east, north],
                    [west, north],
                ],
                force_refresh=True,
            )
        except Exception as exc:  # network/detector hiccups shouldn't void the run
            errors.append(f"cell {cy},{cx}: {exc}")
            continue
        cells_scanned += 1
        covered_labels.extend(cell_labels)
        for element in result.get("elements", []):
            if not isinstance(element, dict):
                continue
            if _is_solar_detection(element):
                center = _centroid(element)
                if center:
                    solar_detections.append(center)
            elif _is_satellite_detection(element):
                center = _centroid(element)
                if center:
                    detections.append(center)

    scores = _score_labels(covered_labels, detections, solar_detections)
    per_area = {
        area: _score_labels(
            [label for label in covered_labels if str(label.get("area") or "(blank)") == area],
            detections,
            solar_detections,
        )
        for area in sorted({str(label.get("area") or "(blank)") for label in covered_labels})
    }
    marketing_audit = _marketing_audit(
        covered_labels,
        evaluation_split=evaluation_split,
        errors=errors,
    )
    if len(ordered) < len(cells):
        marketing_audit["pool_national_claim_eligible"] = False
        marketing_audit["solar_claim_eligible"] = False
        marketing_audit["claim_scope"] = "development_only"
        marketing_audit["blockers"].append(
            f"The run covered {len(ordered)} of {len(cells)} labelled cells; publishable evaluation must cover all holdout cells."
        )
    approved_wording = (
        "National holdout result; quote only together with per-city results and 95% confidence intervals."
        if marketing_audit["pool_national_claim_eligible"]
        else (
            "Development benchmark only; national pool recall is unknown. Quote results only with the "
            "evaluated areas, ground-truth sources, and 95% confidence interval."
        )
    )

    report = {
        "created_at": _now_iso(),
        "build": build_tag,
        "detector_model": detector_model,
        "evaluation_split": evaluation_split,
        "cells_scanned": cells_scanned,
        "cells_total_labelled": len(cells),
        "labels_covered": len(covered_labels),
        "detections_seen": len(detections),
        **scores,
        "solar_detections_seen": len(solar_detections),
        "dataset_profile": {
            "all_labels": dataset_profile(all_labels),
            "selected_split": dataset_profile(labels),
            "evaluated_subset": dataset_profile(covered_labels),
        },
        "per_area": per_area,
        "marketing_audit": marketing_audit,
        "approved_wording": approved_wording,
        "errors": errors[:8],
        "note": (
            "precision_proxy judges only labelled points: detections away from any label are not "
            "counted. OSM bootstrap and product feedback are development data and cannot establish "
            "national performance."
        ),
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=1), encoding="utf-8")
    with HISTORY_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(report) + "\n")
    return report


def load_report() -> dict[str, Any] | None:
    try:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(report, dict):
        return None
    # Legacy reports remain useful experiment history, but must never escape
    # without the current dataset limitations attached.
    if "marketing_audit" not in report:
        labels = load_labels()
        report["evaluation_split"] = "development"
        report["dataset_profile"] = {"all_labels": dataset_profile(labels)}
        report["marketing_audit"] = _marketing_audit(labels, evaluation_split="development")
        report["approved_wording"] = (
            "Historical development benchmark only; national pool recall is unknown."
        )
    return report


# ---------------------------------------------------------------------------
# Feedback store (✓ / ✗ from the map UI)
# ---------------------------------------------------------------------------

VALID_VERDICTS = {"correct", "wrong", "missed"}


def add_feedback(record: dict[str, Any]) -> dict[str, Any]:
    """Append one ✓/✗ feedback record and mirror it into the label store.

    ``correct``: detection is a real pool  -> pool label
    ``wrong``:   detection is NOT a pool   -> no_pool label
    ``missed``:  user marks a pool the detector missed -> pool label
    """
    verdict = str(record.get("verdict", "")).strip().lower()
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"verdict must be one of {sorted(VALID_VERDICTS)}")
    lat = float(record.get("lat"))
    lon = float(record.get("lon"))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("lat/lon out of range")

    tags = record.get("tags") if isinstance(record.get("tags"), dict) else {}
    kept_tags = {
        key: str(value)[:60]
        for key, value in tags.items()
        if key.startswith(("detector:", "confidence:", "scan:", "pool:"))
    }
    entry = {
        "ts": _now_iso(),
        "verdict": verdict,
        "lat": round(lat, 7),
        "lon": round(lon, 7),
        "tags": kept_tags,
    }
    with _lock:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        labels = load_labels()
        upsert_label(
            labels,
            lat=lat,
            lon=lon,
            label="pool" if verdict in {"correct", "missed"} else "no_pool",
            source="feedback",
        )
        _save_labels(labels)
    return entry


def feedback_stats() -> dict[str, Any]:
    counts = {v: 0 for v in VALID_VERDICTS}
    total = 0
    try:
        with FEEDBACK_PATH.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    verdict = json.loads(line).get("verdict")
                except ValueError:
                    continue
                if verdict in counts:
                    counts[verdict] += 1
                    total += 1
    except OSError:
        pass
    return {"total": total, **counts, "labels": label_counts(load_labels())}


def export_feedback() -> str:
    """Raw JSONL — one labelled example per line, ready for a fine-tuning pipeline."""
    try:
        return FEEDBACK_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""
