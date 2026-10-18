"""Practical YOLO pool-detection utilities.

This script is intentionally standalone so it can be used with datasets outside
the web app:

  python scripts/pool_yolo_pipeline.py infer --weights runs/detect/train/weights/best.pt --source tiles/
  python scripts/pool_yolo_pipeline.py make-negatives --images hard_negatives/images --labels hard_negatives/labels
  python scripts/pool_yolo_pipeline.py train --data dataset.yaml --weights yolo11s.pt

Negative-only YOLO images are valid when the matching label file exists but is
empty. That is how hard negatives such as roofs, labels, solar panels, shadows,
and map icons teach the detector "no pool here".
"""
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DetectionFilter:
    min_box_px: int = 8
    max_area_ratio: float = 0.18
    min_area_ratio: float = 0.000015
    max_aspect: float = 8.5
    min_confidence: float = 0.55


def iter_images(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        yield path
        return
    for candidate in sorted(path.rglob("*")):
        if candidate.suffix.lower() in IMAGE_SUFFIXES:
            yield candidate


def passes_box_filter(
    xyxy: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
    confidence: float,
    config: DetectionFilter,
) -> tuple[bool, str]:
    x1, y1, x2, y2 = xyxy
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if confidence < config.min_confidence:
        return False, "low_confidence"
    if width < config.min_box_px or height < config.min_box_px:
        return False, "too_small_px"
    aspect = max(width, height) / max(min(width, height), 1.0)
    if aspect > config.max_aspect:
        return False, "bad_aspect"
    area_ratio = (width * height) / max(float(image_width * image_height), 1.0)
    if area_ratio < config.min_area_ratio:
        return False, "too_small_area"
    if area_ratio > config.max_area_ratio:
        return False, "too_large_area"
    return True, "ok"


def crop_xyxy(image, xyxy: tuple[float, float, float, float], pad_ratio: float = 0.20):
    import cv2  # noqa: F401 - imported here to keep module import lightweight

    height, width = image.shape[:2]
    x1, y1, x2, y2 = xyxy
    pad = max(4, int(max(x2 - x1, y2 - y1) * pad_ratio))
    left = max(0, int(x1) - pad)
    top = max(0, int(y1) - pad)
    right = min(width, int(x2) + pad)
    bottom = min(height, int(y2) + pad)
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right]


def classifier_pool_probability(classifier, crop, imgsz: int) -> float:
    """Return pool probability from an optional Ultralytics classification model."""
    if classifier is None:
        return 1.0
    result = classifier.predict(crop, imgsz=imgsz, verbose=False)[0]
    probs = getattr(result, "probs", None)
    if probs is None:
        return 0.0
    names = getattr(result, "names", {}) or {}
    pool_indices = [idx for idx, name in names.items() if str(name).lower() in {"pool", "swimming_pool"}]
    if not pool_indices:
        pool_indices = [0]
    return max(float(probs.data[int(idx)]) for idx in pool_indices)


def run_inference(args: argparse.Namespace) -> None:
    import cv2
    from ultralytics import YOLO

    detector = YOLO(args.weights)
    classifier = YOLO(args.classifier) if args.classifier else None
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir = Path(args.review_crops) if args.review_crops else None
    rejected_dir = Path(args.rejected_crops) if args.rejected_crops else None
    if review_dir:
        review_dir.mkdir(parents=True, exist_ok=True)
    if rejected_dir:
        rejected_dir.mkdir(parents=True, exist_ok=True)

    config = DetectionFilter(
        min_box_px=args.min_box_px,
        max_area_ratio=args.max_area_ratio,
        min_area_ratio=args.min_area_ratio,
        max_aspect=args.max_aspect,
        min_confidence=args.conf,
    )

    accepted = 0
    rejected = 0
    for image_path in iter_images(Path(args.source)):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        img_h, img_w = image.shape[:2]
        results = detector.predict(
            str(image_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            verbose=False,
        )
        annotated = image.copy()
        for det_index, box in enumerate(results[0].boxes):
            conf = float(box.conf[0])
            xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
            ok, reason = passes_box_filter(
                xyxy,
                image_width=img_w,
                image_height=img_h,
                confidence=conf,
                config=config,
            )
            crop = crop_xyxy(image, xyxy)
            pool_prob = classifier_pool_probability(classifier, crop, args.classifier_imgsz) if crop is not None else 0.0
            if ok and pool_prob >= args.classifier_conf:
                accepted += 1
                x1, y1, x2, y2 = (int(v) for v in xyxy)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    annotated,
                    f"pool {conf:.2f}/{pool_prob:.2f}",
                    (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                if review_dir and crop is not None:
                    cv2.imwrite(str(review_dir / f"{image_path.stem}_pool_{det_index}_{conf:.2f}.jpg"), crop)
            else:
                rejected += 1
                if rejected_dir and crop is not None:
                    cv2.imwrite(str(rejected_dir / f"{image_path.stem}_{reason}_{det_index}_{conf:.2f}.jpg"), crop)
        cv2.imwrite(str(output_dir / image_path.name), annotated)

    print(f"accepted={accepted} rejected={rejected} output={output_dir}")


def make_negative_labels(args: argparse.Namespace) -> None:
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    images = list(iter_images(images_dir))
    # Preflight before writing anything: this command must never erase a real
    # positive annotation when pointed at a mixed or incorrect directory.
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists() and label_path.read_text(encoding="utf-8").strip():
            raise ValueError(
                f"Refusing to overwrite non-empty label for {image_path.name}: {label_path}"
            )
    labels_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        label_path.write_text("", encoding="utf-8")
        count += 1
    print(f"created_empty_labels={count} labels_dir={labels_dir}")


def _reviewed_negative_pairs(src_images: Path, src_labels: Path) -> list[tuple[Path, Path]]:
    """Return explicitly labelled negative pairs or fail before dataset writes.

    A missing label is not evidence that an image is a negative. Requiring an
    existing, empty YOLO label makes human review / ``make-negatives`` an
    explicit step and prevents the runtime review queue from being imported by
    accident.
    """
    pairs: list[tuple[Path, Path]] = []
    for image_path in iter_images(src_images):
        label_path = src_labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(
                "Refusing to copy an unreviewed negative without an explicit empty label: "
                f"{image_path.name} (expected {label_path})"
            )
        if label_path.read_text(encoding="utf-8").strip():
            raise ValueError(
                f"Refusing to copy {image_path.name}: negative label must be explicitly empty"
            )
        pairs.append((image_path, label_path))
    return pairs


def copy_negatives_into_dataset(args: argparse.Namespace) -> None:
    src_images = Path(args.images)
    src_labels = Path(args.labels)
    dst_images = Path(args.dataset_images)
    dst_labels = Path(args.dataset_labels)
    pairs = _reviewed_negative_pairs(src_images, src_labels)
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path, label_path in pairs:
        shutil.copy2(image_path, dst_images / image_path.name)
        shutil.copy2(label_path, dst_labels / label_path.name)
        count += 1
    print(f"copied_negative_samples={count}")


def train_detector(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO swimming-pool detection helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    infer = sub.add_parser("infer", help="Run YOLO inference with precision filters")
    infer.add_argument("--weights", required=True)
    infer.add_argument("--source", required=True)
    infer.add_argument("--output", default="runs/pool_infer")
    infer.add_argument("--conf", type=float, default=0.55)
    infer.add_argument("--iou", type=float, default=0.45)
    infer.add_argument("--imgsz", type=int, default=1280)
    infer.add_argument("--max-det", type=int, default=80)
    infer.add_argument("--min-box-px", type=int, default=8)
    infer.add_argument("--min-area-ratio", type=float, default=0.000015)
    infer.add_argument("--max-area-ratio", type=float, default=0.18)
    infer.add_argument("--max-aspect", type=float, default=8.5)
    infer.add_argument("--review-crops", help="Save accepted crops for quick review")
    infer.add_argument("--rejected-crops", help="Save rejected crops for hard-negative review")
    infer.add_argument("--classifier", help="Optional Ultralytics pool/non-pool classifier weights")
    infer.add_argument("--classifier-conf", type=float, default=0.70)
    infer.add_argument("--classifier-imgsz", type=int, default=224)
    infer.set_defaults(func=run_inference)

    neg = sub.add_parser(
        "make-negatives",
        help="Create empty YOLO labels after images have been human-confirmed as negatives",
    )
    neg.add_argument("--images", required=True)
    neg.add_argument("--labels", required=True)
    neg.set_defaults(func=make_negative_labels)

    copy_neg = sub.add_parser(
        "copy-negatives",
        help="Copy reviewed hard negatives that already have explicit empty labels",
    )
    copy_neg.add_argument("--images", required=True)
    copy_neg.add_argument("--labels", required=True)
    copy_neg.add_argument("--dataset-images", required=True)
    copy_neg.add_argument("--dataset-labels", required=True)
    copy_neg.set_defaults(func=copy_negatives_into_dataset)

    train = sub.add_parser("train", help="Retrain YOLO after adding hard negatives")
    train.add_argument("--data", required=True)
    train.add_argument("--weights", default="yolo11s.pt")
    train.add_argument("--imgsz", type=int, default=1280)
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--patience", type=int, default=20)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--project", default="runs/pool_train")
    train.add_argument("--name", default="pool_hard_negative")
    train.set_defaults(func=train_detector)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
