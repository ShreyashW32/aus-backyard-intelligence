# Pool Detection Precision Pipeline

Satellite swimming-pool detection has high false-positive risk because many urban objects share the same top-down cues as pools: blue roofs, tarpaulins, white or cyan roof patches, water tanks, solar panels, map labels, road shields, icons, shadows, ponds, and small rooftop structures.

This project should prefer precision over raw recall for lead generation. A smaller set of trustworthy pool boxes is better than flooding the map with weak candidates.

## Current App Pipeline

The backend detector uses clean Google satellite tiles with `lyrs=s`, not the hybrid labeled map layer. That avoids road names, POI icons, and Google UI overlays in backend inference. If training data is captured from the browser, do not use screenshots of the Leaflet/Google map view because those can include labels, icons, red boxes, controls, and attribution. Use raw satellite tiles or exported clean imagery instead.

The app now follows this structure:

1. Download max-zoom satellite tiles.
2. Stitch tiles into detection chips.
3. Generate broad water/shape candidates.
4. Estimate real-world area.
5. Apply size-aware pool verification for tiny, standard, lap, large, and very large pools.
6. Reject rooftop, dark-panel, and roof-color context.
7. Deduplicate nearby fragments.
8. Render only verified pool boxes.

## Recommended YOLO Inference Settings

For YOLOv8/YOLOv11 pool detection on aerial images:

```text
imgsz: 1024-1536
conf: 0.50-0.70 for precision mode
iou: 0.40-0.50
max_det: 50-100 per tile/chip
```

Start with:

```bash
python scripts/pool_yolo_pipeline.py infer ^
  --weights runs/detect/pool/weights/best.pt ^
  --source data/pool_tiles/val/images ^
  --output runs/pool_precision_check ^
  --conf 0.60 ^
  --iou 0.45 ^
  --imgsz 1280 ^
  --review-crops runs/pool_review/accepted ^
  --rejected-crops runs/pool_review/rejected
```

Raise `--conf` if the map has too many false positives. Lower it only after adding hard negatives and a second-stage verifier.

## Negative-Only YOLO Images

YOLO supports negative images by using an empty `.txt` label file with the same basename as the image.

Example:

```text
dataset/images/train/blue_roof_001.jpg
dataset/labels/train/blue_roof_001.txt  # empty file
```

Create empty labels:

```bash
python scripts/pool_yolo_pipeline.py make-negatives ^
  --images data/hard_negatives/images ^
  --labels data/hard_negatives/labels
```

Copy them into a training split:

```bash
python scripts/pool_yolo_pipeline.py copy-negatives ^
  --images data/hard_negatives/images ^
  --labels data/hard_negatives/labels ^
  --dataset-images data/pool_dataset/images/train ^
  --dataset-labels data/pool_dataset/labels/train
```

## Hard Negative Mining Loop

Use this loop repeatedly:

1. Train an initial YOLO model.
2. Run inference on difficult urban areas.
3. Save false-positive crops: roofs, solar panels, map icons, shadows, road labels, white/blue roof patches, tanks, tarps.
4. Add those crops to the dataset as negative-only images with empty labels.
5. Retrain.
6. Compare precision, recall, F1, and mAP.

Runtime rejected-candidate capture in the app is available but disabled by
default. A detector rejection is not a verified negative, so these crops are
quarantined for human review and are never given an empty YOLO label
automatically:

```powershell
$env:QLDR_POOL_SAVE_REJECTED_CROPS = "1"
$env:QLDR_POOL_HARD_NEGATIVE_DIR = "D:\aus_backyard_intelligence_workspace\aus_backyard_intelligence_workspace\data\hard_negatives"
$env:QLDR_POOL_MAX_NEGATIVE_CROPS = "200"
python -m aus_backyard_intelligence --host 127.0.0.1 --port 8048
```

Rejected candidate crops and their review metadata are saved under:

```text
data/hard_negatives/review_queue/images
data/hard_negatives/review_queue/metadata
```

No `.txt` training labels are created in the review queue. After a human
confirms that a crop contains no pool, copy it to an approved image directory
and use `pool_yolo_pipeline.py make-negatives` to create its empty label.

## Post-Processing Rules

Use geometric rules after YOLO:

```python
if confidence < 0.55:
    reject()

if width_px < 8 or height_px < 8:
    reject()

aspect = max(width_px, height_px) / min(width_px, height_px)
if aspect > 8.5:
    reject()

area_ratio = (width_px * height_px) / (image_width * image_height)
if area_ratio < 0.000015 or area_ratio > 0.18:
    reject()
```

For segmentation masks:

```python
compactness = 4 * math.pi * area / (perimeter ** 2)
if compactness < 0.20:
    reject()
```

The app uses stricter, size-aware rules in `aus_backyard_intelligence/yolo_detector.py`.

## Two-Stage Pipeline

For better precision, use YOLO only as a candidate generator:

```text
satellite tile
  -> YOLO pool candidate detector
  -> bbox/shape/size filters
  -> crop each candidate
  -> CNN classifier: pool vs non-pool
  -> final map box
```

Train the classifier on crops:

```text
classifier_dataset/
  train/
    pool/
    non_pool/
  val/
    pool/
    non_pool/
```

Non-pool should include roofs, solar panels, road labels, icons, shadows, tarps, tanks, blue cars, rooftop structures, ponds, and industrial roofs.

Use the classifier during inference:

```bash
python scripts/pool_yolo_pipeline.py infer ^
  --weights runs/detect/pool/weights/best.pt ^
  --classifier runs/classify/pool_verify/weights/best.pt ^
  --classifier-conf 0.70 ^
  --source data/pool_tiles/test/images ^
  --output runs/two_stage_pool
```

## Retraining

After adding hard negatives:

```bash
python scripts/pool_yolo_pipeline.py train ^
  --data data/pool_dataset/dataset.yaml ^
  --weights yolo11s.pt ^
  --imgsz 1280 ^
  --epochs 80 ^
  --batch 8
```

Recommended dataset balance:

```text
positive pool images: all pool sizes and colors
negative images: at least 1-3x the number of positives
validation negatives: include the same hard categories as training
```

Report:

```text
precision before/after hard negatives
recall before/after hard negatives
F1 score
mAP50 and mAP50-95
false-positive counts per category
```
