# Pool detection accuracy log

> **Development experiments only — not sales claims.** As audited on 2026-07-19,
> the label store is 94.4% OSM bootstrap data, is dominated by `pitch-bench` and
> Sydney, has no untouched holdout split, and contains one solar label. These
> results measure iteration on biased development ground, not national recall or
> solar accuracy. The current report schema blocks marketing eligibility until
> independently hand-labelled, multi-metro holdout requirements pass.

Running record of measured accuracy changes. Every entry is a controlled
experiment on the eval harness (`aus_backyard_intelligence/evaluation.py`):
same labelled cells, same production detection path, one variable changed.
Development labels are OSM bootstrap + in-product feedback only — pseudo-labels
never enter `eval_data/labels.json`. Neither source is an independent holdout.

Metrics: `recall` = labelled pools found; `fp_rate` = no-pool points flagged;
`precision_proxy` is pessimistic (some "false positives" are real pools OSM
never mapped). Numbers are only comparable within the same label set — the
label store grows over time, so each session re-measures its own baseline.

## Definitive baseline (2026-07-05, bench-r18-final, 581 labelled points)

| config | recall | fp_rate | precision | F1 |
|---|---|---|---|---|
| conf 0.20 (shipped default) | 0.888 (229/258) | 0.127 (41/323) | 0.848 | 0.867 |

Context: StephanST YOLOv8l-P2 at native imgsz 1024, detection zoom z20,
solar/pool detection types separated. Prior findings (2026-07-05 session):
TTA gave identical detections at 2x cost (rejected); imgsz 1280 was worse on
all axes (rejected); conf 0.28 -> 0.20 lifted recall 0.890 -> 0.936 on the
older 193-label set at equal speed (shipped).

Known recall ceiling without retraining: ~10% of labelled pools are invisible
to the raw model even at conf 0.10 (tree canopy / indoor / enclosed) — the
next real step past that ceiling is fine-tuning on the auto-pipeline's
`eval_data/training_candidates.jsonl` (needs GPU time).

## 2026-07-09 — loop iteration 1: eval-metric bug found and fixed

First runs (conf 0.20 vs 0.15, 7 cells, 581 points) produced recall 0.353 /
0.314 — a collapse from 0.888 on identical ground truth, with fp_rate
unchanged (0.124 vs 0.127). Diagnosis, verified step by step:

1. Weights/model healthy: `models/pool_yolo.pt` resolves, v8l-P2 at 1024,
   correct classes. Direct `detect_pools_in_bbox` on the dense Dover Heights
   cell: 158 pools — detection itself is fine.
2. The eval counts only satellite-tagged detections (mapped OSM pools must
   not count — they are the ground truth). But `fetch_osm_features` DEDUPES
   satellite boxes within 20 m of a mapped pool out of the merged output.
   Labels ARE mapped pools, so when Overpass is healthy the merge hides
   exactly the detections the eval needs to count.
3. Result: measured recall tracked Overpass availability, not detector
   quality. bench-r18-final (0.888) ran minutes after a heavy label
   bootstrap had rate-limited Overpass — mapped pools were missing from
   fetches, so satellite boxes survived the dedupe and recall read high.
   Today Overpass answered normally and recall read 0.35 on identical code.

Fix (`external_api._deduplicate_pools` + `evaluation._is_satellite_detection`):
when the dedupe drops a satellite box that duplicates a mapped pool, the kept
pool is stamped `detector:satellite=<source>`; the eval counts the stamp as a
detector hit. Measured recall is now Overpass-independent. UI behaviour is
unchanged (still one element per pool). Regression tests:
`test_dedupe_stamps_mapped_pool_when_satellite_duplicate_dropped`,
`test_dedupe_does_not_stamp_for_mapped_on_mapped_duplicates`,
`test_satellite_detection_counts_stamped_mapped_pool`.

Consequence for prior numbers: all pre-fix eval recalls (including
bench-r18-final 0.888) were measured in the Overpass-degraded regime. They
are approximately comparable to post-fix numbers (both count "detector fired
at a labelled pool") but the post-fix metric is the trustworthy one.

Re-measure on the fixed metric (7 cells, 258 pool / 323 no-pool points):

| config | recall | fp_rate | precision | F1 |
|---|---|---|---|---|
| conf 0.20 (default) | 0.864 (223/258) | 0.127 (41/323) | 0.845 | 0.854 |
| conf 0.15 | 0.872 (225/258) | 0.130 (42/323) | 0.843 | 0.857 |

Decision: conf floor stays 0.20 — the 0.15 gain (+2 tp / +1 fp) is within
run-to-run noise and not worth moving a shipped default for.

**Trustworthy baseline going forward: recall 0.864 / fp_rate 0.127 /
F1 0.854 at conf 0.20.** Gap to the 90-95% target: 35 missed pools.

## 2026-07-09 — loop iteration 2: false-negative anatomy

Bucketed 322 pool labels (7 pool-densest cells) by best RAW model box
(conf >= 0.10, no publish gates) within 15 m:

| bucket | labels | share |
|---|---|---|
| no box at all (model-blind) | 33 | 10% |
| conf 0.10-0.20 (below floor) | 13 | 4% |
| conf 0.20-0.50 | 40 | 12% |
| conf 0.50-0.80 | 179 | 56% |
| conf >= 0.80 | 57 | 18% |

Median best-conf where a box exists: 0.70.

Conclusions:
- The publish pipeline is effectively lossless: published recall (0.864)
  matches the raw above-floor share (276/322 = 0.857). Gates/merge are NOT
  the bottleneck — no code-level gating fix will move recall.
- In-code headroom is only the 13 sub-floor labels (+4 pts max).
- The 33 model-blind pools (tree canopy, indoor, enclosed) put the ceiling
  for THIS model at ~90%. Reaching 90-95% reliably requires fine-tuning.

Threshold curve (7 cells, 581 points, identical code):

| conf | recall | fp_rate | precision | F1 |
|---|---|---|---|---|
| 0.20 | 0.864 | 0.127 | 0.845 | 0.854 |
| 0.15 | 0.872 | 0.130 | 0.843 | **0.857** |
| 0.12 | 0.880 | 0.142 | 0.832 | 0.855 |
| 0.10 | 0.880 | 0.142 | 0.832 | 0.855 |

Recall saturates at 0.880 below conf 0.12 (nothing further survives the
gates); F1 peaks at 0.15. **Shipped: default QLDR_POOL_YOLO_CONF 0.20 ->
0.15** (yolo_detector.py). 0.12 rejected: +0.8 recall costs +1.2 fp_rate
with F1 flat. The threshold lever is now exhausted — 88% is this model's
practical ceiling on free Google imagery.

## 2026-07-09 — loop iteration 3: fine-tune preparation

Path to 90-95% recall = fine-tune the v8l-P2 checkpoint on Google-tile chips
(the model was trained on different imagery; domain shift costs the visible-
pool misses). Work done:

- `scripts/build_pool_dataset.py` upgraded: 2-class labels matching the
  checkpoint ({0: solarpanel, 1: pool} — single-class fine-tune would reset
  the head and silently kill solar detection); chips now label EVERY visible
  feature, not just the anchor (unlabeled in-frame pools previously trained
  the model to suppress pools — dense suburbs cluster them); solar harvested
  from OSM power=generator; negatives avoid both classes. Smoke-tested.
- Train/eval separation: dataset regions are Gold Coast (Broadbeach Waters),
  Moreton Bay (Mango Hill), Perth western suburbs — no overlap with the
  Sydney-east eval cells, so eval ground truth never leaks into training.
- Dataset built (data/pool_ft/): 929 chips — Gold Coast 312 (162 pools),
  Moreton Bay 162 (12 pools; OSM sparse there), Perth 455 (304 pools; also
  a different roof/soil palette for generalization), plus 150 negatives per
  region. Combined dataset.yaml with 2-class names.
- Solar-head preservation: OSM had 1 mapped solar element across all three
  regions, but the chips are full of visible rooftop panels — training with
  them unlabeled would teach the model to suppress class 0. Fix: class 0 is
  self-distilled (pseudo-labelled by the current model at conf >= 0.50)
  before training; pools stay OSM ground truth. Eval labels are untouched.
- Fine-tune launched (CPU, hours): v8l-P2 from models/pool_yolo.pt, 6 epochs,
  imgsz 1024, batch 2, backbone frozen (freeze=10), lr0 1e-4, patience 3,
  output runs/ft/poolft-r21. Acceptance gate: the candidate ships ONLY if it
  beats the current default on the fixed-metric eval (recall AND F1, fp_rate
  not worse than +1 pt) — same 7 cells, same harness, subprocess-isolated.
- 6,221 solar boxes pseudo-labelled across the 929 chips before training
  (chips were full of unlabeled panels — suppression risk confirmed real).
- Training: 6 epochs / 3.64 h CPU; fine-tune-val improved monotonically to
  precision 0.710 / recall 0.885 / mAP50 0.798.

## 2026-07-10 — loop iteration 3 result: candidate REJECTED

Acceptance benchmark (7 cells, conf 0.15, production path):

| model | recall | fp_rate | precision | F1 | solar seen |
|---|---|---|---|---|---|
| live (models/pool_yolo.pt) | 0.872 | 0.130 | 0.843 | 0.857 | ~1032 |
| candidate (poolft-r21 best.pt) | 0.857 | 0.115 | 0.857 | 0.857 | 1096 |

The fine-tune traded recall for precision at identical F1 — fails the gate
(recall down). The solar head survived (pseudo-label strategy worked).
Interpretation: 6 frozen-backbone CPU epochs on 929 chips adapt the model to
Google-tile textures (fewer FPs) but cannot out-recall a checkpoint trained
on a large dedicated dataset. Artifact kept at runs/ft/poolft-r21 for a
future GPU run (more epochs, unfrozen, larger dataset).

Next measurement: are the two models' misses COMPLEMENTARY? Raw-box analysis
(conf 0.15, same cells/labels):

| model (raw) | recall | fp_rate |
|---|---|---|
| base | 0.864 | 0.077 |
| fine-tuned | 0.891 | 0.019 |
| union | **0.950** | 0.077 |

The fine-tune recovers 22 of the base's 35 misses, and adds zero new false
positives at the raw level. The rejection verdict inverts at the union level:
the fine-tune is not a replacement, it is a complement.

## 2026-07-10 — loop iteration 4: two-model ensemble SHIPPED (build r21)

Implementation (`yolo_detector.py`): secondary weights resolve from
`models/pool_yolo_b.pt` (or QLDR_POOL_YOLO_WEIGHTS_B); `_yolo_detect_pools`
runs a second pool-only pass and unions boxes via `_merge_secondary_pool_dets`
(secondary box dropped if its centre lies within half a box-side of a primary
pool box; solar boxes never block adds; primary's solar-array veto applies to
both sets). Solar stays single-model — the fine-tune's solar head is a
distillation of the base's. Kill switch: `QLDR_POOL_YOLO_ENSEMBLE=0`.
Unit test: `test_merge_secondary_pool_dets_unions_and_dedupes`. 116 tests pass.

Production acceptance benchmark (7 cells, 581 points, conf 0.15):

| config | recall | fp_rate | precision | F1 | eval wall time |
|---|---|---|---|---|---|
| single model | 0.872 | 0.130 | 0.843 | 0.857 | ~250 s |
| **ensemble (shipped)** | **0.938 (242/258)** | 0.130 | 0.852 | **0.893** | 345 s |

Recall +6.6 pts at identical fp_rate; ~1.4x wall time (tile fetch dominates,
so the second inference does not double it). **The 90-95% target band is
reached: 93.8% measured on the Overpass-independent production-path eval.**
Client pool cache bumped to v15 (ensemble output must not mix with cached
single-model boxes); badge `build r21 · two-model ensemble · pool recall 94%`.

Remaining misses (16/258): dominated by the model-blind set (tree canopy,
indoor/covered pools) — further recall needs better imagery (licensed
high-res) or GPU-scale training; both out of scope for the free stack.

## 2026-07-10 — precision pass: strict-water CV supplement gated off (build r22)

User directive: focus on precision (r21 fp_rate 0.130, precision 0.852).

FP anatomy (every eval violation attributed to its source): of 42 FPs,
**28 came from the strict-water CV supplement** (conf 0.89-0.95, clustered
near the Dover Heights clifftop — reading the OCEAN as clear pool water) and
14 from the YOLO ensemble. TP mix: YOLO 211, CV supplement 32 (nearest-match
attribution).

Ablation (same fetch pass, scored with vs without CV detections):

| config | recall | fp_rate | precision | F1 |
|---|---|---|---|---|
| with CV supplement | 0.953 | 0.130 | 0.854 | 0.901 |
| without | 0.915 | 0.053 | 0.933 | 0.924 |

Only 10 of the supplement's TPs are unique; the ensemble already finds the
bright pools it was built for (it predates the ensemble). Shipped: supplement
mode env `QLDR_ENABLE_STRICT_WATER_SUPPLEMENT` now defaults to `auto` — the
supplement runs ONLY when no second ensemble model resolves ("1"/"0" force
on/off). Tests: `test_strict_water_supplement_auto_disabled_when_ensemble_active`
(+2 pipeline tests pinned to the no-ensemble scenario). 117 tests pass.

Production confirmation (7 cells, 581 points, conf 0.15):

| build | recall | fp_rate | precision | F1 |
|---|---|---|---|---|
| r21 ensemble | 0.938 | 0.130 | 0.852 | 0.893 |
| **r22 shipped** | **0.926 (239/258)** | **0.065 (21/323)** | **0.919** | **0.923** |

False positives halved, precision +6.7 pts, recall stays in the 90-95 band.
Client pool cache bumped to v16. Note: eval history shows metric variance of
roughly ±0.01 between identical runs (tile-fetch nondeterminism) — deltas
smaller than that are noise.

## 2026-07-13 — expanded-label report and selected-area-only inference (build r23/r24)

The r23 report superseded the old 94% development result at the time. Neither is
a headline or marketing metric. It evaluated a larger covered label set than r22:

| build | covered labels | recall | fp_rate | precision proxy | F1 proxy |
|---|---:|---:|---:|---:|---:|
| r22 | 581 | 0.926 | 0.065 | 0.919 | 0.923 |
| r23 development run | 774 | 0.882 | 0.071 | 0.917 | 0.899 |

The larger set may explain part of the recall change, but both runs are historical
development experiments and must not be presented as national validation. Solar remains under-labelled:
one solar label and 1,647 detections seen cannot establish solar precision.

Build r24 changes runtime scope and telemetry rather than model weights:

- no inference on page load, pan, zoom, layer toggle, or heartbeat;
- satellite inference requires an explicit analysis polygon and outputs are
  clipped to it;
- confidence values are deterministic (response jitter removed);
- cache keys include detector pipeline version, weight SHA-256 checksums,
  thresholds, ensemble state, selection polygon, and local-data file version;
- detector status and feature tags report actual pool checkpoint identity rather
  than the optional generic context-model choice.

## 2026-07-13 — pool/solar runtime compatibility and strict input masking

A live CPU scan exposed `AttributeError: 'Conv' object has no attribute 'bn'`
after the primary checkpoint loaded. The checkpoint itself is valid; the failure
is an Ultralytics first-prediction fusion race: one request can remove a Conv
batch-normalization layer while another is still executing the unfused forward.

Runtime hardening now:

- serializes primary and secondary predictions through one lock, preventing the
  fusion race and bounding concurrent Torch activation memory;
- detects the precise broken state (active forward references a missing `bn`),
  switches that module to its own `forward_fuse`, and retries once;
- parses integer-keyed, string-keyed, and list class metadata. The bundled
  checkpoint mapping is verified as `0=solarpanel, 1=pool`;
- records primary/secondary success, safe error details, discovered class names,
  compatibility repairs, retries, and elapsed inference time in scan/status
  diagnostics. Runtime errors no longer appear as genuine zero detections;
- blacks out every satellite pixel outside the selected polygon before primary
  YOLO, secondary YOLO, optional context YOLO, and CV processing. Centre clipping
  remains as a second inclusive boundary check.

Real-weight smoke validation on a labelled 1024×768 local satellite chip loaded
both 1024-pixel checkpoints successfully, completed both inference passes, and
parsed 7 `solarpanel` plus 5 `pool` boxes under their correct class IDs. This is
a runtime/class-routing smoke test, not a solar precision estimate. The focused
detector/evaluation suite passes 37 tests.
