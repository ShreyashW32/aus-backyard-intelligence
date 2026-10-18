# AUS Backyard Intelligence

A backyard-opportunity research platform that ranks private residential
properties for pool, fence, deck, and landscaping opportunities and produces
jurisdiction-aware lead context. The interactive viewer combines government and
commercial basemaps, mapped features, experimental selected-area pool detection,
live elevation contours, and QGIS exports. National detector performance has not
yet been established.

Package: `aus_backyard_intelligence` (renamed from `qld_backyard_recommender`).
Env vars use the `ABI_` prefix; legacy `QLDR_` names still work. State-specific
pool-compliance rules live in `aus_backyard_intelligence/jurisdiction.py` —
registration/inspection is state law, not national. Set `ABI_DEV_API=0` to
disable the dev/eval endpoints in a public deployment.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .[dev]
```

## Run

```powershell
.\scripts\run_demo.ps1
```

Open `http://127.0.0.1:8000/` for the map viewer or
`http://127.0.0.1:8000/dashboard` for the ranking dashboard.

### Exact lead addresses

The lead dashboard can use Google's official Geocoding API for address lookup.
Enable the Geocoding API for a server-restricted Google Maps key, then put it
in a local `.env` file (git-ignored, loaded by `run_demo.ps1`):

```powershell
Copy-Item .env.example .env
# edit .env and set ABI_GOOGLE_MAPS_API_KEY=your-server-side-key
.\scripts\run_demo.ps1
```

Never commit a real key. See [SECURITY.md](SECURITY.md).

Only Australian `ROOFTOP` results with a numbered route and a
`street_address`, `premise`, or `subpremise` result type, whose result point is
within 35 metres of the selected footprint, are marked verified.
A unit number is displayed only when Google supplies the explicit `subpremise`
component (or the selected building has an explicit OpenStreetMap `addr:unit`
tag). Without a key, the rate-limited OpenStreetMap Nominatim fallback is shown
as approximate and never presented as an exact property/unit match.

Google geocoding content is kept only in the in-memory scan session; the app
does not persist or pre-fetch it. Before enabling this in a public deployment,
publish Terms of Use and a Privacy Policy that meet the Google Maps Platform
requirements and retain the dashboard's visible `Google Maps` attribution.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest -q
corepack enable
pnpm install --frozen-lockfile
pnpm exec playwright install chromium
pnpm test
```

The Playwright regression seeds every map overlay with known geometry and
requires `queryRenderedFeatures({layers:[id]})` to return at least one feature
for all layers.

## Main API routes

- `GET /api/health`
- `GET /api/rankings?module=pool_3d&top_k=5`
- `GET /api/viewer/map?module=pool_3d&top_k=15`
- `GET /api/viewer/property/{property_id}`
- `POST /api/analyze/area`
- `GET /api/osm/features?south=...&west=...&north=...&east=...&zoom=...` (mapped data; satellite inference additionally requires `analysis_polygon`)
- `GET /api/terrain/elevation-grid?south=...&west=...&north=...&east=...&size=5`
- `GET /api/geocode?q=Moreton+Bay`
- `GET /api/osint/iss`
- `GET /api/osint/earthquakes`
- `GET /api/osint/flights?limit=1500`
- `GET /api/osint/wildfires`
- `GET /api/osint/cameras`
- `GET /api/osint/incidents`
- `GET /api/osint/ip`
- `GET /api/export/qgis?module=pool_3d`

## Pool detection (hybrid)

Pool display is a hybrid of two evidence sources:

1. **OpenStreetMap mapped pools** — useful mapped geometry, but incomplete and
   subject to community-data errors and geographic coverage bias.
2. **Pool-trained YOLO** — detects *new/unmapped* pools from satellite chips.

Interactive satellite inference is selection-scoped: the viewer does not run
YOLO on load, pan, zoom, or a timer. Draw an analysis polygon and run analysis;
the polygon is sent to the server and detections outside it are discarded.

Generic COCO YOLO weights (`yolo11n.pt`, etc.) **cannot** detect pools — COCO has
no pool class — so the visual detector needs pool-trained weights. Until those
exist, the app serves OSM-mapped pools only. The legacy colour/water CV detector
is disabled by default because solar panels, blue-grey roofs, and shaded turf are
colourimetrically identical to muted pool water and produce heavy false positives;
enable it for experimentation with `QLDR_ENABLE_CV_POOL_FALLBACK=1`.

Each pool-YOLO box is additionally **colour-confirmed** (a real pool is blue/cyan)
to drop the grey-roof and greenhouse boxes imported models sometimes emit, and
boxes are de-duplicated against each other and OSM. Detection confidence floor is
tunable with `QLDR_POOL_YOLO_CONF` (default `0.15`; fast/auto modes apply higher
effective floors). `GET /api/detection/status` reports the exact active filenames,
SHA-256 checksums, thresholds, ensemble state, and cache signature.

Detector inference is serialized through a bounded FIFO serviced by one
long-lived worker, so overlapping requests in one server process queue instead
of failing because the model is busy. Queue depth and activity are exposed by
`GET /api/detection/status`; `QLDR_DETECTOR_QUEUE_SIZE` controls the bound
(default 32), and an optional `QLDR_DETECTOR_QUEUE_TIMEOUT` enables explicit
load shedding. The machine-wide lease remains as an out-of-memory guard against
accidentally starting two model-owning server processes.

### Model weights

Weights (`*.pt`, `*.onnx`), training runs and the satellite-chip dataset are not
stored in git because of their size. Generic YOLO weights download automatically
through Ultralytics; pool-trained weights must be supplied or trained as below.

### Use a pre-trained pool model (fastest)

Drop an Ultralytics pool detector (`.pt` or exported `.onnx`, single class
`swimming_pool`) at `models/pool_yolo.pt` / `models/pool_yolo.onnx`, or point
`QLDR_POOL_YOLO_WEIGHTS` at any weights file. The environment override is checked
first, followed by the conventional project paths. The app picks it up automatically
and YOLO becomes the primary detector for unmapped pools.

### Train pool weights from your OSM pools

OSM-mapped pools are useful weak labels for training. They are not independent
ground truth for a production recall claim. Bootstrap a training dataset and fine-tune:

```powershell
# 1. Build a YOLO dataset (satellite chips + labels) from mapped pools in a bbox
.\.venv\Scripts\python.exe scripts\build_pool_dataset.py `
    --bbox -27.30 152.95 -27.20 153.10 --out data\pool_dataset --zoom 20 --negatives 200

# 2. Fine-tune YOLO on it
.\.venv\Scripts\python.exe scripts\pool_yolo_pipeline.py train `
    --data data\pool_dataset\dataset.yaml --weights yolo11s.pt
```

The app auto-loads `runs/.../best.pt` (or set `QLDR_POOL_YOLO_WEIGHTS` to any
weights path). Once weights are present, YOLO becomes the primary detector for
unmapped pools and its boxes are merged and de-duplicated against OSM pools.
`GET /api/detection/status` reports which detector is active.

## Current evaluation

There is currently **no defensible national accuracy number**. The label store is
94.4% OSM bootstrap data, 78.5% is named `pitch-bench`, 52 labels lack area
provenance, and there is only one solar-positive label. Historical recall/FP
figures in `docs/accuracy_log.md` are development experiments, not marketing
metrics. Solar candidates are therefore experimental and off by default.

Evaluation reports now include label composition, per-area results, Wilson 95%
confidence intervals, and a machine-readable marketing audit. A national pool
claim is blocked unless a run uses independently hand-labelled `holdout` data
covering at least three metros with 150+ pool/no-pool labels per metro and scans
every holdout cell. Solar additionally requires at least 100 `solar` and 100
`no_solar` holdout labels.

Import independent labels through the dev-only endpoint, then evaluate only the
holdout split:

```text
POST /api/eval/holdout-labels  {"records":[{"lat":-31.95,"lon":115.86,"label":"pool","area":"Perth WA"}]}
POST /api/eval/run             {"split":"holdout","max_cells":40,"build":"candidate-rN"}
```

Until those gates pass, approved wording is: “Development benchmark only;
national pool recall is unknown.” Quote any development number only with its
areas, ground-truth sources, and confidence interval.

## Data notes

The **3D Property Model** is scoped to the current property/area selection. It
preserves the supplied parcel, building, backyard, pool, and point-tree geometry,
uses the recorded/derived building height, and does not invent a placeholder
house when geometry is missing. Roof form is procedural massing. The bundled
demo records use illustrative geocoded geometry, so the viewer labels them as
illustrative rather than presenting them as survey-grade digital twins.

New/unmapped pool candidates and unvalidated experimental solar candidates come
from the pool-trained, multi-class satellite model. Structure footprints, trees/vegetation, and mapped pools
come primarily from OpenStreetMap and Microsoft/public footprint data. Canopy display,
building height/3D meshes, and roof-solar suitability are heuristics over those mapped
features, not fresh neural inference. Geometry and confidence gates reject implausible
features, but all results remain screening evidence rather than proof of cadastral
ownership or a substitute for licensed current imagery and a site inspection. The
detector currently downloads configured Google satellite tiles as model input; operators
are responsible for ensuring their imagery use complies with the applicable terms.

Optional public feeds can be rate-limited or temporarily unavailable. Live
panels use Open Notify/Where The ISS, USGS, NASA EONET, OpenSky, IPAPI, and
Queensland Traffic public data. The CVE panel only opens official NVD reference
pages and does not probe third-party systems.
