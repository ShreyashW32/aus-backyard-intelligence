# Project Guide

## What This Project Does

This project is a Queensland-first backyard recommendation platform.

The main product is:

- pool suitability

The supporting products are:

- fence opportunity
- deck and patio opportunity

The current implementation is a strong starter and demo system:

- sample Brisbane City and Moreton Bay properties already work end to end
- QGIS is configured for 2D and 3D viewing
- the pipeline can also transform real parcel, building, and DEM layers into app-ready data

## Core Idea

Each property gets:

- parcel geometry
- building geometry
- backyard candidate geometry
- elevation-derived slope and relief
- access and coverage metrics
- optional vegetation and planning constraints

Those features are scored into:

- `pool_2d`
- `pool_3d`
- `fence`
- `deck`

## Main Files

- `aus_backyard_intelligence/app.py`
  - local web app and scoring engine
- `aus_backyard_intelligence/pipeline/feature_builder.py`
  - converts GIS layers into the JSON schema used by the app
- `aus_backyard_intelligence/map_export.py`
  - exports selected ranked properties into QGIS layers
- `scripts/export_ranked_map.ps1`
  - creates the GeoPackage for QGIS
- `scripts/prepare_qgis_3d_project.ps1`
  - creates a QGIS project with 3D styling
- `scripts/launch_qgis_3d_ready.ps1`
  - opens the QGIS project and auto-creates a full-screen 3D view

## Normal Workflow

### 1. Run the app

```powershell
.\scripts\run_demo.ps1
```

Then use:

- `http://127.0.0.1:8000`

### 2. Export ranked map layers for QGIS

```powershell
.\scripts\export_ranked_map.ps1 -Module pool_3d -Lga "Brisbane City" -TopK 3
```

This creates:

- `artifacts/qgis/ranked_selection.gpkg`

### 3. Prepare the QGIS 3D project

```powershell
.\scripts\prepare_qgis_3d_project.ps1
```

This creates:

- `artifacts/qgis/ranked_selection_3d.qgz`

### 4. Open the ready 3D project

```powershell
.\scripts\launch_qgis_3d_ready.ps1
```

That opens QGIS with:

- the selected ranked parcels
- building extrusions
- backyard score extrusions
- a full-screen 3D view

## How The Demo 3D View Works

For the sample demo data:

- parcel locations are anchored near their real suburbs
- parcel footprints are illustrative, not actual cadastral boundaries
- buildings are extruded using `building_height_m`
- backyard patches are extruded using `score_height_m`

So the 3D scene is real QGIS 3D rendering, but the geometry is still demo geometry unless you build from real GIS layers.

## How To Upgrade To Real Data

Use real:

- parcel polygons
- building polygons
- DEM raster
- optional vegetation, flood, heritage, and powerline layers

Then run:

```powershell
.\scripts\build_property_features.ps1 `
  -Parcels "C:\path\to\parcels.gpkg" `
  -Buildings "C:\path\to\buildings.gpkg" `
  -Dem "C:\path\to\dem.tif" `
  -Output "C:\Users\Shreyash\Downloads\aus_backyard_intelligence_workspace\aus_backyard_intelligence\data\derived\property_features_v1.json"
```

After that:

- the app reads real property geometry
- the QGIS export uses real parcel, building, and backyard shapes
- the 3D view becomes much more meaningful

## What To Tell In A Demo

You can describe it like this:

- the platform ranks backyard opportunity at parcel level
- pool suitability is the flagship use case
- the system is terrain-aware rather than only area-based
- the same feature table supports fence and deck recommendations
- the QGIS scene lets you inspect the selected ranked parcels spatially in 2D and 3D
