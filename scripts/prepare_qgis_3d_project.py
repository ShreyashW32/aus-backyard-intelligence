from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgis._3d import QgsPhongMaterialSettings, QgsPolygon3DSymbol, QgsVectorLayer3DRenderer
from qgis.core import (
    Qgis,
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsFillSymbol,
    QgsPalLayerSettings,
    QgsProject,
    QgsProperty,
    QgsPropertyCollection,
    QgsRasterLayer,
    QgsReferencedRectangle,
    QgsRectangle,
    QgsSingleSymbolRenderer,
    QgsTextBufferSettings,
    QgsTextFormat,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtGui import QColor


def _set_2d_style(layer: QgsVectorLayer, fill_color: str, stroke_color: str, opacity: float) -> None:
    symbol = QgsFillSymbol.createSimple(
        {
            "color": fill_color,
            "outline_color": stroke_color,
            "outline_width": "0.5",
        }
    )
    symbol.setOpacity(opacity)
    layer.setRenderer(QgsSingleSymbolRenderer(symbol))


def _set_3d_style(layer: QgsVectorLayer, diffuse_color: str, extrusion_expression: str, height_expression: str = "base_elev_m") -> None:
    symbol = QgsPolygon3DSymbol()
    symbol.setAltitudeClamping(Qgis.AltitudeClamping.Absolute)
    symbol.setAltitudeBinding(Qgis.AltitudeBinding.Centroid)

    properties = QgsPropertyCollection()
    properties.setProperty(QgsPolygon3DSymbol.PropertyHeight, QgsProperty.fromExpression(height_expression))
    properties.setProperty(QgsPolygon3DSymbol.PropertyExtrusionHeight, QgsProperty.fromExpression(extrusion_expression))
    symbol.setDataDefinedProperties(properties)

    material = QgsPhongMaterialSettings()
    material.setAmbient(QColor(diffuse_color))
    material.setDiffuse(QColor(diffuse_color))
    material.setDiffuseCoefficient(0.85)
    symbol.setMaterialSettings(material)

    renderer = QgsVectorLayer3DRenderer()
    renderer.setSymbol(symbol)
    renderer.setLayer(layer)
    layer.setRenderer3D(renderer)


def _set_labels(layer: QgsVectorLayer, expression: str) -> None:
    settings = QgsPalLayerSettings()
    settings.fieldName = expression
    settings.isExpression = True
    settings.placement = Qgis.LabelPlacement.OverPoint

    label_format = QgsTextFormat()
    label_format.setSize(10)
    label_format.setColor(QColor("#113922"))

    buffer = QgsTextBufferSettings()
    buffer.setEnabled(True)
    buffer.setSize(1.2)
    buffer.setColor(QColor("#fff6d6"))
    label_format.setBuffer(buffer)

    settings.setFormat(label_format)
    layer.setLabelsEnabled(True)
    layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))


def build_project(package_path: str | Path, output_project_path: str | Path, terrain_path: str | Path | None = None) -> dict[str, str]:
    package = Path(package_path).resolve()
    project_path = Path(output_project_path).resolve()
    project_path.parent.mkdir(parents=True, exist_ok=True)

    project = QgsProject.instance()
    project.clear()
    project.setFileName(str(project_path))
    project.setTitle("Private Residential Pool and Fence Candidates")
    project.setCrs(QgsProject.instance().crs())

    parcels = QgsVectorLayer(f"{package}|layername=ranked_parcels", "ranked_parcels", "ogr")
    buildings = QgsVectorLayer(f"{package}|layername=ranked_buildings", "ranked_buildings", "ogr")
    backyards = QgsVectorLayer(f"{package}|layername=ranked_backyards", "ranked_backyards", "ogr")

    if not parcels.isValid() or not buildings.isValid() or not backyards.isValid():
        raise RuntimeError("One or more layers in the exported GeoPackage could not be loaded.")

    _set_2d_style(parcels, "255,214,10,35", "167,116,0,255", 0.35)
    _set_2d_style(buildings, "120,120,120,200", "60,60,60,255", 0.9)
    _set_2d_style(backyards, "20,180,90,90", "20,110,70,255", 0.6)

    _set_3d_style(parcels, "#d4a017", "0.3")
    _set_3d_style(buildings, "#7a7a7a", "building_height_m")
    _set_3d_style(backyards, "#1d9b5f", "score_height_m")
    _set_labels(parcels, "\"candidate_label\" || '\n' || \"address\"")

    # Google Satellite — sharp up to zoom 21 (street-level)
    google_sat_uri = (
        "type=xyz"
        "&url=https://mt1.google.com/vt/lyrs%3Ds%26x%3D{x}%26y%3D{y}%26z%3D{z}"
        "&zmin=0&zmax=21"
    )
    google_satellite = QgsRasterLayer(google_sat_uri, "Google Satellite", "wms")

    # Queensland Government high-res aerial imagery
    qld_imagery_uri = (
        "type=xyz"
        "&url=https://spatial-gis.information.qld.gov.au/arcgis/rest/services/Basemaps/QldBase_Imagery/MapServer/tile/{z}/{y}/{x}"
        "&zmin=0&zmax=21"
    )
    qld_imagery = QgsRasterLayer(qld_imagery_uri, "QLD Aerial Imagery", "wms")


    layers = [parcels, buildings, backyards]
    if terrain_path:
        terrain = QgsRasterLayer(str(Path(terrain_path).resolve()), "terrain")
        if terrain.isValid():
            layers.insert(0, terrain)
    # Add basemap layers at the bottom (QLD imagery first, then Google on top for sharper zoom)
    if qld_imagery.isValid():
        layers.append(qld_imagery)
    if google_satellite.isValid():
        layers.append(google_satellite)
    for layer in layers:
        project.addMapLayer(layer)

    if parcels.crs().isValid():
        project.setCrs(parcels.crs())
        extent = QgsRectangle(parcels.extent())
        # Grow extent more so satellite context is visible around the parcels
        extent.grow(max(extent.width(), extent.height()) * 1.5)
        referenced_extent = QgsReferencedRectangle(extent, parcels.crs())
        project.viewSettings().setDefaultViewExtent(referenced_extent)
        project.viewSettings().setPresetFullExtent(referenced_extent)
    project.write(str(project_path))
    return {"project_path": str(project_path), "package_path": str(package)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a QGIS project with 3D-ready styling for the ranked backyard layers.")
    parser.add_argument("--package", required=True, help="Path to the GeoPackage created by map_export.py.")
    parser.add_argument("--output-project", required=True, help="Path to the output QGIS project file (.qgz).")
    parser.add_argument("--terrain", help="Optional terrain raster to add to the project.")
    args = parser.parse_args()

    qgs = QgsApplication([], False)
    qgs.initQgis()
    try:
        result = build_project(args.package, args.output_project, args.terrain)
        print(json.dumps(result, indent=2))
    finally:
        qgs.exitQgis()


if __name__ == "__main__":
    main()
