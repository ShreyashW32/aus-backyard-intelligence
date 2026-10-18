import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd

from shapely.geometry import Polygon

from aus_backyard_intelligence.map_export import (
    analyze_area,
    export_area_geopackage,
    export_ranked_map,
    is_point_in_polygon_py,
    osm_feature_intersects_polygon,
    validate_analysis_polygon,
)


class MapExportTestCase(unittest.TestCase):
    def test_export_ranked_map_creates_qgis_package(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            output_path = Path(temp_dir_name) / "ranked_selection.gpkg"
            result = export_ranked_map(
                output_path=output_path,
                module_name="pool_3d",
                lga="Moreton Bay",
                top_k=2,
            )
            self.assertTrue(output_path.exists())
            self.assertEqual(result["layers"]["ranked_parcels"], 2)
            parcels = gpd.read_file(output_path, layer="ranked_parcels")
            backyards = gpd.read_file(output_path, layer="ranked_backyards")
            self.assertEqual(len(parcels), 2)
            self.assertEqual(parcels.iloc[0]["lga"], "Moreton Bay")
            self.assertTrue(parcels.geometry.iloc[0].area > 0)
            self.assertTrue(bool(parcels.iloc[0]["private_property"]))
            self.assertIn("Private residential", parcels.iloc[0]["candidate_label"])
            self.assertEqual(len(backyards), 2)

    def test_is_point_in_polygon_py(self):
        # Coordinates of a simple box: (0,0) to (10,10)
        poly = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]
        
        # Point inside
        self.assertTrue(is_point_in_polygon_py(5.0, 5.0, poly))
        # Point outside
        self.assertFalse(is_point_in_polygon_py(15.0, 5.0, poly))
        # Point on border or corner
        self.assertFalse(is_point_in_polygon_py(0.0, 0.0, poly))
        # Point outside
        self.assertFalse(is_point_in_polygon_py(-1.0, -1.0, poly))

    def test_analyze_area_with_osm_features(self):
        # A simple box property polygon in coordinates: (153.03, -27.47) to (153.031, -27.469)
        poly = [
            [153.030, -27.470],
            [153.031, -27.470],
            [153.031, -27.469],
            [153.030, -27.469],
            [153.030, -27.470]
        ]
        
        # Mock osm features: one building inside, one building outside, one wood inside, one tree outside
        osm_features = {
            "buildings": [
                {
                    "geometry": [
                        {"lon": 153.0302, "lat": -27.4698},
                        {"lon": 153.0308, "lat": -27.4698},
                        {"lon": 153.0308, "lat": -27.4692},
                        {"lon": 153.0302, "lat": -27.4692},
                        {"lon": 153.0302, "lat": -27.4698}
                    ],
                    "tags": {"building": "yes"}
                },
                {
                    "geometry": [
                        {"lon": 153.0400, "lat": -27.4700},
                        {"lon": 153.0405, "lat": -27.4700},
                        {"lon": 153.0405, "lat": -27.4695},
                        {"lon": 153.0400, "lat": -27.4695},
                        {"lon": 153.0400, "lat": -27.4700}
                    ],
                    "tags": {"building": "yes"}
                }
            ],
            "pools": [
                {
                    "geometry": [
                        {"lon": 153.0303, "lat": -27.4697},
                        {"lon": 153.0304, "lat": -27.4697},
                        {"lon": 153.0304, "lat": -27.4696},
                        {"lon": 153.0303, "lat": -27.4696},
                        {"lon": 153.0303, "lat": -27.4697}
                    ],
                    "tags": {"leisure": "swimming_pool"}
                }
            ],
            "trees": [
                {
                    "type": "node",
                    "lon": 153.0305,
                    "lat": -27.4695,
                    "tags": {"natural": "tree"}
                },
                {
                    "type": "node",
                    "lon": 153.0405,
                    "lat": -27.4695,
                    "tags": {"natural": "tree"}
                },
                {
                    "type": "way",
                    "geometry": [
                        {"lon": 153.0301, "lat": -27.4694},
                        {"lon": 153.0303, "lat": -27.4694},
                        {"lon": 153.0303, "lat": -27.4692},
                        {"lon": 153.0301, "lat": -27.4692},
                        {"lon": 153.0301, "lat": -27.4694}
                    ],
                    "tags": {"natural": "wood"}
                }
            ]
        }
        
        body = {
            "polygon": poly,
            "local_features": osm_features,
            "features": {
                "slope_pct": 2.0,
                "relief_m": 0.5,
                "solar_access_pct": 82.0,
            },
        }
        
        result = analyze_area(body)
        
        # Area Analysis is suitability-only; pool geometry remains available for
        # export/overlay but must not suppress new-pool scoring.
        self.assertFalse(result["has_existing_pool"])
        self.assertEqual(result["selected_area_class"], "site_area")
        # Verify only the features strictly inside the polygon were kept
        self.assertEqual(len(result["geometries"]["buildings"]), 1)
        self.assertEqual(len(result["geometries"]["pools"]), 1)
        self.assertEqual(len(result["geometries"]["trees"]), 2)
        # Verify vegetation_pct is calculated and reflects both the tree and the wood area
        self.assertGreater(result["features"]["vegetation_pct"], 0.0)

    def test_pool_boundary_overlap_is_detected_even_when_centroid_is_outside(self):
        selection = Polygon([[153.0, -27.5], [153.001, -27.5], [153.001, -27.499], [153.0, -27.499]])
        crossing_pool = {
            "type": "way",
            "geometry": [
                {"lon": 152.9998, "lat": -27.4996},
                {"lon": 153.0002, "lat": -27.4996},
            ],
        }
        self.assertTrue(osm_feature_intersects_polygon(crossing_pool, selection))

    def test_large_lap_pool_footprint_is_scored_as_suitability_area(self):
        result = analyze_area(
            {
                "polygon": [
                    [151.2090, -33.8690],
                    [151.20955, -33.8690],
                    [151.20955, -33.86876],
                    [151.2090, -33.86876],
                ],
                "features": {"slope_pct": 9.5, "relief_m": 1.0, "solar_access_pct": 80},
                "local_features": {
                    "buildings": [
                        {
                            "type": "way",
                            "tags": {"building": "roof"},
                            "geometry": [
                                {"lon": 151.20908, "lat": -33.86898},
                                {"lon": 151.20918, "lat": -33.86898},
                                {"lon": 151.20918, "lat": -33.86890},
                                {"lon": 151.20908, "lat": -33.86890},
                            ],
                        }
                    ],
                    "trees": [],
                    "pools": [],
                },
            }
        )
        self.assertFalse(result["has_existing_pool"])
        self.assertEqual(result["selected_area_class"], "site_area")
        self.assertGreater(result["scores"]["pool_3d"], 0.0)
        self.assertGreater(result["scores"]["pool_2d"], 0.0)
        self.assertNotIn("Existing pool present", result["explanation"])
        self.assertNotIn("Challenging terrain", result["explanation"])

    def test_detected_pool_in_square_selection_is_not_reported_unsuitable(self):
        # A detector-supplied pool geometry must not change the suitability workflow.
        result = analyze_area(
            {
                "polygon": [
                    [151.1916, -33.8866],
                    [151.1928, -33.8866],
                    [151.1928, -33.8855],
                    [151.1916, -33.8855],
                ],
                "features": {"slope_pct": 9.2, "relief_m": 5.0, "solar_access_pct": 72},
                "local_features": {
                    "buildings": [],
                    "trees": [],
                    "pools": [
                        {
                            "type": "way",
                            "tags": {
                                "leisure": "swimming_pool",
                                "detector:source": "satellite-pool-detector",
                            },
                            "geometry": [
                                {"lon": 151.19205, "lat": -33.88625},
                                {"lon": 151.19255, "lat": -33.88625},
                                {"lon": 151.19255, "lat": -33.88600},
                                {"lon": 151.19205, "lat": -33.88600},
                            ],
                        }
                    ],
                },
            }
        )
        self.assertFalse(result["has_existing_pool"])
        self.assertEqual(result["selected_area_class"], "site_area")
        self.assertGreater(result["scores"]["pool_3d"], 0.0)
        self.assertGreater(result["scores"]["pool_2d"], 0.0)
        self.assertNotIn("Existing swimming pool confirmed", result["explanation"])
        self.assertNotIn("Challenging terrain", result["explanation"])

    def test_practical_pool_score_does_not_collapse_on_workable_slope_and_flood_index(self):
        result = analyze_area(
            {
                "polygon": [
                    [151.1916, -33.8866],
                    [151.1922, -33.8866],
                    [151.1922, -33.8861],
                    [151.1916, -33.8861],
                ],
                "features": {
                    "building_area_sqm": 171.9,
                    "max_side_clearance_m": 4.6,
                    "slope_pct": 13.9,
                    "relief_m": 7.0,
                    "solar_access_pct": 72.7,
                    "vegetation_pct": 18.6,
                    "flood_risk_pct": 19.8,
                    "soil_drainage_pct": 70.0,
                },
                "local_features": {"buildings": [], "trees": [], "pools": []},
            }
        )
        self.assertFalse(result["has_existing_pool"])
        self.assertGreaterEqual(result["scores"]["pool_3d"], 60.0)
        self.assertEqual(result["features"]["nearby_pool_count"], 0)

    def test_nearby_pool_context_improves_feasibility_without_marking_area_occupied(self):
        body = {
            "polygon": [
                [151.1916, -33.8866],
                [151.1922, -33.8866],
                [151.1922, -33.8861],
                [151.1916, -33.8861],
            ],
            "features": {
                "building_area_sqm": 171.9,
                "max_side_clearance_m": 4.6,
                "slope_pct": 13.9,
                "relief_m": 7.0,
                "solar_access_pct": 72.7,
                "vegetation_pct": 18.6,
                "flood_risk_pct": 19.8,
                "soil_drainage_pct": 70.0,
            },
            "local_features": {"buildings": [], "trees": [], "pools": []},
        }
        baseline = analyze_area(body)
        body["context_features"] = {"nearby_pools": [{"id": "nearby-1"}, {"id": "nearby-2"}]}
        contextual = analyze_area(body)
        self.assertFalse(contextual["has_existing_pool"])
        self.assertEqual(contextual["features"]["nearby_pool_count"], 2)
        self.assertGreater(contextual["scores"]["pool_3d"], baseline["scores"]["pool_3d"])
        self.assertIn("Neighbourhood context", contextual["explanation"])

    def test_large_open_area_is_classified_non_residential_and_reasoned_at_scale(self):
        # A large (~7000 sqm) selection with no buildings is open/recreation land, not a
        # suburban backyard. The report must classify it as non-residential and must not
        # blame a residential "tight side clearance" for a low pool score.
        result = analyze_area(
            {
                "polygon": [
                    [153.0300, -27.4700],
                    [153.0308, -27.4700],
                    [153.0308, -27.4692],
                    [153.0300, -27.4692],
                ],
                "features": {"slope_pct": 2.0, "relief_m": 0.4, "solar_access_pct": 82},
                "local_features": {"buildings": [], "trees": [], "pools": []},
            }
        )
        profile = result["site_profile"]
        self.assertFalse(profile["is_residential"])
        self.assertEqual(profile["kind"], "open_space")
        self.assertIn("AREA GEOGRAPHY", result["explanation"])
        self.assertNotIn("tight side clearance", result["explanation"])

    def test_residential_lot_with_house_is_classified_residential(self):
        result = analyze_area(
            {
                "polygon": [
                    [153.0300, -27.4700],
                    [153.03035, -27.4700],
                    [153.03035, -27.46972],
                    [153.0300, -27.46972],
                ],
                "features": {"slope_pct": 3.0, "relief_m": 0.4, "solar_access_pct": 80},
                "local_features": {
                    "buildings": [
                        {
                            "type": "way",
                            "tags": {"building": "house"},
                            "geometry": [
                                {"lon": 153.03008, "lat": -27.46998},
                                {"lon": 153.03020, "lat": -27.46998},
                                {"lon": 153.03020, "lat": -27.46988},
                                {"lon": 153.03008, "lat": -27.46988},
                            ],
                        }
                    ],
                    "trees": [],
                    "pools": [],
                },
            }
        )
        profile = result["site_profile"]
        self.assertTrue(profile["is_residential"])
        self.assertEqual(profile["kind"], "residential")

    def test_export_area_geopackage_writes_selected_area_and_features(self):
        # The Area-Analysis QGIS export must contain the user's drawn polygon (with the
        # suitability scores as attributes) and the features inside it — not the demo's
        # ranked properties.
        analysis = analyze_area(
            {
                "polygon": [
                    [153.0300, -27.4700],
                    [153.03035, -27.4700],
                    [153.03035, -27.46972],
                    [153.0300, -27.46972],
                ],
                "features": {"slope_pct": 3.0, "relief_m": 0.4, "solar_access_pct": 80},
                "local_features": {
                    "buildings": [
                        {
                            "type": "way",
                            "tags": {"building": "house"},
                            "geometry": [
                                {"lon": 153.03008, "lat": -27.46998},
                                {"lon": 153.03020, "lat": -27.46998},
                                {"lon": 153.03020, "lat": -27.46988},
                                {"lon": 153.03008, "lat": -27.46988},
                            ],
                        }
                    ],
                    "trees": [{"type": "node", "lat": -27.46990, "lon": 153.03025, "tags": {"natural": "tree"}}],
                    "pools": [],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "area.gpkg"
            result = export_area_geopackage(analysis, out)
            self.assertTrue(out.exists())
            self.assertIn("analysis_area", result["layers"])
            area_gdf = gpd.read_file(out, layer="analysis_area")
            self.assertEqual(len(area_gdf), 1)
            self.assertEqual(str(area_gdf.crs).upper(), "EPSG:4326")
            self.assertIn("score_fence", area_gdf.columns)
            self.assertIn("site_kind", area_gdf.columns)
            # The building inside the area is exported as its own layer.
            self.assertEqual(result["layers"].get("buildings"), 1)

    def test_export_area_geopackage_rejects_missing_polygon(self):
        with self.assertRaises(ValueError):
            export_area_geopackage({"scores": {}}, Path(tempfile.gettempdir()) / "nope.gpkg")

    def test_invalid_analysis_polygon_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_analysis_polygon([[153.0, -27.5], [153.1, -27.4], [153.0, -27.4], [153.1, -27.5]])

    @patch("aus_backyard_intelligence.map_export.get_solar_access_with_source")
    @patch("aus_backyard_intelligence.map_export.get_elevation_corners_with_source")
    def test_analysis_reports_external_data_provenance(self, mock_elevation, mock_solar):
        mock_elevation.return_value = (
            {"sw": 10.0, "se": 10.1, "nw": 10.2, "ne": 10.3},
            "open_meteo",
        )
        mock_solar.return_value = (84.0, "pvgis")
        result = analyze_area(
            {
                "polygon": [
                    [153.030, -27.470],
                    [153.031, -27.470],
                    [153.031, -27.469],
                    [153.030, -27.469],
                ]
            }
        )
        self.assertEqual(result["data_sources"]["elevation"], "open_meteo")
        self.assertEqual(result["data_sources"]["solar"], "pvgis")
        self.assertEqual(result["data_warnings"], [])


if __name__ == "__main__":
    unittest.main()
