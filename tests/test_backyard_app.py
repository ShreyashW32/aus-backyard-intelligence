import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from wsgiref.util import setup_testing_defaults

from aus_backyard_intelligence.app import create_app, derive_features


def call_app(app, path, method="GET", query="", body=None):
    environ = {}
    setup_testing_defaults(environ)
    raw = json.dumps(body).encode("utf-8") if body is not None else b""
    environ["REQUEST_METHOD"] = method
    environ["PATH_INFO"] = path
    environ["QUERY_STRING"] = query
    environ["CONTENT_TYPE"] = "application/json"
    environ["CONTENT_LENGTH"] = str(len(raw))
    environ["wsgi.input"] = io.BytesIO(raw)
    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = dict(headers)

    response = b"".join(app(environ, start_response)).decode("utf-8")
    return captured["status"], captured["headers"], response


class BackyardAppTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.data_path = Path(cls.temp_dir.name) / "test_properties.json"
        
        mock_data = [
            {
                "id": "MBRC-1001",
                "lga": "Moreton Bay",
                "suburb": "North Lakes",
                "address": "1 Mock Street",
                "parcel_rect": [0.0, 0.0, 20.0, 40.0],
                "building_rect": [3.0, 5.0, 16.0, 20.0],
                "elevation_corners_m": {"sw": 12.0, "se": 12.4, "nw": 13.0, "ne": 13.3},
                "vegetation_pct": 10.0,
                "road_frontage_m": 20.0,
                "corner_lot": False,
                "flood_risk": 0,
                "heritage_overlay": False,
                "overhead_powerline": False,
                "solar_access_pct": 82.0,
                "view_quality": 5,
                "land_use": "Residential",
                "ownership_class": "Private",
                "real_lat": -27.24,
                "real_lon": 153.03,
            },
            {
                "id": "MBRC-1003",
                "lga": "Moreton Bay",
                "suburb": "North Lakes",
                "address": "3 Mock Street",
                "parcel_rect": [0.0, 0.0, 20.0, 40.0],
                "building_rect": [3.0, 5.0, 16.0, 20.0],
                "elevation_corners_m": {"sw": 15.0, "se": 15.4, "nw": 16.0, "ne": 16.3},
                "vegetation_pct": 10.0,
                "road_frontage_m": 20.0,
                "corner_lot": False,
                "flood_risk": 0,
                "heritage_overlay": False,
                "overhead_powerline": False,
                "solar_access_pct": 82.0,
                "view_quality": 5,
                "land_use": "Residential",
                "ownership_class": "Private",
                "real_lat": -27.241,
                "real_lon": 153.031,
            },
            {
                "id": "MBRC-2001",
                "lga": "Moreton Bay",
                "suburb": "North Lakes",
                "address": "21 North Lakes Drive",
                "parcel_rect": [0.0, 0.0, 20.0, 40.0],
                "building_rect": [3.0, 5.0, 16.0, 20.0],
                "elevation_corners_m": {"sw": 12.0, "se": 12.4, "nw": 13.0, "ne": 13.3},
                "vegetation_pct": 10.0,
                "road_frontage_m": 20.0,
                "corner_lot": False,
                "flood_risk": 0,
                "heritage_overlay": False,
                "overhead_powerline": False,
                "solar_access_pct": 82.0,
                "view_quality": 5,
                "land_use": "Residential",
                "ownership_class": "Private",
                "real_lat": -27.242,
                "real_lon": 153.032,
                "has_existing_pool": True,
            },
            {
                "id": "MBRC-2002",
                "lga": "Moreton Bay",
                "suburb": "North Lakes",
                "address": "22 North Lakes Drive",
                "parcel_rect": [0.0, 0.0, 20.0, 40.0],
                "building_rect": [3.0, 5.0, 16.0, 20.0],
                "elevation_corners_m": {"sw": 12.0, "se": 12.4, "nw": 13.0, "ne": 13.3},
                "vegetation_pct": 10.0,
                "road_frontage_m": 20.0,
                "corner_lot": False,
                "flood_risk": 0,
                "heritage_overlay": False,
                "overhead_powerline": False,
                "solar_access_pct": 82.0,
                "view_quality": 5,
                "land_use": "Residential",
                "ownership_class": "Private",
                "real_lat": -27.243,
                "real_lon": 153.033,
            },
            {
                "id": "MBRC-2003",
                "lga": "Moreton Bay",
                "suburb": "North Lakes",
                "address": "23 North Lakes Drive",
                "parcel_rect": [0.0, 0.0, 20.0, 40.0],
                "building_rect": [3.0, 5.0, 16.0, 20.0],
                "elevation_corners_m": {"sw": 12.0, "se": 12.4, "nw": 13.0, "ne": 13.3},
                "vegetation_pct": 10.0,
                "road_frontage_m": 20.0,
                "corner_lot": False,
                "flood_risk": 0,
                "heritage_overlay": False,
                "overhead_powerline": False,
                "solar_access_pct": 82.0,
                "view_quality": 5,
                "land_use": "Residential",
                "ownership_class": "Private",
                "real_lat": -27.244,
                "real_lon": 153.034,
            },
            {
                "id": "MBRC-2004",
                "lga": "Moreton Bay",
                "suburb": "North Lakes",
                "address": "24 North Lakes Drive",
                "parcel_rect": [0.0, 0.0, 20.0, 40.0],
                "building_rect": [3.0, 5.0, 16.0, 20.0],
                "elevation_corners_m": {"sw": 12.0, "se": 12.4, "nw": 13.0, "ne": 13.3},
                "vegetation_pct": 10.0,
                "road_frontage_m": 20.0,
                "corner_lot": False,
                "flood_risk": 0,
                "heritage_overlay": False,
                "overhead_powerline": False,
                "solar_access_pct": 82.0,
                "view_quality": 5,
                "land_use": "Residential",
                "ownership_class": "Private",
                "real_lat": -27.245,
                "real_lon": 153.035,
            }
        ]
        cls.data_path.write_text(json.dumps(mock_data, indent=2), encoding="utf-8")
        cls.app = create_app(data_path=cls.data_path)
        cls.sample_property = cls.app.get_property("MBRC-1001")

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_derived_features_include_backyard_and_slope(self):
        features = derive_features(self.sample_property["property"])
        self.assertGreater(features["backyard_area_sqm"], 300.0)
        self.assertLess(features["slope_pct"], 4.0)

    def test_pool_3d_ranking_prefers_flatter_moreton_bay_site(self):
        rows = self.app.rankings("pool_3d", lga="Moreton Bay", top_k=4)
        self.assertEqual(rows[0]["property_id"], "MBRC-1001")
        self.assertGreaterEqual(rows[0]["module_score"], rows[-1]["module_score"])

    def test_health_endpoint(self):
        status, _, body = call_app(self.app, "/api/health")
        payload = json.loads(body)
        self.assertEqual(status, "200 OK")
        self.assertEqual(payload["status"], "ok")
        self.assertIn("Moreton Bay", payload["lgas"])

    def test_selected_property_3d_payload_is_scoped_and_provenanced(self):
        status, _, body = call_app(self.app, "/api/viewer/property/MBRC-1001")
        payload = json.loads(body)
        self.assertEqual(status, "200 OK")
        self.assertEqual(payload["property_id"], "MBRC-1001")
        self.assertEqual(payload["geometries"]["parcel_geom"]["type"], "Polygon")
        self.assertEqual(payload["geometries"]["building_geom"]["type"], "Polygon")
        self.assertGreater(payload["elevations"]["building_height_m"], 0)
        self.assertEqual(payload["elevations"]["building_height_source"], "derived estimate")
        self.assertGreater(payload["features"]["land_area_sqm"], 0)
        self.assertGreater(payload["features"]["building_area_sqm"], 0)
        self.assertEqual(payload["geometry_provenance"]["scope"], "selected-property-only")
        self.assertIn(payload["geometry_provenance"]["geometry_source"], {
            "real", "geocoded-anchor", "suburb-anchor"
        })
        self.assertEqual(payload["geometry_provenance"]["roof_source"], "procedural massing")

    def test_3d_page_describes_an_indicative_selected_property_model(self):
        status, _, body = call_app(self.app, "/3d?propertyId=MBRC-1001")
        self.assertEqual(status, "200 OK")
        self.assertIn("Selected property · interactive 3D", body)
        self.assertIn("Indicative model.", body)
        self.assertIn("Open property leads", body)
        self.assertNotIn("Ground-Truth 3D Model", body)

    def test_detection_status_endpoint(self):
        status, _, body = call_app(self.app, "/api/detection/status")
        payload = json.loads(body)
        self.assertEqual(status, "200 OK")
        self.assertEqual(payload["default_model"], "yolov11n")
        self.assertIn("yolov11n", payload["supported_models"])
        # The hybrid reports its active detector. The exact value depends on
        # whether pool-trained weights are installed in this environment.
        self.assertIn("pool_yolo_available", payload)
        self.assertEqual(
            payload["detector_process_capacity"]["queue"]["mode"],
            "single_worker_fifo",
        )
        self.assertIn(payload["primary_pool_detector"], {
            "pool_trained_yolo",
            "satellite_water_shape_cv",
            "openstreetmap_mapped_pools_only",
        })
        if payload["pool_yolo_available"]:
            self.assertEqual(payload["primary_pool_detector"], "pool_trained_yolo")
            self.assertTrue(payload["yolo_available"])
        self.assertEqual(len(payload["detector_signature"]), 20)
        self.assertIn("pool_base", payload["detector_thresholds"])
        for weights in payload["active_pool_weights"]:
            self.assertEqual(len(weights["sha256"]), 64)
            self.assertTrue(weights["name"])
        self.assertIn("sam_geo_available", payload)
        self.assertIn("mask_rcnn_available", payload)
        self.assertIn("detector_process_capacity", payload)
        self.assertIn("available", payload["detector_process_capacity"])

    def test_viewer_explains_selected_area_detector_scope(self):
        status, headers, body = call_app(self.app, "/")
        self.assertEqual(status, "200 OK")
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        self.assertIn("Finishing a selection only saves the area", body)
        self.assertIn("Press Enter to open Area Analysis", body)
        self.assertIn("AI Detection Scans", body)
        self.assertIn("Choose Property for 3D", body)
        self.assertNotIn("realtime pool-trained YOLO", body)
        self.assertIn("Enter-only area analysis", body)
        self.assertNotIn('id="run-area-analysis-btn"', body)
        self.assertIn('id="run-ai-detection-btn"', body)
        self.assertIn('id="run-agent-scan-btn"', body)
        self.assertIn("AI detection scan", body)
        self.assertIn("Deep pool scan", body)
        self.assertNotIn('id="auto-agent-scan"', body)
        self.assertIn('id="overlay-solar-panels"', body)
        self.assertNotIn('id="overlay-solar-panels" checked', body)
        self.assertIn("Experimental solar candidates (unvalidated)", body)
        self.assertIn('id="osiris-place-scan-btn"', body)
        self.assertIn('id="osiris-intel-panel"', body)
        self.assertIn('id="osiris-pool-list"', body)
        self.assertNotIn('id="detector-model"', body)
        self.assertNotIn('id="segmentation-mode"', body)
        self.assertNotIn("Worldwide OSM & Feeds", body)
        self.assertNotIn("yolo12", body.lower())

    def test_client_has_no_automatic_viewport_inference_loop(self):
        root = Path(__file__).resolve().parents[1]
        map_source = (root / "aus_backyard_intelligence" / "static" / "js" / "components" / "map.js").read_text(encoding="utf-8")
        api_source = (root / "aus_backyard_intelligence" / "static" / "js" / "services" / "api.js").read_text(encoding="utf-8")
        self.assertNotIn("continuousDetectionTimer", map_source)
        self.assertNotIn("realtime-heartbeat", map_source)
        self.assertIn("options.source !== 'selected-area-ai-scan'", api_source)
        self.assertIn("analysis_polygon: JSON.stringify(selectionCoords)", api_source)

    def test_selected_area_client_keeps_analysis_and_ai_scans_separate(self):
        root = Path(__file__).resolve().parents[1]
        static_js = root / "aus_backyard_intelligence" / "static" / "js"
        api_source = (static_js / "services" / "api.js").read_text(encoding="utf-8")
        drawing_source = (static_js / "components" / "drawing.js").read_text(encoding="utf-8")
        ui_source = (static_js / "components" / "ui.js").read_text(encoding="utf-8")

        self.assertIn("const activeLayers = ['pools', 'buildings', 'trees']", api_source)
        analysis_source = api_source[api_source.index("export async function analyzePolygon"):api_source.index("export async function scanSelectedAreaForAIDetections")]
        detection_source = api_source[api_source.index("export async function scanSelectedAreaForAIDetections"):api_source.index("// Dev-mode")]
        self.assertNotIn("fetchAIDataInView", analysis_source)
        self.assertIn("fetch('/api/analyze/area'", analysis_source)
        self.assertIn("fetchAIDataInView", detection_source)
        self.assertNotIn("analyzePolygon", detection_source)
        self.assertNotIn("startAreaScan", detection_source)
        finalize_source = drawing_source[drawing_source.index("export function finalizeSelection"):drawing_source.index("export function finishPolygon")]
        enter_source = drawing_source[drawing_source.index("export function analyzeSelection"):drawing_source.index("function showSelectionActionHint")]
        self.assertNotIn("analyzePolygon", finalize_source)
        self.assertIn("coords = finalizeSelection()", enter_source)
        self.assertIn("analyzePolygon(coords)", enter_source)
        self.assertNotIn("triggerAutoScan", api_source)
        self.assertNotIn("autoScanSelections", api_source)
        self.assertNotIn("clearAnalysisScopedFeatures({ render: false })", api_source[api_source.index("export async function startAreaScan(area"):api_source.index("export async function startPlaceScan")])
        self.assertNotIn("analyzeSelectedArea", ui_source)
        self.assertIn("await api.scanSelectedAreaForAIDetections", ui_source)
        self.assertIn("await api.startAreaScanFromCurrentView", ui_source)

    def test_new_selection_retires_old_scan_and_invalidates_late_polls(self):
        root = Path(__file__).resolve().parents[1]
        static_js = root / "aus_backyard_intelligence" / "static" / "js"
        api_source = (static_js / "services" / "api.js").read_text(encoding="utf-8")
        drawing_source = (static_js / "components" / "drawing.js").read_text(encoding="utf-8")

        retirement_source = api_source[
            api_source.index("export async function retireActiveAreaScan"):
            api_source.index("function selectedScanPasses")
        ]
        self.assertIn("++state.agentScanPollGeneration", retirement_source)
        self.assertIn("clearTimeout(state.agentScanPollTimer)", retirement_source)
        self.assertIn("/cancel", retirement_source)
        self.assertIn("scanPollIsCurrent", api_source)
        self.assertIn("payload.replace_active = options.replaceActive !== false", api_source)

        finalize_source = drawing_source[
            drawing_source.index("export function finalizeSelection"):
            drawing_source.index("export function finishPolygon")
        ]
        clear_source = drawing_source[
            drawing_source.index("export function clearAnalysisPolygon"):
            drawing_source.index("export function startDrawing")
        ]
        start_source = drawing_source[
            drawing_source.index("export function startDrawing"):
            drawing_source.index("export function startPoolMarking")
        ]
        self.assertIn("retireActiveAreaScan('new-selection-finalized')", finalize_source)
        self.assertIn("retireActiveAreaScan('selection-cleared')", clear_source)
        self.assertIn("retireActiveAreaScan('new-selection-started')", start_source)

        polling_source = api_source[
            api_source.index("function startAreaScanPolling"):
            api_source.index("function featureCountText")
        ]
        catch_position = polling_source.index("} catch (error) {")
        missed_position = polling_source.index("missedPolls += 1", catch_position)
        delay_position = polling_source.index("const delay =", missed_position)
        self.assertIn("scanPollIsCurrent", polling_source[catch_position:missed_position])
        self.assertIn("scanPollIsCurrent", polling_source[missed_position:delay_position])

    def test_scan_requests_are_bounded_and_snapshots_reconcile(self):
        root = Path(__file__).resolve().parents[1]
        api_source = (
            root / "aus_backyard_intelligence" / "static" / "js" / "services" / "api.js"
        ).read_text(encoding="utf-8")

        self.assertIn("const SCAN_CANCEL_TIMEOUT_MS", api_source)
        self.assertIn("const SCAN_POLL_TIMEOUT_MS", api_source)
        self.assertIn("const SCAN_START_TIMEOUT_MS", api_source)
        self.assertIn("const PLACE_SCAN_START_TIMEOUT_MS", api_source)
        self.assertGreaterEqual(api_source.count("fetchWithDeadline("), 5)
        self.assertIn(
            "JSON.stringify({ area: place, passes, radius_km: radiusKm, replace_active: true })",
            api_source,
        )
        self.assertGreaterEqual(api_source.count("state.agentScanStartInFlight = false"), 2)
        self.assertIn("state.agentScanCancellationInFlight = false", api_source)

        ingest_source = api_source[
            api_source.index("function ingestScanFeatures"):
            api_source.index("function layerSet")
        ]
        self.assertIn("authoritative snapshot", ingest_source)
        self.assertIn("ingestFeatureElements(elements, { scanId: job.scan_id })", ingest_source)
        self.assertNotIn("if (!elements.length) return", ingest_source)
        self.assertIn("reconcileOwnedSnapshot", api_source)
        self.assertIn("state.scanOwnedFeatureSnapshots.clear()", api_source)
        self.assertIn("function restoreAndClearPriorScanSnapshots", api_source)

        start_source = api_source[
            api_source.index("export async function startAreaScan"):
            api_source.index("export async function startPlaceScan")
        ]
        restore_position = start_source.index("restoreAndClearPriorScanSnapshots()")
        self.assertIn("state.agentScanActiveId = result.scan_id", start_source[restore_position:])
        place_source = api_source[
            api_source.index("export async function startPlaceScan"):
            api_source.index("export async function startAreaScanFromCurrentView")
        ]
        self.assertIn("await retireActiveAreaScan('place-scan-started')", place_source)

    def test_dashboard_accepts_area_input_and_agent_scan(self):
        status, headers, body = call_app(self.app, "/dashboard")
        self.assertEqual(status, "200 OK")
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        self.assertIn('id="area-query"', body)
        self.assertIn("Sydney NSW", body)
        self.assertIn("/api/agent/scan-place", body)
        # The worldwide OSINT feeds panel was removed; it must not regress back in.
        self.assertNotIn("Worldwide Data Feeds", body)

    def test_standalone_osm_dashboard_is_removed(self):
        status, _, body = call_app(self.app, "/osm-dashboard")
        self.assertEqual(status, "404 Not Found")
        self.assertIn("Route not found", body)

    @patch("aus_backyard_intelligence.scan_agent.AreaScanAgent._run_job", return_value=None)
    def test_agent_scan_endpoint_starts_background_job(self, _mock_run):
        status, _, body = call_app(
            self.app,
            "/api/agent/scan-area",
            method="POST",
            body={
                "polygon": [[153.03, -27.47], [153.031, -27.47], [153.031, -27.469]],
                "passes": 5,
                "source": "test",
                "label": "Test scan",
            },
        )
        payload = json.loads(body)
        self.assertEqual(status, "202 Accepted")
        self.assertEqual(payload["passes_requested"], 5)
        self.assertEqual(payload["label"], "Test scan")
        self.assertTrue(payload["scan_id"].startswith("scan-"))

        status, _, body = call_app(self.app, "/api/agent/scans", query="limit=3")
        scans = json.loads(body)["items"]
        self.assertEqual(status, "200 OK")
        self.assertTrue(any(item["scan_id"] == payload["scan_id"] for item in scans))

    def test_agent_scan_cancel_endpoint_is_idempotent_and_clears_area_results(self):
        scan_id = "scan-endpoint-cancel"
        with self.app.scan_agent._lock:
            self.app.scan_agent._jobs[scan_id] = {
                "scan_id": scan_id,
                "status": "running",
                "source": "test",
                "label": "Old selected area",
                "passes_requested": 3,
                "passes_completed": 1,
                "created_at": 1,
                "updated_at": 1,
                "bbox": {"south": -33.89, "west": 151.17, "north": -33.88, "east": 151.18},
                "polygon": [[151.17, -33.89], [151.18, -33.89], [151.18, -33.88]],
                "counts": {"pools": 1, "solar": 1},
                "leads": [{"id": "old-lead"}],
                "feature_elements": [{"id": "old-detection"}],
                "errors": [],
                "summary": "Running",
            }

        first_status, _, first_body = call_app(
            self.app,
            f"/api/agent/scans/{scan_id}/cancel",
            method="POST",
            body={"reason": "selection-changed"},
        )
        second_status, _, second_body = call_app(
            self.app,
            f"/api/agent/scans/{scan_id}/cancel",
            method="POST",
            body={"reason": "selection-changed"},
        )

        first = json.loads(first_body)
        second = json.loads(second_body)
        self.assertEqual(first_status, "200 OK")
        self.assertEqual(second_status, "200 OK")
        self.assertEqual(first["status"], "cancelled")
        self.assertEqual(second["status"], "cancelled")
        self.assertEqual(first["feature_elements"], [])
        self.assertEqual(first["leads"], [])
        self.assertEqual(first["counts"], {"pools": 0, "solar": 0})

    def test_agent_scan_rejects_invalid_passes(self):
        status, _, body = call_app(
            self.app,
            "/api/agent/scan-area",
            method="POST",
            body={
                "polygon": [[153.03, -27.47], [153.031, -27.47], [153.031, -27.469]],
                "passes": 99,
            },
        )
        self.assertEqual(status, "400 Bad Request")
        self.assertIn("passes must be between 3 and 40", json.loads(body)["error"])

    @patch("aus_backyard_intelligence.app.google_geocoding_configured", return_value=False)
    def test_agent_leads_reports_exact_address_provider_status(self, _mock_google_configured):
        status, _, body = call_app(self.app, "/api/agent/leads", query="limit=5")

        payload = json.loads(body)
        self.assertEqual(status, "200 OK")
        self.assertEqual(payload["items"], [])
        self.assertFalse(payload["address_lookup"]["google_configured"])
        self.assertIsNone(payload["address_lookup"]["exact_provider"])
        self.assertEqual(payload["address_lookup"]["unit_policy"], "explicit_source_only")

    @patch("aus_backyard_intelligence.scan_agent.AreaScanAgent._run_job", return_value=None)
    @patch("aus_backyard_intelligence.external_api.geocode_places")
    def test_agent_scan_place_starts_australian_area_scan(self, mock_geocode, _mock_run):
        mock_geocode.return_value = [
            {
                "display_name": "Visa, Barangaroo, Sydney, New South Wales, Australia",
                "lat": "-33.8644304",
                "lon": "151.2022166",
                "type": "financial",
                "class": "office",
                "country_code": "au",
                "boundingbox": ["-33.8644804", "-33.8643804", "151.2021666", "151.2022666"],
            },
            {
                "display_name": "Sydney, Council of the City of Sydney, New South Wales, Australia",
                "lat": "-33.8688",
                "lon": "151.2093",
                "type": "city",
                "class": "place",
                "country_code": "au",
                "boundingbox": ["-33.95", "-33.80", "151.15", "151.28"],
            }
        ]
        status, _, body = call_app(
            self.app,
            "/api/agent/scan-place",
            method="POST",
            body={"area": "Sydney NSW", "passes": 6, "radius_km": 3},
        )
        payload = json.loads(body)
        self.assertEqual(status, "202 Accepted")
        self.assertEqual(payload["passes_requested"], 6)
        self.assertEqual(payload["source"], "dashboard-place")
        self.assertEqual(payload["place"]["country_code"], "au")
        self.assertIn("Council of the City of Sydney", payload["place"]["display_name"])

    @patch("aus_backyard_intelligence.scan_agent.AreaScanAgent.start_scan")
    @patch("aus_backyard_intelligence.external_api.geocode_places")
    def test_agent_scan_place_forwards_only_explicit_boolean_replacement(
        self,
        mock_geocode,
        mock_start_scan,
    ):
        mock_geocode.return_value = [{
            "display_name": "Sydney, New South Wales, Australia",
            "lat": "-33.8688",
            "lon": "151.2093",
            "type": "city",
            "class": "place",
            "country_code": "au",
            "boundingbox": ["-33.95", "-33.80", "151.15", "151.28"],
        }]
        mock_start_scan.return_value = {
            "scan_id": "scan-place-replace",
            "status": "queued",
            "polygon": [[151.15, -33.95], [151.28, -33.95], [151.28, -33.80]],
            "source": "dashboard-place",
        }

        status, _, _body = call_app(
            self.app,
            "/api/agent/scan-place",
            method="POST",
            body={"area": "Sydney NSW", "passes": 6, "replace_active": True},
        )
        self.assertEqual(status, "202 Accepted")
        self.assertIs(mock_start_scan.call_args.args[0]["replace_active"], True)

        status, _, _body = call_app(
            self.app,
            "/api/agent/scan-place",
            method="POST",
            body={"area": "Sydney NSW", "passes": 6, "replace_active": "true"},
        )
        self.assertEqual(status, "202 Accepted")
        self.assertIs(mock_start_scan.call_args.args[0]["replace_active"], False)

    def test_rankings_endpoint_returns_requested_module(self):
        status, _, body = call_app(self.app, "/api/rankings", query="module=fence&lga=Moreton+Bay&top_k=2")
        payload = json.loads(body)
        self.assertEqual(status, "200 OK")
        self.assertEqual(payload["metadata"]["module"], "fence")
        self.assertEqual(len(payload["items"]), 2)

    def test_property_endpoint(self):
        status, _, body = call_app(self.app, "/api/property/MBRC-1003")
        payload = json.loads(body)
        self.assertEqual(status, "200 OK")
        self.assertEqual(payload["property"]["id"], "MBRC-1003")
        self.assertIn("pool_3d", payload["scores"])

    def test_csv_endpoint(self):
        status, headers, body = call_app(self.app, "/api/rankings.csv", query="module=pool_3d&lga=Moreton+Bay&top_k=3")
        self.assertEqual(status, "200 OK")
        self.assertEqual(headers["Content-Type"], "text/csv; charset=utf-8")
        self.assertIn("property_id", body)
        self.assertIn("MBRC-1001", body)

    def test_refresh_area_endpoint(self):
        status, _, body = call_app(self.app, "/api/refresh-area", method="POST", body={"lga": "Moreton Bay"})
        payload = json.loads(body)
        self.assertEqual(status, "200 OK")
        self.assertEqual(payload["properties_recomputed"], 6)

    def test_unknown_module_is_rejected(self):
        status, _, body = call_app(self.app, "/api/rankings", query="module=solar")
        payload = json.loads(body)
        self.assertEqual(status, "400 Bad Request")
        self.assertIn("Unsupported module", payload["error"])

    def test_invalid_top_k_is_rejected(self):
        status, _, body = call_app(self.app, "/api/rankings", query="top_k=not-a-number")
        self.assertEqual(status, "400 Bad Request")
        self.assertIn("top_k must be an integer", json.loads(body)["error"])

    def test_empty_analysis_request_is_rejected(self):
        status, _, body = call_app(self.app, "/api/analyze/area", method="POST", body={})
        self.assertEqual(status, "400 Bad Request")
        self.assertIn("Provide a polygon", json.loads(body)["error"])

    def test_analysis_rejects_invalid_numeric_features(self):
        status, _, body = call_app(
            self.app,
            "/api/analyze/area",
            method="POST",
            body={
                "polygon": [[153.03, -27.47], [153.031, -27.47], [153.031, -27.469]],
                "features": {"slope_pct": "steep"},
            },
        )
        self.assertEqual(status, "400 Bad Request")
        self.assertIn("features.slope_pct must be a number", json.loads(body)["error"])

    def test_analysis_rejects_invalid_bounds_without_500(self):
        status, _, body = call_app(
            self.app,
            "/api/analyze/area",
            method="POST",
            body={"bounds": {"south": "bad", "north": -27.46, "west": 153.03, "east": 153.04}},
        )
        self.assertEqual(status, "400 Bad Request")
        self.assertIn("bounds south, north, west, and east", json.loads(body)["error"])

    def test_manual_precise_pool_marks_existing_pool(self):
        status, _, body = call_app(
            self.app,
            "/api/analyze/area",
            method="POST",
            body={
                "polygon": [[153.03, -27.47], [153.031, -27.47], [153.031, -27.469], [153.03, -27.469]],
                "features": {"slope_pct": 2, "relief_m": 0.2, "solar_access_pct": 80},
                "local_features": {
                    "buildings": [],
                    "trees": [],
                    "pools": [
                        {
                            "type": "way",
                            "id": "manual-pool-1",
                            "tags": {"leisure": "swimming_pool", "source": "manual_user_trace"},
                            "geometry": [
                                {"lon": 153.0302, "lat": -27.4698},
                                {"lon": 153.0304, "lat": -27.4698},
                                {"lon": 153.0304, "lat": -27.4696},
                                {"lon": 153.0302, "lat": -27.4696},
                            ],
                        }
                    ],
                },
            },
        )
        payload = json.loads(body)
        self.assertEqual(status, "200 OK")
        self.assertFalse(payload["has_existing_pool"])
        self.assertFalse(payload["features"]["has_existing_pool"])
        self.assertGreater(payload["scores"]["pool_3d"], 0.0)
        self.assertGreater(payload["scores"]["pool_2d"], 0.0)

    def test_pool_sized_selected_area_is_not_reported_as_unsuitable(self):
        status, _, body = call_app(
            self.app,
            "/api/analyze/area",
            method="POST",
            body={
                "polygon": [
                    [153.0300, -27.4700],
                    [153.0301, -27.4700],
                    [153.0301, -27.46992],
                    [153.0300, -27.46992],
                ],
                "features": {"slope_pct": 2, "relief_m": 0.2, "solar_access_pct": 80},
                "local_features": {"buildings": [], "trees": [], "pools": []},
            },
        )
        payload = json.loads(body)
        self.assertEqual(status, "200 OK")
        self.assertFalse(payload["has_existing_pool"])
        self.assertEqual(payload["selected_area_class"], "site_area")
        self.assertGreater(payload["scores"]["pool_3d"], 0.0)
        self.assertNotIn("Existing pool present", payload["explanation"])
        self.assertNotIn("Challenging terrain", payload["explanation"])

    def test_malformed_osm_features_do_not_crash_area_analysis(self):
        status, _, body = call_app(
            self.app,
            "/api/analyze/area",
            method="POST",
            body={
                "polygon": [[153.03, -27.47], [153.031, -27.47], [153.031, -27.469]],
                "features": {"slope_pct": 2, "relief_m": 0.2, "solar_access_pct": 80},
                "local_features": {"buildings": ["bad"], "pools": [{"type": "way", "geometry": "bad"}], "trees": []},
            },
        )
        self.assertEqual(status, "200 OK")
        self.assertFalse(json.loads(body)["has_existing_pool"])

    def test_area_qgis_export_requires_a_polygon(self):
        # The area export endpoint must reject a payload with no polygon (400), not crash.
        status, _, body = call_app(
            self.app, "/api/export/qgis/area", method="POST", body={"scores": {}}
        )
        self.assertEqual(status, "400 Bad Request")
        self.assertIn("polygon", json.loads(body)["error"].lower())

    @patch("aus_backyard_intelligence.external_api.execute_overpass_query")
    def test_local_features_endpoint_returns_recorded_pool_provenance(self, mock_overpass):
        mock_overpass.return_value = {"elements": [], "endpoint": "local"}
        status, _, body = call_app(
            self.app,
            "/api/local/features",
            query="south=-27.28&west=153.00&north=-27.20&east=153.06&zoom=17",
        )
        self.assertEqual(status, "200 OK")
        payload = json.loads(body)
        self.assertIn("elements", payload)
        self.assertIsInstance(payload["elements"], list)
        self.assertEqual(payload["source"], "local-properties")
        
        # Recorded pools must not pretend to be fresh YOLO inference.
        pools = [e for e in payload["elements"] if e.get("tags", {}).get("leisure") == "swimming_pool"]
        if pools:
            self.assertEqual(pools[0]["tags"].get("detector:yolo"), "no")
            self.assertEqual(pools[0]["tags"].get("detector:source"), "local-property-record")
            self.assertTrue("confidence:record" in pools[0]["tags"])

    def test_local_features_endpoint_rejects_invalid_bbox(self):
        status, _, body = call_app(
            self.app,
            "/api/local/features",
            query="south=-27.40&west=153.02&north=-27.48&east=153.04",
        )
        self.assertEqual(status, "400 Bad Request")
        self.assertIn("bounding box", json.loads(body)["error"].lower())

    @patch("aus_backyard_intelligence.external_api.fetch_osm_features", side_effect=MemoryError)
    def test_detector_memory_error_returns_service_unavailable_without_crashing(self, _mock_fetch):
        status, _, body = call_app(
            self.app,
            "/api/local/features",
            query="south=-27.28&west=153.00&north=-27.20&east=153.06&zoom=17",
        )

        payload = json.loads(body)
        self.assertEqual(status, "503 Service Unavailable")
        self.assertEqual(payload["code"], "resource_exhausted")
        self.assertIn("still running", payload["error"])

    def test_local_features_endpoint_rejects_unsupported_yolo_options(self):
        status, _, body = call_app(
            self.app,
            "/api/local/features",
            query="south=-27.28&west=153.00&north=-27.20&east=153.06&zoom=17&detector_model=yolo12s",
        )
        self.assertEqual(status, "400 Bad Request")
        self.assertIn("Unsupported detector_model", json.loads(body)["error"])

        status, _, body = call_app(
            self.app,
            "/api/local/features",
            query="south=-27.28&west=153.00&north=-27.20&east=153.06&zoom=17&segmentation_mode=maskrcnn",
        )
        self.assertEqual(status, "400 Bad Request")
        self.assertIn("Unsupported segmentation_mode", json.loads(body)["error"])

    @patch("aus_backyard_intelligence.external_api.geocode_places")
    def test_geocode_endpoint_returns_sanitized_items(self, mock_geocode):
        mock_geocode.return_value = [{"display_name": "Moreton Bay, Queensland", "lat": "-27.248", "lon": "153.025"}]
        status, _, body = call_app(self.app, "/api/geocode", query="q=Moreton Bay")
        self.assertEqual(status, "200 OK")
        payload = json.loads(body)
        self.assertEqual(payload["source"], "nominatim")
        self.assertEqual(payload["items"][0]["display_name"], "Moreton Bay, Queensland")

    def test_public_sites_are_filtered_from_loaded_rankings(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            data_path = Path(temp_dir_name) / "mixed_properties.json"
            data_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "PRIV-1",
                            "lga": "Moreton Bay",
                            "suburb": "North Lakes",
                            "address": "1 Test Street, North Lakes QLD",
                            "parcel_rect": [0.0, 0.0, 20.0, 40.0],
                            "building_rect": [3.0, 5.0, 16.0, 20.0],
                            "elevation_corners_m": {"sw": 12.0, "se": 12.4, "nw": 13.0, "ne": 13.3},
                            "vegetation_pct": 10.0,
                            "road_frontage_m": 20.0,
                            "corner_lot": False,
                            "flood_risk": 0,
                            "heritage_overlay": False,
                            "overhead_powerline": False,
                            "solar_access_pct": 82.0,
                            "view_quality": 5,
                            "land_use": "Residential",
                            "ownership_class": "Private",
                        },
                        {
                            "id": "PUB-1",
                            "lga": "Moreton Bay",
                            "suburb": "North Lakes",
                            "address": "10 Park Office Road, North Lakes QLD",
                            "parcel_rect": [30.0, 0.0, 50.0, 40.0],
                            "building_rect": [34.0, 6.0, 46.0, 18.0],
                            "elevation_corners_m": {"sw": 11.0, "se": 11.1, "nw": 11.3, "ne": 11.6},
                            "vegetation_pct": 18.0,
                            "road_frontage_m": 20.0,
                            "corner_lot": False,
                            "flood_risk": 0,
                            "heritage_overlay": False,
                            "overhead_powerline": False,
                            "solar_access_pct": 76.0,
                            "view_quality": 4,
                            "land_use": "Public park and recreation",
                            "ownership_class": "Moreton Bay Regional Council",
                        },
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )

            filtered_app = create_app(data_path=data_path)
            self.assertEqual(len(filtered_app.snapshots), 1)
            self.assertIsNotNone(filtered_app.get_property("PRIV-1"))
            self.assertIsNone(filtered_app.get_property("PUB-1"))
            self.assertEqual(filtered_app.rankings("pool_3d", top_k=3)[0]["property_id"], "PRIV-1")

    def test_pool_score_override_when_has_existing_pool(self):
        prop = {
            "id": "MOCK-1",
            "lga": "Moreton Bay",
            "suburb": "North Lakes",
            "address": "1 Mock Street",
            "parcel_rect": [0.0, 0.0, 20.0, 40.0],
            "building_rect": [3.0, 5.0, 16.0, 20.0],
            "elevation_corners_m": {"sw": 12.0, "se": 12.4, "nw": 13.0, "ne": 13.3},
            "vegetation_pct": 10.0,
            "road_frontage_m": 20.0,
            "corner_lot": False,
            "flood_risk": 0,
            "heritage_overlay": False,
            "overhead_powerline": False,
            "solar_access_pct": 82.0,
            "view_quality": 5,
            "land_use": "Residential",
            "ownership_class": "Private",
            "has_existing_pool": True,
        }
        from aus_backyard_intelligence.app import score_pool_2d, score_pool_3d, explain_pool_2d, explain_pool_3d
        features = derive_features(prop)
        score_2d = score_pool_2d(features, prop)
        score_3d = score_pool_3d(features, prop, score_2d)
        self.assertEqual(score_2d, 0.0)
        self.assertEqual(score_3d, 0.0)
        
        exp_2d = explain_pool_2d(features, prop)
        exp_3d = explain_pool_3d(features, prop)
        self.assertIn("already has an existing swimming pool", exp_2d)
        self.assertIn("already has an existing swimming pool", exp_3d)

    def test_existing_pool_stays_eligible_for_non_pool_modules(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            data_path = Path(temp_dir_name) / "existing_pool.json"
            record = dict(self.sample_property["property"])
            record.update({"id": "POOL-1", "has_existing_pool": True, "exclude_from_pool_candidates": True})
            data_path.write_text(json.dumps([record]), encoding="utf-8")
            app = create_app(data_path=data_path)
            self.assertEqual(len(app.snapshots), 1)
            self.assertEqual(app.rankings("pool_3d", top_k=5), [])
            self.assertEqual(app.rankings("fence", top_k=5)[0]["property_id"], "POOL-1")

    def test_legacy_generated_pool_flag_without_evidence_is_ignored(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            data_path = Path(temp_dir_name) / "legacy_pool.json"
            record = dict(self.sample_property["property"])
            record.update({"id": "OSM-99", "has_existing_pool": True, "exclude_from_pool_candidates": True})
            record.pop("existing_pool_source", None)
            data_path.write_text(json.dumps([record]), encoding="utf-8")
            app = create_app(data_path=data_path)
            snapshot = app.get_property("OSM-99")
            self.assertFalse(snapshot["property"]["has_existing_pool"])
            self.assertGreater(snapshot["scores"]["pool_3d"], 0)

    def test_small_pool_candidate_scoring(self):
        prop = {
            "id": "MOCK-SMALL",
            "lga": "Moreton Bay",
            "suburb": "North Lakes",
            "address": "5 Mock Street",
            # Total parcel is 15m x 15m = 225 sqm
            "parcel_rect": [0.0, 0.0, 15.0, 15.0],
            # House covers 15m x 10m = 150 sqm, leaving 15m x 5m = 75 sqm backyard
            "building_rect": [0.0, 0.0, 15.0, 10.0],
            "elevation_corners_m": {"sw": 12.0, "se": 12.0, "nw": 12.0, "ne": 12.0},
            "vegetation_pct": 5.0,
            "road_frontage_m": 15.0,
            "corner_lot": False,
            "flood_risk": 0,
            "heritage_overlay": False,
            "overhead_powerline": False,
            "solar_access_pct": 85.0,
            "view_quality": 5,
            "land_use": "Residential",
            "ownership_class": "Private",
        }
        from aus_backyard_intelligence.app import score_pool_2d, score_pool_3d, explain_pool_2d, build_scores
        features = derive_features(prop)
        
        # Verify that small pool candidate criteria is met
        # backyard_area = 75 sqm (< 120 sqm)
        # buildable_rear_zone = 75 - 24 = 51 sqm (>= 15 sqm)
        self.assertLess(features["backyard_area_sqm"], 120.0)
        self.assertGreaterEqual(features["buildable_rear_zone_sqm"], 15.0)

        score_2d = score_pool_2d(features, prop)
        score_3d = score_pool_3d(features, prop, score_2d)
        scores = build_scores(prop, features)

        # Capped score at 75
        self.assertLessEqual(score_2d, 75.0)
        self.assertLessEqual(score_3d, 75.0)
        self.assertGreater(score_2d, 0.0)
        self.assertEqual(scores["pool_type"], "small")

        exp_2d = explain_pool_2d(features, prop)
        self.assertIn("plunge or splash pool", exp_2d)


if __name__ == "__main__":
    unittest.main()
