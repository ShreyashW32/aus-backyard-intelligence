import json
import os
import threading
import time
import unittest
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from aus_backyard_intelligence.external_api import (
    _deduplicate_pools,
    build_overpass_query,
    fetch_elevation_grid,
    fetch_osm_features,
    geocode_places,
    google_geocoding_configured,
    google_reverse_geocode,
    request_json_url,
    validate_bbox,
)
from aus_backyard_intelligence.fetch_real_properties import _assign_pools_to_nearest_buildings


def json_response(payload):
    response = MagicMock()
    response.headers = {"Content-Length": str(len(json.dumps(payload)))}
    response.read.return_value = json.dumps(payload).encode("utf-8")
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


class ExternalApiTestCase(unittest.TestCase):
    def test_detector_job_queue_serializes_concurrent_requests(self):
        from aus_backyard_intelligence.external_api import DetectorJobQueue

        queue = DetectorJobQueue(maxsize=4)
        guard = threading.Lock()
        active = 0
        max_active = 0

        def work(job_id):
            def run():
                nonlocal active, max_active
                with guard:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.03)
                with guard:
                    active -= 1
                return job_id

            return queue.submit(run)

        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(work, "first")
            time.sleep(0.01)
            second = callers.submit(work, "second")
            first_result, first_meta = first.result(timeout=2)
            second_result, second_meta = second.result(timeout=2)

        self.assertEqual((first_result, second_result), ("first", "second"))
        self.assertEqual(max_active, 1)
        self.assertEqual(first_meta["worker"], "abi-detector-worker")
        self.assertGreaterEqual(second_meta["queued_ms"], 1)
        self.assertEqual(queue.status()["completed"], 2)

    def test_bbox_validation_rejects_oversized_or_reversed_requests(self):
        with self.assertRaises(ValueError):
            validate_bbox([-27.4, 153.0, -27.5, 153.1])
        with self.assertRaises(ValueError):
            validate_bbox([-27.8, 153.0, -27.4, 153.1])

    def test_overpass_query_contains_real_pool_and_building_tags(self):
        query = build_overpass_query((-27.48, 153.02, -27.46, 153.04), zoom=18)
        self.assertIn('nwr["leisure"~"^(swimming_pool|water_park|swimming_area)$"]', query)
        self.assertIn('nwr["building"]', query)
        self.assertIn('node["natural"="tree"]', query)
        self.assertIn('nwr["swimming_pool"]', query)
        self.assertIn('nwr["sport"="swimming"]', query)
        self.assertIn('nwr["amenity"~"^(public_bath|swimming_pool)$"]', query)
        self.assertIn('nwr["water"~"^(pool|swimming_pool|wading_pool|lagoon)$"]', query)
        self.assertIn('nwr["tourism"="water_park"]', query)
        self.assertIn('nwr["name"~"(pool|swim|aquatic|lagoon|water ?park|streets beach|boat pool)",i]', query)
        self.assertIn('nwr["landuse"~"^(forest|orchard|vineyard|meadow|grass|recreation_ground|greenfield)$"]', query)
        self.assertIn('nwr["leisure"~"^(garden|park|nature_reserve)$"]', query)

    @patch("aus_backyard_intelligence.external_api.execute_overpass_query", return_value={
        "endpoint": "mock-overpass",
        "elements": [],
    })
    def test_fetch_osm_features_generates_local_buildings(self, _mock_overpass):
        """Local feature generator should produce building elements for properties in bbox."""
        import os
        import tempfile
        from aus_backyard_intelligence.external_api import _osm_feature_cache
        _osm_feature_cache.clear()
        props = [
            {
                "id": "TEST-1", "real_lat": -27.47, "real_lon": 153.03,
                "parcel_rect": [0, 0, 20, 30], "building_rect": [2, 5, 18, 20],
                "has_existing_pool": False, "vegetation_pct": 0, "overhead_powerline": False,
            }
        ]
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(props, f)
            result = fetch_osm_features(
                [-27.48, 153.02, -27.46, 153.04], zoom=17, data_path=path
            )
            self.assertIsInstance(result["elements"], list)
            buildings = [e for e in result["elements"] if e.get("tags", {}).get("building")]
            self.assertGreaterEqual(len(buildings), 1)
            self.assertEqual(buildings[0]["type"], "way")
            self.assertEqual(result["source"], "local-properties")
        finally:
            os.unlink(path)

    def test_fetch_osm_features_generates_pools_and_trees(self):
        """Properties with has_existing_pool and vegetation should produce pool/tree elements."""
        import os
        import tempfile
        from aus_backyard_intelligence.external_api import _osm_feature_cache
        _osm_feature_cache.clear()
        props = [
            {
                "id": "TEST-2", "real_lat": -27.47, "real_lon": 153.03,
                "parcel_rect": [0, 0, 24, 40], "building_rect": [2, 6, 22, 22],
                "has_existing_pool": True, "vegetation_pct": 35, "overhead_powerline": True,
            }
        ]
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(props, f)
            result = fetch_osm_features(
                [-27.48, 153.02, -27.46, 153.04], zoom=17, data_path=path
            )
            pools = [e for e in result["elements"] if e.get("tags", {}).get("leisure") == "swimming_pool"]
            trees = [e for e in result["elements"] if e.get("tags", {}).get("natural") == "tree"]
            powerlines = [e for e in result["elements"] if e.get("tags", {}).get("power") == "line"]
            self.assertGreaterEqual(len(pools), 1, "Should generate at least one pool")
            self.assertEqual(pools[0]["tags"].get("detector:yolo"), "no")
            self.assertEqual(pools[0]["tags"].get("detector:source"), "local-property-record")
            self.assertEqual(pools[0]["tags"].get("detector:model"), "not-applicable")
            self.assertTrue("confidence:record" in pools[0]["tags"])
            self.assertGreaterEqual(len(trees), 1, "Should generate tree nodes")
            self.assertGreaterEqual(len(powerlines), 1, "Should generate powerline")
        finally:
            os.unlink(path)

    def test_recorded_pool_does_not_inherit_context_yolo_options(self):
        import os
        import tempfile
        from aus_backyard_intelligence.external_api import _osm_feature_cache
        _osm_feature_cache.clear()
        props = [
            {
                "id": "TEST-3", "real_lat": -27.47, "real_lon": 153.03,
                "parcel_rect": [0, 0, 24, 40], "building_rect": [2, 6, 22, 22],
                "has_existing_pool": True, "vegetation_pct": 0, "overhead_powerline": False,
            }
        ]
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(props, f)
            result = fetch_osm_features(
                [-27.48, 153.02, -27.46, 153.04], zoom=17, data_path=path,
                detector_model="yolov11s", segmentation_mode="bbox"
            )
            pools = [e for e in result["elements"] if e.get("tags", {}).get("leisure") == "swimming_pool"]
            self.assertEqual(len(pools), 1)
            self.assertEqual(pools[0]["tags"].get("detector:model"), "not-applicable")
            self.assertEqual(pools[0]["tags"].get("detector:segmentation"), "recorded-footprint")
        finally:
            os.unlink(path)

    def test_fetch_osm_features_rejects_unsupported_detection_options(self):
        with self.assertRaises(ValueError):
            fetch_osm_features(
                [-27.48, 153.02, -27.46, 153.04],
                zoom=17,
                detector_model="yolo12s",
            )
        with self.assertRaises(ValueError):
            fetch_osm_features(
                [-27.48, 153.02, -27.46, 153.04],
                zoom=17,
                segmentation_mode="maskrcnn",
            )

    @patch("aus_backyard_intelligence.external_api.execute_overpass_query")
    def test_force_refresh_bypasses_osm_feature_cache_for_scan_passes(self, mock_overpass):
        import os
        import tempfile
        from aus_backyard_intelligence.external_api import _osm_feature_cache

        _osm_feature_cache.clear()
        mock_overpass.return_value = {"endpoint": "mock-overpass", "elements": []}
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump([], f)

            first = fetch_osm_features(
                [-33.888, 151.17675, -33.88575, 151.182],
                zoom=20,
                layers="pools",
                detector_precision="precise",
                data_path=path,
                force_refresh=True,
            )
            second = fetch_osm_features(
                [-33.888, 151.17675, -33.88575, 151.182],
                zoom=20,
                layers="pools",
                detector_precision="precise",
                data_path=path,
                force_refresh=True,
            )
        finally:
            os.unlink(path)

        self.assertEqual(first["cache"], "bypass")
        self.assertEqual(second["cache"], "bypass")
        self.assertEqual(mock_overpass.call_count, 2)
        self.assertEqual(len(_osm_feature_cache), 0)

    @patch("aus_backyard_intelligence.external_api.execute_overpass_query")
    def test_detector_signature_invalidates_feature_cache(self, mock_overpass):
        import os
        import tempfile
        from aus_backyard_intelligence.external_api import _osm_feature_cache

        _osm_feature_cache.clear()
        mock_overpass.return_value = {"endpoint": "mock-overpass", "elements": []}
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump([], handle)
            with patch(
                "aus_backyard_intelligence.external_api.get_detector_signature",
                side_effect=[
                    {"signature": "detector-a", "weights": []},
                    {"signature": "detector-b", "weights": []},
                ],
            ):
                fetch_osm_features([-27.48, 153.02, -27.46, 153.04], data_path=path)
                fetch_osm_features([-27.48, 153.02, -27.46, 153.04], data_path=path)
        finally:
            os.unlink(path)

        self.assertEqual(mock_overpass.call_count, 2)

    @patch("aus_backyard_intelligence.external_api.get_detector_status", return_value={
        "pool_yolo_available": True,
        "primary_pool_detector": "pool_trained_yolo",
    })
    @patch("aus_backyard_intelligence.external_api.execute_overpass_query")
    @patch("aus_backyard_intelligence.yolo_detector.detect_pools_in_bbox")
    @patch("aus_backyard_intelligence.yolo_detector.is_available", return_value=True)
    @patch("aus_backyard_intelligence.yolo_detector.get_last_detector_diagnostics", return_value={
        "reason": "pools_detected",
        "pool_yolo_active": True,
    })
    def test_selected_area_results_are_clipped_and_confidence_is_deterministic(
        self,
        _mock_diagnostics,
        _mock_available,
        mock_satellite,
        mock_overpass,
        _mock_status,
    ):
        import os
        import tempfile
        from unittest.mock import patch as mock_patch
        from aus_backyard_intelligence.external_api import _osm_feature_cache

        _osm_feature_cache.clear()
        mock_overpass.return_value = {"endpoint": "mock-overpass", "elements": []}

        def detection(element_id, lat, lon):
            return {
                "type": "way",
                "id": element_id,
                "tags": {
                    "leisure": "swimming_pool",
                    "detector:source": "satellite-yolo",
                    "detector:method": "pool-trained-yolo",
                    "confidence:yolo": "0.83",
                },
                "geometry": [
                    {"lat": lat - 0.00002, "lon": lon - 0.00002},
                    {"lat": lat - 0.00002, "lon": lon + 0.00002},
                    {"lat": lat + 0.00002, "lon": lon + 0.00002},
                    {"lat": lat + 0.00002, "lon": lon - 0.00002},
                ],
            }

        mock_satellite.return_value = [
            detection(1, -33.88725, 151.17975),
            detection(2, -33.88725, 151.18025),
            {
                "type": "way",
                "id": 3,
                "tags": {
                    "power": "generator",
                    "generator:source": "solar",
                    "detector:source": "satellite-yolo-solar",
                    "confidence:yolo": "0.82",
                },
                "geometry": [
                    {"lat": -33.88730, "lon": 151.17970},
                    {"lat": -33.88730, "lon": 151.17980},
                    {"lat": -33.88720, "lon": 151.17980},
                    {"lat": -33.88720, "lon": 151.17970},
                ],
            },
        ]
        polygon = [
            [151.1795, -33.8875], [151.1800, -33.8875],
            [151.1800, -33.8870], [151.1795, -33.8870],
        ]
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump([], handle)
            with mock_patch("sys.argv", ["runner"]), mock_patch.dict(os.environ, {}), \
                    mock_patch(
                        "aus_backyard_intelligence.external_api._claim_detector_process_lease",
                        return_value=(True, {"pid": os.getpid()}),
                    ):
                os.environ.pop("PYTEST_CURRENT_TEST", None)
                first = fetch_osm_features(
                    [-33.8875, 151.1795, -33.8870, 151.1805],
                    zoom=20,
                    layers="pools",
                    detector_precision="precise",
                    analysis_polygon=polygon,
                    data_path=path,
                )
                second = fetch_osm_features(
                    [-33.8875, 151.1795, -33.8870, 151.1805],
                    zoom=20,
                    layers="pools",
                    detector_precision="precise",
                    analysis_polygon=polygon,
                    data_path=path,
                )
        finally:
            os.unlink(path)

        first_pools = [item for item in first["elements"] if item.get("tags", {}).get("leisure") == "swimming_pool"]
        second_pools = [item for item in second["elements"] if item.get("tags", {}).get("leisure") == "swimming_pool"]
        self.assertEqual([item["id"] for item in first_pools], [1])
        self.assertEqual(first_pools[0]["tags"]["confidence:yolo"], "0.83")
        self.assertEqual(second_pools[0]["tags"]["confidence:yolo"], "0.83")
        self.assertEqual(second["cache"], "hit")
        self.assertTrue(first["pipeline"]["cacheable"])
        self.assertEqual(first["counts"]["solar"], 1)
        self.assertEqual(
            len([item for item in first["elements"] if item.get("tags", {}).get("generator:source") == "solar"]),
            1,
        )
        mock_satellite.assert_called_once()

    @patch("aus_backyard_intelligence.external_api.get_detector_status", return_value={
        "pool_yolo_available": True,
        "primary_pool_detector": "pool_trained_yolo",
    })
    @patch("aus_backyard_intelligence.external_api.execute_overpass_query", return_value={
        "endpoint": "mock-overpass",
        "elements": [],
    })
    @patch("aus_backyard_intelligence.external_api._claim_detector_process_lease", return_value=(
        False,
        {"pid": 4321, "claimed_at": 1.0},
    ))
    @patch("aus_backyard_intelligence.yolo_detector.detect_pools_in_bbox")
    def test_second_server_reports_detector_owner_without_loading_model(
        self,
        mock_satellite,
        _mock_claim,
        _mock_overpass,
        _mock_status,
    ):
        import os
        import tempfile
        from unittest.mock import patch as mock_patch

        polygon = [
            [151.1795, -33.8875], [151.1800, -33.8875],
            [151.1800, -33.8870], [151.1795, -33.8870],
        ]
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump([], handle)
            with mock_patch("sys.argv", ["runner"]), mock_patch.dict(os.environ, {}):
                os.environ.pop("PYTEST_CURRENT_TEST", None)
                result = fetch_osm_features(
                    [-33.8875, 151.1795, -33.8870, 151.1805],
                    zoom=20,
                    layers="pools",
                    detector_precision="precise",
                    analysis_polygon=polygon,
                    data_path=path,
                    force_refresh=True,
                )
        finally:
            os.unlink(path)

        step = next(
            item
            for item in result["pipeline"]["steps"]
            if item["name"] == "satellite_pool_detector"
        )
        self.assertEqual(step["status"], "busy_other_process")
        self.assertEqual(step["detector_owner"]["pid"], 4321)
        self.assertIn("out-of-memory", step["error"])
        self.assertFalse(result["pipeline"]["cacheable"])
        mock_satellite.assert_not_called()

    @patch("aus_backyard_intelligence.external_api.get_detector_status", return_value={
        "pool_yolo_available": True,
        "primary_pool_detector": "pool_trained_yolo",
    })
    @patch("aus_backyard_intelligence.external_api.execute_overpass_query", return_value={
        "endpoint": "mock-overpass",
        "elements": [],
    })
    @patch("aus_backyard_intelligence.external_api._claim_detector_process_lease", return_value=(
        True,
        {"pid": 1234, "claimed_at": 1.0},
    ))
    @patch("aus_backyard_intelligence.yolo_detector.detect_pools_in_bbox", return_value=[])
    @patch("aus_backyard_intelligence.yolo_detector.is_available", return_value=True)
    @patch(
        "aus_backyard_intelligence.yolo_detector.get_last_detector_diagnostics",
        return_value={
            "reason": "primary_inference_failed",
            "primary_inference_error": "checkpoint is incompatible",
            "pool_yolo_active": True,
        },
    )
    def test_primary_inference_failure_is_not_reported_as_valid_zero_result(
        self,
        _mock_diagnostics,
        _mock_available,
        _mock_satellite,
        _mock_claim,
        _mock_overpass,
        _mock_status,
    ):
        import os
        import tempfile
        from unittest.mock import patch as mock_patch

        polygon = [
            [151.1795, -33.8875], [151.1800, -33.8875],
            [151.1800, -33.8870], [151.1795, -33.8870],
        ]
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump([], handle)
            with mock_patch("sys.argv", ["runner"]), mock_patch.dict(os.environ, {}):
                os.environ.pop("PYTEST_CURRENT_TEST", None)
                result = fetch_osm_features(
                    [-33.8875, 151.1795, -33.8870, 151.1805],
                    zoom=20,
                    layers="pools",
                    detector_precision="precise",
                    analysis_polygon=polygon,
                    data_path=path,
                    force_refresh=True,
                )
        finally:
            os.unlink(path)

        step = next(
            item
            for item in result["pipeline"]["steps"]
            if item["name"] == "satellite_pool_detector"
        )
        self.assertEqual(step["status"], "error")
        self.assertEqual(step["error"], "checkpoint is incompatible")

    @patch("aus_backyard_intelligence.external_api.get_detector_status", return_value={"pool_yolo_available": False})
    @patch("aus_backyard_intelligence.external_api.execute_overpass_query")
    def test_fast_mode_prefers_public_pools_over_satellite_boxes(self, mock_overpass, _mock_status):
        import os
        import tempfile
        from aus_backyard_intelligence.external_api import _osm_feature_cache

        _osm_feature_cache.clear()
        mock_overpass.return_value = {
            "endpoint": "mock-overpass",
            "elements": [
                {
                    "type": "way",
                    "id": 123,
                    "tags": {"leisure": "swimming_pool"},
                    "geometry": [
                        {"lat": -27.4704, "lon": 153.0301},
                        {"lat": -27.4704, "lon": 153.0304},
                        {"lat": -27.4701, "lon": 153.0304},
                        {"lat": -27.4701, "lon": 153.0301},
                    ],
                }
            ],
        }
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump([], f)
            result = fetch_osm_features(
                [-27.48, 153.02, -27.46, 153.04],
                zoom=18,
                layers="pools",
                detector_precision="fast",
                data_path=path,
                analysis_polygon=[
                    [153.02, -27.48], [153.04, -27.48],
                    [153.04, -27.46], [153.02, -27.46],
                ],
            )
        finally:
            os.unlink(path)

        pools = [e for e in result["elements"] if e.get("tags", {}).get("leisure") == "swimming_pool"]
        self.assertEqual(len(pools), 1)
        self.assertEqual(pools[0]["tags"].get("detector:source"), "public-mapped")
        satellite_step = next(step for step in result["pipeline"]["steps"] if step["name"] == "satellite_pool_detector")
        self.assertEqual(satellite_step["status"], "skipped_public_mapped_pools_present")

    @patch("aus_backyard_intelligence.external_api.get_detector_status", return_value={"pool_yolo_available": False})
    @patch("aus_backyard_intelligence.external_api.execute_overpass_query")
    @patch("aus_backyard_intelligence.yolo_detector.detect_pools_in_bbox")
    def test_fast_mode_disables_unverified_satellite_pool_boxes(self, mock_satellite, mock_overpass, _mock_status):
        import os
        import tempfile
        from aus_backyard_intelligence.external_api import _osm_feature_cache

        _osm_feature_cache.clear()
        mock_overpass.return_value = {"endpoint": "mock-overpass", "elements": []}
        mock_satellite.return_value = [
            {
                "type": "way",
                "id": 90001,
                "tags": {"leisure": "swimming_pool", "detector:source": "satellite-cv"},
                "geometry": [
                    {"lat": -33.886, "lon": 151.18},
                    {"lat": -33.886, "lon": 151.181},
                    {"lat": -33.885, "lon": 151.181},
                    {"lat": -33.885, "lon": 151.18},
                ],
            }
        ]
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump([], f)
            result = fetch_osm_features(
                [-33.888, 151.17675, -33.88575, 151.182],
                zoom=20,
                layers="pools",
                detector_precision="fast",
                data_path=path,
            )
        finally:
            os.unlink(path)

        pools = [e for e in result["elements"] if e.get("tags", {}).get("leisure") == "swimming_pool"]
        self.assertEqual(pools, [])
        mock_satellite.assert_not_called()
        satellite_step = next(step for step in result["pipeline"]["steps"] if step["name"] == "satellite_pool_detector")
        self.assertEqual(satellite_step["status"], "skipped_no_selected_analysis_area")
        self.assertIn("selected analysis polygon", satellite_step["reason"])

    @patch("aus_backyard_intelligence.external_api.get_detector_status", return_value={
        "pool_yolo_available": True,
        "primary_pool_detector": "pool_trained_yolo",
    })
    @patch("aus_backyard_intelligence.external_api.execute_overpass_query")
    @patch("aus_backyard_intelligence.yolo_detector.detect_pools_in_bbox")
    @patch("aus_backyard_intelligence.yolo_detector.is_available", return_value=True)
    @patch("aus_backyard_intelligence.yolo_detector.get_last_detector_diagnostics", return_value={
        "reason": "pools_detected",
        "pool_yolo_active": True,
    })
    def test_fast_mode_runs_pool_trained_yolo_when_available(
        self,
        _mock_diagnostics,
        _mock_available,
        mock_satellite,
        mock_overpass,
        _mock_status,
    ):
        import os
        import tempfile
        from unittest.mock import patch as mock_patch
        from aus_backyard_intelligence.external_api import _osm_feature_cache

        _osm_feature_cache.clear()
        mock_overpass.return_value = {"endpoint": "mock-overpass", "elements": []}
        mock_satellite.return_value = [
            {
                "type": "way",
                "id": 950000001,
                "tags": {
                    "leisure": "swimming_pool",
                    "detector:source": "satellite-yolo",
                    "detector:method": "pool-trained-yolo",
                },
                "geometry": [
                    {"lat": -33.886, "lon": 151.1800},
                    {"lat": -33.886, "lon": 151.1802},
                    {"lat": -33.8858, "lon": 151.1802},
                    {"lat": -33.8858, "lon": 151.1800},
                ],
            }
        ]
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump([], f)
            with mock_patch("sys.argv", ["runner"]), mock_patch.dict(os.environ, {}), \
                    mock_patch(
                        "aus_backyard_intelligence.external_api._claim_detector_process_lease",
                        return_value=(True, {"pid": os.getpid()}),
                    ):
                os.environ.pop("PYTEST_CURRENT_TEST", None)
                result = fetch_osm_features(
                    [-33.888, 151.17675, -33.88575, 151.182],
                    zoom=20,
                    layers="pools",
                    detector_precision="fast",
                    data_path=path,
                    analysis_polygon=[
                        [151.17675, -33.888], [151.182, -33.888],
                        [151.182, -33.88575], [151.17675, -33.88575],
                    ],
                )
        finally:
            os.unlink(path)

        pools = [e for e in result["elements"] if e.get("tags", {}).get("leisure") == "swimming_pool"]
        self.assertEqual(len(pools), 1)
        self.assertEqual(pools[0]["tags"].get("detector:source"), "satellite-yolo")
        self.assertEqual(result["source"], "satellite-yolo")
        mock_satellite.assert_called_once()
        satellite_step = next(step for step in result["pipeline"]["steps"] if step["name"] == "satellite_pool_detector")
        self.assertEqual(satellite_step["status"], "ok")

    @patch("aus_backyard_intelligence.external_api.execute_overpass_query")
    @patch("aus_backyard_intelligence.yolo_detector.detect_pools_in_bbox")
    @patch("aus_backyard_intelligence.yolo_detector.is_available", return_value=True)
    @patch("aus_backyard_intelligence.yolo_detector.get_last_detector_diagnostics", return_value={"reason": "pools_detected"})
    def test_precise_mode_filters_satellite_pools_inside_buildings(
        self,
        _mock_diagnostics,
        _mock_available,
        mock_satellite,
        mock_overpass,
    ):
        import os
        import tempfile
        from unittest.mock import patch as mock_patch
        from aus_backyard_intelligence.external_api import _osm_feature_cache

        _osm_feature_cache.clear()
        mock_overpass.return_value = {
            "endpoint": "mock-overpass",
            "elements": [
                {
                    "type": "way",
                    "id": 501,
                    "tags": {"building": "yes"},
                    "geometry": [
                        {"lat": -33.8874, "lon": 151.1796},
                        {"lat": -33.8874, "lon": 151.1799},
                        {"lat": -33.8871, "lon": 151.1799},
                        {"lat": -33.8871, "lon": 151.1796},
                    ],
                }
            ],
        }
        mock_satellite.return_value = [
            {
                "type": "way",
                "id": 950000001,
                "tags": {"leisure": "swimming_pool", "detector:source": "satellite-cv"},
                "geometry": [
                    {"lat": -33.88732, "lon": 151.17968},
                    {"lat": -33.88732, "lon": 151.17978},
                    {"lat": -33.88722, "lon": 151.17978},
                    {"lat": -33.88722, "lon": 151.17968},
                ],
            }
        ]
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump([], f)
            # This test mocks the satellite detector and intentionally exercises
            # the real detection path, so suppress the in-test skip guard that
            # would otherwise short-circuit detect_pools_in_bbox under pytest.
            with mock_patch("sys.argv", ["runner"]), \
                    mock_patch.dict(os.environ, {}), \
                    mock_patch(
                        "aus_backyard_intelligence.external_api._claim_detector_process_lease",
                        return_value=(True, {"pid": os.getpid()}),
                    ):
                os.environ.pop("PYTEST_CURRENT_TEST", None)
                os.environ.pop("QLDR_SKIP_SATELLITE_DETECTOR_FOR_TESTS", None)
                result = fetch_osm_features(
                    [-33.8875, 151.1795, -33.8870, 151.1800],
                    zoom=20,
                    layers="pools,buildings",
                    detector_precision="precise",
                    data_path=path,
                    analysis_polygon=[
                        [151.1795, -33.8875], [151.1800, -33.8875],
                        [151.1800, -33.8870], [151.1795, -33.8870],
                    ],
                )
        finally:
            os.unlink(path)

        pools = [e for e in result["elements"] if e.get("tags", {}).get("leisure") == "swimming_pool"]
        self.assertEqual(pools, [])
        mock_satellite.assert_called_once()


    @patch("aus_backyard_intelligence.external_api.request_json_url")
    def test_elevation_grid_returns_real_sample_matrix(self, mock_request_json):
        mock_request_json.return_value = {"elevation": [1, 2, 3, 4, 5, 6, 7, 8, 9]}
        result = fetch_elevation_grid([-27.48, 153.02, -27.46, 153.04], size=3)
        self.assertEqual(result["elevations"], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self.assertEqual(result["minimum_m"], 1.0)
        self.assertEqual(result["maximum_m"], 9.0)
        self.assertIn("latitude=", mock_request_json.call_args.args[0])

    @patch("aus_backyard_intelligence.external_api.urllib.request.urlopen")
    def test_geocoder_performs_global_search(self, mock_urlopen):
        mock_urlopen.return_value = json_response(
            [{"lat": "-27.248", "lon": "153.025", "display_name": "Moreton Bay, Queensland"}]
        )
        results = geocode_places("Moreton Bay unique adapter test")
        self.assertEqual(results[0]["display_name"], "Moreton Bay, Queensland")
        request_url = mock_urlopen.call_args.args[0].full_url
        self.assertIn("q=Moreton+Bay", request_url)

    @patch("aus_backyard_intelligence.external_api._request_json")
    def test_google_reverse_geocoder_returns_verified_explicit_unit(self, mock_request_json):
        mock_request_json.return_value = {
            "status": "OK",
            "results": [
                {
                    "formatted_address": "47FH+6WW Dover Heights NSW, Australia",
                    "place_id": "plus-code-result",
                    "types": ["plus_code"],
                    "address_components": [],
                    "geometry": {"location_type": "APPROXIMATE"},
                },
                {
                    "formatted_address": "Unit 12, 8 Example Street, Dover Heights NSW 2030, Australia",
                    "place_id": "exact-unit-place",
                    "types": ["subpremise"],
                    "address_components": [
                        {"long_name": "12", "short_name": "12", "types": ["subpremise"]},
                        {"long_name": "8", "short_name": "8", "types": ["street_number"]},
                        {"long_name": "Example Street", "short_name": "Example St", "types": ["route"]},
                        {"long_name": "Dover Heights", "short_name": "Dover Heights", "types": ["locality"]},
                        {"long_name": "New South Wales", "short_name": "NSW", "types": ["administrative_area_level_1"]},
                        {"long_name": "2030", "short_name": "2030", "types": ["postal_code"]},
                        {"long_name": "Australia", "short_name": "AU", "types": ["country"]},
                    ],
                    "geometry": {
                        "location_type": "ROOFTOP",
                        "location": {"lat": -33.87688, "lng": 151.27982},
                    },
                },
            ],
        }

        with patch.dict(
            os.environ,
            {"ABI_GOOGLE_MAPS_API_KEY": "test-secret", "GOOGLE_MAPS_API_KEY": ""},
            clear=False,
        ):
            self.assertTrue(google_geocoding_configured())
            result = google_reverse_geocode(-33.8768814, 151.2798189)

        self.assertEqual(result["source"], "google")
        self.assertEqual(result["components"]["unit"], "12")
        self.assertEqual(result["components"]["street_number"], "8")
        self.assertEqual(result["place_id"], "exact-unit-place")
        self.assertEqual(result["precision"], "rooftop")
        self.assertEqual(result["location"], {"lat": -33.87688, "lng": 151.27982})
        self.assertLess(result["distance_m"], 1.0)
        self.assertTrue(result["verified"])
        self.assertTrue(result["exact"])
        self.assertNotIn("test-secret", json.dumps(result))
        request = mock_request_json.call_args.args[0]
        parsed = urllib.parse.urlparse(request.full_url)
        query = urllib.parse.parse_qs(parsed.query)
        self.assertEqual(f"{parsed.scheme}://{parsed.netloc}{parsed.path}", "https://maps.googleapis.com/maps/api/geocode/json")
        self.assertEqual(query["latlng"], ["-33.8768814,151.2798189"])
        self.assertEqual(query["key"], ["test-secret"])

    @patch("aus_backyard_intelligence.external_api._request_json")
    def test_google_reverse_geocoder_never_infers_unit_and_rejects_weak_exactness(
        self, mock_request_json
    ):
        mock_request_json.return_value = {
            "status": "OK",
            "results": [{
                "formatted_address": "Unit 9, 8 Example Street, Sydney NSW 2000, Australia",
                "place_id": "interpolated-place",
                "types": ["street_address"],
                "address_components": [
                    {"long_name": "8", "types": ["street_number"]},
                    {"long_name": "Example Street", "types": ["route"]},
                    {"long_name": "Australia", "short_name": "AU", "types": ["country"]},
                ],
                "geometry": {"location_type": "RANGE_INTERPOLATED"},
            }],
        }
        with patch.dict(os.environ, {"ABI_GOOGLE_MAPS_API_KEY": "test-key"}, clear=False):
            result = google_reverse_geocode(-33.8, 151.2)

        self.assertEqual(result["components"]["unit"], "")
        self.assertEqual(result["precision"], "interpolated")
        self.assertFalse(result["verified"])
        self.assertFalse(result["exact"])

    @patch("aus_backyard_intelligence.external_api._request_json")
    def test_google_reverse_geocoder_requires_spatial_match_for_exactness(
        self, mock_request_json
    ):
        mock_request_json.return_value = {
            "status": "OK",
            "results": [{
                "formatted_address": "8 Example Street, Sydney NSW 2000, Australia",
                "place_id": "distant-rooftop",
                "types": ["street_address"],
                "address_components": [
                    {"long_name": "8", "types": ["street_number"]},
                    {"long_name": "Example Street", "types": ["route"]},
                    {"long_name": "Australia", "short_name": "AU", "types": ["country"]},
                ],
                "geometry": {
                    "location_type": "ROOFTOP",
                    "location": {"lat": -33.799, "lng": 151.2},
                },
            }],
        }
        with patch.dict(os.environ, {"ABI_GOOGLE_MAPS_API_KEY": "test-key"}, clear=False):
            result = google_reverse_geocode(-33.8, 151.2)

        self.assertGreater(result["distance_m"], 35.0)
        self.assertEqual(result["location"], {"lat": -33.799, "lng": 151.2})
        self.assertFalse(result["verified"])
        self.assertFalse(result["exact"])

        mock_request_json.return_value["results"][0]["geometry"].pop("location")
        with patch.dict(os.environ, {"ABI_GOOGLE_MAPS_API_KEY": "test-key"}, clear=False):
            missing = google_reverse_geocode(-33.8, 151.2)
        self.assertEqual(missing["location"], {})
        self.assertIsNone(missing["distance_m"])
        self.assertFalse(missing["exact"])

        mock_request_json.return_value["results"][0]["geometry"]["location"] = {
            "lat": -33.8,
            "lng": 151.2,
        }
        mock_request_json.return_value["results"][0]["partial_match"] = True
        with patch.dict(os.environ, {"ABI_GOOGLE_MAPS_API_KEY": "test-key"}, clear=False):
            partial = google_reverse_geocode(-33.8, 151.2)
        self.assertTrue(partial["partial_match"])
        self.assertFalse(partial["exact"])

    @patch("aus_backyard_intelligence.external_api._request_json")
    def test_google_reverse_geocoder_is_disabled_without_key(self, mock_request_json):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(google_geocoding_configured())
            self.assertEqual(google_reverse_geocode(-33.8, 151.2), {})
        mock_request_json.assert_not_called()

    @patch("aus_backyard_intelligence.external_api._request_json")
    def test_google_reverse_geocoder_validates_coordinate_ranges(self, mock_request_json):
        with patch.dict(os.environ, {"ABI_GOOGLE_MAPS_API_KEY": "test-key"}, clear=False):
            self.assertEqual(google_reverse_geocode(91, 151.2), {})
            self.assertEqual(google_reverse_geocode(-33.8, float("nan")), {})
            self.assertEqual(google_reverse_geocode("not-a-coordinate", 151.2), {})
        mock_request_json.assert_not_called()

    def test_generic_json_adapter_rejects_unapproved_hosts(self):
        with self.assertRaises(ValueError):
            request_json_url("https://example.com/data.json")

    def test_dedupe_stamps_mapped_pool_when_satellite_duplicate_dropped(self):
        """The eval measures the detector via satellite detections; when the merge
        drops a satellite box that duplicates a mapped pool, the kept pool must
        carry detector:satellite or measured recall collapses whenever Overpass
        is healthy (it read 0.35 vs 0.89 on identical code, 2026-07-09)."""
        mapped_pool = {
            "type": "way",
            "id": 1,
            "tags": {"leisure": "swimming_pool", "detector:source": "public-mapped"},
            "geometry": [
                {"lat": -33.8760, "lon": 151.2800},
                {"lat": -33.8760, "lon": 151.2801},
                {"lat": -33.8761, "lon": 151.2801},
                {"lat": -33.8761, "lon": 151.2800},
            ],
        }
        satellite_duplicate = {
            "type": "node",
            "id": 2,
            "lat": -33.87605,
            "lon": 151.28005,
            "tags": {"leisure": "swimming_pool", "detector:source": "satellite-yolo"},
        }
        far_satellite = {
            "type": "node",
            "id": 3,
            "lat": -33.8790,
            "lon": 151.2830,
            "tags": {"leisure": "swimming_pool", "detector:source": "satellite-yolo"},
        }
        merged = _deduplicate_pools([mapped_pool], [satellite_duplicate, far_satellite])

        ids = {el["id"] for el in merged}
        self.assertEqual(ids, {1, 3}, "duplicate dropped, distant detection kept")
        self.assertEqual(mapped_pool["tags"].get("detector:satellite"), "satellite-yolo")
        self.assertNotIn("detector:satellite", far_satellite["tags"])

    def test_dedupe_does_not_stamp_for_mapped_on_mapped_duplicates(self):
        """Call 1 of the merge dedupes Overpass pools against local records —
        a dropped mapped pool is not detector evidence and must not stamp."""
        local_pool = {
            "type": "node",
            "id": 1,
            "lat": -33.8760,
            "lon": 151.2800,
            "tags": {"leisure": "swimming_pool", "detector:source": "local-property-record"},
        }
        overpass_pool = {
            "type": "node",
            "id": 2,
            "lat": -33.87601,
            "lon": 151.28001,
            "tags": {"leisure": "swimming_pool", "detector:source": "public-mapped"},
        }
        merged = _deduplicate_pools([local_pool], [overpass_pool])
        self.assertEqual([el["id"] for el in merged], [1])
        self.assertNotIn("detector:satellite", local_pool["tags"])

    def test_pool_is_assigned_to_only_the_nearest_building(self):
        buildings = [
            {"id": 1, "geometry": [{"lat": -27.47, "lon": 153.03}]},
            {"id": 2, "geometry": [{"lat": -27.47, "lon": 153.0302}]},
        ]
        pools = [(-27.47, 153.03018)]
        assignments = _assign_pools_to_nearest_buildings(buildings, pools)
        self.assertEqual(set(assignments), {2})


if __name__ == "__main__":
    unittest.main()
