import math
import unittest
from unittest.mock import patch

from aus_backyard_intelligence.scan_agent import (
    AreaScanAgent,
    MAX_SCAN_JOBS,
    MAX_TOTAL_SCAN_WINDOWS,
    SCAN_CELL_TILE_BUDGET,
    _bbox_intersects_polygon,
    _build_scan_plan,
    _coverage_grid_dims,
    _feature_intersects_polygon,
    _grid_bboxes,
    _is_residential_building,
    _max_zoom_tile_count,
    _scan_bboxes_for_passes,
)


class ScanAgentTestCase(unittest.TestCase):
    def test_area_scan_rejects_unselected_viewport_bounds(self):
        agent = AreaScanAgent(data_path="sample.json")
        with self.assertRaisesRegex(ValueError, "selected polygon"):
            agent.start_scan(
                {"bounds": {"south": -33.89, "west": 151.17, "north": -33.88, "east": 151.18}},
                passes=3,
            )

    @patch("aus_backyard_intelligence.scan_agent.threading.Thread")
    def test_duplicate_scan_request_reuses_single_active_job(self, mock_thread):
        agent = AreaScanAgent(data_path="sample.json")
        polygon = [
            [151.1796, -33.8875],
            [151.1800, -33.8875],
            [151.1800, -33.8872],
            [151.1796, -33.8872],
        ]

        first = agent.start_scan({"polygon": polygon}, passes=3)
        # Ring start point/direction do not change the selected scope.
        equivalent_polygon = [polygon[2], polygon[1], polygon[0], polygon[3]]
        second = agent.start_scan({"polygon": equivalent_polygon}, passes=3)

        self.assertEqual(second["scan_id"], first["scan_id"])
        self.assertTrue(second["reused_active_scan"])
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()

    @patch("aus_backyard_intelligence.scan_agent.threading.Thread")
    def test_different_scope_requires_explicit_replacement(self, mock_thread):
        agent = AreaScanAgent(data_path="sample.json")
        first_polygon = [[151.1796, -33.8875], [151.1800, -33.8875], [151.1800, -33.8872]]
        second_polygon = [[151.1810, -33.8885], [151.1814, -33.8885], [151.1814, -33.8882]]
        first = agent.start_scan({"polygon": first_polygon}, passes=3)

        with self.assertRaisesRegex(ValueError, "replace_active"):
            agent.start_scan({"polygon": second_polygon}, passes=3)

        self.assertEqual(agent.get_scan(first["scan_id"])["status"], "queued")
        mock_thread.assert_called_once()

    @patch("aus_backyard_intelligence.scan_agent.threading.Thread")
    def test_different_scope_atomically_cancels_and_replaces_active_job(self, mock_thread):
        agent = AreaScanAgent(data_path="sample.json")
        first_polygon = [[151.1796, -33.8875], [151.1800, -33.8875], [151.1800, -33.8872]]
        second_polygon = [[151.1810, -33.8885], [151.1814, -33.8885], [151.1814, -33.8882]]
        first = agent.start_scan({"polygon": first_polygon}, passes=3)
        with agent._lock:
            agent._jobs[first["scan_id"]]["feature_elements"] = [{"id": "old-area"}]
            agent._jobs[first["scan_id"]]["leads"] = [{"id": "old-lead"}]
            agent._jobs[first["scan_id"]]["counts"]["pools"] = 1

        replacement = agent.start_scan(
            {"polygon": second_polygon, "replace_active": True},
            passes=3,
        )

        self.assertNotEqual(replacement["scan_id"], first["scan_id"])
        self.assertEqual(replacement["superseded_scan_ids"], [first["scan_id"]])
        old = agent.get_scan(first["scan_id"])
        self.assertEqual(old["status"], "cancelled")
        self.assertEqual(old["superseded_by"], replacement["scan_id"])
        self.assertEqual(old["feature_elements"], [])
        self.assertEqual(old["leads"], [])
        self.assertEqual(old["counts"]["pools"], 0)
        self.assertEqual(mock_thread.call_count, 2)

    @patch("aus_backyard_intelligence.scan_agent.threading.Thread")
    def test_evicted_queued_workers_remain_retired_after_rapid_replacements(self, mock_thread):
        agent = AreaScanAgent(data_path="sample.json")
        first_polygon = [[151.10, -33.90], [151.101, -33.90], [151.101, -33.899]]
        first = agent.start_scan({"polygon": first_polygon}, passes=3)
        first_worker = mock_thread.call_args_list[0].kwargs["target"]
        first_worker_args = mock_thread.call_args_list[0].kwargs["args"]

        for index in range(1, MAX_SCAN_JOBS + 8):
            offset = index * 0.002
            polygon = [
                [151.10 + offset, -33.90],
                [151.101 + offset, -33.90],
                [151.101 + offset, -33.899],
            ]
            agent.start_scan(
                {"polygon": polygon, "replace_active": True},
                passes=3,
            )

        self.assertIsNone(agent.get_scan(first["scan_id"]))
        self.assertTrue(agent._is_cancelled(first["scan_id"]))
        self.assertEqual(len(agent._jobs), MAX_SCAN_JOBS)
        with patch.object(agent, "_run_job") as mock_run:
            first_worker(*first_worker_args)
        mock_run.assert_not_called()

    def test_cancel_is_idempotent_and_cancelled_job_cannot_be_resurrected(self):
        bbox = (-33.8875, 151.1796, -33.8872, 151.18)
        agent = AreaScanAgent(data_path="sample.json")
        self._seed_job(agent, "scan-cancel", bbox, 3)
        with agent._lock:
            agent._jobs["scan-cancel"].update({
                "status": "running",
                "feature_elements": [{"id": "old-area"}],
                "leads": [{"id": "old-lead"}],
            })

        first = agent.cancel_scan("scan-cancel", reason="Selection changed")
        second = agent.cancel_scan("scan-cancel", reason="Selection changed again")
        agent._update_job(
            "scan-cancel",
            status="completed",
            feature_elements=[{"id": "late-result"}],
        )

        final = agent.get_scan("scan-cancel")
        self.assertEqual(first["status"], "cancelled")
        self.assertEqual(second["status"], "cancelled")
        self.assertEqual(final["status"], "cancelled")
        self.assertEqual(final["feature_elements"], [])
        self.assertTrue(final["cancellation_accepted"])

    def test_background_exception_is_recorded_as_pollable_failure(self):
        bbox = (-33.8875, 151.1796, -33.8872, 151.18)
        agent = AreaScanAgent(data_path="sample.json")
        self._seed_job(agent, "scan-memory", bbox, 3)

        with patch.object(agent, "_run_job", side_effect=MemoryError("allocation failed")):
            agent._run_job_safely("scan-memory", bbox, 3, None)

        job = agent.get_scan("scan-memory")
        self.assertEqual(job["status"], "failed")
        self.assertIn("memory was exhausted", job["summary"])
        self.assertIn("allocation failed", job["errors"][-1])

    def test_passes_emit_varying_windows_over_selected_area(self):
        bbox = (-33.8875, 151.1796, -33.8872, 151.18)

        windows = _scan_bboxes_for_passes(bbox, 4)

        self.assertEqual(len(windows), 4)
        # The first window is always a full baseline sweep of the selected area.
        self.assertEqual(windows[0], bbox)
        # Passes are no longer identical re-scans — they shift so tile-seam pools
        # missed under one alignment are caught under another.
        self.assertGreater(len({tuple(window) for window in windows}), 1)

    def test_coverage_grid_keeps_each_cell_within_the_detector_budget(self):
        # A selection that fits in one detector call stays a single window…
        self.assertEqual(_coverage_grid_dims(1), (1, 1))
        self.assertEqual(_coverage_grid_dims(SCAN_CELL_TILE_BUDGET), (1, 1))
        # …a larger one is split into enough cells that each holds <= the per-call budget.
        big = SCAN_CELL_TILE_BUDGET * 18
        nx, ny = _coverage_grid_dims(big)
        self.assertGreaterEqual(nx * ny, math.ceil(big / SCAN_CELL_TILE_BUDGET))
        self.assertLessEqual(big / (nx * ny), SCAN_CELL_TILE_BUDGET)

        # The grid cells together tile the whole bbox exactly.
        big_bbox = (-33.90, 151.18, -33.88, 151.20)
        cells = _grid_bboxes(big_bbox, 3, 3)
        self.assertEqual(len(cells), 9)
        self.assertAlmostEqual(min(c[0] for c in cells), big_bbox[0], places=6)
        self.assertAlmostEqual(min(c[1] for c in cells), big_bbox[1], places=6)
        self.assertAlmostEqual(max(c[2] for c in cells), big_bbox[2], places=6)
        self.assertAlmostEqual(max(c[3] for c in cells), big_bbox[3], places=6)

    def test_scan_covers_every_block_even_when_few_passes_requested(self):
        # A large area is split into coverage blocks; every block must be scanned at least
        # once (the first sweep visits all of them) or parts of the selection are silently
        # skipped and their pools missed.
        big_bbox = (-33.89, 151.25, -33.86, 151.29)
        nx, ny = _coverage_grid_dims(_max_zoom_tile_count(big_bbox))
        self.assertGreater(nx * ny, 1)  # this area really does need subdividing
        plan = _build_scan_plan(big_bbox, 3)  # 3 refinement sweeps requested
        self.assertEqual(plan.blocks, nx * ny)
        self.assertGreaterEqual(len(plan.windows), plan.blocks)  # ≥ one full coverage sweep
        self.assertEqual(len(plan.windows), plan.blocks * plan.sweeps)
        # A small area is unaffected: it stays one block and runs the requested sweeps.
        small = (-33.8875, 151.1796, -33.8872, 151.18)
        small_plan = _build_scan_plan(small, 4)
        self.assertEqual(small_plan.blocks, 1)
        self.assertEqual(len(small_plan.windows), 4)
        self.assertEqual(small_plan.windows[0], small)

    def test_scan_windows_stay_within_selected_polygon(self):
        # A polygon covering only one corner of its bounding box must not spawn scan
        # blocks over the empty rest of the box — the scan stays inside what was drawn.
        big_bbox = (-33.89, 151.25, -33.86, 151.29)
        full = _build_scan_plan(big_bbox, 1)
        self.assertGreater(full.blocks, 1)
        corner = [
            [151.25, -33.89],
            [151.262, -33.89],
            [151.262, -33.878],
            [151.25, -33.878],
        ]
        clipped = _build_scan_plan(big_bbox, 1, polygon=corner)
        self.assertLess(clipped.blocks, full.blocks)
        self.assertGreaterEqual(clipped.blocks, 1)

    def test_scan_window_budget_is_bounded_for_huge_selections(self):
        # An enormous selection can't run an unbounded number of detector calls: coverage
        # is guaranteed (≥1 sweep over every block) but total calls stay under the ceiling,
        # and the plan reports the reduced sweep count so the UI can explain it.
        huge_bbox = (-34.10, 151.00, -33.60, 151.60)
        plan = _build_scan_plan(huge_bbox, 12)
        self.assertGreater(plan.blocks, 1)
        self.assertLessEqual(len(plan.windows), MAX_TOTAL_SCAN_WINDOWS)
        self.assertGreaterEqual(len(plan.windows), plan.blocks)  # full coverage kept
        self.assertLessEqual(plan.sweeps, plan.requested_passes)

    def test_bbox_intersects_polygon_rejects_disjoint_cells(self):
        polygon = [[151.25, -33.89], [151.26, -33.89], [151.26, -33.88], [151.25, -33.88]]
        inside = (-33.885, 151.252, -33.882, 151.256)  # cell within the polygon
        outside = (-33.87, 151.28, -33.865, 151.285)  # cell well clear of it
        self.assertTrue(_bbox_intersects_polygon(inside, polygon))
        self.assertFalse(_bbox_intersects_polygon(outside, polygon))

    def _seed_job(self, agent, scan_id, bbox, passes):
        with agent._lock:
            agent._jobs[scan_id] = {
                "scan_id": scan_id,
                "status": "queued",
                "source": "test",
                "label": "Selected area",
                "passes_requested": passes,
                "passes_completed": 0,
                "created_at": 1,
                "updated_at": 1,
                "bbox": {"south": bbox[0], "west": bbox[1], "north": bbox[2], "east": bbox[3]},
                "polygon": None,
                "counts": {"pools": 0, "buildings": 0, "trees": 0, "powerlines": 0},
                "leads": [],
                "feature_elements": [],
                "errors": [],
                "summary": "Scan queued.",
            }

    @patch("aus_backyard_intelligence.scan_agent.fetch_australia_buildings", return_value=[])
    @patch("aus_backyard_intelligence.scan_agent.execute_overpass_query", return_value={"elements": []})
    @patch("aus_backyard_intelligence.scan_agent.fetch_osm_features")
    def test_run_job_votes_repeated_detections_into_one_confirmed_pool(self, mock_fetch, mock_overpass, mock_ms):
        bbox = (-33.8875, 151.1796, -33.8872, 151.18)
        pool = {
            "type": "way",
            "id": 950001,
            "tags": {"leisure": "swimming_pool", "detector:source": "satellite-cv", "confidence:cv": "0.9"},
            "geometry": [
                {"lat": -33.8874, "lon": 151.1797},
                {"lat": -33.8874, "lon": 151.1798},
                {"lat": -33.8873, "lon": 151.1798},
                {"lat": -33.8873, "lon": 151.1797},
            ],
        }
        mock_fetch.return_value = {"elements": [pool]}
        agent = AreaScanAgent(data_path="sample.json")
        scan_id = "scan-test"
        self._seed_job(agent, scan_id, bbox, 3)

        agent._run_job(scan_id, bbox, 3, None)
        job = agent.get_scan(scan_id)

        self.assertEqual(mock_fetch.call_count, 3)
        # First pass scans the full baseline bbox at max zoom; all passes are precise.
        self.assertEqual(mock_fetch.call_args_list[0].args[0], bbox)
        for call in mock_fetch.call_args_list:
            self.assertEqual(call.kwargs["zoom"], 20)
            self.assertEqual(call.kwargs["detector_precision"], "precise")
            self.assertIs(call.kwargs["force_refresh"], True)
            self.assertEqual(call.kwargs["detector_queue_timeout"], 30.0)
        self.assertEqual(job["passes_completed"], 3)
        # The same pool seen across 3 passes collapses to ONE confirmed pool.
        self.assertEqual(job["counts"]["pools"], 1)
        self.assertEqual(len(job["feature_elements"]), 1)
        kept = job["feature_elements"][0]
        self.assertEqual(kept["tags"]["leisure"], "swimming_pool")
        self.assertEqual(kept["tags"]["scan:verdict"], "confirmed")
        self.assertEqual(kept["tags"]["scan:passes_seen"], "3")
        self.assertIn("Ran 3/3 detection passes", job["summary"])
        self.assertNotIn("cells", job["summary"])

    @patch("aus_backyard_intelligence.scan_agent.fetch_australia_buildings", return_value=[])
    @patch("aus_backyard_intelligence.scan_agent.execute_overpass_query", return_value={"elements": []})
    @patch("aus_backyard_intelligence.scan_agent.fetch_osm_features")
    def test_cancellation_during_inference_drops_late_area_results(
        self,
        mock_fetch,
        _mock_overpass,
        _mock_ms,
    ):
        bbox = (-33.8875, 151.1796, -33.8872, 151.18)
        agent = AreaScanAgent(data_path="sample.json")
        self._seed_job(agent, "scan-in-flight", bbox, 3)
        late_pool = {
            "type": "way",
            "id": 999,
            "tags": {"leisure": "swimming_pool"},
            "geometry": [{"lat": -33.8874, "lon": 151.1797}],
        }

        def finish_after_cancel(*_args, **_kwargs):
            agent.cancel_scan("scan-in-flight", reason="Selection changed")
            return {"elements": [late_pool]}

        mock_fetch.side_effect = finish_after_cancel
        agent._run_job("scan-in-flight", bbox, 3, None)

        job = agent.get_scan("scan-in-flight")
        self.assertEqual(job["status"], "cancelled")
        self.assertEqual(job["feature_elements"], [])
        self.assertEqual(job["leads"], [])
        mock_fetch.assert_called_once()

    @patch("aus_backyard_intelligence.scan_agent.fetch_australia_buildings", return_value=[])
    @patch("aus_backyard_intelligence.scan_agent.execute_overpass_query", return_value={"elements": []})
    @patch("aus_backyard_intelligence.scan_agent.fetch_osm_features")
    def test_run_job_publishes_solar_detection_and_count(self, mock_fetch, _mock_overpass, _mock_ms):
        bbox = (-33.8875, 151.1796, -33.8872, 151.18)
        solar = {
            "type": "way",
            "id": 950000101,
            "tags": {
                "power": "generator",
                "generator:source": "solar",
                "detector:source": "satellite-yolo-solar",
                "confidence:yolo": "0.81",
            },
            "geometry": [
                {"lat": -33.8874, "lon": 151.1797},
                {"lat": -33.8874, "lon": 151.1798},
                {"lat": -33.8873, "lon": 151.1798},
                {"lat": -33.8873, "lon": 151.1797},
            ],
        }
        mock_fetch.return_value = {"elements": [solar]}
        agent = AreaScanAgent(data_path="sample.json")
        self._seed_job(agent, "scan-solar", bbox, 3)

        agent._run_job("scan-solar", bbox, 3, None)

        job = agent.get_scan("scan-solar")
        self.assertEqual(job["counts"]["solar"], 1)
        self.assertEqual(len(job["feature_elements"]), 1)
        self.assertEqual(job["feature_elements"][0]["tags"]["generator:source"], "solar")
        self.assertIn("1 solar arrays", job["summary"])

    @patch("aus_backyard_intelligence.scan_agent.fetch_australia_buildings", return_value=[])
    @patch("aus_backyard_intelligence.scan_agent.execute_overpass_query", return_value={"elements": []})
    @patch("aus_backyard_intelligence.scan_agent.fetch_osm_features")
    def test_detector_pipeline_error_stops_scan_as_pollable_failure(
        self,
        mock_fetch,
        _mock_overpass,
        _mock_ms,
    ):
        bbox = (-33.8875, 151.1796, -33.8872, 151.18)
        mock_fetch.return_value = {
            "elements": [],
            "pipeline": {
                "steps": [
                    {
                        "name": "satellite_pool_detector",
                        "status": "busy_other_process",
                        "error": "Detector is owned by PID 123.",
                        "retryable": True,
                    }
                ]
            },
        }
        agent = AreaScanAgent(data_path="sample.json")
        self._seed_job(agent, "scan-busy", bbox, 3)

        agent._run_job("scan-busy", bbox, 3, None)

        job = agent.get_scan("scan-busy")
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["passes_completed"], 0)
        self.assertIn("PID 123", job["summary"])
        mock_fetch.assert_called_once()

    @patch("aus_backyard_intelligence.scan_agent.fetch_australia_buildings", return_value=[])
    @patch("aus_backyard_intelligence.scan_agent.execute_overpass_query", return_value={"elements": []})
    @patch("aus_backyard_intelligence.scan_agent.fetch_osm_features")
    def test_run_job_rejects_low_confidence_single_pass_false_positive(self, mock_fetch, mock_overpass, mock_ms):
        bbox = (-33.8875, 151.1796, -33.8872, 151.18)
        # A low-confidence satellite candidate that appears in only ONE pass while
        # several passes covered its location should be rejected as a false positive.
        flicker = {
            "type": "way",
            "id": 950002,
            "tags": {
                "leisure": "swimming_pool",
                "detector:source": "satellite-cv",
                "confidence:yolo": "0.41",
                "confidence:cv": "0.41",
            },
            "geometry": [
                {"lat": -33.88735, "lon": 151.17975},
                {"lat": -33.88735, "lon": 151.17985},
                {"lat": -33.88725, "lon": 151.17985},
                {"lat": -33.88725, "lon": 151.17975},
            ],
        }
        # Only the first pass returns the flicker; later passes (which still cover
        # the same spot) return nothing.
        mock_fetch.side_effect = [{"elements": [flicker]}] + [{"elements": []}] * 7
        agent = AreaScanAgent(data_path="sample.json")
        scan_id = "scan-fp"
        self._seed_job(agent, scan_id, bbox, 6)

        agent._run_job(scan_id, bbox, 6, None)
        job = agent.get_scan(scan_id)

        self.assertEqual(job["counts"]["pools"], 0)
        self.assertEqual(job["feature_elements"], [])
        self.assertIn("Rejected 1 false-positive candidate", job["summary"])

    @patch("aus_backyard_intelligence.scan_agent.fetch_australia_buildings", return_value=[])
    @patch("aus_backyard_intelligence.scan_agent.fetch_osm_features")
    def test_run_job_drops_satellite_pool_detected_on_a_building_roof(self, mock_fetch, mock_ms):
        bbox = (-33.8875, 151.1796, -33.8872, 151.18)
        building = {
            "type": "way",
            "id": 700001,
            "tags": {"building": "house"},
            "geometry": [
                {"lat": -33.88735, "lon": 151.17975},
                {"lat": -33.88735, "lon": 151.17985},
                {"lat": -33.88725, "lon": 151.17985},
                {"lat": -33.88725, "lon": 151.17975},
            ],
        }
        # A confident satellite "pool" whose centroid lands ON the building roof (FP)…
        roof_pool = {
            "type": "way",
            "id": 950003,
            "tags": {"leisure": "swimming_pool", "detector:source": "satellite-cv", "confidence:yolo": "0.88"},
            "geometry": [
                {"lat": -33.88732, "lon": 151.17978},
                {"lat": -33.88732, "lon": 151.17982},
                {"lat": -33.88728, "lon": 151.17982},
                {"lat": -33.88728, "lon": 151.17978},
            ],
        }
        # …and a genuine pool in the open (no building over it) — must be kept.
        backyard_pool = {
            "type": "way",
            "id": 950004,
            "tags": {"leisure": "swimming_pool", "detector:source": "satellite-cv", "confidence:yolo": "0.88"},
            "geometry": [
                {"lat": -33.88742, "lon": 151.17962},
                {"lat": -33.88742, "lon": 151.17966},
                {"lat": -33.88738, "lon": 151.17966},
                {"lat": -33.88738, "lon": 151.17962},
            ],
        }
        with patch(
            "aus_backyard_intelligence.scan_agent.execute_overpass_query",
            return_value={"elements": [building]},
        ):
            mock_fetch.return_value = {"elements": [roof_pool, backyard_pool]}
            agent = AreaScanAgent(data_path="sample.json")
            scan_id = "scan-roof"
            self._seed_job(agent, scan_id, bbox, 2)
            agent._run_job(scan_id, bbox, 2, None)
            job = agent.get_scan(scan_id)

        # Only the open-yard pool survives; the rooftop false positive is dropped.
        self.assertEqual(job["counts"]["pools"], 1)
        kept_ids = {el["id"] for el in job["feature_elements"] if el["tags"].get("leisure") == "swimming_pool"}
        self.assertIn(950004, kept_ids)
        self.assertNotIn(950003, kept_ids)

    @patch("aus_backyard_intelligence.scan_agent.fetch_australia_buildings", return_value=[])
    @patch("aus_backyard_intelligence.scan_agent.fetch_osm_features")
    def test_run_job_drops_isolated_open_water_detection_but_keeps_when_no_buildings(self, mock_fetch, mock_ms):
        # ~500 m bbox so a detection can be far from any building.
        bbox = (-33.890, 151.270, -33.886, 151.276)
        building = {
            "type": "way",
            "id": 700002,
            "tags": {"building": "house"},
            "geometry": [
                {"lat": -33.8896, "lon": 151.2704},
                {"lat": -33.8896, "lon": 151.2706},
                {"lat": -33.8894, "lon": 151.2706},
                {"lat": -33.8894, "lon": 151.2704},
            ],
        }
        # A confident "pool" far out over open water — no building anywhere near it (FP).
        open_water = {
            "type": "way",
            "id": 950005,
            "tags": {"leisure": "swimming_pool", "detector:source": "satellite-cv", "confidence:yolo": "0.9"},
            "geometry": [
                {"lat": -33.88655, "lon": 151.27545},
                {"lat": -33.88655, "lon": 151.27555},
                {"lat": -33.88645, "lon": 151.27555},
                {"lat": -33.88645, "lon": 151.27545},
            ],
        }
        agent = AreaScanAgent(data_path="sample.json")

        # With building coverage present, the isolated open-water detection is dropped.
        with patch(
            "aus_backyard_intelligence.scan_agent.execute_overpass_query",
            return_value={"elements": [building]},
        ):
            mock_fetch.return_value = {"elements": [open_water]}
            self._seed_job(agent, "scan-water", bbox, 2)
            agent._run_job("scan-water", bbox, 2, None)
            self.assertEqual(agent.get_scan("scan-water")["counts"]["pools"], 0)

        # But with NO building footprints available, the filter can't judge isolation,
        # so it must NOT drop detections (no recall regression in unmapped areas).
        with patch(
            "aus_backyard_intelligence.scan_agent.execute_overpass_query",
            return_value={"elements": []},
        ):
            mock_fetch.return_value = {"elements": [open_water]}
            self._seed_job(agent, "scan-water2", bbox, 2)
            agent._run_job("scan-water2", bbox, 2, None)
            self.assertEqual(agent.get_scan("scan-water2")["counts"]["pools"], 1)

    @patch("aus_backyard_intelligence.scan_agent.fetch_australia_buildings", return_value=[])
    @patch("aus_backyard_intelligence.scan_agent.execute_overpass_query", return_value={"elements": []})
    @patch("aus_backyard_intelligence.scan_agent.fetch_osm_features")
    def test_run_job_rejects_implausibly_large_satellite_pool(self, mock_fetch, mock_overpass, mock_ms):
        bbox = (-33.890, 151.270, -33.886, 151.276)
        # A confident "pool" the size of an ocean rock shelf (~150 m x 100 m) — reject.
        ocean_blob = {
            "type": "way",
            "id": 950010,
            "tags": {"leisure": "swimming_pool", "detector:source": "satellite-cv", "confidence:cv": "0.95"},
            "geometry": [
                {"lat": -33.8880, "lon": 151.2740},
                {"lat": -33.8880, "lon": 151.2756},
                {"lat": -33.8871, "lon": 151.2756},
                {"lat": -33.8871, "lon": 151.2740},
            ],
        }
        # A normal backyard-sized pool (~9 m x 7 m) — keep.
        backyard = {
            "type": "way",
            "id": 950011,
            "tags": {"leisure": "swimming_pool", "detector:source": "satellite-cv", "confidence:cv": "0.9"},
            "geometry": [
                {"lat": -33.8891, "lon": 151.2701},
                {"lat": -33.8891, "lon": 151.27018},
                {"lat": -33.88902, "lon": 151.27018},
                {"lat": -33.88902, "lon": 151.2701},
            ],
        }
        mock_fetch.return_value = {"elements": [ocean_blob, backyard]}
        agent = AreaScanAgent(data_path="sample.json")
        self._seed_job(agent, "scan-area-gate", bbox, 3)
        agent._run_job("scan-area-gate", bbox, 3, None)
        job = agent.get_scan("scan-area-gate")

        kept_ids = {el["id"] for el in job["feature_elements"]}
        self.assertIn(950011, kept_ids)
        self.assertNotIn(950010, kept_ids)

    @patch("aus_backyard_intelligence.scan_agent.fetch_australia_buildings", return_value=[])
    @patch("aus_backyard_intelligence.scan_agent.execute_overpass_query", return_value={"elements": []})
    @patch("aus_backyard_intelligence.scan_agent.fetch_osm_features")
    def test_run_job_gates_pool_size_on_detector_area_not_refined_outline(
        self,
        mock_fetch,
        _mock_overpass,
        _mock_ms,
    ):
        bbox = (-33.890, 151.270, -33.886, 151.276)
        # YOLO's original box is a valid 300 m² pool. Its optional padded
        # water-mask outline spans roughly 900 m², which is display geometry and
        # must not make the Leads-only 400 m² plausibility gate discard a strong
        # ensemble detection.
        expanded_outline = {
            "type": "way",
            "id": 950012,
            "tags": {
                "leisure": "swimming_pool",
                "detector:source": "satellite-yolo",
                "detector:model_role": "secondary",
                "confidence:yolo": "0.96",
                "pool:area_sqm": "300.0",
            },
            "geometry": [
                {"lat": -33.88850, "lon": 151.27200},
                {"lat": -33.88850, "lon": 151.27230},
                {"lat": -33.88820, "lon": 151.27230},
                {"lat": -33.88820, "lon": 151.27200},
            ],
        }
        mock_fetch.return_value = {"elements": [expanded_outline]}
        agent = AreaScanAgent(data_path="sample.json")
        self._seed_job(agent, "scan-refined-area", bbox, 3)

        agent._run_job("scan-refined-area", bbox, 3, None)

        job = agent.get_scan("scan-refined-area")
        kept_ids = {el["id"] for el in job["feature_elements"]}
        self.assertIn(950012, kept_ids)
        self.assertEqual(job["feature_elements"][0]["tags"]["scan:verdict"], "confirmed")

    @patch("aus_backyard_intelligence.scan_agent.fetch_australia_buildings", return_value=[])
    @patch("aus_backyard_intelligence.scan_agent.execute_overpass_query", return_value={"elements": []})
    @patch("aus_backyard_intelligence.scan_agent.fetch_osm_features")
    def test_run_job_rejects_detector_oversize_even_when_refined_outline_is_small(
        self,
        mock_fetch,
        _mock_overpass,
        _mock_ms,
    ):
        bbox = (-33.890, 151.270, -33.886, 151.276)
        # Conversely, a small visible-water component must not hide the fact that
        # the model's original candidate box is an implausible 900 m² blob.
        contracted_outline = {
            "type": "way",
            "id": 950013,
            "tags": {
                "leisure": "swimming_pool",
                "detector:source": "satellite-yolo",
                "confidence:yolo": "0.97",
                "pool:area_sqm": "900.0",
            },
            "geometry": [
                {"lat": -33.88850, "lon": 151.27200},
                {"lat": -33.88850, "lon": 151.27208},
                {"lat": -33.88842, "lon": 151.27208},
                {"lat": -33.88842, "lon": 151.27200},
            ],
        }
        mock_fetch.return_value = {"elements": [contracted_outline]}
        agent = AreaScanAgent(data_path="sample.json")
        self._seed_job(agent, "scan-detector-area", bbox, 3)

        agent._run_job("scan-detector-area", bbox, 3, None)

        self.assertEqual(agent.get_scan("scan-detector-area")["counts"]["pools"], 0)

    def test_scan_features_are_clipped_to_selected_polygon(self):
        polygon = [
            [151.1796, -33.8875],
            [151.1799, -33.8875],
            [151.1799, -33.8872],
            [151.1796, -33.8872],
        ]
        inside_pool = {
            "type": "way",
            "id": 1,
            "tags": {"leisure": "swimming_pool"},
            "geometry": [
                {"lat": -33.88742, "lon": 151.17968},
                {"lat": -33.88742, "lon": 151.17978},
                {"lat": -33.88732, "lon": 151.17978},
                {"lat": -33.88732, "lon": 151.17968},
            ],
        }
        outside_pool = {
            "type": "way",
            "id": 2,
            "tags": {"leisure": "swimming_pool"},
            "geometry": [
                {"lat": -33.88742, "lon": 151.1801},
                {"lat": -33.88742, "lon": 151.1802},
                {"lat": -33.88732, "lon": 151.1802},
                {"lat": -33.88732, "lon": 151.1801},
            ],
        }

        self.assertTrue(_feature_intersects_polygon(inside_pool, polygon))
        self.assertFalse(_feature_intersects_polygon(outside_pool, polygon))

    @staticmethod
    def _house(house_id=1, lat=-33.88740, lon=151.17970, tags=None):
        return {
            "type": "way",
            "id": house_id,
            "tags": {"building": "house", **(tags or {})},
            "geometry": [
                {"lat": lat, "lon": lon},
                {"lat": lat, "lon": lon + 0.00011},
                {"lat": lat + 0.00009, "lon": lon + 0.00011},
                {"lat": lat + 0.00009, "lon": lon},
            ],
        }

    def test_residential_filter_excludes_sheds_and_commercial(self):
        self.assertTrue(_is_residential_building(self._house()))
        self.assertFalse(_is_residential_building(self._house(2, tags={"building": "shed"})))
        self.assertFalse(_is_residential_building(self._house(3, tags={"building": "yes", "shop": "bakery"})))

    def test_summarize_builds_residential_lead_with_published_contact(self):
        agent = AreaScanAgent(data_path="sample.json")
        house = self._house(
            1,
            tags={
                "addr:housenumber": "12",
                "addr:street": "Vine Street",
                "phone": "+61 2 5550 1234",
                "website": "https://example.com",
            },
        )
        shed = self._house(2, lat=-33.88760, lon=151.17990, tags={"building": "shed"})

        counts, leads = agent._summarize_features([house, shed], polygon=None)

        self.assertEqual(counts["residential"], 1)
        self.assertEqual(len(leads), 1)
        lead = leads[0]
        self.assertIn("Vine Street", lead["title"])
        self.assertEqual(lead["contact"]["status"], "published_osm")
        self.assertEqual(lead["contact"]["phone"], "+61 2 5550 1234")
        self.assertGreaterEqual(lead["score"], 42.0)
        for key in ("land_titles", "realestate", "planning"):
            self.assertIn(key, lead["links"])

    def test_summarize_preserves_explicit_osm_unit_and_provenance(self):
        agent = AreaScanAgent(data_path="sample.json")
        house = self._house(
            15,
            tags={
                "addr:unit": "5",
                "addr:housenumber": "12",
                "addr:street": "Vine Street",
                "addr:suburb": "Sydney",
                "addr:state": "NSW",
                "addr:postcode": "2000",
            },
        )

        _, leads = agent._summarize_features([house], polygon=None)

        self.assertEqual(len(leads), 1)
        lead = leads[0]
        self.assertEqual(lead["title"], "Unit 5, 12 Vine Street, Sydney")
        self.assertEqual(lead["address_details"]["unit"], "5")
        self.assertEqual(lead["address_details"]["source"], "openstreetmap")
        self.assertEqual(lead["address_details"]["precision"], "unit")
        self.assertTrue(lead["address_details"]["verified"])

    def test_summarize_does_not_select_a_unit_from_osm_flats_range(self):
        agent = AreaScanAgent(data_path="sample.json")
        house = self._house(
            18,
            tags={
                "addr:flats": "1-12",
                "addr:housenumber": "20",
                "addr:street": "Bay Road",
            },
        )

        _, leads = agent._summarize_features([house], polygon=None)

        lead = leads[0]
        self.assertEqual(lead["title"], "20 Bay Road")
        self.assertEqual(lead["address_details"]["unit"], "")
        self.assertEqual(lead["address_details"]["available_units"], "1-12")
        self.assertFalse(lead["address_details"]["unit_identified"])
        self.assertEqual(lead["address_details"]["precision"], "premise")

    @patch("aus_backyard_intelligence.cadastre.lookup_parcel", return_value=None)
    @patch("aus_backyard_intelligence.scan_agent.reverse_geocode")
    @patch("aus_backyard_intelligence.scan_agent.google_reverse_geocode")
    @patch("aus_backyard_intelligence.scan_agent.google_geocoding_configured", return_value=True)
    def test_enrichment_uses_verified_google_unit_address(
        self,
        _mock_configured,
        mock_google,
        mock_nominatim,
        _mock_parcel,
    ):
        agent = AreaScanAgent(data_path="sample.json")
        _, leads = agent._summarize_features([self._house(16)], polygon=None)
        mock_google.return_value = {
            "formatted_address": "Unit 5, 12 Vine Street, Sydney NSW 2000, Australia",
            "components": {
                "unit": "5",
                "street_number": "12",
                "route": "Vine Street",
                "locality": "Sydney",
                "state": "New South Wales",
                "postcode": "2000",
                "country_code": "AU",
            },
            "source": "google",
            "place_id": "google-place-123",
            "precision": "rooftop",
            "verified": True,
            "exact": True,
        }

        enriched = agent._enrich_lead_addresses(leads)

        self.assertEqual(len(enriched), 1)
        lead = enriched[0]
        self.assertEqual(lead["address_details"]["unit"], "5")
        self.assertEqual(lead["address_details"]["source"], "google")
        self.assertTrue(lead["address_details"]["verified"])
        self.assertEqual(lead["title"], mock_google.return_value["formatted_address"])
        self.assertIn("query_place_id=google-place-123", lead["links"]["google_maps"])
        self.assertNotIn("_precise_osm_addr", lead)
        mock_nominatim.assert_not_called()

    @patch("aus_backyard_intelligence.cadastre.lookup_parcel", return_value=None)
    @patch("aus_backyard_intelligence.scan_agent.google_reverse_geocode")
    @patch("aus_backyard_intelligence.scan_agent.google_geocoding_configured", return_value=True)
    def test_google_free_form_text_cannot_invent_a_unit_without_subpremise(
        self,
        _mock_configured,
        mock_google,
        _mock_parcel,
    ):
        agent = AreaScanAgent(data_path="sample.json")
        _, leads = agent._summarize_features([self._house(19)], polygon=None)
        mock_google.return_value = {
            "formatted_address": "Unit 9, 8 Example Street, Sydney NSW 2000, Australia",
            "components": {
                "unit": "",
                "street_number": "8",
                "route": "Example Street",
                "locality": "Sydney",
                "state": "New South Wales",
                "postcode": "2000",
                "country_code": "AU",
            },
            "source": "google",
            "place_id": "premise-without-unit",
            "precision": "rooftop",
            "verified": True,
            "exact": True,
        }

        lead = agent._enrich_lead_addresses(leads)[0]

        self.assertEqual(lead["address_details"]["unit"], "")
        self.assertFalse(lead["address_details"]["unit_identified"])
        self.assertNotIn("Unit 9", lead["title"])
        self.assertEqual(lead["title"], "8 Example Street, Sydney, New South Wales, 2000")

    @patch("aus_backyard_intelligence.cadastre.lookup_parcel", return_value=None)
    @patch("aus_backyard_intelligence.scan_agent.reverse_geocode")
    @patch("aus_backyard_intelligence.scan_agent.google_reverse_geocode")
    @patch("aus_backyard_intelligence.scan_agent.google_geocoding_configured", return_value=True)
    def test_google_plus_code_only_result_falls_back_to_nominatim(
        self,
        _mock_configured,
        mock_google,
        mock_nominatim,
        _mock_parcel,
    ):
        agent = AreaScanAgent(data_path="sample.json")
        _, leads = agent._summarize_features([self._house(20)], polygon=None)
        mock_google.return_value = {
            "formatted_address": "47FH+6WW Dover Heights NSW, Australia",
            "components": {},
            "source": "google",
            "precision": "approximate",
            "verified": False,
            "exact": False,
        }
        mock_nominatim.return_value = {
            "address": {"road": "Military Road", "suburb": "Dover Heights", "state": "NSW"}
        }

        lead = agent._enrich_lead_addresses(leads)[0]

        self.assertEqual(lead["address_details"]["source"], "openstreetmap_nominatim")
        self.assertEqual(lead["title"], "Military Road, Dover Heights, NSW")
        self.assertFalse(lead["address_details"]["verified"])

    @patch("aus_backyard_intelligence.cadastre.lookup_parcel", return_value=None)
    @patch("aus_backyard_intelligence.scan_agent.reverse_geocode")
    @patch("aus_backyard_intelligence.scan_agent.google_geocoding_configured", return_value=False)
    def test_nominatim_address_is_approximate_but_preserves_explicit_unit(
        self,
        _mock_configured,
        mock_nominatim,
        _mock_parcel,
    ):
        agent = AreaScanAgent(data_path="sample.json")
        _, leads = agent._summarize_features([self._house(17)], polygon=None)
        mock_nominatim.return_value = {
            "display_name": "Unit 8, 20 Bay Road, Sydney NSW 2000, Australia",
            "address": {
                "unit": "8",
                "house_number": "20",
                "road": "Bay Road",
                "suburb": "Sydney",
                "state": "New South Wales",
                "postcode": "2000",
                "country_code": "au",
            },
        }

        lead = agent._enrich_lead_addresses(leads)[0]

        self.assertEqual(lead["title"], "Unit 8, 20 Bay Road, Sydney, New South Wales, 2000")
        self.assertEqual(lead["address_details"]["unit"], "8")
        self.assertEqual(lead["address_details"]["status"], "approximate")
        self.assertFalse(lead["address_details"]["verified"])
        for key in ("realestate", "domain", "planning"):
            self.assertNotIn(key, lead["links"])

    def test_public_lead_payload_strips_internal_address_flags(self):
        agent = AreaScanAgent(data_path="sample.json")
        with agent._lock:
            agent._jobs["scan-test"] = {
                "scan_id": "scan-test",
                "label": "Test area",
                "leads": [{"id": "lead-1", "score": 50, "_precise_osm_addr": False}],
            }

        public = agent.aggregate_leads(limit=5)

        self.assertEqual(public[0]["id"], "lead-1")
        self.assertNotIn("_precise_osm_addr", public[0])

    def test_summarize_excludes_property_with_pool_on_its_footprint(self):
        agent = AreaScanAgent(data_path="sample.json")
        house = self._house(1)
        pool_on_lot = {
            "type": "way",
            "id": 9,
            "tags": {"leisure": "swimming_pool"},
            "geometry": [
                {"lat": -33.887355, "lon": 151.179755},
                {"lat": -33.887355, "lon": 151.179770},
                {"lat": -33.887345, "lon": 151.179770},
                {"lat": -33.887345, "lon": 151.179755},
            ],
        }
        counts, leads = agent._summarize_features([house, pool_on_lot], polygon=None)
        self.assertEqual(len(leads), 0)


if __name__ == "__main__":
    unittest.main()
