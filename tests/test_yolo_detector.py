import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from aus_backyard_intelligence import yolo_detector


class FakeImage:
    shape = (256, 256, 3)


class YoloDetectorViewportCoverageTest(unittest.TestCase):
    def test_detector_status_does_not_load_heavy_runtime(self):
        with patch.object(yolo_detector, "_lazy_import") as lazy_import:
            status = yolo_detector.get_detector_status()

        lazy_import.assert_not_called()
        self.assertIn("detector_signature", status)
        self.assertIn("pool_yolo_runtime_loaded", status)
        self.assertIn("pool_yolo_runtime", status)
        self.assertIn("pool_yolo_last_inference_success", status)

    def test_analysis_polygon_boundary_is_inclusive_and_excludes_outside_points(self):
        polygon = [(151.0, -34.0), (151.1, -34.0), (151.1, -33.9), (151.0, -33.9)]
        self.assertTrue(yolo_detector._point_in_analysis_polygon(151.05, -33.95, polygon))
        self.assertTrue(yolo_detector._point_in_analysis_polygon(151.0, -33.95, polygon))
        self.assertFalse(yolo_detector._point_in_analysis_polygon(151.11, -33.95, polygon))

    def test_analysis_polygon_mask_blacks_out_pixels_outside_selection(self):
        yolo_detector._lazy_import()
        if yolo_detector._np is None:
            self.skipTest("NumPy is required for polygon masking")
        image = yolo_detector._np.full((10, 10, 3), 200, dtype=yolo_detector._np.uint8)
        masked, inside_pixels = yolo_detector._mask_image_to_analysis_polygon(
            image,
            (0.0, 0.0, 10.0, 10.0),
            [(0.0, 0.0), (5.0, 0.0), (5.0, 10.0), (0.0, 10.0)],
        )

        self.assertGreater(inside_pixels, 0)
        self.assertTrue((masked[5, 2] == 200).all())
        self.assertTrue((masked[5, 8] == 0).all())
        self.assertTrue((image == 200).all(), "masking must not mutate the cached tile image")

    def test_selected_area_is_masked_before_yolo_and_cv_inference(self):
        yolo_detector._lazy_import()
        if yolo_detector._np is None:
            self.skipTest("NumPy is required for polygon masking")
        source = yolo_detector._np.full((10, 10, 3), 180, dtype=yolo_detector._np.uint8)
        captured = []

        def capture_yolo(image, **kwargs):
            captured.append(image.copy())
            return []

        def capture_cv(image, **kwargs):
            captured.append(image.copy())
            return []

        with patch.object(yolo_detector, "pool_yolo_available", return_value=True), \
             patch.object(yolo_detector, "_choose_tile_zoom", return_value=20), \
             patch.object(yolo_detector, "_iter_detection_bboxes", return_value=[(0.0, 0.0, 10.0, 10.0)]), \
             patch.object(yolo_detector, "_planned_chunk_count", return_value=1), \
             patch.object(yolo_detector, "_stitch_tiles_with_stats", return_value=(source, (0.0, 0.0, 10.0, 10.0), {"attempted_tiles": 1, "loaded_tiles": 1})), \
             patch.object(yolo_detector, "_STRICT_WATER_SUPPLEMENT_MODE", "1"), \
             patch.object(yolo_detector, "_yolo_detect_pools", side_effect=capture_yolo), \
             patch.object(yolo_detector, "_detect_blue_pools_cv", side_effect=capture_cv):
            yolo_detector.detect_pools_in_bbox(
                (0.0, 0.0, 10.0, 10.0),
                zoom=20,
                detector_precision="precise",
                analysis_polygon=[[0.0, 0.0], [5.0, 0.0], [5.0, 10.0], [0.0, 10.0]],
            )

        self.assertEqual(len(captured), 2)
        for inference_image in captured:
            self.assertTrue((inference_image[5, 2] == 180).all())
            self.assertTrue((inference_image[5, 8] == 0).all())
        diagnostics = yolo_detector.get_last_detector_diagnostics()
        self.assertTrue(diagnostics["analysis_mask_applied"])
        self.assertGreater(diagnostics["analysis_mask_pixels"], 0)

    def test_repairs_fused_conv_missing_batch_norm_forward(self):
        class BrokenConv:
            def forward(self, value=None):
                return self.bn

            def forward_fuse(self, value=None):
                return "fused"

        class NativeModel:
            def __init__(self, module):
                self.module = module

            def modules(self):
                return [self.module]

        class Wrapper:
            def __init__(self, module):
                self.model = NativeModel(module)

        module = BrokenConv()
        repaired = yolo_detector._repair_fused_conv_forwards(Wrapper(module))

        self.assertEqual(repaired, 1)
        self.assertEqual(module.forward(), "fused")

    def test_inference_errors_are_single_line_and_bounded_for_status(self):
        safe = yolo_detector._safe_inference_error(RuntimeError("first line\n" + "x" * 1000))

        self.assertNotIn("\n", safe)
        self.assertLessEqual(len(safe), 550)
        self.assertTrue(safe.startswith("RuntimeError: first line"))

    def test_pool_prediction_repairs_bn_failure_and_retries_once(self):
        class BrokenConv:
            def __init__(self):
                self.bn = object()

            def forward(self, value=None):
                return self.bn

            def forward_fuse(self, value=None):
                return "fused"

        class NativeModel:
            def __init__(self, module):
                self.module = module

            def modules(self):
                return [self.module]

        class Model:
            def __init__(self):
                self.module = BrokenConv()
                self.model = NativeModel(self.module)
                self.calls = 0

            def predict(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    del self.module.bn
                    raise AttributeError("'Conv' object has no attribute 'bn'")
                return ["recovered"]

        diagnostics = {}
        model = Model()
        result = yolo_detector._predict_pool_yolo_model(
            model,
            object(),
            role="primary",
            imgsz=1024,
            conf=0.15,
            diagnostics=diagnostics,
        )

        self.assertEqual(result, ["recovered"])
        self.assertEqual(model.calls, 2)
        self.assertEqual(diagnostics["inference_successes"], 1)
        self.assertEqual(diagnostics["inference_retries"], 1)
        self.assertEqual(diagnostics["compatibility_repairs"], 1)

    def test_pool_predictions_are_serialized_to_bound_peak_memory(self):
        class Model:
            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.guard = threading.Lock()

            def predict(self, *args, **kwargs):
                with self.guard:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.02)
                with self.guard:
                    self.active -= 1
                return []

        model = Model()
        threads = [
            threading.Thread(
                target=yolo_detector._predict_pool_yolo_model,
                args=(model, object()),
                kwargs={"role": "primary", "imgsz": 640, "conf": 0.15},
            )
            for _ in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(model.max_active, 1)

    def test_checkpoint_class_mapping_emits_solar_zero_and_pool_one(self):
        class Row(list):
            def tolist(self):
                return list(self)

        class Box:
            def __init__(self, cls_id, confidence, xyxy):
                self.cls = [cls_id]
                self.conf = [confidence]
                self.xyxy = [Row(xyxy)]

        class Result:
            names = {"0": "solarpanel", "1": "pool"}
            boxes = [Box(0, 0.82, [10, 10, 30, 24]), Box(1, 0.76, [40, 40, 70, 62])]

        with patch.object(yolo_detector, "_box_colour_stats", return_value={}):
            detections, solar_boxes, names = yolo_detector._parse_pool_yolo_results(
                object(),
                [Result()],
                model_role="primary",
                include_solar=True,
                conf=0.15,
            )

        self.assertEqual(names, ["solarpanel", "pool"])
        self.assertEqual(len(solar_boxes), 1)
        self.assertEqual([d["detector"] for d in detections], ["yolo_solar", "yolo"])
        self.assertEqual([d["class_id"] for d in detections], [0, 1])

    def test_fast_cv_detector_prefers_clear_water_and_rejects_roof_or_turf_blocks(self):
        yolo_detector._lazy_import()
        if not yolo_detector._HAS_CV2 or yolo_detector._np is None:
            self.skipTest("OpenCV and NumPy are required for colour detector tests")

        np = yolo_detector._np
        cv2 = yolo_detector._cv2

        clear_pool = np.full((180, 220, 3), (90, 95, 110), dtype=np.uint8)
        cv2.rectangle(clear_pool, (70, 55), (145, 105), (190, 175, 110), -1)
        self.assertGreaterEqual(
            len(yolo_detector._detect_blue_pools_cv(clear_pool, detector_precision="fast")),
            1,
        )

        pool_image = np.full((180, 220, 3), (90, 95, 110), dtype=np.uint8)
        cv2.rectangle(pool_image, (70, 55), (145, 105), (140, 128, 123), -1)
        self.assertGreaterEqual(len(yolo_detector._detect_blue_pools_cv(pool_image)), 1)
        self.assertEqual(yolo_detector._detect_blue_pools_cv(pool_image, detector_precision="fast"), [])

        grey_roof = np.full((180, 220, 3), (90, 95, 110), dtype=np.uint8)
        cv2.rectangle(grey_roof, (70, 55), (145, 105), (142, 138, 132), -1)
        self.assertEqual(yolo_detector._detect_blue_pools_cv(grey_roof, detector_precision="fast"), [])

        green_turf = np.full((180, 220, 3), (90, 95, 110), dtype=np.uint8)
        cv2.rectangle(green_turf, (70, 55), (145, 105), (120, 150, 120), -1)
        self.assertEqual(yolo_detector._detect_blue_pools_cv(green_turf, detector_precision="fast"), [])

    def test_detection_chunks_cover_full_wide_viewport(self):
        bbox = (-33.875, 151.272, -33.865, 151.286)
        chunks = yolo_detector._iter_detection_bboxes(bbox, 18)

        self.assertGreater(len(chunks), 1)
        self.assertAlmostEqual(min(chunk[0] for chunk in chunks), bbox[0], places=7)
        self.assertAlmostEqual(min(chunk[1] for chunk in chunks), bbox[1], places=7)
        self.assertAlmostEqual(max(chunk[2] for chunk in chunks), bbox[2], places=7)
        self.assertAlmostEqual(max(chunk[3] for chunk in chunks), bbox[3], places=7)

        for chunk in chunks:
            x_min, x_max, y_min, y_max = yolo_detector._tile_range_for_bbox(chunk, 18)
            self.assertLessEqual(x_max - x_min + 1, yolo_detector._MAX_STITCH_TILE_SIDE)
            self.assertLessEqual(y_max - y_min + 1, yolo_detector._MAX_STITCH_TILE_SIDE)

    def test_precision_mode_downshifts_only_broad_viewports(self):
        broad_bbox = (-33.875, 151.273, -33.864, 151.289)
        small_cell_bbox = (-33.872, 151.276, -33.8705, 151.278)
        precise_pool_bbox = (-33.88195, 151.27758, -33.88172, 151.27793)
        selected_scan_bbox = (-33.889, 151.176, -33.8845, 151.1845)

        # The viewer's zoom no longer caps detection: each mode aims for its max
        # zoom (pools are near-invisible below z19) and only the tile/chunk
        # budget shrinks it on broad viewports.
        self.assertEqual(yolo_detector._choose_tile_zoom(broad_bbox, 18, "fast"), 19)
        self.assertEqual(yolo_detector._choose_tile_zoom(broad_bbox, 18, "auto"), 17)
        self.assertEqual(yolo_detector._choose_tile_zoom(broad_bbox, 18, "precise"), 20)
        self.assertEqual(yolo_detector._choose_tile_zoom(small_cell_bbox, 18, "fast"), 20)
        self.assertEqual(yolo_detector._choose_tile_zoom(precise_pool_bbox, 20, "precise"), 20)
        self.assertEqual(yolo_detector._choose_tile_zoom(selected_scan_bbox, 20, "precise"), 20)

    def test_precise_pool_gate_handles_multiple_pool_sizes(self):
        tiny_spa = {
            "confidence": 0.96,
            "water_fraction": 0.9,
            "circularity": 0.62,
            "aspect_ratio": 1.4,
            "extent": 0.64,
            "rotated_extent": 0.72,
            "value_mean": 150,
            "saturation_mean": 60,
        }
        weak_tiny_blob = {
            "confidence": 0.9,
            "water_fraction": 0.76,
            "circularity": 0.42,
            "aspect_ratio": 2.1,
            "extent": 0.44,
            "rotated_extent": 0.49,
            "value_mean": 135,
            "saturation_mean": 55,
        }
        standard_pool = {
            "confidence": 0.9,
            "water_fraction": 0.75,
            "circularity": 0.34,
            "aspect_ratio": 2.4,
            "extent": 0.46,
            "rotated_extent": 0.55,
            "value_mean": 145,
            "saturation_mean": 58,
        }
        lap_pool = {
            "confidence": 0.94,
            "water_fraction": 0.82,
            "circularity": 0.22,
            "aspect_ratio": 7.5,
            "extent": 0.48,
            "rotated_extent": 0.6,
            "value_mean": 132,
            "saturation_mean": 52,
        }
        large_pool = {
            "confidence": 0.93,
            "water_fraction": 0.78,
            "circularity": 0.28,
            "aspect_ratio": 3.5,
            "extent": 0.44,
            "rotated_extent": 0.52,
            "value_mean": 148,
            "saturation_mean": 57,
        }
        roof_panel = {
            **standard_pool,
            "confidence": 0.96,
            "water_fraction": 0.86,
            "context_roof_fraction": 0.52,
        }
        dark_solar_panel = {
            **standard_pool,
            "confidence": 0.91,
            "water_fraction": 0.74,
            "context_dark_panel_fraction": 0.52,
        }

        self.assertTrue(yolo_detector._passes_precise_pool_candidate(tiny_spa, 5.0))
        self.assertFalse(yolo_detector._passes_precise_pool_candidate(weak_tiny_blob, 5.0))
        self.assertTrue(yolo_detector._passes_precise_pool_candidate(standard_pool, 45.0))
        self.assertTrue(yolo_detector._passes_precise_pool_candidate(lap_pool, 80.0))
        self.assertTrue(yolo_detector._passes_precise_pool_candidate(large_pool, 420.0))
        self.assertFalse(yolo_detector._passes_precise_pool_candidate(roof_panel, 45.0))
        self.assertFalse(yolo_detector._passes_precise_pool_candidate(dark_solar_panel, 45.0))

    def test_auto_pool_gate_rejects_common_false_positives(self):
        base_pool = {
            "confidence": 0.84,
            "water_fraction": 0.70,
            "circularity": 0.42,
            "aspect_ratio": 1.8,
            "extent": 0.55,
            "rotated_extent": 0.60,
            "value_mean": 140,
            "saturation_mean": 55,
            "context_roof_fraction": 0.05,
            "context_dark_panel_fraction": 0.04,
        }
        # A clean backyard pool clears the default gate.
        self.assertTrue(yolo_detector._passes_auto_pool_candidate(base_pool, 40.0))

        blue_roof = {**base_pool, "context_roof_fraction": 0.60}
        self.assertFalse(yolo_detector._passes_auto_pool_candidate(blue_roof, 40.0))

        solar_panel = {**base_pool, "context_dark_panel_fraction": 0.60}
        self.assertFalse(yolo_detector._passes_auto_pool_candidate(solar_panel, 40.0))

        shadow = {**base_pool, "value_mean": 50}
        self.assertFalse(yolo_detector._passes_auto_pool_candidate(shadow, 40.0))

        washed_grey = {**base_pool, "saturation_mean": 12}
        self.assertFalse(yolo_detector._passes_auto_pool_candidate(washed_grey, 40.0))

        faint_water = {**base_pool, "water_fraction": 0.35}
        self.assertFalse(yolo_detector._passes_auto_pool_candidate(faint_water, 40.0))

        ragged_sliver = {**base_pool, "circularity": 0.10, "aspect_ratio": 9.5}
        self.assertFalse(yolo_detector._passes_auto_pool_candidate(ragged_sliver, 40.0))

        # Anything strong enough for the strict precise gate also clears auto.
        strong_pool = {**base_pool, "confidence": 0.92, "water_fraction": 0.80}
        self.assertTrue(yolo_detector._passes_auto_pool_candidate(strong_pool, 40.0))

    def test_yolo_water_verifier_rejects_roofs_lawn_and_shadow_boxes(self):
        clear_pool = {
            "confidence": 0.50,
            "green_minus_red": 34,
            "blue_minus_red": 42,
            "saturation_mean": 72,
            "water_fraction": 0.38,
            "strong_water_fraction": 0.16,
            "vegetation_fraction": 0.08,
            "warm_roof_fraction": 0.02,
            "dark_fraction": 0.04,
        }
        self.assertTrue(yolo_detector._yolo_box_is_watery(clear_pool))

        lawn_shadow = {
            **clear_pool,
            "confidence": 0.56,
            "green_minus_red": 10,
            "blue_minus_red": 8,
            "water_fraction": 0.08,
            "strong_water_fraction": 0.02,
            "vegetation_fraction": 0.68,
        }
        self.assertFalse(yolo_detector._yolo_box_is_watery(lawn_shadow))

        warm_roof = {
            **clear_pool,
            "green_minus_red": -4,
            "blue_minus_red": -8,
            "water_fraction": 0.12,
            "strong_water_fraction": 0.03,
            "warm_roof_fraction": 0.52,
        }
        self.assertFalse(yolo_detector._yolo_box_is_watery(warm_roof))

        weak_blue_roof = {
            **clear_pool,
            "confidence": 0.42,
            "water_fraction": 0.22,
            "strong_water_fraction": 0.04,
        }
        self.assertFalse(yolo_detector._yolo_box_is_watery(weak_blue_roof))

    def test_strict_water_supplement_gate_rejects_blue_roof_fragments(self):
        clear_pool = {
            "confidence": 0.94,
            "water_fraction": 0.96,
            "circularity": 0.46,
            "aspect_ratio": 1.7,
            "extent": 0.48,
            "rotated_extent": 0.56,
            "value_mean": 145,
            "saturation_mean": 70,
            "green_minus_red": 36,
            "blue_minus_red": 48,
            "context_roof_fraction": 0.08,
            "context_dark_panel_fraction": 0.04,
        }
        self.assertTrue(yolo_detector._passes_strict_water_supplement_candidate(clear_pool, 42.0, "fast"))

        blue_roof = {
            **clear_pool,
            "green_minus_red": 15,
            "blue_minus_red": 29,
            "context_roof_fraction": 0.33,
        }
        self.assertFalse(yolo_detector._passes_strict_water_supplement_candidate(blue_roof, 42.0, "fast"))

        solar_panel = {
            **clear_pool,
            "green_minus_red": 18,
            "blue_minus_red": 31,
            "context_dark_panel_fraction": 0.30,
        }
        self.assertFalse(yolo_detector._passes_strict_water_supplement_candidate(solar_panel, 42.0, "fast"))

    def test_rejected_candidate_capture_is_quarantined_without_training_label(self):
        class FakeArray:
            ndim = 3
            shape = (64, 64, 3)

            def __getitem__(self, _item):
                return self

        class FakeNumpy:
            @staticmethod
            def asarray(value):
                return value

        class FakeCV2:
            @staticmethod
            def imwrite(path, _image):
                Path(path).write_bytes(b"review crop")
                return True

        image = FakeArray()
        candidate = {
            "x": 16,
            "y": 18,
            "w": 20,
            "h": 14,
            "area_px": 280,
            "confidence": 0.61,
            "water_fraction": 0.14,
        }
        with tempfile.TemporaryDirectory() as temp_dir, \
             patch.object(yolo_detector, "_SAVE_REJECTED_POOL_CROPS", True), \
             patch.object(yolo_detector, "_HARD_NEGATIVE_DIR", temp_dir), \
             patch.object(yolo_detector, "_MAX_NEGATIVE_CROPS_PER_RUN", 5), \
             patch.object(yolo_detector, "_negative_crops_saved_this_run", 0), \
             patch.object(yolo_detector, "_np", FakeNumpy()), \
             patch.object(yolo_detector, "_HAS_CV2", True), \
             patch.object(yolo_detector, "_cv2", FakeCV2()):
            saved = yolo_detector._save_rejected_candidate_crop(
                image,
                candidate,
                reason="yolo_not_watery",
                chunk_index=3,
                tile_zoom=20,
            )
            root = Path(temp_dir)
            images = list(root.rglob("*.jpg"))
            metadata_files = list(root.rglob("*.json"))

            self.assertTrue(saved)
            self.assertEqual(len(images), 1)
            self.assertEqual(len(metadata_files), 1)
            self.assertEqual(list(root.rglob("*.txt")), [])
            self.assertIn("review_queue", images[0].parts)
            metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
            self.assertEqual(metadata["review_status"], "unreviewed")
            self.assertFalse(metadata["eligible_for_training"])
            self.assertFalse(metadata["training_label_created"])
            self.assertEqual(metadata["reason"], "yolo_not_watery")
            self.assertEqual(metadata["candidate"]["water_fraction"], 0.14)

    def test_stitch_tiles_returns_none_when_every_tile_fails(self):
        yolo_detector._lazy_import()
        if yolo_detector._np is None:
            self.skipTest("NumPy is required for tile stitching")

        with patch.object(yolo_detector, "_download_tile", return_value=None):
            result = yolo_detector._stitch_tiles_with_stats(
                (-33.875, 151.272, -33.874, 151.273),
                zoom=18,
            )

        self.assertIsNone(result)

    def test_detect_pools_keeps_confident_normal_pool_candidates_across_chunks(self):
        request_bbox = (-33.875, 151.27, -33.865, 151.29)
        chunks = [
            (-33.875, 151.27, -33.87, 151.28),
            (-33.87, 151.28, -33.865, 151.29),
        ]
        pool_candidate = {
            "cx": 128,
            "cy": 128,
            "x": 104,
            "y": 108,
            "w": 48,
            "h": 40,
            "area_px": 900,
            "confidence": 0.9,
            "circularity": 0.72,
            "aspect_ratio": 1.2,
            "water_fraction": 0.75,
            "extent": 0.7,
            "rotated_extent": 0.74,
            "value_mean": 145,
            "saturation_mean": 60,
            "context_roof_fraction": 0.05,
            "context_dark_panel_fraction": 0.03,
        }

        def fake_stitch(chunk_bbox, zoom):
            return FakeImage(), chunk_bbox, {"attempted_tiles": 4, "loaded_tiles": 4}

        # The colour-CV path is an opt-in experimental fallback now, so enable
        # it explicitly and ensure no pool-trained weights short-circuit it.
        with patch.object(yolo_detector, "_lazy_import", return_value=None), \
             patch.object(yolo_detector, "_np", object()), \
             patch.object(yolo_detector, "_Image", object()), \
             patch.object(yolo_detector, "_HAS_CV2", False), \
             patch.object(yolo_detector, "_HAS_YOLO", False), \
             patch.object(yolo_detector, "_ENABLE_CV_POOL_FALLBACK", True), \
             patch.object(yolo_detector, "pool_yolo_available", return_value=False), \
             patch.object(yolo_detector, "_choose_tile_zoom", return_value=18), \
             patch.object(yolo_detector, "_iter_detection_bboxes", return_value=chunks), \
             patch.object(yolo_detector, "_planned_chunk_count", return_value=2), \
             patch.object(yolo_detector, "_stitch_tiles_with_stats", side_effect=fake_stitch), \
             patch.object(yolo_detector, "_detect_blue_pools_cv", return_value=[pool_candidate]), \
             patch.object(yolo_detector, "_estimate_pool_area_sqm", side_effect=[18.0, 120.0]):
            elements = yolo_detector.detect_pools_in_bbox(request_bbox, zoom=18, detector_precision="precise")

        self.assertEqual(len(elements), 2)
        # Order is not guaranteed (dedup ranks by confidence), so compare as a set.
        areas = sorted(float(el["tags"]["pool:area_sqm"]) for el in elements)
        self.assertEqual(areas, [18.0, 120.0])
        diagnostics = yolo_detector.get_last_detector_diagnostics()
        self.assertEqual(diagnostics["chunks_scanned"], 2)
        self.assertEqual(diagnostics["tiles_loaded"], 8)
        self.assertEqual(diagnostics["reason"], "pools_detected")

    def test_detect_pools_returns_empty_when_no_pool_model_configured(self):
        """Default behaviour: no pool weights + CV fallback off => no boxes."""
        with patch.object(yolo_detector, "_lazy_import", return_value=None), \
             patch.object(yolo_detector, "_np", object()), \
             patch.object(yolo_detector, "_Image", object()), \
             patch.object(yolo_detector, "_ENABLE_CV_POOL_FALLBACK", False), \
             patch.object(yolo_detector, "pool_yolo_available", return_value=False), \
             patch.object(yolo_detector, "_choose_tile_zoom", return_value=18), \
             patch.object(yolo_detector, "_iter_detection_bboxes",
                          return_value=[(-33.875, 151.27, -33.87, 151.28)]), \
             patch.object(yolo_detector, "_detect_blue_pools_cv") as mock_cv:
            elements = yolo_detector.detect_pools_in_bbox(
                (-33.875, 151.27, -33.865, 151.29), zoom=18, detector_precision="precise"
            )

        self.assertEqual(elements, [])
        mock_cv.assert_not_called()
        diagnostics = yolo_detector.get_last_detector_diagnostics()
        self.assertEqual(diagnostics["reason"], "no_pool_model_configured")

    def test_inference_failure_is_reported_as_runtime_error_not_zero_detections(self):
        request_bbox = (-33.875, 151.27, -33.865, 151.29)

        def fail_inference(image, **kwargs):
            yolo_detector._update_scan_inference_diagnostics(
                kwargs.get("runtime_diagnostics"),
                "primary",
                success=False,
                error=AttributeError("'Conv' object has no attribute 'bn'"),
            )
            return []

        with patch.object(yolo_detector, "_lazy_import", return_value=None), \
             patch.object(yolo_detector, "_np", object()), \
             patch.object(yolo_detector, "_Image", object()), \
             patch.object(yolo_detector, "pool_yolo_available", return_value=True), \
             patch.object(yolo_detector, "_choose_tile_zoom", return_value=20), \
             patch.object(yolo_detector, "_iter_detection_bboxes", return_value=[request_bbox]), \
             patch.object(yolo_detector, "_planned_chunk_count", return_value=1), \
             patch.object(yolo_detector, "_stitch_tiles_with_stats", return_value=(FakeImage(), request_bbox, {"attempted_tiles": 1, "loaded_tiles": 1})), \
             patch.object(yolo_detector, "_STRICT_WATER_SUPPLEMENT_MODE", "0"), \
             patch.object(yolo_detector, "_yolo_detect_pools", side_effect=fail_inference):
            elements = yolo_detector.detect_pools_in_bbox(
                request_bbox,
                zoom=20,
                detector_precision="precise",
            )

        self.assertEqual(elements, [])
        diagnostics = yolo_detector.get_last_detector_diagnostics()
        self.assertEqual(diagnostics["reason"], "detector_runtime_error")
        self.assertEqual(diagnostics["inference_failures"], 1)
        self.assertIn("Conv", diagnostics["primary_inference_error"])

    def test_detect_pools_uses_pool_trained_yolo_when_available(self):
        """A pool-trained YOLO box flows through as a trusted detection."""
        request_bbox = (-33.875, 151.27, -33.865, 151.29)
        yolo_box = {
            "cx": 128, "cy": 128, "x": 104, "y": 108, "w": 48, "h": 40,
            "area_px": 1920, "confidence": 0.62, "circularity": 0.0,
            "aspect_ratio": 1.2, "detector": "yolo",
        }

        def fake_stitch(chunk_bbox, zoom):
            return FakeImage(), chunk_bbox, {"attempted_tiles": 4, "loaded_tiles": 4}

        with patch.object(yolo_detector, "_lazy_import", return_value=None), \
             patch.object(yolo_detector, "_np", object()), \
             patch.object(yolo_detector, "_Image", object()), \
             patch.object(yolo_detector, "pool_yolo_available", return_value=True), \
             patch.object(yolo_detector, "_choose_tile_zoom", return_value=18), \
             patch.object(yolo_detector, "_iter_detection_bboxes",
                          return_value=[(-33.875, 151.27, -33.87, 151.28)]), \
             patch.object(yolo_detector, "_stitch_tiles_with_stats", side_effect=fake_stitch), \
             patch.object(yolo_detector, "_yolo_detect_pools", return_value=[yolo_box]), \
             patch.object(yolo_detector, "_detect_blue_pools_cv", return_value=[]) as mock_cv, \
             patch.object(yolo_detector, "_estimate_pool_area_sqm", return_value=45.0):
            elements = yolo_detector.detect_pools_in_bbox(
                request_bbox, zoom=18, detector_precision="auto"
            )

        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0]["tags"]["detector:source"], "satellite-yolo")
        self.assertEqual(elements[0]["tags"]["detector:method"], "pool-trained-yolo")
        # Auto mode keeps the broad colour-CV path disabled when YOLO is active;
        # realtime fast and manual precise scans add the strict supplement.
        mock_cv.assert_not_called()

    def test_strict_clear_water_supplement_recovers_obvious_pool_when_yolo_misses(self):
        request_bbox = (-33.875, 151.27, -33.865, 151.29)
        strict_pool = {
            "cx": 128, "cy": 128, "x": 104, "y": 108, "w": 48, "h": 40,
            "area_px": 1920, "confidence": 0.82, "circularity": 0.62,
            "aspect_ratio": 1.2, "water_fraction": 0.94,
            "extent": 0.62, "rotated_extent": 0.68,
            "value_mean": 150, "saturation_mean": 64,
            "green_minus_red": 36, "blue_minus_red": 42,
            "context_roof_fraction": 0.04, "context_dark_panel_fraction": 0.02,
        }

        def fake_stitch(chunk_bbox, zoom):
            return FakeImage(), chunk_bbox, {"attempted_tiles": 4, "loaded_tiles": 4}

        with patch.object(yolo_detector, "_lazy_import", return_value=None), \
             patch.object(yolo_detector, "_np", object()), \
             patch.object(yolo_detector, "_Image", object()), \
             patch.object(yolo_detector, "pool_yolo_available", return_value=True), \
             patch.object(yolo_detector, "_choose_tile_zoom", return_value=20), \
             patch.object(yolo_detector, "_iter_detection_bboxes",
                          return_value=[(-33.875, 151.27, -33.87, 151.28)]), \
             patch.object(yolo_detector, "_stitch_tiles_with_stats", side_effect=fake_stitch), \
             patch.object(yolo_detector, "_yolo_detect_pools", return_value=[]), \
             patch.object(yolo_detector, "resolve_pool_yolo_weights_b", return_value=None), \
             patch.object(yolo_detector, "_detect_blue_pools_cv", return_value=[strict_pool]), \
             patch.object(yolo_detector, "_estimate_pool_area_sqm", return_value=42.0):
            elements = yolo_detector.detect_pools_in_bbox(
                request_bbox, zoom=20, detector_precision="fast"
            )

        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0]["tags"]["detector:source"], "satellite-cv")
        self.assertEqual(elements[0]["tags"]["detector:method"], "strict-clear-water-cv")
        self.assertEqual(elements[0]["tags"]["detector:version"], "strict-clear-water-v1")

    def test_strict_water_supplement_auto_disabled_when_ensemble_active(self):
        # With a second ensemble model resolved, auto mode must skip the CV
        # supplement: it contributed 28 of 42 eval false positives (mostly
        # ocean water) for only 10 unique true positives once the ensemble
        # covered the bright pools it was built for (2026-07-10 ablation).
        request_bbox = (-33.875, 151.27, -33.865, 151.29)

        def fake_stitch(chunk_bbox, zoom):
            return FakeImage(), chunk_bbox, {"attempted_tiles": 4, "loaded_tiles": 4}

        with patch.object(yolo_detector, "_lazy_import", return_value=None), \
             patch.object(yolo_detector, "_np", object()), \
             patch.object(yolo_detector, "_Image", object()), \
             patch.object(yolo_detector, "pool_yolo_available", return_value=True), \
             patch.object(yolo_detector, "_choose_tile_zoom", return_value=20), \
             patch.object(yolo_detector, "_iter_detection_bboxes",
                          return_value=[(-33.875, 151.27, -33.87, 151.28)]), \
             patch.object(yolo_detector, "_stitch_tiles_with_stats", side_effect=fake_stitch), \
             patch.object(yolo_detector, "_yolo_detect_pools", return_value=[]), \
             patch.object(yolo_detector, "resolve_pool_yolo_weights_b", return_value="models/pool_yolo_b.pt"), \
             patch.object(yolo_detector, "_detect_blue_pools_cv") as mock_cv:
            yolo_detector.detect_pools_in_bbox(
                request_bbox, zoom=20, detector_precision="precise"
            )

        mock_cv.assert_not_called()

    def test_merge_secondary_pool_dets_unions_and_dedupes(self):
        # Ensemble merge: secondary boxes covering a primary pool are dropped;
        # genuinely new secondary boxes are added; solar dets never block adds.
        from aus_backyard_intelligence.yolo_detector import _merge_secondary_pool_dets

        primary = [
            {"cx": 100.0, "cy": 100.0, "w": 20.0, "h": 20.0, "detector": "yolo"},
            {"cx": 300.0, "cy": 300.0, "w": 40.0, "h": 40.0, "detector": "yolo_solar"},
        ]
        secondary = [
            # duplicate of the primary pool (centre within half box side)
            {"cx": 104.0, "cy": 98.0, "w": 18.0, "h": 18.0, "detector": "yolo", "model": "b"},
            # new pool the primary missed
            {"cx": 500.0, "cy": 500.0, "w": 22.0, "h": 22.0, "detector": "yolo", "model": "b"},
            # overlaps only the SOLAR det — solar must not block a pool add
            {"cx": 302.0, "cy": 301.0, "w": 20.0, "h": 20.0, "detector": "yolo", "model": "b"},
        ]
        merged = _merge_secondary_pool_dets(primary, secondary)
        centres = sorted((d["cx"], d["cy"]) for d in merged)
        self.assertEqual(
            centres,
            [(100.0, 100.0), (300.0, 300.0), (302.0, 301.0), (500.0, 500.0)],
        )

    def test_solar_prediction_does_not_hard_veto_ensemble_pool(self):
        """An unvalidated solar box must not delete a water-verified pool proposal."""
        primary = object()
        secondary = object()
        solar = {
            "cx": 100.0, "cy": 100.0, "x": 80.0, "y": 80.0,
            "w": 40.0, "h": 40.0, "detector": "yolo_solar",
        }
        pool = {
            "cx": 100.0, "cy": 100.0, "x": 84.0, "y": 84.0,
            "w": 32.0, "h": 32.0, "detector": "yolo",
        }
        diagnostics = {}

        with patch.object(yolo_detector, "_get_pool_yolo_model", return_value=primary), \
             patch.object(yolo_detector, "_get_pool_yolo_model_b", return_value=secondary), \
             patch.object(yolo_detector, "_predict_pool_yolo_model", side_effect=[["primary"], ["secondary"]]), \
             patch.object(
                 yolo_detector,
                 "_parse_pool_yolo_results",
                 side_effect=[([solar], [(80.0, 80.0, 120.0, 120.0)], ["solarpanel", "pool"]),
                              ([pool], [], ["solarpanel", "pool"])],
             ):
            detections = yolo_detector._yolo_detect_pools(
                object(), conf=0.15, imgsz=1024, runtime_diagnostics=diagnostics
            )

        self.assertIn(pool, detections)
        self.assertIn(solar, detections)
        self.assertEqual(diagnostics["solar_pool_overlap_candidates"], 1)
        self.assertEqual(diagnostics["pool_candidates_after_ensemble"], 1)

    def test_solar_detection_emitted_as_generator_element(self):
        request_bbox = (-33.875, 151.27, -33.865, 151.29)
        solar_det = {
            "cx": 128, "cy": 128, "x": 100, "y": 100, "w": 56, "h": 56,
            "area_px": 3136, "confidence": 0.81, "circularity": 0.0,
            "aspect_ratio": 1.0, "detector": "yolo_solar",
        }

        def fake_stitch(chunk_bbox, zoom):
            return FakeImage(), chunk_bbox, {"attempted_tiles": 4, "loaded_tiles": 4}

        with patch.object(yolo_detector, "_lazy_import", return_value=None), \
             patch.object(yolo_detector, "_np", object()), \
             patch.object(yolo_detector, "_Image", object()), \
             patch.object(yolo_detector, "pool_yolo_available", return_value=True), \
             patch.object(yolo_detector, "_choose_tile_zoom", return_value=20), \
             patch.object(yolo_detector, "_iter_detection_bboxes",
                          return_value=[(-33.875, 151.27, -33.87, 151.28)]), \
             patch.object(yolo_detector, "_stitch_tiles_with_stats", side_effect=fake_stitch), \
             patch.object(yolo_detector, "_yolo_detect_pools", return_value=[solar_det]), \
             patch.object(yolo_detector, "_detect_blue_pools_cv", return_value=[]), \
             patch.object(yolo_detector, "_estimate_pool_area_sqm", return_value=30.0):
            elements = yolo_detector.detect_pools_in_bbox(
                request_bbox, zoom=20, detector_precision="precise"
            )

        solar = [e for e in elements if e["tags"].get("generator:source") == "solar"]
        self.assertEqual(len(solar), 1)
        self.assertEqual(solar[0]["tags"]["detector:source"], "satellite-yolo-solar")
        self.assertEqual(solar[0]["tags"]["power"], "generator")
        self.assertEqual(solar[0]["tags"]["solar:area_sqm"], "30.0")
        self.assertEqual(len(solar[0]["geometry"]), 5)
        diagnostics = yolo_detector.get_last_detector_diagnostics()
        self.assertEqual(diagnostics.get("solar_elements"), 1)
        self.assertEqual(diagnostics.get("reason"), "solar_detected")
        # Solar arrays never count as (or displace) pool elements.
        self.assertEqual(diagnostics.get("pool_elements"), 0)

    def test_is_large_clear_water_pool_gates_on_strong_water_signature(self):
        big_clear = {
            "water_fraction": 0.92, "strong_water_fraction": 0.55, "value_mean": 160,
            "saturation_mean": 80, "green_minus_red": 22, "blue_minus_red": 60,
        }
        # A large, bright, strongly-blue basin is accepted above the normal cap.
        self.assertTrue(yolo_detector._is_large_clear_water_pool(big_clear, 1250.0))
        # A large but weakly-blue / low-water region (roof, court, field) is not.
        weak = dict(big_clear, blue_minus_red=12, water_fraction=0.30, strong_water_fraction=0.05)
        self.assertFalse(yolo_detector._is_large_clear_water_pool(weak, 1250.0))
        # The bypass only applies to genuinely large basins (bigger than any backyard
        # pool), with a sane upper bound.
        self.assertFalse(yolo_detector._is_large_clear_water_pool(big_clear, 150.0))
        self.assertFalse(yolo_detector._is_large_clear_water_pool(big_clear, 9000.0))

    def test_large_water_gate_rejects_known_feedback_false_positive(self):
        known_false_positive = {
            "water_fraction": 0.967,
            "value_mean": 97.0,
            "saturation_mean": 85.2,
            "green_minus_red": 17.5,
            "blue_minus_red": 31.2,
        }
        self.assertFalse(
            yolo_detector._is_large_clear_water_pool(known_false_positive, 356.6)
        )

        # Preserve the measured signatures of confirmed large pools.
        for green_minus_red, blue_minus_red, area_sqm in (
            (73.3, 85.3, 679.4),
            (65.5, 70.7, 1065.6),
        ):
            confirmed_pool = {
                "water_fraction": 0.95,
                "value_mean": 130.0,
                "saturation_mean": 70.0,
                "green_minus_red": green_minus_red,
                "blue_minus_red": blue_minus_red,
            }
            self.assertTrue(
                yolo_detector._is_large_clear_water_pool(confirmed_pool, area_sqm)
            )

    def test_strict_water_supplement_recovers_large_public_pool_over_size_cap(self):
        request_bbox = (-33.875, 151.27, -33.865, 151.29)
        # An Olympic-size pool: ~1250 m² (far over the 220 m² fast cap), touching the
        # chunk edge, with an unmistakable strong clear-water signature.
        big_pool = {
            "cx": 128, "cy": 128, "x": 2, "y": 40, "w": 240, "h": 150,
            "area_px": 80000, "confidence": 0.90, "circularity": 0.55,
            "aspect_ratio": 1.9, "water_fraction": 0.90,
            "extent": 0.70, "rotated_extent": 0.72,
            "value_mean": 160, "saturation_mean": 78,
            "green_minus_red": 22, "blue_minus_red": 60,
            "strong_water_fraction": 0.50,
            "context_roof_fraction": 0.02, "context_dark_panel_fraction": 0.01,
        }

        def fake_stitch(chunk_bbox, zoom):
            return FakeImage(), chunk_bbox, {"attempted_tiles": 4, "loaded_tiles": 4}

        with patch.object(yolo_detector, "_lazy_import", return_value=None), \
             patch.object(yolo_detector, "_np", object()), \
             patch.object(yolo_detector, "_Image", object()), \
             patch.object(yolo_detector, "pool_yolo_available", return_value=True), \
             patch.object(yolo_detector, "_choose_tile_zoom", return_value=20), \
             patch.object(yolo_detector, "_iter_detection_bboxes",
                          return_value=[(-33.875, 151.27, -33.87, 151.28)]), \
             patch.object(yolo_detector, "_stitch_tiles_with_stats", side_effect=fake_stitch), \
             patch.object(yolo_detector, "_yolo_detect_pools", return_value=[]), \
             patch.object(yolo_detector, "resolve_pool_yolo_weights_b", return_value=None), \
             patch.object(yolo_detector, "_detect_blue_pools_cv", return_value=[big_pool]), \
             patch.object(yolo_detector, "_estimate_pool_area_sqm", return_value=1250.0):
            elements = yolo_detector.detect_pools_in_bbox(
                request_bbox, zoom=20, detector_precision="fast"
            )

        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0]["tags"]["detector:source"], "satellite-cv")


if __name__ == "__main__":
    unittest.main()
