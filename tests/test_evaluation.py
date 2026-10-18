import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from wsgiref.util import setup_testing_defaults


def _fresh_eval_module(tmpdir):
    """Import evaluation with its storage redirected into a temp dir."""
    os.environ["QLDR_EVAL_DIR"] = str(tmpdir)
    import importlib

    from aus_backyard_intelligence import evaluation

    importlib.reload(evaluation)
    return evaluation


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


class EvaluationHarnessTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.evaluation = _fresh_eval_module(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("QLDR_EVAL_DIR", None)

    def test_upsert_label_dedupes_and_feedback_overrides(self):
        labels = []
        ev = self.evaluation
        self.assertTrue(ev.upsert_label(labels, lat=-33.87, lon=151.28, label="pool", source="osm-bootstrap"))
        # Same spot again -> no duplicate.
        self.assertFalse(ev.upsert_label(labels, lat=-33.870001, lon=151.280001, label="pool", source="osm-bootstrap"))
        self.assertEqual(len(labels), 1)
        # Feedback at the same spot overrides the bootstrapped label.
        self.assertTrue(ev.upsert_label(labels, lat=-33.87, lon=151.28, label="no_pool", source="feedback"))
        self.assertEqual(labels[0]["label"], "no_pool")
        self.assertEqual(labels[0]["source"], "feedback")

    def test_holdout_labels_are_explicit_and_cannot_be_overwritten_by_feedback(self):
        ev = self.evaluation
        labels = []
        self.assertTrue(ev.upsert_label(
            labels,
            lat=-31.95,
            lon=115.86,
            label="pool",
            source="human-holdout",
            area="Perth WA",
            split="holdout",
        ))
        self.assertFalse(ev.upsert_label(
            labels,
            lat=-31.95,
            lon=115.86,
            label="no_pool",
            source="feedback",
        ))
        self.assertEqual(labels[0]["label"], "pool")
        self.assertEqual(labels[0]["split"], "holdout")
        with self.assertRaises(ValueError):
            ev.upsert_label(
                [], lat=-31.95, lon=115.86, label="pool",
                source="feedback", area="Perth WA", split="holdout",
            )

    def test_dataset_profile_and_marketing_audit_expose_bias(self):
        ev = self.evaluation
        labels = [
            {"lat": -33.87, "lon": 151.28, "label": "pool", "source": "osm-bootstrap", "area": "pitch-bench"},
            {"lat": -33.88, "lon": 151.29, "label": "no_pool", "source": "feedback", "area": "Sydney NSW"},
            {"lat": -33.89, "lon": 151.30, "label": "solar", "source": "osm-bootstrap", "area": ""},
        ]
        profile = ev.dataset_profile(labels)
        self.assertEqual(profile["by_source"]["osm-bootstrap"], 2)
        self.assertEqual(profile["by_area"]["pitch-bench"], 1)
        self.assertEqual(profile["by_split"]["development"], 3)
        audit = ev._marketing_audit(labels, evaluation_split="development")
        self.assertFalse(audit["pool_national_claim_eligible"])
        self.assertFalse(audit["solar_claim_eligible"])
        self.assertTrue(any("pitch-bench" in blocker for blocker in audit["blockers"]))

    def test_marketing_audit_requires_each_pool_class_per_metro(self):
        ev = self.evaluation

        def holdout(area, label, count):
            return [
                {
                    "lat": -31.0,
                    "lon": 120.0,
                    "label": label,
                    "source": "human-holdout",
                    "area": area,
                    "split": "holdout",
                }
                for _ in range(count)
            ]

        labels = []
        labels += holdout("Perth WA", "pool", 2) + holdout("Perth WA", "no_pool", 2)
        labels += holdout("Darwin NT", "pool", 2) + holdout("Darwin NT", "no_pool", 2)
        # The combined total meets the old gate, but this metro lacks one negative.
        labels += holdout("Adelaide SA", "pool", 3) + holdout("Adelaide SA", "no_pool", 1)

        with (
            patch.object(ev, "MIN_HOLDOUT_METROS", 3),
            patch.object(ev, "MIN_HOLDOUT_LABELS_PER_METRO", 2),
        ):
            audit = ev._marketing_audit(labels, evaluation_split="holdout")
            self.assertFalse(audit["pool_national_claim_eligible"])
            self.assertEqual(audit["qualified_metros"], ["Darwin NT", "Perth WA"])
            self.assertTrue(any("2+ pool and 2+ no-pool" in blocker for blocker in audit["blockers"]))

            labels += holdout("Adelaide SA", "no_pool", 1)
            audit = ev._marketing_audit(labels, evaluation_split="holdout")
            self.assertTrue(audit["pool_national_claim_eligible"])
            self.assertEqual(audit["qualified_metros"], ["Adelaide SA", "Darwin NT", "Perth WA"])

    def test_marketing_audit_requires_solar_classes_in_every_qualified_metro(self):
        ev = self.evaluation

        def holdout(area, label, count=2):
            return [
                {
                    "lat": -31.0,
                    "lon": 120.0,
                    "label": label,
                    "source": "human-holdout",
                    "area": area,
                    "split": "holdout",
                }
                for _ in range(count)
            ]

        metros = ["Adelaide SA", "Darwin NT", "Perth WA"]
        labels = []
        for metro in metros:
            labels += holdout(metro, "pool") + holdout(metro, "no_pool")
        # Aggregate solar minima pass, but only one of the three claimed metros is covered.
        labels += holdout("Perth WA", "solar") + holdout("Perth WA", "no_solar")

        with (
            patch.object(ev, "MIN_HOLDOUT_METROS", 3),
            patch.object(ev, "MIN_HOLDOUT_LABELS_PER_METRO", 2),
            patch.object(ev, "MIN_SOLAR_POSITIVES", 2),
            patch.object(ev, "MIN_SOLAR_NEGATIVES", 2),
        ):
            audit = ev._marketing_audit(labels, evaluation_split="holdout")
            self.assertTrue(audit["pool_national_claim_eligible"])
            self.assertFalse(audit["solar_claim_eligible"])
            coverage_blocker = next(
                blocker for blocker in audit["solar_blockers"] if "Every qualified metro" in blocker
            )
            self.assertIn("Adelaide SA (0 solar, 0 no-solar)", coverage_blocker)
            self.assertIn("Darwin NT (0 solar, 0 no-solar)", coverage_blocker)

            for metro in ["Adelaide SA", "Darwin NT"]:
                labels += holdout(metro, "solar") + holdout(metro, "no_solar")
            audit = ev._marketing_audit(labels, evaluation_split="holdout")
            self.assertTrue(audit["solar_claim_eligible"])
            self.assertEqual(audit["solar_blockers"], [])

    def test_import_holdout_labels_requires_area_and_records_split(self):
        ev = self.evaluation
        result = ev.import_holdout_labels([
            {"lat": -31.95, "lon": 115.86, "label": "pool", "area": "Perth WA"},
            {"lat": -31.96, "lon": 115.87, "label": "no_solar", "area": "Perth WA"},
        ])
        self.assertEqual(result["added"], 2)
        labels = ev.load_labels()
        self.assertTrue(all(label["source"] == "human-holdout" for label in labels))
        self.assertTrue(all(label["split"] == "holdout" for label in labels))
        with self.assertRaises(ValueError):
            ev.import_holdout_labels([
                {"lat": -31.95, "lon": 115.86, "label": "pool", "area": "pitch-bench"}
            ])

    def test_bootstrap_labels_positive_and_clear_negative(self):
        ev = self.evaluation
        pool = {
            "type": "way", "id": 1, "tags": {"leisure": "swimming_pool"},
            "geometry": [
                {"lat": -33.8700, "lon": 151.2800}, {"lat": -33.8700, "lon": 151.2801},
                {"lat": -33.8699, "lon": 151.2801}, {"lat": -33.8699, "lon": 151.2800},
            ],
        }
        # House right next to the pool (within clearance) must NOT become a negative...
        near_house = {
            "type": "way", "id": 2, "tags": {"building": "house"},
            "geometry": [
                {"lat": -33.87005, "lon": 151.28005}, {"lat": -33.87005, "lon": 151.28015},
                {"lat": -33.86995, "lon": 151.28015}, {"lat": -33.86995, "lon": 151.28005},
            ],
        }
        # ...but a house well away becomes a no_pool label.
        far_house = {
            "type": "way", "id": 3, "tags": {"building": "house"},
            "geometry": [
                {"lat": -33.8730, "lon": 151.2840}, {"lat": -33.8730, "lon": 151.2841},
                {"lat": -33.8729, "lon": 151.2841}, {"lat": -33.8729, "lon": 151.2840},
            ],
        }
        with patch.object(ev, "execute_overpass_query", return_value={"elements": [pool, near_house, far_house]}):
            result = ev.bootstrap_labels((-33.88, 151.27, -33.86, 151.29), area="test")
        self.assertEqual(result["added_pool_labels"], 1)
        self.assertEqual(result["added_no_pool_labels"], 1)
        labels = ev.load_labels()
        kinds = sorted(label["label"] for label in labels)
        self.assertEqual(kinds, ["no_pool", "pool"])

    def test_satellite_detection_counts_stamped_mapped_pool(self):
        # A mapped pool stamped detector:satellite by the merge dedupe counts as a
        # satellite detection, so eval recall stays independent of Overpass health.
        ev = self.evaluation
        stamped = {
            "tags": {
                "leisure": "swimming_pool",
                "detector:source": "public-mapped",
                "detector:satellite": "satellite-yolo",
            }
        }
        plain_mapped = {"tags": {"leisure": "swimming_pool", "detector:source": "public-mapped"}}
        solar = {"tags": {"detector:source": "satellite-yolo-solar"}}
        self.assertTrue(ev._is_satellite_detection(stamped))
        self.assertFalse(ev._is_satellite_detection(plain_mapped))
        self.assertFalse(ev._is_satellite_detection(solar))

    def test_label_cells_bbox_contains_southern_hemisphere_label(self):
        # Regression: int() truncation binned negative latitudes one cell north,
        # so the eval scanned bboxes that did not contain their own labels.
        ev = self.evaluation
        label = {"lat": -33.878, "lon": 151.282, "label": "pool"}
        ((cy, cx), members), = ev._label_cells([label]).items()
        self.assertIn(label, members)
        self.assertLessEqual(cy * ev.EVAL_CELL_DEG, label["lat"])
        self.assertGreaterEqual((cy + 1) * ev.EVAL_CELL_DEG, label["lat"])
        self.assertLessEqual(cx * ev.EVAL_CELL_DEG, label["lon"])
        self.assertGreaterEqual((cx + 1) * ev.EVAL_CELL_DEG, label["lon"])

    def test_run_evaluation_scores_tp_fn_fp(self):
        ev = self.evaluation
        labels = []
        ev.upsert_label(labels, lat=-33.8700, lon=151.2800, label="pool", source="osm-bootstrap")     # found
        ev.upsert_label(labels, lat=-33.8705, lon=151.2810, label="pool", source="osm-bootstrap")     # missed
        ev.upsert_label(labels, lat=-33.8702, lon=151.2805, label="no_pool", source="osm-bootstrap")  # violated
        ev.upsert_label(labels, lat=-33.8708, lon=151.2815, label="no_pool", source="osm-bootstrap")  # clean
        ev.upsert_label(labels, lat=-33.8703, lon=151.2820, label="solar", source="osm-bootstrap")    # solar found
        ev._save_labels(labels)

        detections = {
            "elements": [
                {  # matches pool label 1 (within 15 m)
                    "type": "way", "id": 10,
                    "tags": {"leisure": "swimming_pool", "detector:source": "satellite-yolo"},
                    "geometry": [{"lat": -33.87001, "lon": 151.28001}, {"lat": -33.87002, "lon": 151.28003},
                                 {"lat": -33.87003, "lon": 151.28001}],
                },
                {  # violates the first no_pool label (within 20 m)
                    "type": "way", "id": 11,
                    "tags": {"leisure": "swimming_pool", "detector:source": "satellite-cv"},
                    "geometry": [{"lat": -33.87021, "lon": 151.28051}, {"lat": -33.87022, "lon": 151.28053},
                                 {"lat": -33.87023, "lon": 151.28051}],
                },
                {  # mapped OSM pool must be IGNORED by the evaluation
                    "type": "way", "id": 12,
                    "tags": {"leisure": "swimming_pool"},
                    "geometry": [{"lat": -33.87050, "lon": 151.28100}, {"lat": -33.87051, "lon": 151.28102},
                                 {"lat": -33.87052, "lon": 151.28100}],
                },
                {  # solar det right next to the MISSED pool label — must NOT rescue it
                    "type": "way", "id": 13,
                    "tags": {"detector:source": "satellite-yolo-solar", "power": "generator"},
                    "geometry": [{"lat": -33.87051, "lon": 151.28101}, {"lat": -33.87052, "lon": 151.28103},
                                 {"lat": -33.87053, "lon": 151.28101}],
                },
                {  # solar det matching the solar label
                    "type": "way", "id": 14,
                    "tags": {"detector:source": "satellite-yolo-solar", "power": "generator"},
                    "geometry": [{"lat": -33.87031, "lon": 151.28201}, {"lat": -33.87032, "lon": 151.28203},
                                 {"lat": -33.87033, "lon": 151.28201}],
                },
            ]
        }
        with patch.object(ev, "fetch_osm_features", return_value=detections):
            report = ev.run_evaluation(data_path="sample.json", build_tag="test-build")

        self.assertEqual(report["tp"], 1)
        self.assertEqual(report["fn"], 1)  # solar det near the missed pool did NOT rescue it
        self.assertEqual(report["fp"], 1)
        self.assertEqual(report["tn"], 1)
        self.assertEqual(report["recall"], 0.5)
        self.assertEqual(report["solar_labels"], 1)
        self.assertEqual(report["solar_tp"], 1)
        self.assertEqual(report["solar_recall"], 1.0)
        self.assertEqual(report["solar_detections_seen"], 2)
        self.assertEqual(report["fp_rate"], 0.5)
        self.assertEqual(report["precision_proxy"], 0.5)
        self.assertEqual(report["build"], "test-build")
        self.assertEqual(report["evaluation_split"], "development")
        self.assertEqual(report["recall_ci_95"]["n"], 2)
        self.assertIn("(blank)", report["per_area"])
        self.assertFalse(report["marketing_audit"]["pool_national_claim_eligible"])
        # Report persisted for the dashboard.
        self.assertIsNotNone(ev.load_report())

    def test_feedback_appends_and_mirrors_into_labels(self):
        ev = self.evaluation
        ev.add_feedback({"verdict": "wrong", "lat": -33.9, "lon": 151.2,
                         "tags": {"detector:source": "satellite-yolo", "irrelevant": "dropped"}})
        ev.add_feedback({"verdict": "correct", "lat": -33.91, "lon": 151.21, "tags": {}})
        stats = ev.feedback_stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["wrong"], 1)
        self.assertEqual(stats["correct"], 1)
        labels = ev.load_labels()
        by_label = {label["label"] for label in labels}
        self.assertEqual(by_label, {"pool", "no_pool"})
        self.assertTrue(all(label["source"] == "feedback" for label in labels))
        # Irrelevant tags are not persisted into the JSONL.
        export = ev.export_feedback()
        self.assertIn("detector:source", export)
        self.assertNotIn("irrelevant", export)
        with self.assertRaises(ValueError):
            ev.add_feedback({"verdict": "maybe", "lat": 0, "lon": 0})

    def test_training_candidates_harvest_and_pipeline_status(self):
        ev = self.evaluation
        elements = [
            {  # scan-verified pool -> harvested
                "type": "way", "id": 1,
                "tags": {"leisure": "swimming_pool", "detector:source": "satellite-yolo",
                         "scan:verdict": "confirmed", "scan:passes_seen": "3", "scan:confidence": "0.88"},
                "geometry": [{"lat": -27.50, "lon": 153.00}, {"lat": -27.5001, "lon": 153.0001},
                             {"lat": -27.5002, "lon": 153.00}],
            },
            {  # one-shot pool (no verdict) -> too noisy, skipped
                "type": "way", "id": 2,
                "tags": {"leisure": "swimming_pool", "detector:source": "satellite-yolo",
                         "confidence:yolo": "0.95"},
                "geometry": [{"lat": -27.51, "lon": 153.01}, {"lat": -27.5101, "lon": 153.0101},
                             {"lat": -27.5102, "lon": 153.01}],
            },
            {  # confident solar array -> harvested
                "type": "way", "id": 3,
                "tags": {"detector:source": "satellite-yolo-solar", "confidence:yolo": "0.82",
                         "power": "generator"},
                "geometry": [{"lat": -27.52, "lon": 153.02}, {"lat": -27.5201, "lon": 153.0201},
                             {"lat": -27.5202, "lon": 153.02}],
            },
        ]
        added = ev.record_training_candidates(elements, source="scan:test")
        self.assertEqual(added, {"pool": 1, "solar": 1})
        # Re-harvesting the same ground dedupes (8 m radius).
        again = ev.record_training_candidates(elements, source="scan:test2")
        self.assertEqual(again, {"pool": 0, "solar": 0})
        status = ev.pipeline_status()
        self.assertEqual(status["training_candidates"]["total"], 2)
        self.assertEqual(status["training_candidates"]["pool"], 1)
        self.assertEqual(status["training_candidates"]["solar"], 1)
        export = ev.export_training_candidates()
        self.assertIn("scan:test", export)
        self.assertNotIn("0.95", export)  # the one-shot pool stayed out

    def test_feedback_api_roundtrip(self):
        from aus_backyard_intelligence.app import create_app

        with tempfile.TemporaryDirectory() as data_dir:
            data_path = Path(data_dir) / "props.json"
            data_path.write_text("[]", encoding="utf-8")
            app = create_app(data_path=data_path)
            status, _, body = call_app(
                app, "/api/feedback/pool", method="POST",
                body={"verdict": "correct", "lat": -33.87, "lon": 151.28, "tags": {"detector:source": "satellite-yolo"}},
            )
            self.assertEqual(status, "200 OK")
            payload = json.loads(body)
            self.assertEqual(payload["stats"]["correct"], 1)
            status, _, body = call_app(app, "/api/eval/labels")
            self.assertEqual(status, "200 OK")
            self.assertEqual(json.loads(body)["counts"]["pool"], 1)

    def test_holdout_label_api_roundtrip(self):
        from aus_backyard_intelligence.app import create_app

        with tempfile.TemporaryDirectory() as data_dir:
            data_path = Path(data_dir) / "props.json"
            data_path.write_text("[]", encoding="utf-8")
            app = create_app(data_path=data_path)
            status, _, body = call_app(
                app,
                "/api/eval/holdout-labels",
                method="POST",
                body={"records": [
                    {"lat": -31.95, "lon": 115.86, "label": "pool", "area": "Perth WA"}
                ]},
            )
            self.assertEqual(status, "200 OK")
            self.assertEqual(json.loads(body)["counts"]["holdout"], 1)


class CadastreTestCase(unittest.TestCase):
    def test_qld_and_nsw_parsing(self):
        from aus_backyard_intelligence import cadastre

        os.environ["QLDR_ALLOW_CADASTRE_IN_TESTS"] = "1"
        try:
            cadastre._cache.clear()
            qld_rows = [{"lot": "597", "plan": "SP317627", "lotplan": "597SP317627"}]
            nsw_rows = [{"lotnumber": "1749", "sectionnumber": None, "planlabel": "DP728661"}]

            with patch.object(cadastre, "_point_query", return_value=qld_rows):
                parcel = cadastre.lookup_parcel(-27.4753, 153.0281)
            self.assertEqual(parcel["state"], "QLD")
            self.assertEqual(parcel["lotplan"], "597SP317627")

            cadastre._cache.clear()
            with patch.object(cadastre, "_point_query", return_value=nsw_rows):
                parcel = cadastre.lookup_parcel(-33.8920, 151.2745)
            self.assertEqual(parcel["state"], "NSW")
            self.assertEqual(parcel["lotplan"], "1749/-/DP728661")
            self.assertIn("DP728661", parcel["display"])

            # Cache: identical (rounded) point does not re-query.
            with patch.object(cadastre, "_point_query", side_effect=AssertionError("should be cached")):
                cached = cadastre.lookup_parcel(-33.8920, 151.2745)
            self.assertEqual(cached["lotplan"], "1749/-/DP728661")
        finally:
            os.environ.pop("QLDR_ALLOW_CADASTRE_IN_TESTS", None)
            cadastre._cache.clear()

    def test_disabled_under_pytest_by_default(self):
        from aus_backyard_intelligence import cadastre

        cadastre._cache.clear()
        with patch.object(cadastre, "_point_query", side_effect=AssertionError("must not hit network")):
            self.assertIsNone(cadastre.lookup_parcel(-27.47, 153.02))


if __name__ == "__main__":
    unittest.main()
