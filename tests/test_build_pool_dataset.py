import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_pool_dataset


class FakeImage:
    shape = (1024, 1024, 3)


class PoolDatasetBuilderTestCase(unittest.TestCase):
    def test_spatial_split_is_stable_for_identical_and_grouped_bboxes(self):
        first = (10.10, 20.10, 10.20, 20.20)
        nearby_same_group = (10.60, 20.60, 10.70, 20.70)

        first_split = build_pool_dataset._spatial_split_for_bbox(
            first,
            val_frac=0.35,
            seed=7,
            group_span_deg=1.0,
        )
        self.assertEqual(
            first_split,
            build_pool_dataset._spatial_split_for_bbox(
                first,
                val_frac=0.35,
                seed=7,
                group_span_deg=1.0,
            ),
        )
        self.assertEqual(
            first_split,
            build_pool_dataset._spatial_split_for_bbox(
                nearby_same_group,
                val_frac=0.35,
                seed=7,
                group_span_deg=1.0,
            ),
        )
        self.assertEqual(
            build_pool_dataset._spatial_split_for_bbox(
                first, val_frac=0.0, seed=7, group_span_deg=1.0
            ),
            "train",
        )
        self.assertEqual(
            build_pool_dataset._spatial_split_for_bbox(
                first, val_frac=1.0, seed=7, group_span_deg=1.0
            ),
            "val",
        )

    def test_known_feature_intersection_is_inclusive(self):
        pool = [
            (-27.51, 153.01),
            (-27.51, 153.02),
            (-27.50, 153.02),
            (-27.50, 153.01),
        ]
        features = [(pool, build_pool_dataset.CLASS_POOL)]

        self.assertTrue(
            build_pool_dataset._known_feature_intersects_bbox(
                features,
                (-27.52, 153.00, -27.49, 153.03),
            )
        )
        self.assertTrue(
            build_pool_dataset._known_feature_intersects_bbox(
                features,
                (-27.60, 152.90, -27.51, 153.01),
            ),
            "touching the stitched boundary must block an empty-label negative",
        )
        self.assertFalse(
            build_pool_dataset._known_feature_intersects_bbox(
                features,
                (-27.70, 152.80, -27.60, 152.90),
            )
        )

    def test_build_discards_negative_when_stitched_bbox_contains_known_feature(self):
        pool_geometry = [
            {"lat": 0.004, "lon": 0.004},
            {"lat": 0.004, "lon": 0.006},
            {"lat": 0.006, "lon": 0.006},
            {"lat": 0.006, "lon": 0.004},
        ]
        feature_bbox = (0.0, 0.0, 0.01, 0.01)
        clean_bbox = (0.50, 0.50, 0.51, 0.51)
        stitched = [
            (FakeImage(), feature_bbox, {}),  # positive chip
            (FakeImage(), feature_bbox, {}),  # proposed negative: must be discarded
            (FakeImage(), clean_bbox, {}),    # genuinely clear proposed negative
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                bbox=(0.0, 0.0, 1.0, 1.0),
                out=temp_dir,
                zoom=20,
                chip_m=45.0,
                negatives=1,
                val_frac=0.0,
                split_group_m=250.0,
                timeout=1.0,
                seed=7,
            )
            with patch.object(
                build_pool_dataset, "_pool_elements", return_value=[{"geometry": pool_geometry}]
            ), patch.object(
                build_pool_dataset, "_solar_elements", return_value=[]
            ), patch.object(
                build_pool_dataset.yd, "_lazy_import", return_value=None
            ), patch.object(
                build_pool_dataset.yd, "_np", object()
            ), patch.object(
                build_pool_dataset.yd, "_HAS_CV2", False
            ), patch.object(
                build_pool_dataset.yd, "_Image", object()
            ), patch.object(
                build_pool_dataset.yd, "_latlon_distance_m", return_value=10_000.0
            ), patch.object(
                build_pool_dataset.yd, "_stitch_tiles_with_stats", side_effect=stitched
            ) as stitch, patch.object(
                build_pool_dataset, "_save", return_value=True
            ) as save:
                build_pool_dataset.build(args)

        self.assertEqual(stitch.call_count, 3)
        self.assertEqual(save.call_count, 2)
        positive_call, negative_call = save.call_args_list
        self.assertTrue(positive_call.args[1].startswith("1 "))
        self.assertEqual(positive_call.args[2], "chip_00000")
        self.assertEqual(negative_call.args[1], "")
        self.assertEqual(negative_call.args[2], "neg_00000")
        self.assertEqual(Path(negative_call.args[3]).name, "train")


if __name__ == "__main__":
    unittest.main()
