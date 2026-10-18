import gzip
import json
import os
import tempfile
import unittest

from aus_backyard_intelligence import ms_buildings


class MsBuildingsTestCase(unittest.TestCase):
    def test_quadkey_matches_known_bondi_z9_tile(self):
        # Bondi (-33.89, 151.27) sits in Microsoft's Australia z9 tile 311230133.
        x, y = ms_buildings._tile_xy(-33.89, 151.27, 9)
        self.assertEqual(ms_buildings._quadkey(x, y, 9), "311230133")

    def test_parse_tile_filters_to_bbox_and_tags_source(self):
        inside = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[
                [151.2740, -33.8885], [151.2741, -33.8885],
                [151.2741, -33.8884], [151.2740, -33.8884], [151.2740, -33.8885],
            ]]},
            "properties": {"height": 6.2, "confidence": -1.0},
        }
        outside = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[
                [151.3000, -33.9000], [151.3001, -33.9000],
                [151.3001, -33.8999], [151.3000, -33.8999], [151.3000, -33.9000],
            ]]},
            "properties": {"height": 4.0},
        }
        fd, path = tempfile.mkstemp(suffix=".geojsonl.gz")
        os.close(fd)
        try:
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(inside) + "\n")
                handle.write(json.dumps(outside) + "\n")
            bbox = (-33.8895, 151.2730, -33.8875, 151.2770)  # south, west, north, east
            elements = ms_buildings._parse_tile_for_bbox(path, bbox, start_id=800_000_000)
        finally:
            os.remove(path)

        self.assertEqual(len(elements), 1)
        element = elements[0]
        self.assertEqual(element["type"], "way")
        self.assertEqual(element["tags"]["building"], "yes")
        self.assertEqual(element["tags"]["detector:source"], "microsoft-footprint")
        self.assertEqual(element["tags"]["height"], "6.2")
        self.assertGreaterEqual(len(element["geometry"]), 4)
        self.assertIn("lat", element["geometry"][0])

    def test_quadkeys_for_small_bbox_is_bounded(self):
        bbox = (-33.8895, 151.2730, -33.8875, 151.2770)
        keys = ms_buildings._quadkeys_for_bbox(bbox)
        self.assertGreaterEqual(len(keys), 1)
        self.assertLessEqual(len(keys), ms_buildings.MAX_TILES_PER_REQUEST)
        self.assertIn("311230133", keys)


if __name__ == "__main__":
    unittest.main()
