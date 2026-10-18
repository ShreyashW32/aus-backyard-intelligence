import unittest

from aus_backyard_intelligence.eligibility import assess_private_property


class EligibilityTestCase(unittest.TestCase):
    def test_building_rect_alone_is_not_reported_as_backyard_geometry(self):
        assessment = assess_private_property({
            "address": "12 Test Street",
            "use": "residential",
            "building_rect": [0, 0, 8, 12],
        })

        self.assertTrue(assessment["is_private_property_candidate"])
        self.assertNotIn("rear-yard geometry", assessment["eligibility_reason"])

    def test_parcel_and_building_rects_count_as_derived_backyard_geometry(self):
        assessment = assess_private_property({
            "address": "14 Test Street",
            "use": "residential",
            "parcel_rect": [0, 0, 18, 30],
            "building_rect": [3, 2, 15, 16],
        })

        self.assertTrue(assessment["is_private_property_candidate"])
        self.assertIn("rear-yard geometry", assessment["eligibility_reason"])


if __name__ == "__main__":
    unittest.main()
