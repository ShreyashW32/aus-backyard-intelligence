import json
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from aus_backyard_intelligence.app import derive_features
from aus_backyard_intelligence.pipeline.feature_builder import build_property_records


class FeatureBuilderTestCase(unittest.TestCase):
    def test_build_property_records_from_local_layers(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            parcels_path = temp_dir / "parcels.geojson"
            buildings_path = temp_dir / "buildings.geojson"
            vegetation_path = temp_dir / "vegetation.geojson"
            pools_path = temp_dir / "pools.geojson"
            dem_path = temp_dir / "dem.tif"
            output_path = temp_dir / "property_features.json"

            parcels = gpd.GeoDataFrame(
                [
                    {
                        "parcel_id": "MBRC-T1",
                        "lga_name": "Moreton Bay",
                        "suburb": "North Lakes",
                        "address": "1 Test Street, North Lakes QLD",
                    }
                ],
                geometry=[box(0, 0, 20, 40)],
                crs="EPSG:7856",
            )
            buildings = gpd.GeoDataFrame(
                [{"name": "House"}],
                geometry=[box(3, 5, 16, 20)],
                crs="EPSG:7856",
            )
            vegetation = gpd.GeoDataFrame(
                [{"name": "Trees"}],
                geometry=[box(0, 25, 10, 40)],
                crs="EPSG:7856",
            )
            pools = gpd.GeoDataFrame(
                [{"name": "Mapped pool"}],
                geometry=[box(4, 26, 9, 31)],
                crs="EPSG:7856",
            )

            parcels.to_file(parcels_path, driver="GeoJSON")
            buildings.to_file(buildings_path, driver="GeoJSON")
            vegetation.to_file(vegetation_path, driver="GeoJSON")
            pools.to_file(pools_path, driver="GeoJSON")

            dem_values = np.array(
                [
                    [14.0, 15.0],
                    [13.0, 14.0],
                    [12.0, 13.0],
                    [11.0, 12.0],
                ],
                dtype="float32",
            )
            transform = from_origin(0, 40, 10, 10)
            with rasterio.open(
                dem_path,
                "w",
                driver="GTiff",
                width=2,
                height=4,
                count=1,
                dtype="float32",
                crs="EPSG:7856",
                transform=transform,
            ) as dataset:
                dataset.write(dem_values, 1)

            result = build_property_records(
                parcels_path=parcels_path,
                buildings_path=buildings_path,
                dem_path=dem_path,
                output_path=output_path,
                id_field="parcel_id",
                lga_field="lga_name",
                suburb_field="suburb",
                address_field="address",
                vegetation_path=vegetation_path,
                pools_path=pools_path,
            )

            self.assertEqual(len(result.records), 1)
            self.assertEqual(result.skipped_without_buildings, 0)
            self.assertTrue(output_path.exists())

            record = result.records[0]
            self.assertEqual(record["id"], "MBRC-T1")
            self.assertEqual(record["lga"], "Moreton Bay")
            self.assertGreater(record["solar_access_pct"], 40.0)
            self.assertAlmostEqual(record["vegetation_pct"], 18.8, places=1)
            self.assertTrue(record["has_existing_pool"])
            self.assertEqual(record["existing_pool_source"], "configured_pool_layer")

            features = derive_features(record)
            self.assertGreater(features["backyard_area_sqm"], 350.0)
            self.assertGreater(features["slope_pct"], 0.0)

            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(written[0]["id"], "MBRC-T1")

    def test_build_property_records_skips_public_sites_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            parcels_path = temp_dir / "parcels.geojson"
            buildings_path = temp_dir / "buildings.geojson"
            dem_path = temp_dir / "dem.tif"

            parcels = gpd.GeoDataFrame(
                [
                    {
                        "parcel_id": "MBRC-PRIVATE",
                        "lga_name": "Moreton Bay",
                        "suburb": "North Lakes",
                        "address": "2 Private Street, North Lakes QLD",
                        "land_use": "Residential",
                        "owner_type": "Private",
                    },
                    {
                        "parcel_id": "MBRC-PUBLIC",
                        "lga_name": "Moreton Bay",
                        "suburb": "North Lakes",
                        "address": "10 Park Office Road, North Lakes QLD",
                        "land_use": "Public park and recreation",
                        "owner_type": "Moreton Bay Regional Council",
                    },
                ],
                geometry=[box(0, 0, 20, 40), box(30, 0, 50, 40)],
                crs="EPSG:7856",
            )
            buildings = gpd.GeoDataFrame(
                [{"name": "House"}, {"name": "Amenities"}],
                geometry=[box(3, 5, 16, 20), box(34, 5, 46, 18)],
                crs="EPSG:7856",
            )

            parcels.to_file(parcels_path, driver="GeoJSON")
            buildings.to_file(buildings_path, driver="GeoJSON")

            dem_values = np.full((4, 5), 12.0, dtype="float32")
            transform = from_origin(0, 40, 10, 10)
            with rasterio.open(
                dem_path,
                "w",
                driver="GTiff",
                width=5,
                height=4,
                count=1,
                dtype="float32",
                crs="EPSG:7856",
                transform=transform,
            ) as dataset:
                dataset.write(dem_values, 1)

            result = build_property_records(
                parcels_path=parcels_path,
                buildings_path=buildings_path,
                dem_path=dem_path,
                id_field="parcel_id",
                lga_field="lga_name",
                suburb_field="suburb",
                address_field="address",
                land_use_field="land_use",
                ownership_field="owner_type",
            )

            self.assertEqual(len(result.records), 1)
            self.assertEqual(result.records[0]["id"], "MBRC-PRIVATE")
            self.assertEqual(result.skipped_non_private, 1)


if __name__ == "__main__":
    unittest.main()
