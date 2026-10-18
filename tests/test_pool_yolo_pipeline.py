import argparse
import tempfile
import unittest
from pathlib import Path

from scripts import pool_yolo_pipeline


class PoolYoloPipelineNegativeReviewTestCase(unittest.TestCase):
    @staticmethod
    def _copy_args(root: Path) -> argparse.Namespace:
        return argparse.Namespace(
            images=str(root / "source_images"),
            labels=str(root / "source_labels"),
            dataset_images=str(root / "dataset_images"),
            dataset_labels=str(root / "dataset_labels"),
        )

    def test_copy_negatives_refuses_missing_unreviewed_label_before_writing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "source_images"
            images.mkdir()
            (images / "candidate.jpg").write_bytes(b"image")
            args = self._copy_args(root)

            with self.assertRaisesRegex(FileNotFoundError, "unreviewed negative"):
                pool_yolo_pipeline.copy_negatives_into_dataset(args)

            self.assertFalse(Path(args.dataset_images).exists())
            self.assertFalse(Path(args.dataset_labels).exists())

    def test_copy_negatives_refuses_nonempty_positive_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "source_images"
            labels = root / "source_labels"
            images.mkdir()
            labels.mkdir()
            (images / "candidate.jpg").write_bytes(b"image")
            (labels / "candidate.txt").write_text(
                "1 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
            args = self._copy_args(root)

            with self.assertRaisesRegex(ValueError, "must be explicitly empty"):
                pool_yolo_pipeline.copy_negatives_into_dataset(args)

            self.assertFalse(Path(args.dataset_images).exists())
            self.assertFalse(Path(args.dataset_labels).exists())

    def test_copy_negatives_accepts_explicit_empty_reviewed_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "source_images"
            labels = root / "source_labels"
            images.mkdir()
            labels.mkdir()
            (images / "confirmed_roof.jpg").write_bytes(b"image")
            (labels / "confirmed_roof.txt").write_text("", encoding="utf-8")
            args = self._copy_args(root)

            pool_yolo_pipeline.copy_negatives_into_dataset(args)

            self.assertTrue((Path(args.dataset_images) / "confirmed_roof.jpg").is_file())
            copied_label = Path(args.dataset_labels) / "confirmed_roof.txt"
            self.assertTrue(copied_label.is_file())
            self.assertEqual(copied_label.read_text(encoding="utf-8"), "")

    def test_make_negatives_never_overwrites_a_positive_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            (images / "mixed.jpg").write_bytes(b"image")
            label = labels / "mixed.txt"
            original = "1 0.5 0.5 0.2 0.2\n"
            label.write_text(original, encoding="utf-8")
            args = argparse.Namespace(images=str(images), labels=str(labels))

            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                pool_yolo_pipeline.make_negative_labels(args)

            self.assertEqual(label.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
