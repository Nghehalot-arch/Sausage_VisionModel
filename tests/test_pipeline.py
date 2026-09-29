import io
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import yaml
from PIL import Image
from sausage_vision.geometry import checked_parts, join_parts, normalized_line, line_to_points, raster, area
from sausage_vision.prepare import prepare, assignments


class ConversionTests(unittest.TestCase):
    def test_disconnected_parts_stay_one_instance_without_filling_gap(self):
        segments = [[10, 10, 40, 10, 40, 60, 10, 60], [10, 110, 40, 110, 40, 160, 10, 160]]
        parts = checked_parts(segments, 100, 200)
        joined = join_parts(parts)
        line = normalized_line(joined, 100, 200)
        recovered = line_to_points(line, 100, 200)
        self.assertNotIn("\n", line)
        self.assertAlmostEqual(abs(area(joined)), sum(abs(area(p)) for p in parts))
        mask = raster([recovered], 100, 200, 200)
        self.assertTrue(mask[30, 25])
        self.assertTrue(mask[130, 25])
        self.assertFalse(mask[85, 25])  # the occluded middle remains unfilled
        self.assertGreater(np.count_nonzero(mask), 3000)

    def test_rejects_invalid_coordinate_and_duplicate_assignment(self):
        with self.assertRaises(ValueError):
            checked_parts([[0, 0, 120, 0, 0, 20]], 100, 100)
        with self.assertRaises(ValueError):
            checked_parts([[0, 0, float("nan"), 10, 0, 20]], 100, 100)
        with self.assertRaises(ValueError):
            assignments({"groups": [{"name": "a", "split": "train", "files": ["a.jpg"]},
                                    {"name": "b", "split": "val", "files": ["a.jpg"]}]})

    def test_zip_preparation_handles_rotation_exclusion_and_instance_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_data = []
            for colour in ("red", "green", "blue"):
                image = Image.new("RGB", (200, 100), colour)
                exif = Image.Exif()
                exif[274] = 6  # displayed size is 100 x 200
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", exif=exif)
                image_data.append(buffer.getvalue())
            data = {"categories": [{"id": 1, "name": "sausage"}],
                    "images": [{"id": i, "file_name": f"{i}.jpg", "width": 100, "height": 200} for i in (1, 2, 3)],
                    "annotations": [{"id": i, "image_id": i, "category_id": 1,
                                     "segmentation": [[10, 10, 40, 10, 40, 80, 10, 80]]} for i in (1, 2, 3)]}
            archive = root / "input.zip"
            with ZipFile(archive, "w") as z:
                z.writestr("annotations/instances_default.json", json.dumps(data))
                for i, payload in enumerate(image_data, 1):
                    z.writestr(f"images/{i}.jpg", payload)
            config = {"class_name": "sausage", "exclude": {"3.jpg": "review"},
                      "groups": [{"name": "first", "split": "train", "files": ["1.jpg"]},
                                 {"name": "second", "split": "val", "files": ["2.jpg"]}],
                      "conversion": {"min_preview_iou": 0.98, "preview_long_side": 640}}
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            out = root / "prepared"
            report = prepare(archive, config_path, out)
            self.assertEqual(report["counts"], {"train": {"images": 1, "instances": 1}, "val": {"images": 1, "instances": 1}})
            self.assertEqual(len(report["excluded"]), 1)
            with Image.open(out / "images/train/1.jpg") as oriented:
                self.assertEqual(oriented.size, (100, 200))
                self.assertIn(oriented.getexif().get(274, 1), (1, None))
            self.assertFalse((out / "images/train/3.jpg").exists())
            self.assertTrue((out / "review/index.html").is_file())
            with self.assertRaises(ValueError):
                prepare(archive, config_path, out)


if __name__ == "__main__":
    unittest.main()
