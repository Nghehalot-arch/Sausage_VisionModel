"""Prepare the supplied COCO ZIP, including explicit scene splits and review output."""
import hashlib
import html
import io
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps
from .geometry import checked_parts, join_parts, normalized_line, line_to_points, compare_masks

COLOURS = [(235, 66, 66), (35, 185, 100), (30, 185, 230), (230, 185, 30), (200, 60, 220)]


def assignments(config):
    result = {}
    excluded = config.get("exclude", {})
    for group in config["groups"]:
        if group["split"] not in {"train", "val", "test"}:
            raise ValueError("Unknown dataset split.")
        for name in group["files"]:
            if name in result or name in excluded:
                raise ValueError(f"Conflicting split/exclusion assignment: {name}")
            result[name] = (group["split"], group["name"])
    return result


def overlay(image, contours, caption):
    view = image.copy()
    view.thumbnail((540, 720))
    scale_x, scale_y = view.width / image.width, view.height / image.height
    rgba = view.convert("RGBA")
    layer = Image.new("RGBA", view.size)
    pen = ImageDraw.Draw(layer)
    for i, parts in enumerate(contours):
        colour = COLOURS[i % len(COLOURS)]
        for part in parts:
            points = [(x * scale_x, y * scale_y) for x, y in part]
            pen.polygon(points, fill=(*colour, 35), outline=(*colour, 255), width=2)
    view = Image.alpha_composite(rgba, layer).convert("RGB")
    panel = Image.new("RGB", (540, 756), "white")
    panel.paste(view, ((540 - view.width) // 2, 36))
    ImageDraw.Draw(panel).text((8, 10), caption, fill="black")
    return panel


def prepare(archive_path, config_path, output):
    archive_path, config_path, output = Path(archive_path), Path(config_path), Path(output).resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new --out directory to preserve previous work.")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mapping = assignments(config)
    excluded = config.get("exclude", {})
    report = {"source_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
              "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "counts": {}, "images": [], "excluded": [], "multipart_instances": 0,
              "conversion": "Retraced-edge polygon connections; Pillow geometric audit, not YOLO model accuracy.",
              "split_limitation": "Pilot layout groups inferred visually; no independent final test set in this handoff."}
    with ZipFile(archive_path) as archive:
        coco_name = "annotations/instances_default.json"
        data = json.loads(archive.read(coco_name))
        images = data["images"]
        annotations = data["annotations"]
        if len({i["id"] for i in images}) != len(images) or len({a["id"] for a in annotations}) != len(annotations):
            raise ValueError("Duplicate COCO image or annotation IDs.")
        if len({i["file_name"] for i in images}) != len(images):
            raise ValueError("Duplicate COCO image filenames.")
        categories = data["categories"]
        if len(categories) != 1 or categories[0]["name"] != config["class_name"]:
            raise ValueError("This pilot expects exactly one category named sausage.")
        image_ids = {i["id"] for i in images}
        by_image = defaultdict(list)
        for ann in annotations:
            if ann["image_id"] not in image_ids or ann["category_id"] != categories[0]["id"] or ann.get("iscrowd", 0):
                raise ValueError(f"Unexpected annotation linkage/category/crowd flag: {ann['id']}")
            by_image[ann["image_id"]].append(ann)
        names = {i["file_name"] for i in images}
        if names != set(mapping) | set(excluded):
            raise ValueError(f"Every image needs an explicit split or exclusion; unassigned={names-set(mapping)-set(excluded)}, stale={set(mapping)|set(excluded)-names}")
        output.mkdir(parents=True)
        review = output / "review"
        review.mkdir()
        (output / "source_coco.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        (output / "split_config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        hashes, stem_keys, cards = {}, set(), []
        for info in images:
            name = info["file_name"]
            if PurePosixPath(name).name != name or "\\" in name:
                raise ValueError("Only simple image filenames are supported.")
            payload = archive.read("images/" + name)
            with Image.open(io.BytesIO(payload)) as raw:
                orientation = raw.getexif().get(274, 1)
                image = ImageOps.exif_transpose(raw).convert("RGB")
            width, height = info["width"], info["height"]
            if image.size != (width, height):
                raise ValueError(f"COCO dimensions do not match EXIF-oriented image: {name}")
            source_contours = [checked_parts(a["segmentation"], width, height) for a in by_image[info["id"]]]
            if name in excluded:
                report["excluded"].append({"file": name, "reason": excluded[name], "instances": len(source_contours)})
                overlay(image, source_contours, "EXCLUDED: " + name).save(review / (Path(name).stem + ".jpg"), quality=90)
                cards.append(f'<article><h2>{html.escape(name)} — excluded</h2><p>{html.escape(excluded[name])}</p><img src="{html.escape(Path(name).stem)}.jpg"></article>')
                continue
            split, group = mapping[name]
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest in hashes and hashes[digest] != split:
                raise ValueError("Identical decoded images cross dataset splits.")
            hashes[digest] = split
            key = (split, Path(name).stem)
            if key in stem_keys:
                raise ValueError("Two images would share a YOLO label filename.")
            stem_keys.add(key)
            converted, rows, audits = [], [], []
            for ann, parts in zip(by_image[info["id"]], source_contours):
                joined = join_parts(parts)
                row = normalized_line(joined, width, height)
                points = line_to_points(row, width, height)
                high = compare_masks(parts, points, width, height, config["conversion"]["preview_long_side"])
                low = compare_masks(parts, points, width, height, 160)
                if high["iou"] < config["conversion"]["min_preview_iou"]:
                    raise ValueError(f"Conversion needs review: annotation {ann['id']}, IoU={high['iou']:.5f}. Original labels remain in source_coco.json.")
                report["multipart_instances"] += len(parts) > 1
                audits.append({"annotation_id": ann["id"], "source_parts": len(parts), "preview": high, "low_resolution": low})
                rows.append(row)
                converted.append([points])
            for kind in ("images", "labels"):
                (output / kind / split).mkdir(parents=True, exist_ok=True)
            image_name = Path(name).stem + ".jpg"
            image.save(output / "images" / split / image_name, quality=95)
            (output / "labels" / split / (Path(name).stem + ".txt")).write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            pair = Image.new("RGB", (1080, 756), "white")
            pair.paste(overlay(image, source_contours, "COCO source: " + name), (0, 0))
            pair.paste(overlay(image, converted, "YOLO conversion: " + name), (540, 0))
            pair.save(review / (Path(name).stem + ".jpg"), quality=90)
            minimum = min((a["preview"]["iou"] for a in audits), default=1)
            cards.append(f'<article><h2>{html.escape(name)} — {split}, {len(rows)} instances</h2><p>Geometric preview IoU: {minimum:.4f}</p><img src="{html.escape(Path(name).stem)}.jpg"></article>')
            counts = report["counts"].setdefault(split, {"images": 0, "instances": 0})
            counts["images"] += 1
            counts["instances"] += len(rows)
            report["images"].append({"file": name, "output_image": f"images/{split}/{image_name}", "split": split,
                                      "group": group, "width": width, "height": height,
                                      "source_exif_orientation": orientation, "annotations": audits})
        if any(report["counts"].get(s, {}).get("images", 0) == 0 for s in ("train", "val")):
            raise ValueError("Training and validation must both contain images.")
        dataset = {"path": str(output), "train": "images/train", "val": "images/val", "names": {0: config["class_name"]}}
        if "test" in report["counts"]:
            dataset["test"] = "images/test"
        (output / "data.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
        (output / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        intro = '<!doctype html><html><meta charset="utf-8"><title>Sausage label review</title><style>body{font:16px system-ui;background:#f2f4f7;margin:24px}article{background:white;padding:18px;margin:20px 0;max-width:1100px}img{max-width:100%;height:auto}h2{font-size:18px}</style><h1>COCO and YOLO label review</h1><p>Left: original visible parts. Right: converted single polygon. Multipart outlines may acquire thin connecting lines. These are geometric previews, not model predictions.</p>'
        (review / "index.html").write_text(intro + "\n".join(cards) + "</html>", encoding="utf-8")
    return report
