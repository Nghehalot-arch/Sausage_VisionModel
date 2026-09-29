"""Polygon conversion and raster comparison, independent of GPU/YOLO packages."""
import math
import numpy as np
from PIL import Image, ImageDraw


def area(points):
    return sum(x * yn - xn * y for (x, y), (xn, yn) in zip(points, points[1:] + points[:1])) / 2


def checked_parts(segments, width, height):
    if not isinstance(segments, list) or not segments:
        raise ValueError("Expected nonempty COCO polygon segments; RLE is not supported by this converter.")
    result = []
    for segment in segments:
        if len(segment) < 6 or len(segment) % 2:
            raise ValueError("A polygon requires at least three coordinate pairs.")
        points = [(float(x), float(y)) for x, y in zip(segment[::2], segment[1::2])]
        if any(not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= width or not 0 <= y <= height for x, y in points):
            raise ValueError("Polygon coordinates are nonfinite or outside the COCO image dimensions.")
        if points[-1] == points[0]:
            points.pop()
        if len(set(points)) < 3 or abs(area(points)) < 1e-6:
            raise ValueError("Degenerate polygon.")
        result.append(points if area(points) > 0 else list(reversed(points)))
    return result


def join_parts(parts):
    """Join contours by retraced edges; one output polygon per original instance.

    Retraced connecting edges have zero continuous area but can add thin raster
    lines. This is an approximation to disconnected COCO masks and must be audited.
    """
    remaining = sorted((list(p) for p in parts), key=lambda p: abs(area(p)), reverse=True)
    result = remaining.pop(0)
    while remaining:
        candidates = []
        for k, part in enumerate(remaining):
            distances = ((np.asarray(result)[:, None, :] - np.asarray(part)[None, :, :]) ** 2).sum(axis=2)
            i, j = np.unravel_index(distances.argmin(), distances.shape)
            candidates.append((float(distances[i, j]), k, int(i), int(j)))
        _, k, i, j = min(candidates)
        part = remaining.pop(k)
        loop = part[j:] + part[:j]
        result = result[:i + 1] + loop + [loop[0], result[i]] + result[i + 1:]
    return result


def normalized_line(points, width, height):
    return "0 " + " ".join(f"{v:.10f}" for x, y in points for v in (x / width, y / height))


def line_to_points(line, width, height):
    values = [float(v) for v in line.split()]
    if len(values) < 7 or (len(values) - 1) % 2 or values[0] != 0:
        raise ValueError("Expected a class-0 segmentation polygon.")
    return [(x * width, y * height) for x, y in zip(values[1::2], values[2::2])]


def raster(parts, width, height, side):
    scale = side / max(width, height)
    image = Image.new("1", (max(1, round(width * scale)), max(1, round(height * scale))))
    pen = ImageDraw.Draw(image)
    for part in parts:
        pen.polygon([(x * scale, y * scale) for x, y in part], fill=1)
    return np.asarray(image, dtype=bool)


def compare_masks(parts, converted, width, height, side):
    reference = raster(parts, width, height, side)
    predicted = raster([converted], width, height, side)
    union = int((reference | predicted).sum())
    return {"long_side": side, "iou": float((reference & predicted).sum() / union) if union else 1.0,
            "added_pixels": int((predicted & ~reference).sum()),
            "removed_pixels": int((reference & ~predicted).sum())}
