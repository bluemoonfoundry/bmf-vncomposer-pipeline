"""Estimate useful lighting metadata from a background plate without an LLM."""

from __future__ import annotations

import colorsys
import math
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageStat

from .schemas import SceneLighting


def estimate_lighting(path: str | Path | BinaryIO) -> SceneLighting:
    """Return a conservative lighting estimate from image gradients and color.

    Brightness is treated as coming from the upper half of the plate. The result is
    intentionally a hint for a Blender light, not a claim of physical reconstruction.
    """
    with Image.open(path).convert("RGB") as image:
        image.thumbnail((256, 256))
        width, height = image.size
        mean = tuple(channel / 255.0 for channel in ImageStat.Stat(image).mean)
        upper = [image.getpixel((x, y)) for y in range(max(1, height // 2)) for x in range(width)]
        upper_mean = tuple(sum(p[i] for p in upper) / max(1, len(upper)) / 255.0 for i in range(3))

        # Weighted centroid of luminance gives a stable, explainable light direction.
        weighted = [(x, y, 0.2126 * r + 0.7152 * g + 0.0722 * b) for y in range(height) for x in range(width) for r, g, b in [image.getpixel((x, y))]]
        total = sum(item[2] for item in weighted) or 1.0
        cx = sum(x * weight for x, _, weight in weighted) / total / max(1, width - 1)
        cy = sum(y * weight for _, y, weight in weighted) / total / max(1, height - 1)
        direction = [float((cx - 0.5) * 2), float((cy - 0.5) * 2), -1.0]
        length = math.sqrt(sum(axis * axis for axis in direction)) or 1.0
        direction = [axis / length for axis in direction]
        saturation = colorsys.rgb_to_hsv(*mean)[1]
        intensity = max(0.1, min(3.0, (sum(upper_mean) / 3.0) * 2.0))
        confidence = max(0.0, min(1.0, (sum(item[2] for item in weighted) / len(weighted) / 255.0) * (0.5 + saturation)))
        return SceneLighting(direction=direction, color_rgb=mean, ambient_rgb=upper_mean, intensity=intensity, confidence=confidence)
