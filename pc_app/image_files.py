"""OpenCV image I/O that also works when Windows paths contain Chinese text."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_image(path: Path, image: np.ndarray, params: list[int] | None = None) -> None:
    """Encode with OpenCV, then write bytes through pathlib for Unicode paths."""
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image, params or [])
    if not ok:
        raise OSError(f"图像编码失败: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def read_color_image(path: Path) -> np.ndarray | None:
    """Read image bytes first, avoiding OpenCV's Windows narrow-path API."""
    try:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
