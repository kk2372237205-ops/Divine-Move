"""Small OpenCV camera wrapper with explicit start and stop methods."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


CAMERA_BACKENDS: tuple[tuple[str, int], ...] = (
    ("DirectShow", cv2.CAP_DSHOW),
    ("Media Foundation", cv2.CAP_MSMF),
    ("Windows auto", cv2.CAP_ANY),
)

PREFERRED_WIDTH = 1920
PREFERRED_HEIGHT = 1080
PREFERRED_FPS = 30
CAMERA_WARMUP_FRAMES = 45
PREFERRED_EXPOSURE = -1

# This camera reports a strong blue cast while its driver-side white-balance
# controls remain ineffective.  Keep raw frames untouched for calibration and
# recognition; callers can apply this display/data correction explicitly.
COLOR_BALANCE_GAINS = (0.80, 1.00, 1.15)


class CameraService:
    def __init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None
        self._first_frame: Any | None = None
        self.index: int | None = None
        self.backend_name: str | None = None
        self.frame_width: int | None = None
        self.frame_height: int | None = None
        self.consecutive_read_failures = 0

    @staticmethod
    def available_indexes(limit: int = 5) -> list[int]:
        """Probe camera indexes briefly and release each test capture."""
        indexes: list[int] = []
        for index in range(limit):
            capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            try:
                if capture.isOpened():
                    ok, _ = capture.read()
                    if ok:
                        indexes.append(index)
            finally:
                capture.release()
        return indexes

    def start(self, index: int) -> tuple[bool, str]:
        self.stop()
        failures: list[str] = []
        for backend_name, backend in CAMERA_BACKENDS:
            capture = cv2.VideoCapture(index, backend)
            if not capture.isOpened():
                capture.release()
                failures.append(f"{backend_name} 无法打开")
                continue

            self._configure_capture(capture)

            frame = None
            # Some DirectShow devices expose a valid but overexposed/striped
            # frame while auto-exposure settles. Keep the newest valid frame
            # through a short warm-up window before exposing it to the UI.
            for _ in range(CAMERA_WARMUP_FRAMES):
                ok, candidate = capture.read()
                if ok and candidate is not None and candidate.size:
                    frame = candidate
            if frame is None:
                capture.release()
                failures.append(f"{backend_name} 未返回画面")
                continue

            self._capture = capture
            self._first_frame = frame
            self.index = index
            self.backend_name = backend_name
            self.frame_height, self.frame_width = frame.shape[:2]
            resolution = f"{self.frame_width}x{self.frame_height}"
            quality = "1080P" if self.frame_width >= 1920 and self.frame_height >= 1080 else "低于1080P"
            return True, f"摄像头 {index} 预览中（{backend_name}，{resolution}，{quality}）"

        detail = "；".join(failures)
        return False, f"摄像头 {index} 无法读取画面：{detail}"

    @staticmethod
    def _configure_capture(capture: cv2.VideoCapture) -> None:
        """Request the competition camera mode before reading the first frame."""
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, PREFERRED_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, PREFERRED_HEIGHT)
        capture.set(cv2.CAP_PROP_FPS, PREFERRED_FPS)
        capture.set(cv2.CAP_PROP_EXPOSURE, PREFERRED_EXPOSURE)

    def read(self) -> Any | None:
        if self._first_frame is not None:
            frame = self._first_frame
            self._first_frame = None
            return frame
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if ok and frame is not None and frame.size:
            self.consecutive_read_failures = 0
            return frame
        self.consecutive_read_failures += 1
        return None

    @staticmethod
    def correct_color(frame: np.ndarray) -> np.ndarray:
        """Return a balanced BGR copy for preview and dataset output."""
        if frame is None or frame.size == 0:
            raise ValueError("没有可用的摄像头画面")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("摄像头画面必须是三通道 BGR 图像")
        gains = np.asarray(COLOR_BALANCE_GAINS, dtype=np.float32)
        balanced = frame.astype(np.float32) * gains.reshape(1, 1, 3)
        return np.clip(balanced, 0, 255).astype(np.uint8)

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._first_frame = None
        self.index = None
        self.backend_name = None
        self.frame_width = None
        self.frame_height = None
        self.consecutive_read_failures = 0
