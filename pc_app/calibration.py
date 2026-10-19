"""Board calibration and perspective mapping utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .board import BOARD_SIZE, Move


Point = tuple[float, float]


@dataclass(frozen=True)
class CalibrationAlignment:
    """Result of comparing a live frame with a saved camera calibration."""

    aligned: bool
    reason: str
    mean_corner_error: float | None = None
    max_corner_error: float | None = None


def check_calibration_alignment(
    frame: np.ndarray,
    calibration: "BoardCalibration",
    *,
    max_mean_error: float = 12.0,
    max_corner_error: float = 24.0,
) -> CalibrationAlignment:
    """Check whether the saved board geometry still matches a live frame.

    A missing board or a large corner displacement is treated as unsafe.  The
    caller can use this result as a run-time gate: recognition and motion
    must stop until a fresh calibration is confirmed.
    """
    if frame is None or frame.size == 0:
        return CalibrationAlignment(False, "当前没有可用画面")
    try:
        # During a game, stones, a hand, or changing exposure can introduce
        # additional large contours.  Select the live candidate that is
        # geometrically closest to the already trusted calibration instead
        # of blindly accepting the largest contour in the frame.
        detected = detect_board_corners(frame, reference_corners=calibration.corners)
    except ValueError:
        # A transient contour candidate can occasionally contain duplicate
        # corners.  Treat that frame as unsafe rather than allowing a
        # calibration-monitor exception to interrupt the preview loop.
        return CalibrationAlignment(False, "当前画面的棋盘角点不稳定")
    if detected is None:
        return CalibrationAlignment(False, "当前画面未找到完整棋盘")
    expected = np.asarray(calibration.corners, dtype=np.float32)
    actual = np.asarray(detected, dtype=np.float32)
    errors = np.linalg.norm(actual - expected, axis=1)
    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    aligned = mean_error <= max_mean_error and max_error <= max_corner_error
    if aligned:
        reason = "棋盘几何与已保存标定一致"
    else:
        reason = (
            f"棋盘几何偏移过大（平均 {mean_error:.1f}px，最大 {max_error:.1f}px），"
            "可能是相机被移动"
        )
    return CalibrationAlignment(aligned, reason, mean_error, max_error)


def order_corners(points: Iterable[Point]) -> tuple[Point, Point, Point, Point]:
    """Return corners in top-left, top-right, bottom-right, bottom-left order."""
    values = np.asarray(list(points), dtype=np.float32)
    if values.shape != (4, 2):
        raise ValueError("需要恰好四个棋盘角点")
    sums = values.sum(axis=1)
    differences = np.diff(values, axis=1).ravel()
    ordered = np.array(
        [
            values[np.argmin(sums)],
            values[np.argmin(differences)],
            values[np.argmax(sums)],
            values[np.argmax(differences)],
        ],
        dtype=np.float32,
    )
    if len({(float(x), float(y)) for x, y in ordered}) != 4:
        raise ValueError("棋盘角点不能重复")
    return tuple((float(x), float(y)) for x, y in ordered)  # type: ignore[return-value]


def detect_board_corners(
    frame: np.ndarray,
    *,
    reference_corners: tuple[Point, Point, Point, Point] | None = None,
) -> tuple[Point, Point, Point, Point] | None:
    """Find the outermost grid intersections of the 13 x 13 board.

    ``reference_corners`` is only used for a live consistency check.  A fresh
    calibration intentionally leaves it unset and keeps the original
    largest-candidate behaviour; a live check can use its known geometry to
    avoid being distracted by an unrelated contour.
    """
    if frame is None or frame.size == 0:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(frame.shape[0] * frame.shape[1])
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < frame_area * 0.08:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            candidates.append((area, polygon.reshape(4, 2)))
    if not candidates:
        candidates = _detect_intensity_board_candidates(gray, frame_area)
    if not candidates:
        return None

    # A transient Canny contour can look like a large convex quadrilateral
    # while still lacking four distinct geometric corners.  Do not let that
    # single candidate abort the entire alignment check: try the remaining
    # candidates, then let the caller treat a true absence as unsafe.
    detected_candidates: list[tuple[Point, Point, Point, Point]] = []
    for _, polygon in sorted(candidates, key=lambda item: item[0], reverse=True):
        try:
            outer_corners = order_corners([(float(x), float(y)) for x, y in polygon])
            detected = _refine_grid_corners(frame, outer_corners) or outer_corners
        except ValueError:
            continue
        if reference_corners is None:
            return detected
        detected_candidates.append(detected)
    if reference_corners is not None and detected_candidates:
        reference = np.asarray(reference_corners, dtype=np.float32)
        return min(
            detected_candidates,
            key=lambda corners: float(
                np.linalg.norm(np.asarray(corners, dtype=np.float32) - reference, axis=1).mean()
            ),
        )
    return None


def _detect_intensity_board_candidates(
    gray: np.ndarray,
    frame_area: float,
) -> list[tuple[float, np.ndarray]]:
    """Find the board frame when low-light Canny edges are fragmented."""
    height, width = gray.shape
    candidates: list[tuple[float, np.ndarray]] = []
    # The tan board occupies roughly 40-50% of this camera view.  Scanning a
    # small set of thresholds adapts to low/medium/high exposure without
    # baking an absolute brightness value into calibration.
    lower = max(8.0, float(np.percentile(gray, 8)))
    upper = min(245.0, float(np.percentile(gray, 92)))
    thresholds = np.linspace(lower, upper, 13)
    for threshold in thresholds:
        mask = cv2.threshold(gray, float(threshold), 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            fraction = area / frame_area
            if not 0.20 <= fraction <= 0.70:
                continue
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            points = polygon.reshape(4, 2).astype(np.float32)
            center_x = float(points[:, 0].mean()) / width
            center_y = float(points[:, 1].mean()) / height
            if not (0.35 <= center_x <= 0.75 and 0.30 <= center_y <= 0.75):
                continue
            # Prefer the expected board-sized region and reject large room
            # surfaces that happen to form a quadrilateral.
            try:
                ordered = order_corners([(float(x), float(y)) for x, y in points])
            except ValueError:
                # Threshold leakage can also create a degenerate four-point
                # contour.  It is not evidence of a board, so exclude it
                # rather than allowing one bad threshold to abort detection.
                continue
            left_drift = abs(ordered[3][0] - ordered[0][0])
            right_drift = abs(ordered[2][0] - ordered[1][0])
            # The printed grid is close to vertical in the camera view.  A
            # threshold mask that leaks into the dark left background often
            # creates a quadrilateral with a strongly asymmetric side drift;
            # penalize that shape before refinement.
            drift_penalty = abs(left_drift - right_drift) / max(width, 1) * 2.0
            score = -abs(fraction - 0.43) - abs(center_x - 0.45) * 0.15 - drift_penalty
            candidates.append((score, points))
    return candidates


def _refine_grid_corners(
    frame: np.ndarray,
    outer_corners: tuple[Point, Point, Point, Point],
    *,
    working_size: int = 1000,
) -> tuple[Point, Point, Point, Point] | None:
    """Locate 13 regularly spaced grid lines inside the detected board frame."""
    source = np.asarray(outer_corners, dtype=np.float32)
    last = float(working_size - 1)
    destination = np.asarray(((0, 0), (last, 0), (last, last), (0, last)), dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, destination)
    inverse = np.linalg.inv(matrix)
    warped = cv2.warpPerspective(frame, matrix, (working_size, working_size))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    darkness = 255.0 - gray.astype(np.float32)
    gradient_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gradient_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    margin = int(round(working_size * 0.08))
    vertical_score = darkness[margin:-margin].mean(axis=0) * 0.5 + gradient_x[margin:-margin].mean(axis=0)
    horizontal_score = darkness[:, margin:-margin].mean(axis=1) * 0.5 + gradient_y[:, margin:-margin].mean(axis=1)
    x_lines = _fit_regular_grid_lines(vertical_score, BOARD_SIZE)
    y_lines = _fit_regular_grid_lines(horizontal_score, BOARD_SIZE)
    if x_lines is None or y_lines is None:
        return None

    # The board frame and the printed grid are not guaranteed to share the
    # same four corners (the physical board has rounded margins).  As a
    # result, grid lines can remain a few degrees slanted after the frame is
    # warped.  Fit each boundary line and intersect them instead of crossing
    # the average x/y peaks; this keeps the outer intersections on the real
    # grid rather than on an inner line.
    horizontal = _fit_grid_line_equations(gray, y_lines, axis="horizontal")
    vertical = _fit_grid_line_equations(gray, x_lines, axis="vertical")
    if horizontal is not None and vertical is not None:
        intersections = (
            _intersect_grid_lines(vertical[0], horizontal[0]),
            _intersect_grid_lines(vertical[-1], horizontal[0]),
            _intersect_grid_lines(vertical[-1], horizontal[-1]),
            _intersect_grid_lines(vertical[0], horizontal[-1]),
        )
        grid_in_warp = np.asarray(intersections, dtype=np.float32).reshape(1, 4, 2)
    else:
        # Keep a conservative fallback for low-contrast frames where a line
        # equation cannot be fitted, while the quality checks still reject a
        # calibration that is visibly inconsistent.
        grid_in_warp = np.asarray(
            (((x_lines[0], y_lines[0]), (x_lines[-1], y_lines[0]),
              (x_lines[-1], y_lines[-1]), (x_lines[0], y_lines[-1]))),
            dtype=np.float32,
        ).reshape(1, 4, 2)
    grid_in_frame = cv2.perspectiveTransform(grid_in_warp, inverse)[0]
    return order_corners([(float(x), float(y)) for x, y in grid_in_frame])


def _fit_grid_line_equations(
    gray: np.ndarray,
    lines: tuple[float, ...],
    *,
    axis: str,
) -> tuple[tuple[float, float], ...] | None:
    """Fit slanted grid lines as ``y=a*x+b`` or ``x=a*y+b``.

    A profile peak gives a good approximate line location but loses the
    slope.  For every coordinate along the line, select the strongest local
    gradient in a generous band and robustly fit the resulting points.  The
    band is less than half a cell, so neighboring grid lines cannot be
    selected even when a stone or a highlight interrupts part of a line.
    """
    if gray.ndim != 2 or gray.size == 0 or len(lines) < 2:
        return None
    height, width = gray.shape
    gradient_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gradient_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    spacing = float(np.median(np.diff(lines)))
    if spacing <= 0:
        return None
    band = max(10, min(32, int(round(spacing * 0.36))))
    margin = max(20, int(round(min(width, height) * 0.02)))
    fitted: list[tuple[float, float]] = []

    for expected in lines:
        if axis == "horizontal":
            coordinates = np.arange(margin, width - margin, dtype=np.float32)
            values: list[float] = []
            weights: list[float] = []
            center = int(round(expected))
            for coordinate in coordinates.astype(int):
                low = max(0, center - band)
                high = min(height, center + band + 1)
                local = gradient_y[low:high, coordinate]
                peak = low + int(np.argmax(local))
                values.append(float(peak))
                weights.append(float(local[peak - low]))
        elif axis == "vertical":
            coordinates = np.arange(margin, height - margin, dtype=np.float32)
            values = []
            weights = []
            center = int(round(expected))
            for coordinate in coordinates.astype(int):
                low = max(0, center - band)
                high = min(width, center + band + 1)
                local = gradient_x[coordinate, low:high]
                peak = low + int(np.argmax(local))
                values.append(float(peak))
                weights.append(float(local[peak - low]))
        else:
            raise ValueError("axis 必须是 horizontal 或 vertical")

        fitted_line = _robust_weighted_line(
            coordinates,
            np.asarray(values, dtype=np.float32),
            np.asarray(weights, dtype=np.float32),
        )
        if fitted_line is None:
            return None
        fitted.append(fitted_line)
    return tuple(fitted)


def _robust_weighted_line(
    coordinates: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float] | None:
    """Return a weighted line fit after rejecting stone/edge outliers."""
    if coordinates.size < 40 or values.shape != coordinates.shape:
        return None
    finite = np.isfinite(coordinates) & np.isfinite(values) & np.isfinite(weights)
    if int(finite.sum()) < 40:
        return None
    coordinates = coordinates[finite].astype(np.float64)
    values = values[finite].astype(np.float64)
    weights = np.maximum(weights[finite].astype(np.float64), 1e-3)
    # Ignore the weakest quarter of columns/rows, which are usually plain
    # board background or an occluded section rather than a grid edge.
    mask = weights >= float(np.percentile(weights, 25))
    for _ in range(4):
        if int(mask.sum()) < 30:
            return None
        x = coordinates[mask]
        y = values[mask]
        w = weights[mask]
        design = np.column_stack((x, np.ones_like(x)))
        weighted = design * w[:, None]
        try:
            slope, intercept = np.linalg.lstsq(weighted, y * w, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        residual = np.abs(values - (slope * coordinates + intercept))
        mask = (residual <= 4.0) & (weights >= float(np.percentile(weights, 20)))
    if int(mask.sum()) < 30:
        return None
    return float(slope), float(intercept)


def _intersect_grid_lines(
    vertical: tuple[float, float],
    horizontal: tuple[float, float],
) -> tuple[float, float]:
    """Intersect ``x=av*y+bv`` with ``y=ah*x+bh``."""
    av, bv = vertical
    ah, bh = horizontal
    denominator = 1.0 - ah * av
    if abs(denominator) < 1e-6:
        raise ValueError("棋盘网格线几何退化")
    y = (ah * bv + bh) / denominator
    x = av * y + bv
    return float(x), float(y)


def _fit_regular_grid_lines(score: np.ndarray, count: int) -> tuple[float, ...] | None:
    """Fit equally spaced peaks and reject profiles without a credible grid."""
    length = int(score.shape[0])
    best: tuple[float, list[int]] | None = None
    for spacing in np.arange(length * 0.055, length * 0.083, 0.5):
        # The outermost grid line may be right next to the detected frame
        # boundary.  Starting at 2.5% systematically discarded it on the
        # real board and made an inner 11-line grid look like the full board.
        for start in np.arange(0.0, length * 0.145, 1.0):
            expected = start + spacing * np.arange(count)
            if expected[-1] >= length * 0.995:
                continue
            actual: list[int] = []
            strengths: list[float] = []
            search_radius = max(4, int(round(spacing * 0.08)))
            for position in expected:
                center = int(round(float(position)))
                low = max(0, center - search_radius)
                high = min(length, center + search_radius + 1)
                peak = low + int(np.argmax(score[low:high]))
                actual.append(peak)
                strengths.append(float(score[peak]))
            irregularity = float(np.std(np.diff(actual)))
            value = float(np.mean(strengths)) - irregularity * 1.5
            if best is None or value > best[0]:
                best = value, actual
    if best is None:
        return None

    _, actual = best
    selected = score[np.asarray(actual)]
    baseline = float(np.median(score))
    spread = float(np.std(score))
    span = actual[-1] - actual[0]
    if span < length * 0.72 or float(np.std(np.diff(actual))) > length * 0.008:
        return None
    if float(np.mean(selected)) < baseline + spread * 0.55:
        return None
    return tuple(float(value) for value in actual)


@dataclass(frozen=True)
class BoardCalibration:
    corners: tuple[Point, Point, Point, Point]
    output_size: int = 520
    board_size: int = BOARD_SIZE

    def __post_init__(self) -> None:
        # Normalize caller-provided corners once so perspective mapping is
        # stable even when a manual calibration supplies them in another order.
        object.__setattr__(self, "corners", order_corners(self.corners))
        if self.output_size < 100:
            raise ValueError("校正输出尺寸过小")
        if self.board_size < 2:
            raise ValueError("棋盘至少需要两条网格线")

    @property
    def matrix(self) -> np.ndarray:
        source = np.asarray(self.corners, dtype=np.float32)
        last = float(self.output_size - 1)
        destination = np.asarray(
            [(0, 0), (last, 0), (last, last), (0, last)], dtype=np.float32
        )
        return cv2.getPerspectiveTransform(source, destination)

    def warp(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.matrix, (self.output_size, self.output_size))

    def warp_padded(self, frame: np.ndarray, padding: int = 24) -> np.ndarray:
        """Warp the board while retaining real pixels just outside its grid.

        ``warp`` deliberately maps the outer grid intersections to the edge
        of its 520x520 output because that is the coordinate system used by
        recognition and label projection.  A review image needs a little
        context around those intersections: otherwise a stone centered on an
        edge is clipped before the UI can display it.  This method is a
        display-only variant and does not change ``warp`` or ``board_pixel``.
        """
        if isinstance(padding, bool) or not isinstance(padding, (int, np.integer)):
            raise ValueError("显示补边必须是非负整数")
        padding = int(padding)
        if padding < 0:
            raise ValueError("显示补边必须是非负整数")
        if padding == 0:
            return self.warp(frame)

        source = np.asarray(self.corners, dtype=np.float32)
        last = float(self.output_size - 1)
        destination = np.asarray(
            [
                (padding, padding),
                (padding + last, padding),
                (padding + last, padding + last),
                (padding, padding + last),
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(source, destination)
        size = self.output_size + 2 * padding
        return cv2.warpPerspective(frame, matrix, (size, size))

    def board_pixel(self, move: Move) -> tuple[float, float]:
        column, row = move
        if not 0 <= column < self.board_size or not 0 <= row < self.board_size:
            raise ValueError(f"棋盘坐标越界: {move}")
        spacing = (self.output_size - 1) / (self.board_size - 1)
        return column * spacing, row * spacing

    def to_json(self, path: Path) -> None:
        payload = {
            "corners": [[x, y] for x, y in self.corners],
            "output_size": self.output_size,
            "board_size": self.board_size,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "BoardCalibration":
        payload = json.loads(path.read_text(encoding="utf-8"))
        corners = order_corners(payload["corners"])
        return cls(
            corners=corners,
            output_size=int(payload.get("output_size", 520)),
            board_size=int(payload.get("board_size", BOARD_SIZE)),
        )
