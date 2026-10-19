"""Calibrated, background-referenced Gomoku stone recognition.

The competition board is fixed below the camera.  Comparing a calibrated
current view with an explicitly captured empty-board reference is more
reliable at this stage than guessing a global colour threshold under changing
lighting.  This module has no camera or serial-port I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .board import BOARD_SIZE, Move, Stone
from .calibration import BoardCalibration
from .image_files import read_color_image, write_image


@dataclass(frozen=True)
class StoneDetection:
    """One occupied grid point observed in one calibrated camera frame."""

    move: Move
    stone: Stone
    confidence: float
    difference: float


@dataclass(frozen=True)
class BoardVisionResult:
    """All occupied grid points detected from one frame."""

    detections: tuple[StoneDetection, ...]

    @property
    def stones(self) -> dict[Move, Stone]:
        return {detection.move: detection.stone for detection in self.detections}

    def stone_at(self, move: Move) -> Stone:
        return self.stones.get(move, Stone.EMPTY)


@dataclass(frozen=True)
class PlacementMeasurement:
    """Camera-measured centre error for one expected black stone."""

    move: Move
    expected_pixel: tuple[float, float]
    observed_pixel: tuple[float, float]

    @property
    def offset_pixel(self) -> tuple[float, float]:
        return (
            self.observed_pixel[0] - self.expected_pixel[0],
            self.observed_pixel[1] - self.expected_pixel[1],
        )


@dataclass(frozen=True)
class BoardObstruction:
    """A non-board region found inside the calibrated 13 x 13 grid."""

    x: int
    y: int
    width: int
    height: int
    area: float


@dataclass(frozen=True)
class BoardClearanceResult:
    """Result of checking that the board is empty before a new run."""

    obstructions: tuple[BoardObstruction, ...]

    @property
    def clear(self) -> bool:
        return not self.obstructions


GAMEPLAY_OCCLUSION_MIN_AREA = 6_000.0


def detect_board_obstructions(
    frame: np.ndarray,
    calibration: BoardCalibration,
    *,
    difference_threshold: int = 42,
    minimum_area: float = 90.0,
    ignore_grid_intersections: bool = False,
) -> BoardClearanceResult:
    """Find chess pieces or foreign objects without needing an empty reference.

    The board is first perspective-corrected.  Known printed grid lines are
    excluded, then regions substantially lighter or darker than the board's
    robust median colour are grouped into objects.  This catches a forgotten
    black/white piece and larger foreign objects such as a card or hand-held
    item while ignoring the printed lines and five star points.
    """
    if frame is None or frame.size == 0:
        raise ValueError("没有可用的摄像头画面")
    if difference_threshold <= 0 or minimum_area <= 0:
        raise ValueError("异物检测参数必须为正数")

    warped = calibration.warp(frame)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    size = calibration.output_size
    usable = np.full((size, size), 255, dtype=np.uint8)
    spacing = (size - 1) / (calibration.board_size - 1)
    # Ignore the inked grid itself.  A stone or foreign object remains visible
    # outside this narrow mask, even when it covers an intersection.
    for index in range(calibration.board_size):
        point = int(round(index * spacing))
        cv2.line(usable, (point, 0), (point, size - 1), 0, 5)
        cv2.line(usable, (0, point), (size - 1, point), 0, 5)
    usable[:12, :] = 0
    usable[-12:, :] = 0
    usable[:, :12] = 0
    usable[:, -12:] = 0

    samples = gray[usable > 0]
    colour_samples = warped[usable > 0]
    if samples.size == 0 or colour_samples.size == 0:
        raise ValueError("棋盘有效区域为空")
    board_median = float(np.median(samples))
    board_colour = np.median(colour_samples.astype(np.float32), axis=0)
    brightness_change = np.abs(gray.astype(np.float32) - board_median)
    colour_change = np.linalg.norm(warped.astype(np.float32) - board_colour, axis=2)
    # A white card can have a brightness close to a brightly lit tan board,
    # but its three-channel colour distance is still large.  Combining both
    # measurements catches that case without treating normal board texture as
    # an object.
    changed = np.logical_or(
        brightness_change >= difference_threshold,
        colour_change >= 50.0,
    ).astype(np.uint8) * 255
    changed = cv2.bitwise_and(changed, usable)
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    if ignore_grid_intersections:
        # Adjacent pieces can visually join into one large contour. Its
        # aggregate area may exceed the hand/arm threshold even though every
        # changed region is a legitimate stone. Remove a stone-sized disk at
        # each legal intersection after morphology, leaving real hands and
        # arms visible in the spaces between intersections.
        stone_radius = max(6, int(round(spacing * 0.44)))
        for row in range(calibration.board_size):
            for column in range(calibration.board_size):
                cv2.circle(
                    changed,
                    (int(round(column * spacing)), int(round(row * spacing))),
                    stone_radius,
                    0,
                    -1,
                )

    contours, _ = cv2.findContours(changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found: list[BoardObstruction] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < minimum_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        # Remaining fragments from the printed grid are long, one-pixel-wide
        # strips.  Real pieces and physical objects have visible thickness.
        if min(width, height) < 7:
            continue
        found.append(BoardObstruction(x, y, width, height, area))
    merged = _merge_nearby_obstructions(found)
    # Perspective correction and auto-exposure make the fixed rounded board
    # frame bleed a little way into the corner cells.  The previous guard was
    # narrower than that bleed, so a small dark fragment near a corner could
    # flicker between "clear" and "obstruction" as the exposure changed.
    # Keep the filter limited to regions touching *two* outer edges: objects
    # in the middle and along a single side are still reported.
    edge_guard = max(24, int(round(spacing * 0.65)))
    # The rounded board frame and the camera's corner shadow can form a large
    # dark region where two outer edges meet.  It is printed hardware, not an
    # object on the playable grid.  Ignore only these double-edge corner
    # regions; objects in the middle or along a single side remain reportable.
    reliable = [
        item
        for item in merged
        if sum(
            (
                item.x <= edge_guard,
                item.y <= edge_guard,
                item.x + item.width >= size - edge_guard,
                item.y + item.height >= size - edge_guard,
            )
        ) < 2
    ]
    return BoardClearanceResult(tuple(reliable))


def detect_gameplay_occlusion(
    frame: np.ndarray,
    calibration: BoardCalibration,
    *,
    minimum_area: float = GAMEPLAY_OCCLUSION_MIN_AREA,
) -> BoardObstruction | None:
    """Return a hand/arm-sized board occlusion, but never classify stones as one.

    This deliberately reuses the empty-board difference detector only as a
    *temporary recognition gate* during a game.  Individual chess pieces are
    small; a hand, gripper, or arm segment covers thousands of corrected-board
    pixels.  While such an object is present the caller should keep the last
    confirmed game state and wait for a clear frame rather than treating dark
    hardware as a cluster of black stones.
    """
    if minimum_area <= 0:
        raise ValueError("大遮挡面积阈值必须为正数")
    result = detect_board_obstructions(
        frame,
        calibration,
        ignore_grid_intersections=True,
    )
    candidates = [item for item in result.obstructions if item.area >= minimum_area]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.area)


def _merge_nearby_obstructions(
    obstructions: list[BoardObstruction], *, gap: int = 10
) -> list[BoardObstruction]:
    """Join parts of one object split by the excluded printed grid lines."""
    pending = list(obstructions)
    merged: list[BoardObstruction] = []
    while pending:
        current = pending.pop()
        changed = True
        while changed:
            changed = False
            left, top = current.x, current.y
            right, bottom = current.x + current.width, current.y + current.height
            for index, other in enumerate(pending):
                other_left, other_top = other.x, other.y
                other_right, other_bottom = other.x + other.width, other.y + other.height
                if right + gap < other_left or other_right + gap < left or bottom + gap < other_top or other_bottom + gap < top:
                    continue
                current = BoardObstruction(
                    min(left, other_left),
                    min(top, other_top),
                    max(right, other_right) - min(left, other_left),
                    max(bottom, other_bottom) - min(top, other_top),
                    current.area + other.area,
                )
                pending.pop(index)
                changed = True
                break
        merged.append(current)
    return sorted(merged, key=lambda item: item.area, reverse=True)


@dataclass
class StableBoardTracker:
    """Accept a board state only after consecutive matching observations."""

    required_frames: int = 3
    _candidate: tuple[tuple[Move, Stone], ...] | None = field(default=None, init=False)
    _consecutive: int = field(default=0, init=False)
    _stable: tuple[tuple[Move, Stone], ...] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.required_frames < 1:
            raise ValueError("稳定识别帧数至少为 1")

    @property
    def consecutive_frames(self) -> int:
        return self._consecutive

    @property
    def stable_stones(self) -> dict[Move, Stone]:
        return dict(self._stable or ())

    def reset(self) -> None:
        self._candidate = None
        self._consecutive = 0
        self._stable = None

    def update(self, result: BoardVisionResult) -> bool:
        signature = tuple(sorted(result.stones.items()))
        if signature == self._candidate:
            self._consecutive += 1
        else:
            self._candidate = signature
            self._consecutive = 1

        if self._consecutive >= self.required_frames:
            changed = signature != self._stable
            self._stable = signature
            return changed
        return False


class BoardStoneRecognizer:
    """Detect black and white pieces at calibrated 13 x 13 grid points.

    The default thresholds come from replaying 168 reviewed images captured
    with a fixed camera (2026-08-29) against a per-lighting empty-board
    reference.  At the previous defaults (10.0 / 18) every real stone was found
    but black false positives were frequent; at 24.0 / 26 recall stayed at
    100% while precision rose from 88.6% to 97.1%.  See
    ``tmp/VISION_SELFCHECK_REPORT_20260922.md``.  Re-measure these on the real
    competition camera and lighting before relying on them.
    """

    def __init__(
        self,
        reference_path: Path,
        *,
        minimum_difference: float = 24.0,
        changed_pixel_difference: int = 26,
        minimum_changed_ratio: float = 0.24,
    ) -> None:
        if minimum_difference <= 0:
            raise ValueError("最小差异必须为正数")
        if not 0 <= minimum_changed_ratio <= 1:
            raise ValueError("变化像素比例必须在 0 到 1 之间")
        self.reference_path = reference_path
        self.minimum_difference = minimum_difference
        self.changed_pixel_difference = changed_pixel_difference
        self.minimum_changed_ratio = minimum_changed_ratio
        self._reference: np.ndarray | None = None

    @property
    def has_reference(self) -> bool:
        return self._reference is not None or self.reference_path.exists()

    def capture_empty_board(self, frame: np.ndarray, calibration: BoardCalibration) -> Path:
        """Save the warped current view as the known-empty board reference."""
        if frame is None or frame.size == 0:
            raise ValueError("没有可用的摄像头画面")
        reference = calibration.warp(frame)
        write_image(self.reference_path, reference)
        self._reference = reference
        return self.reference_path

    def reset_reference_cache(self) -> None:
        self._reference = None

    def detect(self, frame: np.ndarray, calibration: BoardCalibration) -> BoardVisionResult:
        """Return pieces in ``frame`` compared with the captured empty board."""
        if frame is None or frame.size == 0:
            raise ValueError("没有可用的摄像头画面")
        reference = self._load_reference()
        current = calibration.warp(frame)
        if current.shape != reference.shape:
            raise ValueError("空棋盘基准图尺寸与当前标定不一致，请重新保存基准图")

        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.int16)
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.int16)
        absolute_difference = np.abs(current_gray - reference_gray)
        signed_difference = current_gray - reference_gray
        spacing = (calibration.output_size - 1) / (calibration.board_size - 1)
        radius = max(6, int(round(spacing * 0.34)))
        detections: list[StoneDetection] = []

        for row in range(calibration.board_size):
            for column in range(calibration.board_size):
                move = (column, row)
                center_x, center_y = calibration.board_pixel(move)
                x0, y0, x1, y1 = self._patch_bounds(center_x, center_y, radius, calibration.output_size)
                mask = self._circular_mask(center_x - x0, center_y - y0, radius, x1 - x0, y1 - y0)
                difference_patch = absolute_difference[y0:y1, x0:x1][mask]
                signed_patch = signed_difference[y0:y1, x0:x1][mask]
                mean_difference = float(difference_patch.mean())
                changed_ratio = float((difference_patch >= self.changed_pixel_difference).mean())
                if (
                    mean_difference < self.minimum_difference
                    or changed_ratio < self.minimum_changed_ratio
                ):
                    continue

                # Black pieces darken the original grid point; white pieces
                # brighten it or mainly alter its local edge/shadow pattern.
                mean_signed_difference = float(signed_patch.mean())
                stone = Stone.BLACK if mean_signed_difference < -4.0 else Stone.WHITE
                confidence = min(
                    1.0,
                    0.55 * (mean_difference / (self.minimum_difference * 2.0))
                    + 0.45 * (changed_ratio / self.minimum_changed_ratio),
                )
                detections.append(StoneDetection(move, stone, confidence, mean_difference))

        return BoardVisionResult(tuple(detections))

    def measure_black_stone_centers(
        self,
        frame: np.ndarray,
        calibration: BoardCalibration,
        moves: tuple[Move, ...],
    ) -> tuple[PlacementMeasurement, ...]:
        """Measure continuous pixel offsets of known black calibration stones.

        This does not alter a calibration.  It compares the live warped image
        with the empty-board reference, then finds the dark changed component
        nearest each requested grid intersection.  It is deliberately used
        only after a controlled five-point placement where each expected
        point is known in advance.
        """
        if frame is None or frame.size == 0:
            raise ValueError("没有可用的摄像头画面")
        if not moves:
            raise ValueError("至少需要一个待测棋盘点")
        reference = self._load_reference()
        current = calibration.warp(frame)
        if current.shape != reference.shape:
            raise ValueError("空棋盘基准图尺寸与当前标定不一致，请重新保存基准图")
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.int16)
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.int16)
        darkening = np.maximum(reference_gray - current_gray, 0).astype(np.uint8)
        spacing = (calibration.output_size - 1) / (calibration.board_size - 1)
        radius = max(14, int(round(spacing * 0.72)))
        measurements: list[PlacementMeasurement] = []
        for move in moves:
            expected = calibration.board_pixel(move)
            x0, y0, x1, y1 = self._patch_bounds(expected[0], expected[1], radius, calibration.output_size)
            local = darkening[y0:y1, x0:x1]
            mask = (local >= 30).astype(np.uint8) * 255
            # The reference already contains the printed grid.  A modest
            # close joins the portions of one stone interrupted by that grid
            # without merging stones at neighbouring intersections.
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                raise ValueError(f"未在棋盘点 {move} 附近找到黑棋")
            candidates = [contour for contour in contours if cv2.contourArea(contour) >= 45.0]
            if not candidates:
                raise ValueError(f"棋盘点 {move} 的黑棋轮廓过小或不稳定")
            contour = max(candidates, key=cv2.contourArea)
            component = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(component, [contour], -1, 255, -1)
            weights = darkening[y0:y1, x0:x1].astype(np.float64) * (component > 0)
            total = float(weights.sum())
            if total <= 0:
                raise ValueError(f"棋盘点 {move} 的黑棋对比度不足")
            yy, xx = np.indices(weights.shape, dtype=np.float64)
            observed = (
                float(x0 + (xx * weights).sum() / total),
                float(y0 + (yy * weights).sum() / total),
            )
            measurements.append(PlacementMeasurement(move, expected, observed))
        return tuple(measurements)

    def _load_reference(self) -> np.ndarray:
        if self._reference is not None:
            return self._reference
        reference = read_color_image(self.reference_path)
        if reference is None:
            raise FileNotFoundError("尚未保存空棋盘基准图")
        self._reference = reference
        return reference

    @staticmethod
    def _patch_bounds(center_x: float, center_y: float, radius: int, size: int) -> tuple[int, int, int, int]:
        x0 = max(0, int(round(center_x)) - radius)
        y0 = max(0, int(round(center_y)) - radius)
        x1 = min(size, int(round(center_x)) + radius + 1)
        y1 = min(size, int(round(center_y)) + radius + 1)
        return x0, y0, x1, y1

    @staticmethod
    def _circular_mask(center_x: float, center_y: float, radius: int, width: int, height: int) -> np.ndarray:
        y, x = np.ogrid[:height, :width]
        return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
