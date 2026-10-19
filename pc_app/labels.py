"""Project calibrated board-grid coordinates into YOLO image annotations.

The helper accepts the same internal board coordinates used by the vision
pipeline and can optionally convert competition coordinates first.  Boxes are
projected through the inverse perspective calibration so labels remain valid
for the original camera image instead of only the 520 x 520 warped preview.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Literal

import cv2
import numpy as np

from .board import BOARD_SIZE, Move
from .calibration import BoardCalibration
from .coordinates import competition_to_board
from .dataset import YOLO_CLASS_NAMES


CoordinateSystem = Literal["board", "competition"]


@dataclass(frozen=True)
class GridAnnotation:
    """One projected stone box, retaining its source grid coordinate."""

    class_id: int
    move: Move
    pixel_box: tuple[float, float, float, float]
    yolo: tuple[int, float, float, float, float]


@dataclass(frozen=True)
class BoxOverride:
    """A human-approved box centre and radius in the calibrated board image."""

    move: Move
    center: tuple[float, float]
    radius: float

    def __post_init__(self) -> None:
        if len(self.center) != 2 or not all(np.isfinite(value) for value in (*self.center, self.radius)):
            raise ValueError("标注框中心和半径必须是有限数值")
        if self.radius <= 0:
            raise ValueError("标注框半径必须大于 0")

    def as_json(self) -> dict[str, object]:
        return {"move": list(self.move), "center": list(self.center), "radius": self.radius}


def box_overrides_from_json(values: object) -> tuple[BoxOverride, ...]:
    """Read optional per-stone box corrections from a truth record."""

    if values is None:
        return ()
    if not isinstance(values, list):
        raise ValueError("box_overrides 必须是列表")
    overrides: list[BoxOverride] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("box_overrides 中的每项必须是对象")
        move, center, radius = value.get("move"), value.get("center"), value.get("radius")
        if not isinstance(move, list) or len(move) != 2 or not isinstance(center, list) or len(center) != 2:
            raise ValueError("box_overrides 的 move 和 center 必须各有两个数值")
        overrides.append(
            BoxOverride(
                move=(int(move[0]), int(move[1])),
                center=(float(center[0]), float(center[1])),
                radius=float(radius),
            )
        )
    if len({item.move for item in overrides}) != len(overrides):
        raise ValueError("同一棋子不能有多个框修正")
    return tuple(overrides)


def fit_grid_box_overrides(
    image: np.ndarray,
    calibration: BoardCalibration,
    moves: Iterable[Move],
    *,
    max_center_offset: float = 12.0,
    min_radius: float = 12.0,
    max_radius: float = 21.0,
) -> tuple[BoxOverride, ...]:
    """Find confirmed stones' circular outlines near their reviewed grid points.

    The reviewed grid coordinates and classes remain authoritative. This only
    moves a box to a nearby visible stone outline, and leaves a move unchanged
    when an unambiguous circle cannot be found.
    """

    if image is None or image.size == 0:
        raise ValueError("不能为没有像素的图片贴合标注框")
    if max_center_offset <= 0 or min_radius <= 0 or max_radius < min_radius:
        raise ValueError("标注框贴合参数无效")
    warped = calibration.warp(image)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=20.0,
        param1=80.0,
        param2=18.0,
        minRadius=int(min_radius),
        maxRadius=int(max_radius),
    )
    if circles is None:
        return ()
    candidates = np.asarray(circles[0], dtype=np.float32)
    spacing = (calibration.output_size - 1) / (calibration.board_size - 1)
    safe_radius = spacing / 2 - 0.5
    used: set[int] = set()
    fitted: list[BoxOverride] = []
    for move in moves:
        expected_x, expected_y = calibration.board_pixel(move)
        distances = np.hypot(candidates[:, 0] - expected_x, candidates[:, 1] - expected_y)
        choices = sorted((float(distance), index) for index, distance in enumerate(distances) if index not in used)
        if not choices or choices[0][0] > max_center_offset:
            continue
        _distance, index = choices[0]
        center_x, center_y, detected_radius = (float(value) for value in candidates[index])
        used.add(index)
        fitted.append(
            BoxOverride(
                move=move,
                center=(center_x, center_y),
                radius=min(safe_radius, detected_radius + 1.0),
            )
        )
    return tuple(fitted)


def parse_grid_coordinates(
    text: str,
    *,
    board_size: int = BOARD_SIZE,
    coordinate_system: CoordinateSystem = "board",
) -> tuple[Move, ...]:
    """Parse ``x,y`` pairs separated by spaces or semicolons.

    ``board`` coordinates are zero-based ``(column,row)`` pairs.  In
    ``competition`` mode pairs are one-based and use the contest's upper-right
    origin; they are converted to the internal image coordinate system.
    """

    if coordinate_system not in ("board", "competition"):
        raise ValueError("coordinate_system 必须是 board 或 competition")
    if board_size < 2:
        raise ValueError("棋盘至少需要两条网格线")
    if not text.strip():
        return ()

    normalized = text.replace("，", ",").replace("；", ";")
    tokens = [token for token in re.split(r"[;\s]+", normalized) if token]
    moves: list[Move] = []
    seen: set[Move] = set()
    for token in tokens:
        match = re.fullmatch(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", token)
        if match is None:
            raise ValueError(f"无法解析棋盘坐标：{token!r}，应写成 x,y")
        coordinate = (int(match.group(1)), int(match.group(2)))
        move = (
            competition_to_board(coordinate, board_size)
            if coordinate_system == "competition"
            else coordinate
        )
        column, row = move
        if not 0 <= column < board_size or not 0 <= row < board_size:
            raise ValueError(f"棋盘坐标越界：{coordinate}")
        if move in seen:
            raise ValueError(f"棋盘坐标重复：{coordinate}")
        seen.add(move)
        moves.append(move)
    return tuple(moves)


def project_grid_annotations(
    calibration: BoardCalibration,
    image_shape: tuple[int, ...],
    black_moves: Iterable[Move] = (),
    white_moves: Iterable[Move] = (),
    *,
    radius: float = 16.0,
    box_overrides: Iterable[BoxOverride] = (),
) -> tuple[GridAnnotation, ...]:
    """Project board intersections and square stone boxes into image pixels.

    ``radius`` is measured in calibrated warped-image pixels.  The default is
    suitable for the current board (grid spacing is about 43 px in the 520 px
    calibration), while the CLI exposes it for a manual fit check.  Boxes are
    clipped to the camera frame and normalized for YOLO.
    """

    if len(image_shape) < 2:
        raise ValueError("image_shape 必须至少包含高和宽")
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    if image_height < 1 or image_width < 1:
        raise ValueError("图像尺寸必须为正数")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("标注框半径必须是正数")
    spacing = (calibration.output_size - 1) / (calibration.board_size - 1)
    if radius * 2 >= spacing:
        raise ValueError(f"标注框半径过大：必须小于半个网格间距 {spacing / 2:.2f}")

    black = tuple(black_moves)
    white = tuple(white_moves)
    all_moves = list(black) + list(white)
    if not all_moves:
        raise ValueError("至少需要一个黑棋或白棋坐标")
    if len(set(all_moves)) != len(all_moves):
        raise ValueError("黑棋和白棋坐标不能重复")

    for move in all_moves:
        column, row = move
        if not 0 <= column < calibration.board_size or not 0 <= row < calibration.board_size:
            raise ValueError(f"棋盘坐标越界：{move}")

    overrides = tuple(box_overrides)
    override_by_move = {item.move: item for item in overrides}
    if len(override_by_move) != len(overrides):
        raise ValueError("同一棋子不能有多个框修正")
    unknown_overrides = set(override_by_move) - set(all_moves)
    if unknown_overrides:
        raise ValueError(f"存在不属于当前棋局的框修正：{sorted(unknown_overrides)}")
    for override in override_by_move.values():
        if not (0 <= override.center[0] < calibration.output_size and 0 <= override.center[1] < calibration.output_size):
            raise ValueError(f"框修正中心越出校正棋盘：{override.move}")
        if override.radius * 2 >= spacing:
            raise ValueError(f"框修正半径过大：{override.move}")

    inverse = np.linalg.inv(calibration.matrix).astype(np.float32)
    projected: list[GridAnnotation] = []
    for class_id, moves in ((0, black), (1, white)):
        for move in moves:
            override = override_by_move.get(move)
            center_x, center_y = override.center if override is not None else calibration.board_pixel(move)
            item_radius = override.radius if override is not None else radius
            corners = np.asarray(
                [[
                    (center_x - item_radius, center_y - item_radius),
                    (center_x + item_radius, center_y - item_radius),
                    (center_x + item_radius, center_y + item_radius),
                    (center_x - item_radius, center_y + item_radius),
                ]],
                dtype=np.float32,
            )
            source = cv2.perspectiveTransform(corners, inverse)[0]
            left = float(np.clip(source[:, 0].min(), 0, image_width))
            top = float(np.clip(source[:, 1].min(), 0, image_height))
            right = float(np.clip(source[:, 0].max(), 0, image_width))
            bottom = float(np.clip(source[:, 1].max(), 0, image_height))
            if right <= left or bottom <= top:
                raise ValueError(f"坐标 {move} 投影后没有有效标注框")
            center_x_px = (left + right) / 2
            center_y_px = (top + bottom) / 2
            width_px = right - left
            height_px = bottom - top
            yolo = (
                class_id,
                center_x_px / image_width,
                center_y_px / image_height,
                width_px / image_width,
                height_px / image_height,
            )
            projected.append(
                GridAnnotation(
                    class_id=class_id,
                    move=move,
                    pixel_box=(left, top, right, bottom),
                    yolo=yolo,
                )
            )
    return tuple(projected)


def draw_grid_annotations(image: np.ndarray, annotations: Iterable[GridAnnotation]) -> np.ndarray:
    """Return a preview image with class-coloured boxes and grid coordinates."""

    if image is None or image.size == 0:
        raise ValueError("不能在空图像上绘制标注预览")
    preview = image.copy()
    colors = ((0, 0, 255), (255, 120, 0))  # black/red, white/cyan-ish in BGR
    for annotation in annotations:
        left, top, right, bottom = annotation.pixel_box
        p1 = (int(round(left)), int(round(top)))
        p2 = (int(round(right)), int(round(bottom)))
        color = colors[annotation.class_id]
        cv2.rectangle(preview, p1, p2, color, 2, cv2.LINE_AA)
        center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.circle(preview, center, 3, color, -1, cv2.LINE_AA)
        name = YOLO_CLASS_NAMES[annotation.class_id]
        label = f"{name} {annotation.move[0]},{annotation.move[1]}"
        text_origin = (p1[0], max(16, p1[1] - 5))
        cv2.putText(preview, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return preview


__all__ = [
    "CoordinateSystem",
    "BoxOverride",
    "GridAnnotation",
    "box_overrides_from_json",
    "draw_grid_annotations",
    "fit_grid_box_overrides",
    "parse_grid_coordinates",
    "project_grid_annotations",
]
