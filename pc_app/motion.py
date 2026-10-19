"""Offline board-to-arm mapping and safe-to-review motion plans.

This module deliberately contains no serial implementation.  A plan is an
auditable description of the robot's two separate rotation mechanisms and the
arm sequence; it becomes executable only after a separate, user-authorized
hardware integration phase.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .board import BOARD_SIZE, Move
from .coordinates import board_to_competition


ArmPoint = tuple[float, float, float]


class MotionStepKind(str, Enum):
    TURNTABLE_RECEIVE = "turntable_receive"
    TURNTABLE_PRESENT = "turntable_present"
    ARM_TO_PICKUP = "arm_to_pickup"
    VACUUM_PICK = "vacuum_pick"
    ARM_LIFT = "arm_lift"
    ARM_TO_TARGET = "arm_to_target"
    VALVE_RELEASE = "valve_release"
    ARM_RETURN = "arm_return"


@dataclass(frozen=True)
class ArmBoardCalibration:
    """Measured arm coordinates for the four board intersections.

    The names use the internal camera board system: ``(column, row)`` with
    ``(0, 0)`` in the calibrated image's upper-left.  Every value must be
    measured on the actual machine; no firmware backup number is assumed here.
    """

    point_0_0: ArmPoint
    point_12_0: ArmPoint
    point_12_12: ArmPoint
    point_0_12: ArmPoint
    pickup_point: ArmPoint
    standby_point: ArmPoint
    safe_z: float
    board_size: int = BOARD_SIZE

    def __post_init__(self) -> None:
        if self.board_size < 2:
            raise ValueError("棋盘至少需要两条网格线")
        for name, point in (
            ("point_0_0", self.point_0_0),
            ("point_12_0", self.point_12_0),
            ("point_12_12", self.point_12_12),
            ("point_0_12", self.point_0_12),
            ("pickup_point", self.pickup_point),
            ("standby_point", self.standby_point),
        ):
            if len(point) != 3:
                raise ValueError(f"{name} 必须含有 X/Y/Z 三个数值")

    def board_to_arm(self, move: Move) -> ArmPoint:
        """Map a grid intersection to arm space using all four measured corners."""
        column, row = move
        if not 0 <= column < self.board_size or not 0 <= row < self.board_size:
            raise ValueError(f"棋盘坐标越界: {move}")
        divisor = float(self.board_size - 1)
        u, v = column / divisor, row / divisor
        weights = (
            ((1 - u) * (1 - v), self.point_0_0),
            (u * (1 - v), self.point_12_0),
            (u * v, self.point_12_12),
            ((1 - u) * v, self.point_0_12),
        )
        return tuple(sum(weight * point[index] for weight, point in weights) for index in range(3))  # type: ignore[return-value]

    def target_safe_point(self, move: Move) -> ArmPoint:
        x, y, _ = self.board_to_arm(move)
        return x, y, self.safe_z

    def to_json(self, path: Path) -> None:
        payload = {
            "point_0_0": list(self.point_0_0),
            "point_12_0": list(self.point_12_0),
            "point_12_12": list(self.point_12_12),
            "point_0_12": list(self.point_0_12),
            "pickup_point": list(self.pickup_point),
            "standby_point": list(self.standby_point),
            "safe_z": self.safe_z,
            "board_size": self.board_size,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "ArmBoardCalibration":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            point_0_0=tuple(float(value) for value in payload["point_0_0"]),
            point_12_0=tuple(float(value) for value in payload["point_12_0"]),
            point_12_12=tuple(float(value) for value in payload["point_12_12"]),
            point_0_12=tuple(float(value) for value in payload["point_0_12"]),
            pickup_point=tuple(float(value) for value in payload["pickup_point"]),
            standby_point=tuple(float(value) for value in payload["standby_point"]),
            safe_z=float(payload["safe_z"]),
            board_size=int(payload.get("board_size", BOARD_SIZE)),
        )


@dataclass(frozen=True)
class MotionStep:
    kind: MotionStepKind
    label: str
    target: ArmPoint | None = None


@dataclass(frozen=True)
class MotionPlan:
    move: Move
    competition_coordinate: tuple[int, int]
    steps: tuple[MotionStep, ...]
    calibrated: bool

    def display_lines(self) -> tuple[str, ...]:
        lines = [
            f"机器人黑棋建议：竞赛坐标 {self.competition_coordinate}（内部 {self.move}）",
            "仅干运行计划，当前没有串口发送功能。",
        ]
        for index, step in enumerate(self.steps, start=1):
            suffix = "" if step.target is None else f" 目标 {self._format_point(step.target)}"
            lines.append(f"{index}. {step.label}{suffix}")
        if not self.calibrated:
            lines.append("机械臂坐标尚未实测标定：目标位置只表示棋盘格，不可执行。")
        return tuple(lines)

    @staticmethod
    def _format_point(point: ArmPoint) -> str:
        return "(" + ", ".join(f"{value:.1f}" for value in point) + ")"


class MotionPlanner:
    """Build the required pick-and-place sequence without driving hardware."""

    def __init__(self, calibration: ArmBoardCalibration | None = None) -> None:
        self.calibration = calibration

    def plan_black_move(self, move: Move) -> MotionPlan:
        competition_coordinate = board_to_competition(move)
        if self.calibration is None:
            return MotionPlan(
                move=move,
                competition_coordinate=competition_coordinate,
                calibrated=False,
                steps=self._uncalibrated_steps(),
            )

        calibration = self.calibration
        pickup_safe = (calibration.pickup_point[0], calibration.pickup_point[1], calibration.safe_z)
        target = calibration.board_to_arm(move)
        target_safe = calibration.target_safe_point(move)
        return MotionPlan(
            move=move,
            competition_coordinate=competition_coordinate,
            calibrated=True,
            steps=(
                MotionStep(MotionStepKind.TURNTABLE_RECEIVE, "取子转盘保持 0 度，圆孔在储子机构下方接子"),
                MotionStep(MotionStepKind.TURNTABLE_PRESENT, "取子转盘转到 90 度，将棋子送到机械臂取子位"),
                MotionStep(MotionStepKind.ARM_TO_PICKUP, "机械臂移动到取子位上方", pickup_safe),
                MotionStep(MotionStepKind.ARM_TO_PICKUP, "机械臂下降到取子位", calibration.pickup_point),
                MotionStep(MotionStepKind.VACUUM_PICK, "气泵吸取棋子并确认棋子未掉落"),
                MotionStep(MotionStepKind.ARM_LIFT, "机械臂抬升到安全高度", pickup_safe),
                MotionStep(MotionStepKind.ARM_TO_TARGET, "机械臂移动到目标棋格上方", target_safe),
                MotionStep(MotionStepKind.ARM_TO_TARGET, "机械臂下降到目标棋格", target),
                MotionStep(MotionStepKind.VALVE_RELEASE, "打开电磁阀放置棋子"),
                MotionStep(MotionStepKind.ARM_LIFT, "机械臂抬升到安全高度", target_safe),
                MotionStep(MotionStepKind.ARM_RETURN, "机械臂返回待命位", calibration.standby_point),
                MotionStep(MotionStepKind.TURNTABLE_RECEIVE, "取子转盘回到 0 度，等待下一颗棋子"),
            ),
        )

    @staticmethod
    def _uncalibrated_steps() -> tuple[MotionStep, ...]:
        return (
            MotionStep(MotionStepKind.TURNTABLE_RECEIVE, "取子转盘保持 0 度，圆孔在储子机构下方接子"),
            MotionStep(MotionStepKind.TURNTABLE_PRESENT, "取子转盘转到 90 度，将棋子送到机械臂取子位"),
            MotionStep(MotionStepKind.ARM_TO_PICKUP, "机械臂移动并下降到已实测的取子位"),
            MotionStep(MotionStepKind.VACUUM_PICK, "气泵吸取棋子并确认棋子未掉落"),
            MotionStep(MotionStepKind.ARM_LIFT, "机械臂抬升到安全高度"),
            MotionStep(MotionStepKind.ARM_TO_TARGET, "机械臂移动并下降到建议目标棋格"),
            MotionStep(MotionStepKind.VALVE_RELEASE, "打开电磁阀放置棋子"),
            MotionStep(MotionStepKind.ARM_RETURN, "机械臂抬升并返回待命位"),
            MotionStep(MotionStepKind.TURNTABLE_RECEIVE, "取子转盘回到 0 度，等待下一颗棋子"),
        )
