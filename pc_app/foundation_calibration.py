"""Measured coordinates used by the foundation-stage five-point test."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path


ArmPoint = tuple[float, float, float]

FOUNDATION_TARGETS: tuple[tuple[str, tuple[int, int], str], ...] = (
    ("upper_left", (3, 3), "左上黑点"),
    ("upper_right", (9, 3), "右上黑点"),
    ("center", (6, 6), "中心黑点"),
    ("lower_left", (3, 9), "左下黑点"),
    ("lower_right", (9, 9), "右下黑点"),
)


def target_label(key: str) -> str:
    for target_key, board_move, label in FOUNDATION_TARGETS:
        if target_key == key:
            return f"{label} {board_move}"
    raise KeyError(key)


@dataclass
class FoundationHardwareCalibration:
    pickup_point: ArmPoint = (215.0, 189.0, 52.0)
    release_point: ArmPoint = (220.0, 187.0, 56.0)
    standby_point: ArmPoint = (215.0, 189.0, 130.0)
    safe_z: float = 130.0
    turntable_receive_duty: int = 6000
    # The physical 90-degree pickup position was measured on this robot at
    # duty 2800. Keep the default aligned with the stored field calibration
    # so a newly created calibration does not revert to the old 3150 value.
    turntable_pickup_duty: int = 2800
    targets: dict[str, ArmPoint] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pickup_point = self._validated_point(self.pickup_point, "取子点")
        self.release_point = self._validated_point(self.release_point, "放子点")
        self.standby_point = self._validated_point(self.standby_point, "待命点")
        if self.safe_z <= 0:
            raise ValueError("安全高度必须大于 0")
        self.turntable_receive_duty = self._validated_turntable_duty(self.turntable_receive_duty, "接子位")
        self.turntable_pickup_duty = self._validated_turntable_duty(self.turntable_pickup_duty, "取子位")
        allowed = {key for key, _, _ in FOUNDATION_TARGETS}
        unknown = set(self.targets) - allowed
        if unknown:
            raise ValueError(f"未知黑点标识: {sorted(unknown)}")
        self.targets = {
            key: self._validated_point(point, target_label(key))
            for key, point in self.targets.items()
        }

    @staticmethod
    def _validated_point(point: ArmPoint, name: str) -> ArmPoint:
        values = tuple(float(value) for value in point)
        if len(values) != 3:
            raise ValueError(f"{name}必须包含 X/Y/Z")
        if values[0] < 0 or values[2] <= 0:
            raise ValueError(f"{name}要求 X>=0 且 Z>0")
        return values  # type: ignore[return-value]

    @staticmethod
    def _validated_turntable_duty(value: int, name: str) -> int:
        try:
            duty = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name}占空比必须是整数") from error
        if not 1638 <= duty <= 8192:
            raise ValueError(f"{name}占空比必须在 1638～8192 之间")
        return duty

    @property
    def complete(self) -> bool:
        return all(key in self.targets for key, _, _ in FOUNDATION_TARGETS)

    @property
    def missing_labels(self) -> tuple[str, ...]:
        return tuple(label for key, _, label in FOUNDATION_TARGETS if key not in self.targets)

    def set_target(self, key: str, point: ArmPoint) -> None:
        target_label(key)
        self.targets[key] = self._validated_point(point, target_label(key))

    def ordered_targets(self) -> tuple[tuple[str, ArmPoint], ...]:
        if not self.complete:
            raise ValueError("五个黑点尚未全部标定: " + "、".join(self.missing_labels))
        return tuple((key, self.targets[key]) for key, _, _ in FOUNDATION_TARGETS)

    def to_json(self, path: Path) -> None:
        payload = {
            "pickup_point": list(self.pickup_point),
            "release_point": list(self.release_point),
            "standby_point": list(self.standby_point),
            "safe_z": self.safe_z,
            "turntable_receive_duty": self.turntable_receive_duty,
            "turntable_pickup_duty": self.turntable_pickup_duty,
            "targets": {key: list(point) for key, point in self.targets.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "FoundationHardwareCalibration":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            pickup_point=tuple(payload.get("pickup_point", (215.0, 189.0, 52.0))),
            release_point=tuple(payload.get("release_point", (220.0, 187.0, 56.0))),
            standby_point=tuple(payload.get("standby_point", (215.0, 189.0, 130.0))),
            safe_z=float(payload.get("safe_z", 130.0)),
            turntable_receive_duty=payload.get("turntable_receive_duty", 6000),
            turntable_pickup_duty=payload.get("turntable_pickup_duty", 2800),
            targets={key: tuple(point) for key, point in payload.get("targets", {}).items()},
        )
