"""Encoding rules for the ESP32 protocol, intentionally without serial I/O."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArmMoveCommand:
    x_mm: float
    y_mm: float
    z_mm: float
    retract: bool = False

    def encode(self) -> bytes:
        """Return the documented line protocol, without transmitting it."""
        return (
            f"{self.x_mm:.2f}/{self.y_mm:.2f}/{self.z_mm:.2f}/"
            f"{'True' if self.retract else 'False'}\n"
        ).encode("utf-8")
