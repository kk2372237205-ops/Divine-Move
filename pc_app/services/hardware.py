"""Explicit hardware debug session for the ESP32 UART1 protocol."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class HardwareReply:
    command: str
    response: str


class HardwareDebugError(RuntimeError):
    """Raised when the debug session cannot be opened or acknowledged."""


class HardwareCancelled(HardwareDebugError):
    """Raised after the current controller command finishes and a stop was requested."""


class HardwareDebugSession:
    """Small, explicit UART1 debug client.

    The session never sends the normal ``x/y/z/retract`` protocol. It enters
    the firmware's DEBUG mode first, then only sends commands documented by
    ``firmware_backup/main.py``.
    """

    def __init__(self, *, port: str = "COM3", baudrate: int = 9600, timeout: float = 8.0) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial = None
        self._pump_started_at: float | None = None

    @property
    def connected(self) -> bool:
        return self._serial is not None and bool(self._serial.is_open)

    def connect(self) -> HardwareReply:
        if self.connected:
            return HardwareReply("DEBUG", "DEBUG_MODE_ENTER")
        try:
            import serial
        except ImportError as error:  # pragma: no cover - exercised on an unconfigured PC
            raise HardwareDebugError(
                "未安装 pyserial，请先运行 .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
            ) from error

        try:
            serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.25,
            )
            serial_port.reset_input_buffer()
            try:
                response = self._write_and_wait(serial_port, "DEBUG", "DEBUG_MODE_ENTER")
            except HardwareDebugError as error:
                # The ESP32 debug loop accepts only movement/actuator commands.
                # When DEBUG is sent while that loop is already active, the
                # firmware correctly returns DEBUG_ERR_FORMAT.  It is a valid
                # connection state, not a failed port or a failed handshake.
                if str(error) != "DEBUG_ERR_FORMAT":
                    raise
                response = "DEBUG_MODE_ALREADY_ACTIVE"
        except Exception as error:
            try:
                serial_port.close()
            except (UnboundLocalError, AttributeError):
                pass
            raise HardwareDebugError(f"连接 {self.port} 失败：{error}") from error

        self._serial = serial_port
        return HardwareReply("DEBUG", response)

    def command(
        self,
        command: str,
        *,
        expected_prefix: str = "DEBUG_",
        timeout: float | None = None,
    ) -> HardwareReply:
        if not self.connected:
            raise HardwareDebugError("请先点击“连接并进入调试”")
        assert self._serial is not None
        try:
            response = self._write_and_wait(self._serial, command, expected_prefix, timeout=timeout)
        except HardwareDebugError:
            # A timeout may mean the motors are still completing the move.
            # Keep the session open so a late DEBUG_OK or deliberate EXIT can
            # be handled without racing the firmware.
            raise
        except Exception as error:
            self.disconnect()
            raise HardwareDebugError(f"发送 {command} 失败：{error}") from error
        return HardwareReply(command, response)

    def move(self, x: float, y: float, z: float) -> HardwareReply:
        # MoveTo waits for three motor completion states; a low pickup point
        # can take considerably longer than the short standby-height test.
        return self.command(f"{x:g}/{y:g}/{z:g}", expected_prefix="DEBUG_OK", timeout=45.0)

    def return_to(self, x: float, y: float, z: float) -> HardwareReply:
        """Retract arm joints before rotating the base toward a safe pose.

        ``RETURN`` is a deliberately separate firmware command.  Unlike the
        normal coordinate command, it maps to the firmware's ``BackTo``
        routine, which folds the two arm joints first and only then rotates
        the base.  A direct return from an extended board pose can otherwise
        sweep into the rear camera support.
        """
        if x < 0 or z <= 0:
            raise HardwareDebugError("安全回收坐标无效：需要 X>=0 且 Z>0")
        return self.command(
            f"RETURN/{x:g}/{y:g}/{z:g}",
            expected_prefix="DEBUG_RETURN_OK",
            timeout=45.0,
        )

    def move_via_approach(
        self,
        point: tuple[float, float, float],
        *,
        approach_z: float = 130.0,
    ) -> HardwareReply:
        """Reach a low point through the same-X/Y high approach point."""
        x, y, z = point
        if x < 0 or z <= 0 or approach_z <= z:
            raise HardwareDebugError("安全接近坐标无效：需要 X>=0、Z>0 且接近高度高于目标高度")
        self.move(x, y, approach_z)
        return self.move(x, y, z)

    def turntable(self, duty: int) -> HardwareReply:
        if not 1638 <= duty <= 8192:
            raise HardwareDebugError("取子机构占空比必须在 1638～8192 之间")
        return self.command(f"P3/{duty}", expected_prefix="DEBUG_P3_")

    def pump(self, state: int) -> HardwareReply:
        """Control pump/valve states; release vents after stopping suction."""
        if state not in (0, 1, 2):
            raise HardwareDebugError("气泵状态只能是 0（放子）、1（吸子）或 2（停止）")
        expected = {
            0: "DEBUG_PUMP_RELEASE_OK",
            1: "DEBUG_PUMP_HOLD_ON",
            2: "DEBUG_PUMP_STOP_OK",
        }[state]
        try:
            reply = self.command(f"PUMP/{state}", expected_prefix=expected, timeout=5.0)
            if state == 1:
                self._pump_started_at = time.monotonic()
            else:
                self._pump_started_at = None
            return reply
        except HardwareDebugError as error:
            if state == 1:
                raise HardwareDebugError(
                    "ESP32 仍是旧版吸子逻辑：未收到 DEBUG_PUMP_HOLD_ON，请先上传持续吸力补丁"
                ) from error
            raise

    def vent(self, opened: bool) -> HardwareReply:
        """Hold the valve open/closed for a stationary release diagnosis."""
        state = 3 if opened else 4
        expected = "DEBUG_PUMP_VENT_ON" if opened else "DEBUG_PUMP_VENT_OFF"
        reply = self.command(f"PUMP/{state}", expected_prefix=expected, timeout=5.0)
        if opened:
            # Firmware PUMP/3 explicitly stops p0 before it opens the vent.
            self._pump_started_at = None
        return reply

    @staticmethod
    def _raise_if_cancelled(cancel_requested: Callable[[], bool] | None) -> None:
        if cancel_requested is not None and cancel_requested():
            raise HardwareCancelled("已收到停止请求")

    @classmethod
    def _wait_or_cancel(cls, seconds: float, cancel_requested: Callable[[], bool] | None) -> None:
        """Wait in short intervals so a five-point task can stop promptly."""
        deadline = time.monotonic() + seconds
        while True:
            cls._raise_if_cancelled(cancel_requested)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.1, remaining))

    def pickup(
        self,
        pickup: tuple[float, float, float],
        *,
        approach_z: float = 130.0,
        turntable_receive_duty: int = 6000,
        turntable_pickup_duty: int = 2800,
        continuous_suction: bool = True,
        pickup_hold_seconds: float = 1.0,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> HardwareReply:
        """Run one guarded pickup cycle.

        The arm is positioned above the pickup XY before the turntable moves.
        This keeps the nozzle out of the side rod while the disk rotates and
        makes the final approach vertical. Continuous mode leaves PUMP/1 on
        for a caller-managed release; short mode stops the pump after the
        measured pickup hold and is used by the automatic placement flow.
        """
        if pickup_hold_seconds < 0:
            raise HardwareDebugError("取子吸附时间不能为负数")
        x, y, z = pickup
        if x < 0 or z <= 0:
            raise HardwareDebugError("取子坐标无效：X 必须非负，Z 必须大于 0")
        for name, duty in (("接子位", turntable_receive_duty), ("取子位", turntable_pickup_duty)):
            if not 1638 <= duty <= 8192:
                raise HardwareDebugError(f"{name}占空比必须在 1638～8192 之间")
        approach = (x, y, approach_z)
        self._raise_if_cancelled(cancel_requested)
        self.move(*approach)
        self._raise_if_cancelled(cancel_requested)
        self.turntable(turntable_receive_duty)
        self._raise_if_cancelled(cancel_requested)
        self.turntable(turntable_pickup_duty)
        self._raise_if_cancelled(cancel_requested)
        self.move(*approach)
        self._raise_if_cancelled(cancel_requested)
        self.move(x, y, z)
        try:
            self._raise_if_cancelled(cancel_requested)
            reply = self.pump(1)
        except HardwareCancelled:
            if self._pump_started_at is not None:
                try:
                    self.pump(2)
                except HardwareDebugError:
                    pass
            raise
        except HardwareDebugError:
            # Never leave the nozzle at the low pickup point after a firmware
            # capability failure or missing pump acknowledgement.
            self.move(*approach)
            raise
        try:
            if continuous_suction:
                self._wait_or_cancel(0.8, cancel_requested)
            else:
                # Match the supplier's stable sequence: build vacuum briefly,
                # then stop the pump before the arm begins its transit.
                self._wait_or_cancel(pickup_hold_seconds, cancel_requested)
                self._raise_if_cancelled(cancel_requested)
                self.pump(2)
            self._raise_if_cancelled(cancel_requested)
            self.move(*approach)
        except HardwareCancelled:
            # A standalone pickup is also cancellable, outside of a complete
            # pick-and-place flow, so it must stop the suction itself.
            if self._pump_started_at is not None:
                try:
                    self.pump(2)
                except HardwareDebugError:
                    pass
            raise
        return reply

    def pick_and_place(
        self,
        pickup: tuple[float, float, float],
        target: tuple[float, float, float],
        standby: tuple[float, float, float],
        *,
        safe_z: float = 130.0,
        turntable_receive_duty: int = 6000,
        turntable_pickup_duty: int = 2800,
        minimum_suction_seconds: float = 10.0,
        release_settle_seconds: float = 1.5,
        continuous_suction: bool = False,
        pickup_hold_seconds: float = 1.0,
        release_open_seconds: float = 1.0,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> HardwareReply:
        """Execute one guarded foundation-stage pickup and placement."""
        for name, point in (("取子点", pickup), ("落子点", target), ("待命点", standby)):
            if len(point) != 3 or point[0] < 0 or point[2] <= 0:
                raise HardwareDebugError(f"{name}坐标无效：需要 X>=0 且 Z>0")
        if safe_z <= max(pickup[2], target[2]):
            raise HardwareDebugError("安全高度必须高于取子点和落子点")
        if minimum_suction_seconds < 0:
            raise HardwareDebugError("最短吸附时间不能为负数")
        if release_settle_seconds < 0:
            raise HardwareDebugError("落子释放等待时间不能为负数")
        if pickup_hold_seconds < 0 or release_open_seconds < 0:
            raise HardwareDebugError("取放执行器等待时间不能为负数")

        recovery_xy = (pickup[0], pickup[1])
        vent_open = False
        started = time.monotonic()
        try:
            self._raise_if_cancelled(cancel_requested)
            self.pickup(
                pickup,
                approach_z=safe_z,
                turntable_receive_duty=turntable_receive_duty,
                turntable_pickup_duty=turntable_pickup_duty,
                continuous_suction=continuous_suction,
                pickup_hold_seconds=pickup_hold_seconds,
                cancel_requested=cancel_requested,
            )
            recovery_xy = (target[0], target[1])
            self._raise_if_cancelled(cancel_requested)
            self.move(target[0], target[1], safe_z)
            self._raise_if_cancelled(cancel_requested)
            self.move(*target)
            if continuous_suction and self._pump_started_at is not None:
                remaining = minimum_suction_seconds - (time.monotonic() - self._pump_started_at)
                if remaining > 0:
                    self._wait_or_cancel(remaining, cancel_requested)
                self._raise_if_cancelled(cancel_requested)
                self.pump(0)
            else:
                # Supplier-validated release: open the vent only after the
                # pump has already stopped at the pickup point.
                self.vent(True)
                vent_open = True
                if release_open_seconds > 0:
                    self._wait_or_cancel(release_open_seconds, cancel_requested)
                # Keep the vent open while lifting vertically. This prevents
                # residual vacuum from dragging a piece during the first lift.
                self._raise_if_cancelled(cancel_requested)
                self.move(target[0], target[1], safe_z)
            # The valve acknowledgement means the vent cycle finished, not
            # that the piece has visibly settled. Keep the nozzle over the
            # target at the safe height for a short, deterministic window.
            if release_settle_seconds > 0:
                self._wait_or_cancel(release_settle_seconds, cancel_requested)
            self._raise_if_cancelled(cancel_requested)
            self.vent(False)
            vent_open = False
            self._raise_if_cancelled(cancel_requested)
            self.return_to(*standby)
            self._raise_if_cancelled(cancel_requested)
            self.turntable(turntable_receive_duty)
        except HardwareCancelled:
            # Do not send a recovery move after the user has requested a stop.
            # The ESP32 finishes the command it is currently executing; then
            # the pump is stopped and no further arm positions are commanded.
            if vent_open:
                try:
                    self.vent(False)
                except HardwareDebugError:
                    pass
            if self._pump_started_at is not None:
                try:
                    self.pump(2)
                except HardwareDebugError:
                    pass
            raise
        except Exception:
            if vent_open:
                try:
                    self.vent(False)
                except Exception:
                    pass
            try:
                if self._pump_started_at is not None:
                    self.pump(2)
            except Exception:
                pass
            try:
                self.move(recovery_xy[0], recovery_xy[1], safe_z)
            except Exception:
                pass
            raise
        elapsed = time.monotonic() - started
        return HardwareReply("FOUNDATION_PICK_PLACE", f"FOUNDATION_PLACE_OK:{elapsed:.2f}s")

    def return_piece_to_pickup(
        self,
        pickup: tuple[float, float, float],
        standby: tuple[float, float, float],
        *,
        safe_z: float = 130.0,
        turntable_receive_duty: int = 6000,
        release_z_offset: float = 2.0,
        release_point: tuple[float, float, float] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> HardwareReply:
        """Return a held piece to the pickup point, release it, and stand by.

        This is the manual counterpart to ``pickup``.  It deliberately does
        not move toward a board point: it returns vertically to the saved
        saved release coordinate, or at Z+2 above the pickup coordinate for
        older calibrations, then returns directly to the standby position.
        """
        for name, point in (("取子点", pickup), ("待命点", standby)):
            if len(point) != 3 or point[0] < 0 or point[2] <= 0:
                raise HardwareDebugError(f"{name}坐标无效：需要 X>=0 且 Z>0")
        if safe_z <= pickup[2]:
            raise HardwareDebugError("安全高度必须高于取子点")
        if release_z_offset < 0:
            raise HardwareDebugError("放子高度偏移不能为负数")
        if not 1638 <= turntable_receive_duty <= 8192:
            raise HardwareDebugError("接子位占空比必须在 1638～8192 之间")

        x, y, z = pickup
        if release_point is None:
            release_point = (x, y, z + release_z_offset)
        if len(release_point) != 3 or release_point[0] < 0 or release_point[2] <= 0:
            raise HardwareDebugError("放子坐标无效：需要 X>=0 且 Z>0")
        release_point = tuple(float(value) for value in release_point)
        if release_point[2] >= safe_z:
            raise HardwareDebugError("放子高度必须低于安全高度")
        approach = (release_point[0], release_point[1], safe_z)
        try:
            self._raise_if_cancelled(cancel_requested)
            self.move(*approach)
            self._raise_if_cancelled(cancel_requested)
            self.move(*release_point)
            self._raise_if_cancelled(cancel_requested)
            # PUMP/0 already opens the valve, stops suction, and holds its
            # own pressure-release period in firmware. Do not add a second
            # PUMP/3/PUMP/4 vent cycle here.
            reply = self.pump(0)
            self._raise_if_cancelled(cancel_requested)
            self.move(*approach)
            if tuple(standby) != approach:
                self._raise_if_cancelled(cancel_requested)
                self.return_to(*standby)
            self._raise_if_cancelled(cancel_requested)
            self.turntable(turntable_receive_duty)
            return reply
        except HardwareCancelled:
            if self._pump_started_at is not None:
                try:
                    self.pump(2)
                except HardwareDebugError:
                    pass
            raise
        except Exception:
            try:
                if self._pump_started_at is not None:
                    self.pump(2)
            except Exception:
                pass
            try:
                self.move(*approach)
            except Exception:
                pass
            raise

    def reset(self) -> HardwareReply:
        return self.command("RESET", expected_prefix="DEBUG_RESET_", timeout=60.0)

    def disconnect(self) -> None:
        serial_port, self._serial = self._serial, None
        self._pump_started_at = None
        if serial_port is not None:
            try:
                if serial_port.is_open:
                    self._write_and_wait(serial_port, "EXIT", "DEBUG_MODE_EXIT", timeout=3.0)
            except Exception:
                # Closing must still release the selected port if the board has already reset.
                pass
            finally:
                serial_port.close()

    def _write_and_wait(
        self,
        serial_port,
        command: str,
        expected_prefix: str,
        *,
        timeout: float | None = None,
    ) -> str:
        # Use a real LF byte. Sending the two characters "\\n" makes the
        # firmware parse the entire string as a coordinate and raises ValueError.
        serial_port.write(command.encode("ascii") + b"\n")
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        replies: list[str] = []
        while time.monotonic() < deadline:
            raw = serial_port.readline()
            if raw:
                response = raw.decode("utf-8", errors="replace").strip()
                replies.append(response)
                if response.startswith("DEBUG_ERR"):
                    raise HardwareDebugError(response)
                if response.startswith(expected_prefix):
                    return response
            else:
                time.sleep(0.02)
        detail = ", ".join(replies[-3:]) if replies else "无回执"
        raise HardwareDebugError(f"等待 {command} 回执超时：{detail}")
