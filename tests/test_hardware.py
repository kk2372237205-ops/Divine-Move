from pc_app.services.hardware import HardwareCancelled, HardwareDebugError, HardwareDebugSession
import sys
from types import SimpleNamespace


def test_debug_session_commands_use_real_line_feed() -> None:
    class FakeSerial:
        def __init__(self) -> None:
            self.written = b""

        def write(self, payload: bytes) -> None:
            self.written += payload

        def readline(self) -> bytes:
            return b"DEBUG_OK\n"

    session = HardwareDebugSession()
    serial_port = FakeSerial()
    assert session._write_and_wait(serial_port, "206/183/110", "DEBUG_") == "DEBUG_OK"
    assert serial_port.written == b"206/183/110\n"


def test_return_to_uses_dedicated_safe_return_command() -> None:
    class FakeSerial:
        is_open = True

        def __init__(self) -> None:
            self.written = b""

        def write(self, payload: bytes) -> None:
            self.written += payload

        def readline(self) -> bytes:
            return b"DEBUG_RETURN_OK\n"

    session = HardwareDebugSession()
    fake = FakeSerial()
    session._serial = fake

    reply = session.return_to(215.0, 189.0, 130.0)

    assert reply.response == "DEBUG_RETURN_OK"
    assert fake.written == b"RETURN/215/189/130\n"


def test_turntable_duty_range_is_checked() -> None:
    session = HardwareDebugSession()
    try:
        session.turntable(1600)
    except HardwareDebugError as error:
        assert "1638" in str(error)
    else:
        raise AssertionError("out-of-range duty must be rejected before serial I/O")


def test_turntable_accepts_calibrated_duty_below_3150() -> None:
    class FakeSerial:
        is_open = True

        def __init__(self) -> None:
            self.written = b""

        def write(self, payload: bytes) -> None:
            self.written += payload

        def readline(self) -> bytes:
            return b"DEBUG_P3_OK:2775\n"

    session = HardwareDebugSession()
    fake = FakeSerial()
    session._serial = fake

    reply = session.turntable(2775)

    assert reply.response == "DEBUG_P3_OK:2775"
    assert fake.written == b"P3/2775\n"


def test_pump_state_range_is_checked() -> None:
    session = HardwareDebugSession()
    try:
        session.pump(3)
    except HardwareDebugError:
        pass
    else:
        raise AssertionError("invalid pump state must be rejected before serial I/O")


def test_debug_error_is_not_waited_until_timeout() -> None:
    class FakeSerial:
        def write(self, payload: bytes) -> None:
            pass

        def readline(self) -> bytes:
            return b"DEBUG_ERR_FORMAT\n"

    session = HardwareDebugSession(timeout=30)
    try:
        session._write_and_wait(FakeSerial(), "DEBUG", "DEBUG_MODE_ENTER")
    except HardwareDebugError as error:
        assert str(error) == "DEBUG_ERR_FORMAT"
    else:
        raise AssertionError("firmware errors must be surfaced immediately")


def test_connect_accepts_firmware_that_is_already_in_debug_mode(monkeypatch) -> None:
    class FakeSerial:
        is_open = True

        def reset_input_buffer(self) -> None:
            pass

        def write(self, _payload: bytes) -> None:
            pass

        def readline(self) -> bytes:
            return b"DEBUG_ERR_FORMAT\n"

        def close(self) -> None:
            self.is_open = False

    fake_port = FakeSerial()
    fake_serial_module = SimpleNamespace(
        Serial=lambda **_kwargs: fake_port,
        EIGHTBITS=8,
        PARITY_NONE="N",
        STOPBITS_ONE=1,
    )
    monkeypatch.setitem(sys.modules, "serial", fake_serial_module)

    session = HardwareDebugSession(port="COM6")
    reply = session.connect()

    assert session.connected
    assert reply.response == "DEBUG_MODE_ALREADY_ACTIVE"


def test_pickup_rejects_negative_x_without_serial_connection() -> None:
    session = HardwareDebugSession()
    try:
        session.pickup((-1.0, 189.0, 55.0))
    except HardwareDebugError as error:
        assert "X" in str(error)
    else:
        raise AssertionError("pickup must reject negative X before serial I/O")


def test_pump_state_has_explicit_continuous_suction_ack() -> None:
    session = HardwareDebugSession()
    class FakeSerial:
        is_open = True

        def __init__(self) -> None:
            self.written = b""

        def write(self, payload: bytes) -> None:
            self.written += payload

        def readline(self) -> bytes:
            return b"DEBUG_PUMP_HOLD_ON\n"

    fake = FakeSerial()
    session._serial = fake
    reply = session.pump(1)
    assert reply.response == "DEBUG_PUMP_HOLD_ON"
    assert fake.written == b"PUMP/1\n"


def test_vent_diagnostic_commands_are_explicit() -> None:
    class FakeSerial:
        is_open = True

        def __init__(self) -> None:
            self.written = b""

        def write(self, payload: bytes) -> None:
            self.written += payload

        def readline(self) -> bytes:
            return b"DEBUG_PUMP_VENT_ON\n"

    fake = FakeSerial()
    session = HardwareDebugSession()
    session._serial = fake
    reply = session.vent(True)
    assert reply.response == "DEBUG_PUMP_VENT_ON"
    assert fake.written == b"PUMP/3\n"


def test_move_via_approach_rejects_unsafe_height_order() -> None:
    session = HardwareDebugSession()
    try:
        session.move_via_approach((204.0, 189.0, 130.0), approach_z=130.0)
    except HardwareDebugError as error:
        assert "接近高度" in str(error)
    else:
        raise AssertionError("approach height must be above target height")


def test_pick_and_place_uses_safe_vertical_routes() -> None:
    class RecordingSession(HardwareDebugSession):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple] = []

        def move(self, x: float, y: float, z: float):
            self.calls.append(("move", x, y, z))
            return None

        def return_to(self, x: float, y: float, z: float):
            self.calls.append(("return_to", x, y, z))
            return None

        def turntable(self, duty: int):
            self.calls.append(("turntable", duty))
            return None

        def pump(self, state: int):
            self.calls.append(("pump", state))
            self._pump_started_at = 0.0 if state == 1 else None
            return None

        def vent(self, opened: bool):
            self.calls.append(("vent", opened))
            return None

    session = RecordingSession()
    reply = session.pick_and_place(
        (204.0, 189.0, 52.0),
        (300.0, 350.0, 70.0),
        (204.0, 189.0, 130.0),
        safe_z=130.0,
        turntable_receive_duty=5900,
        turntable_pickup_duty=3300,
        minimum_suction_seconds=0.0,
        release_settle_seconds=0.0,
        pickup_hold_seconds=0.0,
    )

    assert reply.response.startswith("FOUNDATION_PLACE_OK")
    assert session.calls == [
        ("move", 204.0, 189.0, 130.0),
        ("turntable", 5900),
        ("turntable", 3300),
        ("move", 204.0, 189.0, 130.0),
        ("move", 204.0, 189.0, 52.0),
        ("pump", 1),
        ("pump", 2),
        ("move", 204.0, 189.0, 130.0),
        ("move", 300.0, 350.0, 130.0),
        ("move", 300.0, 350.0, 70.0),
        ("vent", True),
        ("move", 300.0, 350.0, 130.0),
        ("vent", False),
        ("return_to", 204.0, 189.0, 130.0),
        ("turntable", 5900),
    ]


def test_pick_and_place_rejects_safe_height_below_target() -> None:
    session = HardwareDebugSession()

    try:
        session.pick_and_place(
            (204.0, 189.0, 52.0),
            (300.0, 350.0, 140.0),
            (204.0, 189.0, 130.0),
            safe_z=130.0,
        )
    except HardwareDebugError as error:
        assert "安全高度" in str(error)
    else:
        raise AssertionError("safe height must be above every low point")


def test_pick_and_place_cancels_before_sending_later_positions() -> None:
    class RecordingSession(HardwareDebugSession):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple] = []
            self.cancelled = False

        def move(self, x: float, y: float, z: float):
            self.calls.append(("move", x, y, z))
            self.cancelled = True
            return None

        def turntable(self, duty: int):
            self.calls.append(("turntable", duty))
            return None

        def pump(self, state: int):
            self.calls.append(("pump", state))
            return None

        def vent(self, opened: bool):
            self.calls.append(("vent", opened))
            return None

    session = RecordingSession()
    try:
        session.pick_and_place(
            (204.0, 189.0, 52.0),
            (300.0, 350.0, 70.0),
            (204.0, 189.0, 130.0),
            safe_z=130.0,
            cancel_requested=lambda: session.cancelled,
        )
    except HardwareCancelled:
        pass
    else:
        raise AssertionError("cancel request must stop the remaining sequence")

    assert session.calls == [("move", 204.0, 189.0, 130.0)]


def test_return_piece_to_pickup_releases_then_returns_to_standby() -> None:
    class RecordingSession(HardwareDebugSession):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple] = []

        def move(self, x: float, y: float, z: float):
            self.calls.append(("move", x, y, z))
            return None

        def return_to(self, x: float, y: float, z: float):
            self.calls.append(("return_to", x, y, z))
            return None

        def pump(self, state: int):
            self.calls.append(("pump", state))
            self._pump_started_at = None
            return None

        def vent(self, opened: bool):
            self.calls.append(("vent", opened))
            if opened:
                self._pump_started_at = None
            return None

        def turntable(self, duty: int):
            self.calls.append(("turntable", duty))
            return None

    session = RecordingSession()
    reply = session.return_piece_to_pickup(
        (220.0, 187.0, 50.0),
        (215.0, 189.0, 130.0),
        safe_z=130.0,
        turntable_receive_duty=6000,
    )

    assert reply is None
    assert session.calls == [
        ("move", 220.0, 187.0, 130.0),
        ("move", 220.0, 187.0, 52.0),
        ("pump", 0),
        ("move", 220.0, 187.0, 130.0),
        ("return_to", 215.0, 189.0, 130.0),
        ("turntable", 6000),
    ]
