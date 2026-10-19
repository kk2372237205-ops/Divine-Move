from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np

import pc_app.ui as ui_module
from pc_app.services.ports import SerialPortInfo
from pc_app.dataset import DatasetManager
from pc_app.ui import DiagnosticApp


class FakeVariable:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def set(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


class FakeCombo:
    def __init__(self) -> None:
        self.values = ()

    def __setitem__(self, key: str, value) -> None:
        assert key == "values"
        self.values = value


def test_refresh_ports_replaces_missing_legacy_com_port(monkeypatch) -> None:
    monkeypatch.setattr(
        ui_module,
        "list_serial_ports",
        lambda: (
            [
                SerialPortInfo("COM6", "USB-SERIAL CH340", ""),
                SerialPortInfo("COM7", "USB-SERIAL CH340", ""),
            ],
            None,
        ),
    )
    app = SimpleNamespace(
        hardware_port_combo=FakeCombo(),
        hardware_port_var=FakeVariable("COM3"),
        arm_var=FakeVariable(),
        arm_debug_var=FakeVariable(),
        _set_arm_status=lambda status: DiagnosticApp._set_arm_status(app, status),
        _set_port_text=lambda _content: None,
    )

    DiagnosticApp._refresh_ports(app)

    assert app.hardware_port_combo.values == ["COM6", "COM7"]
    assert app.hardware_port_var.get() == "COM6"
    assert "COM6" in app.arm_var.get()


def test_start_camera_does_not_reopen_an_active_preview() -> None:
    starts = []
    control_states = []
    app = SimpleNamespace(
        preview_running=True,
        camera=SimpleNamespace(index=0, start=lambda index: starts.append(index)),
        status_var=FakeVariable(),
    )
    app._set_preview_controls = lambda running: control_states.append(running)

    DiagnosticApp._start_camera(app)

    assert starts == []
    assert control_states == [True]
    assert "已经启动" in app.status_var.value


def test_prepare_next_match_disarms_automation_and_requires_fresh_empty_board() -> None:
    rendered_boards = []
    events = []
    app = SimpleNamespace(
        game_action_in_progress=False,
        hardware_busy=False,
        dry_run_enabled=True,
        game_hardware_enabled_var=FakeVariable(True),
        game_auto_execution_var=FakeVariable(True),
        game_controller=SimpleNamespace(reset=lambda: events.append("controller_reset")),
        board_tracker=SimpleNamespace(reset=lambda: events.append("tracker_reset")),
        game_suggested_move=(6, 6),
        game_motion_plan=object(),
        game_waiting_visual_confirmation=True,
        game_finished=True,
        game_pending_move=(6, 6),
        reference_invalidated=False,
        recognizer=SimpleNamespace(reset_reference_cache=lambda: events.append("reference_reset")),
        game_phase_var=FakeVariable(),
        game_message_var=FakeVariable(),
        game_plan_var=FakeVariable(),
        status_var=FakeVariable(),
        session_log=SimpleNamespace(record=lambda event, **_details: events.append(event)),
    )
    app._cancel_game_timers = lambda: events.append("timers_cancelled")
    app._draw_game_board = lambda board: rendered_boards.append(board)

    DiagnosticApp._prepare_next_match(app)

    assert not app.dry_run_enabled
    assert not app.game_hardware_enabled_var.get()
    assert not app.game_auto_execution_var.get()
    assert app.reference_invalidated
    assert app.board_clearance_in_progress
    assert app.board_obstructions == ()
    assert app.game_phase_var.get() == "准备下一局：请先清空实体棋盘"
    assert "保存空盘基准" in app.game_plan_var.get()
    assert len(rendered_boards) == 1
    assert "advanced_match_reset_for_next_game" in events


def test_preview_watchdog_detects_repeated_frozen_frame_and_resets_on_change() -> None:
    app = SimpleNamespace(preview_last_sample=None, preview_stalled_frames=0)
    frozen = np.zeros((1080, 1920, 3), dtype=np.uint8)

    assert not DiagnosticApp._preview_is_stalled(app, frozen)
    for _ in range(ui_module.PREVIEW_STALL_FRAME_LIMIT - 1):
        assert not DiagnosticApp._preview_is_stalled(app, frozen)
    assert DiagnosticApp._preview_is_stalled(app, frozen)

    changed = frozen.copy()
    changed[0:80, 0:80] = 255
    assert not DiagnosticApp._preview_is_stalled(app, changed)
    assert app.preview_stalled_frames == 0


def test_live_obstruction_refresh_removes_stale_red_box(monkeypatch) -> None:
    stale = SimpleNamespace(x=12, y=15, width=13, height=23, area=180.0)
    app = SimpleNamespace(
        calibration=object(),
        calibration_invalidated=False,
        last_frame=np.zeros((1080, 1920, 3), dtype=np.uint8),
        obstruction_last_checked=0.0,
        obstruction_check_interval=0.6,
        board_obstructions=(stale,),
        corner_overlay_until=float("inf"),
        calibration_var=FakeVariable(),
        status_var=FakeVariable(),
    )
    monkeypatch.setattr(ui_module, "detect_board_obstructions", lambda _frame, _calibration: SimpleNamespace(obstructions=()))

    DiagnosticApp._refresh_board_obstruction_overlay(app, force=True)

    assert app.board_obstructions == ()
    assert app.corner_overlay_until != float("inf")
    assert "过期红框已移除" in app.status_var.value


def test_foundation_placement_updates_turntable_status_to_zero_degrees() -> None:
    app = SimpleNamespace(
        servo_var=FakeVariable(),
        foundation_calibration=SimpleNamespace(
                pickup_point=(204.0, 189.0, 52.0),
                standby_point=(204.0, 189.0, 130.0),
                safe_z=130.0,
                turntable_receive_duty=6000,
                turntable_pickup_duty=3150,
            ),
        _selected_foundation_point=lambda: ("center", (304.0, -3.0, 50.0)),
    )
    app.hardware = SimpleNamespace(pick_and_place=lambda *_args, **_kwargs: object())
    app._hardware_cancel_requested = lambda: False
    app._set_turntable_state = lambda position, duty: DiagnosticApp._set_turntable_state(app, position, duty)
    app._hardware_reply = lambda _action, callback, on_success=None: on_success(callback())

    DiagnosticApp._place_foundation_target(app)

    assert "0°接子" in app.servo_var.value
    assert "6000" in app.servo_var.value


def test_arm_debug_picks_then_places_at_saved_center_point() -> None:
    calls = []
    calibration = SimpleNamespace(
        pickup_point=(215.0, 187.0, 50.0),
        standby_point=(215.0, 189.0, 130.0),
        safe_z=130.0,
        turntable_receive_duty=6000,
        turntable_pickup_duty=2800,
        targets={"center": (304.0, -3.0, 50.0)},
    )
    app = SimpleNamespace(
        foundation_calibration=calibration,
        servo_var=FakeVariable(),
        status_var=FakeVariable(),
        hardware=SimpleNamespace(pick_and_place=lambda *args, **kwargs: calls.append((args, kwargs)) or object()),
    )
    app._hardware_cancel_requested = lambda: False
    app._hardware_reply = lambda _action, callback, on_success=None: on_success(callback())
    app._set_turntable_state = lambda position, duty: DiagnosticApp._set_turntable_state(app, position, duty)
    app._finish_center_arm_debug = lambda receive_duty: DiagnosticApp._finish_center_arm_debug(app, receive_duty)

    DiagnosticApp._debug_arm_to_center_target(app)

    assert calls == [
        (
            ((215.0, 187.0, 50.0), (304.0, -3.0, 50.0), (215.0, 189.0, 130.0)),
            {
                "safe_z": 130.0,
                "turntable_receive_duty": 6000,
                "turntable_pickup_duty": 2800,
                "cancel_requested": app._hardware_cancel_requested,
            },
        )
    ]
    assert "0°接子" in app.servo_var.value
    assert "中心" in app.status_var.value


def test_arm_debug_requires_saved_center_point() -> None:
    app = SimpleNamespace(
        foundation_calibration=SimpleNamespace(targets={}),
        status_var=FakeVariable(),
    )

    DiagnosticApp._debug_arm_to_center_target(app)

    assert "中心黑点尚未保存" in app.status_var.value


def test_visual_five_point_candidate_converts_camera_error_to_opposite_arm_offset() -> None:
    arm = SimpleNamespace(
        board_size=13,
        point_0_0=(170.6, -147.8, 51.6),
        point_12_0=(432.6, -147.8, 51.6),
        point_12_12=(436.6, 142.2, 51.6),
        point_0_12=(174.6, 142.2, 51.6),
    )
    app = SimpleNamespace(
        advanced_arm_calibration=arm,
        foundation_calibration=SimpleNamespace(targets={"center": (304.0, -4.0, 50.0)}),
    )

    candidate = DiagnosticApp._foundation_visual_candidate(app, "center", (6, 6), (0.17, -0.26))

    assert candidate is not None
    # Right/up in the camera requires left/down in the arm's board mapping.
    assert candidate[0] < 304.0
    assert candidate[1] > -4.0
    assert candidate[2] == 50.0


def test_pickup_uses_saved_pickup_point_instead_of_empty_move_inputs() -> None:
    calls = []
    calibration = SimpleNamespace(
        pickup_point=(204.0, 189.0, 52.0),
        standby_point=(204.0, 189.0, 130.0),
        safe_z=130.0,
        turntable_receive_duty=6000,
        turntable_pickup_duty=2800,
    )
    app = SimpleNamespace(
        foundation_calibration=calibration,
        status_var=FakeVariable(),
        hardware=SimpleNamespace(pickup=lambda *args, **kwargs: calls.append((args, kwargs)) or object()),
    )
    app._hardware_cancel_requested = lambda: False
    app._hardware_reply = lambda _action, callback, on_success=None: on_success(callback())

    DiagnosticApp._pickup_hardware(app)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ((204.0, 189.0, 52.0),)
    assert kwargs == {
        "approach_z": 130.0,
        "turntable_receive_duty": 6000,
        "turntable_pickup_duty": 2800,
        "cancel_requested": app._hardware_cancel_requested,
    }
    assert "保存的取子点" in app.status_var.value


def test_stop_button_cancels_busy_hardware_sequence() -> None:
    app = SimpleNamespace(
        hardware_busy=True,
        hardware_cancel_event=threading.Event(),
        status_var=FakeVariable(),
    )
    app._request_hardware_stop = lambda: DiagnosticApp._request_hardware_stop(app)

    DiagnosticApp._pump(app, 2, "停止")

    assert app.hardware_cancel_event.is_set()
    assert "已请求停止" in app.status_var.value


def test_drop_returns_piece_to_saved_release_point() -> None:
    calls = []
    calibration = SimpleNamespace(
        pickup_point=(215.0, 189.0, 52.0),
        release_point=(220.0, 187.0, 56.0),
        standby_point=(215.0, 189.0, 130.0),
        safe_z=130.0,
        turntable_receive_duty=6000,
    )
    app = SimpleNamespace(
        foundation_calibration=calibration,
        status_var=FakeVariable(),
        hardware=SimpleNamespace(return_piece_to_pickup=lambda *args, **kwargs: calls.append((args, kwargs)) or object()),
    )
    app._hardware_cancel_requested = lambda: False
    app._hardware_reply = lambda _action, callback, on_success=None: on_success(callback())

    DiagnosticApp._drop_hardware(app)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ((215.0, 189.0, 52.0), (215.0, 189.0, 130.0))
    assert kwargs == {
        "safe_z": 130.0,
        "turntable_receive_duty": 6000,
        "release_point": (220.0, 187.0, 56.0),
        "cancel_requested": app._hardware_cancel_requested,
    }
    assert "正常泄压" in app.status_var.value


def test_calibration_rejects_below_1080p() -> None:
    app = SimpleNamespace(
        last_frame=np.zeros((720, 1280, 3), dtype=np.uint8),
        calibration_var=FakeVariable(),
        status_var=FakeVariable(),
    )

    DiagnosticApp._calibrate_board(app)

    assert "1280x720" in app.calibration_var.value
    assert "1920x1080" in app.status_var.value


def test_camera_restart_invalidates_saved_calibration(tmp_path: Path) -> None:
    app = SimpleNamespace(
        calibration_invalidated=True,
        calibration=object(),
        calibration_path=tmp_path / "board.json",
    )

    assert DiagnosticApp._current_calibration(app) is None


def test_dataset_batch_saves_metadata_and_tracks_files(tmp_path: Path) -> None:
    app = SimpleNamespace(
        dataset_batch_active=False,
        dataset_batch_files=[],
        dataset_batch_status_var=FakeVariable(),
        dataset=DatasetManager(tmp_path / "dataset"),
        last_frame=np.zeros((1080, 1920, 3), dtype=np.uint8),
        camera=SimpleNamespace(correct_color=lambda frame: frame.copy()),
        dataset_lighting_var=FakeVariable("高光"),
        dataset_edge_var=SimpleNamespace(get=lambda: True),
        dataset_scene_var=FakeVariable("scene_ui"),
        dataset_auto_id_var=SimpleNamespace(get=lambda: False),
        dataset_notes_var=FakeVariable("manual layout"),
        status_var=FakeVariable(),
        _update_dataset_status=lambda: None,
    )
    app._save_current_dataset_frame = lambda **kwargs: DiagnosticApp._save_current_dataset_frame(app, **kwargs)

    DiagnosticApp._begin_dataset_batch(app)
    DiagnosticApp._save_dataset_frame_to_batch(app)
    DiagnosticApp._finish_dataset_batch(app)

    assert len(app.dataset_batch_files) == 1
    record = app.dataset._read_records()[0]
    assert record.scene_id == "scene_ui"
    assert record.notes.startswith("manual layout")
    assert record.lighting == "high"
    assert record.edge_scene is True
    assert "batch_frame=01" in record.notes
    assert "已结束" in app.dataset_batch_status_var.value


def test_dataset_batch_generates_scene_id_when_auto_numbering(tmp_path: Path) -> None:
    app = SimpleNamespace(
        dataset_batch_active=False,
        dataset_batch_files=[],
        dataset_batch_counter=0,
        dataset_batch_status_var=FakeVariable(),
        dataset=DatasetManager(tmp_path / "dataset"),
        dataset_scene_var=FakeVariable(""),
        dataset_auto_id_var=SimpleNamespace(get=lambda: True),
        status_var=FakeVariable(),
    )

    DiagnosticApp._begin_dataset_batch(app)

    assert app.dataset_scene_var.value.startswith("batch_")
    assert "已开始" in app.dataset_batch_status_var.value


def test_board_corner_overlay_marks_saved_calibration() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    app = SimpleNamespace(
        calibration=SimpleNamespace(corners=((10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0))),
        calibration_invalidated=False,
        board_obstructions=(),
        corner_overlay_until=float("inf"),
    )

    overlay = DiagnosticApp._draw_board_corner_overlay(app, frame)

    assert overlay is not frame
    assert np.any(overlay != frame)
    assert np.array_equal(frame, np.zeros((100, 100, 3), dtype=np.uint8))


def test_turntable_status_omits_manual_angle_confirmation_note() -> None:
    app = SimpleNamespace(servo_var=FakeVariable())

    DiagnosticApp._set_turntable_state(app, "90°取子", 2800)

    assert app.servo_var.get() == "取子机构舵机调试台：已发送 90°取子，占空比 2800"
    assert app.turntable_duty == 2800


def test_saving_empty_board_returns_an_extended_turntable_to_storage_first() -> None:
    turntable_calls = []
    saved = []
    app = SimpleNamespace(
        last_frame=object(),
        hardware_busy=False,
        foundation_calibration=SimpleNamespace(turntable_receive_duty=6000),
        hardware=SimpleNamespace(connected=True, turntable=lambda duty: turntable_calls.append(duty) or object()),
        turntable_duty=2800,
        status_var=FakeVariable(),
        servo_var=FakeVariable(),
    )
    app._hardware_reply = lambda _action, callback, on_success=None: on_success(callback())
    app._save_empty_board_reference = lambda: saved.append(True)
    app._empty_board_ready_for_capture = lambda: True

    DiagnosticApp._capture_empty_board(app)

    assert turntable_calls == [6000]
    assert app.turntable_duty == 6000
    assert saved == [True]


def test_board_corner_overlay_hides_after_clean_timeout(monkeypatch) -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    app = SimpleNamespace(
        calibration=SimpleNamespace(corners=((10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0))),
        calibration_invalidated=False,
        board_obstructions=(),
        corner_overlay_until=10.0,
    )
    monkeypatch.setattr(ui_module.time, "monotonic", lambda: 11.0)

    assert DiagnosticApp._draw_board_corner_overlay(app, frame) is frame
