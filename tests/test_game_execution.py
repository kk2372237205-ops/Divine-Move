from types import SimpleNamespace

from pc_app.board import GomokuBoard, Stone
from pc_app.gameflow import GameFlowUpdate, GamePhase
from pc_app.motion import ArmBoardCalibration, MotionPlanner
from pc_app.ui import DiagnosticApp


class _Variable:
    def __init__(self, value=None) -> None:
        self.value = value

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value


def _arm_calibration() -> ArmBoardCalibration:
    return ArmBoardCalibration(
        point_0_0=(170.6, -147.8, 51.6),
        point_12_0=(432.6, -147.8, 51.6),
        point_12_12=(436.6, 142.2, 51.6),
        point_0_12=(174.6, 142.2, 51.6),
        pickup_point=(218.0, 187.0, 51.0),
        standby_point=(215.0, 189.0, 130.0),
        safe_z=130.0,
    )


def test_game_plan_cannot_send_hardware_until_explicitly_enabled() -> None:
    calls = []
    app = SimpleNamespace(
        game_hardware_enabled_var=_Variable(False),
        status_var=_Variable(),
    )
    app.hardware = SimpleNamespace(pick_and_place=lambda *_args, **_kwargs: calls.append(True))

    DiagnosticApp._execute_game_plan(app)

    assert not calls
    assert "受监督执行" in app.status_var.value


def test_automatic_game_plan_cannot_send_hardware_without_match_authorization() -> None:
    calls = []
    app = SimpleNamespace(
        game_auto_execution_var=_Variable(False),
        status_var=_Variable(),
    )
    app.hardware = SimpleNamespace(pick_and_place=lambda *_args, **_kwargs: calls.append(True))

    DiagnosticApp._execute_game_plan(app, automatic=True)

    assert not calls


def test_formal_match_start_is_the_single_automatic_execution_authorization() -> None:
    started = []
    events = []
    app = SimpleNamespace(
        game_action_in_progress=False,
        hardware_busy=False,
        last_frame=object(),
        calibration_invalidated=False,
        reference_invalidated=False,
        hardware=SimpleNamespace(connected=True),
        recognizer=SimpleNamespace(has_reference=True),
        game_hardware_enabled_var=_Variable(True),
        game_auto_execution_var=_Variable(False),
        status_var=_Variable(),
        session_log=SimpleNamespace(record=lambda event, **details: events.append((event, details))),
        dry_run_enabled=False,
    )
    app._current_calibration = lambda: object()
    app._start_dry_run = lambda: (started.append(True), setattr(app, "dry_run_enabled", True))

    DiagnosticApp._start_automatic_match(app)

    assert started == [True]
    assert not app.game_hardware_enabled_var.get()
    assert app.game_auto_execution_var.get()
    assert "自动执行" in app.status_var.get()
    assert ("automatic_execution_authorized", {"source": "formal_match_start"}) in events


def test_game_plan_uses_verified_arm_calibration_after_all_preflight_checks() -> None:
    calibration = _arm_calibration()
    plan = MotionPlanner(calibration).plan_black_move((6, 6))
    calls = []
    app = SimpleNamespace(
        game_hardware_enabled_var=_Variable(True),
        dry_run_enabled=True,
        game_action_in_progress=False,
        hardware_busy=False,
        game_motion_plan=plan,
        game_suggested_move=plan.move,
        advanced_arm_calibration=calibration,
        calibration_invalidated=False,
        reference_invalidated=False,
        last_vision_result=object(),
        board_tracker=SimpleNamespace(consecutive_frames=3, required_frames=3),
        hardware=SimpleNamespace(connected=True, pick_and_place=lambda *args, **kwargs: calls.append((args, kwargs))),
        foundation_calibration=SimpleNamespace(turntable_receive_duty=6000, turntable_pickup_duty=2800),
        game_pending_move=None,
        game_waiting_visual_confirmation=False,
        game_phase_var=_Variable(),
        game_message_var=_Variable(),
        status_var=_Variable(),
    )
    app._schedule_game_action_timeout = lambda _move: None
    app._hardware_cancel_requested = lambda: False
    app._hardware_reply = lambda _action, callback, on_success=None, on_failure=None: callback()
    app._stop_game_after_motion_failure = lambda _error: None

    DiagnosticApp._execute_game_plan(app)

    assert app.game_action_in_progress
    assert app.game_pending_move == (6, 6)
    assert calls == [
        (
            ((218.0, 187.0, 51.0), (303.6, -2.8000000000000114, 51.6), (215.0, 189.0, 130.0)),
            {
                "safe_z": 130.0,
                "turntable_receive_duty": 6000,
                "turntable_pickup_duty": 2800,
                "cancel_requested": app._hardware_cancel_requested,
            },
        )
    ]


def test_automatic_game_plan_uses_same_verified_preflight_checks() -> None:
    calibration = _arm_calibration()
    plan = MotionPlanner(calibration).plan_black_move((6, 6))
    calls = []
    app = SimpleNamespace(
        game_auto_execution_var=_Variable(True),
        dry_run_enabled=True,
        game_action_in_progress=False,
        hardware_busy=False,
        game_motion_plan=plan,
        game_suggested_move=plan.move,
        advanced_arm_calibration=calibration,
        calibration_invalidated=False,
        reference_invalidated=False,
        last_vision_result=object(),
        board_tracker=SimpleNamespace(consecutive_frames=3, required_frames=3),
        hardware=SimpleNamespace(connected=True, pick_and_place=lambda *args, **kwargs: calls.append((args, kwargs))),
        foundation_calibration=SimpleNamespace(turntable_receive_duty=6000, turntable_pickup_duty=2800),
        game_pending_move=None,
        game_waiting_visual_confirmation=False,
        game_phase_var=_Variable(),
        game_message_var=_Variable(),
        status_var=_Variable(),
    )
    app._schedule_game_action_timeout = lambda _move: None
    app._hardware_cancel_requested = lambda: False
    app._hardware_reply = lambda _action, callback, on_success=None, on_failure=None: callback()
    app._stop_game_after_motion_failure = lambda _error: None

    DiagnosticApp._execute_game_plan(app, automatic=True)

    assert app.game_action_in_progress
    assert app.game_pending_move == (6, 6)
    assert len(calls) == 1
    assert calls[0][1]["safe_z"] == 130.0


def test_new_camera_confirmed_black_plan_dispatches_when_match_mode_is_armed() -> None:
    calibration = _arm_calibration()
    plan = MotionPlanner(calibration).plan_black_move((6, 6))
    board = GomokuBoard()
    dispatched = []
    app = SimpleNamespace(
        game_phase_var=_Variable(),
        game_message_var=_Variable(),
        game_suggested_move=None,
        game_motion_plan=None,
        game_plan_var=_Variable(),
        game_hardware_enabled_var=_Variable(False),
        status_var=_Variable(),
    )
    app._game_auto_execution_enabled = lambda: True
    app._draw_game_board = lambda _board: None
    app._execute_game_plan = lambda *, automatic=False: dispatched.append(automatic)

    DiagnosticApp._apply_gameflow_update(
        app,
        GameFlowUpdate(GamePhase.ROBOT_BLACK_TURN, "new black plan", board, plan=plan),
    )

    assert app.game_suggested_move == (6, 6)
    assert dispatched == [True]


def test_game_action_requires_camera_confirmation_of_the_expected_black_move() -> None:
    board = GomokuBoard()
    board.replace_stones({(6, 6): Stone.BLACK})
    app = SimpleNamespace(
        game_waiting_visual_confirmation=True,
        game_pending_move=(6, 6),
        game_phase_var=_Variable(),
        game_message_var=_Variable(),
        status_var=_Variable(),
    )
    app._cancel_game_confirmation_timeout = lambda: None

    DiagnosticApp._check_game_visual_confirmation(
        app,
        GameFlowUpdate(GamePhase.WAITING_FOR_HUMAN_WHITE, "confirmed", board),
    )

    assert not app.game_waiting_visual_confirmation
    assert app.game_pending_move is None
    assert "已由视觉确认" in app.status_var.value


def test_legal_game_finish_cancels_pending_confirmation_without_safety_fault() -> None:
    board = GomokuBoard()
    board.replace_stones({(0, 0): Stone.WHITE, (1, 0): Stone.WHITE, (2, 0): Stone.WHITE, (3, 0): Stone.WHITE, (4, 0): Stone.WHITE})
    app = SimpleNamespace(
        game_action_in_progress=False,
        game_waiting_visual_confirmation=True,
        game_pending_move=(6, 6),
        game_finished=False,
        game_hardware_enabled_var=_Variable(True),
        game_auto_execution_var=_Variable(True),
        game_suggested_move=(6, 6),
        game_motion_plan=object(),
        game_phase_var=_Variable(),
        game_message_var=_Variable(),
        game_plan_var=_Variable(),
        match_result_var=_Variable(),
        match_voice_enabled_var=_Variable(False),
        status_var=_Variable(),
    )
    app._cancel_game_timers = lambda: None
    app._draw_game_board = lambda _board: None

    DiagnosticApp._finish_completed_game(
        app,
        GameFlowUpdate(GamePhase.FINISHED, "白棋五连，人类获胜。", board),
    )

    assert app.game_finished
    assert not app.game_waiting_visual_confirmation
    assert app.game_pending_move is None
    assert not app.game_hardware_enabled_var.get()
    assert not app.game_auto_execution_var.get()
    assert "正常结束" in app.status_var.value
    assert app.match_result_var.get() == "人类获胜！"
