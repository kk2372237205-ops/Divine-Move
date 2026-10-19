from pc_app.board import Stone
from pc_app.gameflow import AutonomousGomokuController, GamePhase
from pc_app.motion import ArmBoardCalibration, MotionPlanner


def test_robot_black_starts_from_empty_board_with_a_plan() -> None:
    update = AutonomousGomokuController().observe({})

    assert update.phase == GamePhase.ROBOT_BLACK_TURN
    assert update.plan is not None
    assert update.plan.move == (6, 6)


def test_robot_waits_after_camera_confirms_its_black_stone() -> None:
    controller = AutonomousGomokuController()
    update = controller.observe({(6, 6): Stone.BLACK})

    assert update.phase == GamePhase.WAITING_FOR_HUMAN_WHITE
    assert update.plan is None


def test_new_human_white_stone_automatically_produces_next_black_plan() -> None:
    controller = AutonomousGomokuController()
    first = controller.observe({})
    assert first.plan is not None

    update = controller.observe({(6, 6): Stone.BLACK, (5, 6): Stone.WHITE})

    assert update.phase == GamePhase.ROBOT_BLACK_TURN
    assert update.plan is not None
    assert update.plan.move != (6, 6)


def test_invalid_stone_counts_are_rejected() -> None:
    update = AutonomousGomokuController().observe({(1, 1): Stone.WHITE})

    assert update.phase == GamePhase.INVALID_BOARD
    assert update.plan is None


def test_white_five_is_accepted_only_with_a_legal_black_first_turn_count() -> None:
    legal_white_win = {
        (0, 0): Stone.WHITE,
        (1, 0): Stone.WHITE,
        (2, 0): Stone.WHITE,
        (3, 0): Stone.WHITE,
        (4, 0): Stone.WHITE,
        (0, 2): Stone.BLACK,
        (2, 2): Stone.BLACK,
        (4, 2): Stone.BLACK,
        (6, 2): Stone.BLACK,
        (8, 2): Stone.BLACK,
    }

    update = AutonomousGomokuController().observe(legal_white_win)

    assert update.phase == GamePhase.FINISHED
    assert "白棋五连" in update.message


def test_illegal_white_five_does_not_override_turn_validation() -> None:
    invalid_white_win = {(column, 0): Stone.WHITE for column in range(5)}

    update = AutonomousGomokuController().observe(invalid_white_win)

    assert update.phase == GamePhase.INVALID_BOARD
    assert "不符合黑棋先行规则" in update.message


def test_dry_run_plan_uses_the_confirmed_four_corner_calibration() -> None:
    calibration = ArmBoardCalibration(
        point_0_0=(170.6, -147.8, 51.6),
        point_12_0=(432.6, -147.8, 51.6),
        point_12_12=(436.6, 142.2, 51.6),
        point_0_12=(174.6, 142.2, 51.6),
        pickup_point=(218.0, 187.0, 51.0),
        standby_point=(215.0, 189.0, 130.0),
        safe_z=130.0,
    )
    controller = AutonomousGomokuController(planner=MotionPlanner(calibration))

    update = controller.observe({})

    assert update.plan is not None
    assert update.plan.calibrated
    assert update.plan.move == (6, 6)
    assert update.plan.steps[7].target == (303.6, -2.8000000000000114, 51.6)
