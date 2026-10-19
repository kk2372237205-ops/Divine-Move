from pc_app.motion import ArmBoardCalibration, MotionPlanner, MotionStepKind


def calibrated_arm() -> ArmBoardCalibration:
    return ArmBoardCalibration(
        point_0_0=(100.0, 200.0, 50.0),
        point_12_0=(220.0, 200.0, 62.0),
        point_12_12=(230.0, 320.0, 74.0),
        point_0_12=(100.0, 320.0, 62.0),
        pickup_point=(50.0, 75.0, 42.0),
        standby_point=(60.0, 80.0, 110.0),
        safe_z=115.0,
    )


def test_four_corner_mapping_uses_all_corners() -> None:
    calibration = calibrated_arm()

    assert calibration.board_to_arm((0, 0)) == (100.0, 200.0, 50.0)
    assert calibration.board_to_arm((12, 12)) == (230.0, 320.0, 74.0)
    assert calibration.board_to_arm((6, 6)) == (162.5, 260.0, 62.0)
    assert calibration.target_safe_point((6, 6)) == (162.5, 260.0, 115.0)


def test_plan_keeps_turntable_and_arm_base_sequence_separate() -> None:
    plan = MotionPlanner(calibrated_arm()).plan_black_move((12, 0))

    assert plan.calibrated is True
    assert plan.competition_coordinate == (1, 1)
    assert plan.steps[0].kind == MotionStepKind.TURNTABLE_RECEIVE
    assert plan.steps[1].kind == MotionStepKind.TURNTABLE_PRESENT
    assert plan.steps[4].kind == MotionStepKind.VACUUM_PICK
    assert plan.steps[8].kind == MotionStepKind.VALVE_RELEASE
    assert plan.steps[-1].kind == MotionStepKind.TURNTABLE_RECEIVE
    assert plan.steps[7].target == (220.0, 200.0, 62.0)


def test_uncalibrated_plan_is_explicitly_not_executable() -> None:
    plan = MotionPlanner().plan_black_move((6, 6))

    assert plan.calibrated is False
    assert "不可执行" in "\n".join(plan.display_lines())
