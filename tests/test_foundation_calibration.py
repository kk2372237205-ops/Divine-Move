from pc_app.foundation_calibration import FoundationHardwareCalibration, FOUNDATION_TARGETS


def test_foundation_calibration_round_trip_and_order(tmp_path) -> None:
    calibration = FoundationHardwareCalibration(turntable_receive_duty=5900, turntable_pickup_duty=3300)
    for index, (key, _, _) in enumerate(FOUNDATION_TARGETS):
        calibration.set_target(key, (250.0 + index, 300.0 + index, 70.0))
    path = tmp_path / "foundation_hardware.json"

    calibration.to_json(path)
    restored = FoundationHardwareCalibration.from_json(path)

    assert restored.complete
    assert tuple(key for key, _ in restored.ordered_targets()) == tuple(item[0] for item in FOUNDATION_TARGETS)
    assert restored.pickup_point == (215.0, 189.0, 52.0)
    assert restored.turntable_receive_duty == 5900
    assert restored.turntable_pickup_duty == 3300


def test_incomplete_calibration_lists_missing_points() -> None:
    calibration = FoundationHardwareCalibration()
    calibration.set_target("center", (300.0, 350.0, 70.0))

    assert calibration.turntable_pickup_duty == 2800
    assert not calibration.complete
    assert "中心黑点" not in calibration.missing_labels
    assert len(calibration.missing_labels) == 4


def test_foundation_calibration_rejects_unsafe_point() -> None:
    calibration = FoundationHardwareCalibration()

    try:
        calibration.set_target("center", (-1.0, 350.0, 70.0))
    except ValueError as error:
        assert "X>=0" in str(error)
    else:
        raise AssertionError("unsafe coordinates must be rejected")
