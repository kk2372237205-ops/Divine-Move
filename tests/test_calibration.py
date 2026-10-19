import cv2
import numpy as np

import pc_app.calibration as calibration_module
from pc_app.calibration import (
    BoardCalibration,
    check_calibration_alignment,
    detect_board_corners,
    order_corners,
)


def test_order_corners_and_cell_pixel() -> None:
    calibration = BoardCalibration(((100, 80), (500, 90), (510, 480), (90, 470)))
    assert calibration.corners[0] == (100.0, 80.0)
    assert calibration.board_pixel((6, 6)) == (259.5, 259.5)


def test_calibration_normalizes_manually_supplied_corner_order() -> None:
    calibration = BoardCalibration(((510, 480), (90, 470), (100, 80), (500, 90)))

    assert calibration.corners == ((100.0, 80.0), (500.0, 90.0), (510.0, 480.0), (90.0, 470.0))


def test_warp_and_detect_synthetic_board() -> None:
    frame = np.zeros((500, 700, 3), dtype=np.uint8)
    polygon = np.array([[100, 80], [560, 100], [520, 430], [120, 450]], dtype=np.int32)
    cv2.fillConvexPoly(frame, polygon, (185, 145, 90))
    cv2.polylines(frame, [polygon], True, (20, 20, 20), 8)

    detected = detect_board_corners(frame)
    assert detected is not None
    calibration = BoardCalibration(detected, output_size=300)
    warped = calibration.warp(frame)
    assert warped.shape == (300, 300, 3)
    assert float(warped.mean()) > 5


def test_warp_padded_keeps_grid_inside_display_image() -> None:
    calibration = BoardCalibration(((100, 80), (500, 90), (510, 480), (90, 470)), output_size=300)
    frame = np.zeros((560, 700, 3), dtype=np.uint8)
    cv2.fillConvexPoly(frame, np.array([[100, 80], [500, 90], [510, 480], [90, 470]], dtype=np.int32), (185, 145, 90))

    padded = calibration.warp_padded(frame, padding=20)

    assert padded.shape == (340, 340, 3)
    # The outer grid intersections map to the inner 300x300 region rather
    # than being placed on the preview image boundary.
    assert float(padded[20, 20].mean()) > 5
    assert float(padded[319, 319].mean()) > 5


def test_warp_padded_rejects_invalid_padding() -> None:
    calibration = BoardCalibration(((100, 80), (500, 90), (510, 480), (90, 470)))
    frame = np.zeros((500, 700, 3), dtype=np.uint8)

    for padding in (-1, True, 1.5):
        try:
            calibration.warp_padded(frame, padding=padding)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_detection_refines_outer_frame_to_grid_intersections() -> None:
    frame = np.zeros((700, 700, 3), dtype=np.uint8)
    cv2.rectangle(frame, (80, 70), (620, 630), (185, 145, 90), -1)
    cv2.rectangle(frame, (80, 70), (620, 630), (20, 20, 20), 8)
    for coordinate in np.linspace(140, 560, 13).round().astype(int):
        cv2.line(frame, (coordinate, 130), (coordinate, 570), (25, 25, 25), 3)
        cv2.line(frame, (130, coordinate), (570, coordinate), (25, 25, 25), 3)

    detected = detect_board_corners(frame)

    assert detected is not None
    expected = ((140, 140), (560, 140), (560, 560), (140, 560))
    assert np.allclose(detected, expected, atol=8)


def test_invalid_corner_count_is_rejected() -> None:
    try:
        order_corners([(0, 0), (1, 1), (2, 2)])
    except ValueError as error:
        assert "四个" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_detect_board_corners_skips_a_degenerate_contour_candidate(monkeypatch) -> None:
    """A malformed large contour must not prevent a later valid board candidate."""
    malformed = object()
    valid = object()
    malformed_points = np.array(((10, 5), (5, 10), (10, 20), (25, 10)), dtype=np.int32).reshape(4, 1, 2)
    valid_points = np.array(((10, 10), (90, 10), (90, 90), (10, 90)), dtype=np.int32).reshape(4, 1, 2)

    monkeypatch.setattr(cv2, "findContours", lambda *_args: ([malformed, valid], None))
    monkeypatch.setattr(cv2, "contourArea", lambda _contour: 5_000.0)
    monkeypatch.setattr(cv2, "arcLength", lambda *_args: 100.0)
    monkeypatch.setattr(
        cv2,
        "approxPolyDP",
        lambda contour, *_args: malformed_points if contour is malformed else valid_points,
    )
    monkeypatch.setattr(cv2, "isContourConvex", lambda _polygon: True)
    monkeypatch.setattr(calibration_module, "_refine_grid_corners", lambda *_args: None)

    detected = detect_board_corners(np.zeros((100, 100, 3), dtype=np.uint8))

    assert detected == ((10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0))


def test_alignment_corner_detection_prefers_candidate_near_saved_board(monkeypatch) -> None:
    """A game-piece/hand contour must not win merely because it is larger."""
    wrong = object()
    board = object()
    wrong_points = np.array(((5, 5), (95, 5), (95, 95), (5, 95)), dtype=np.int32).reshape(4, 1, 2)
    board_points = np.array(((20, 20), (80, 20), (80, 80), (20, 80)), dtype=np.int32).reshape(4, 1, 2)

    monkeypatch.setattr(cv2, "findContours", lambda *_args: ([wrong, board], None))
    monkeypatch.setattr(cv2, "contourArea", lambda contour: 9_000.0 if contour is wrong else 5_000.0)
    monkeypatch.setattr(cv2, "arcLength", lambda *_args: 100.0)
    monkeypatch.setattr(
        cv2,
        "approxPolyDP",
        lambda contour, *_args: wrong_points if contour is wrong else board_points,
    )
    monkeypatch.setattr(cv2, "isContourConvex", lambda _polygon: True)
    monkeypatch.setattr(calibration_module, "_refine_grid_corners", lambda _frame, corners: corners)

    detected = detect_board_corners(
        np.zeros((100, 100, 3), dtype=np.uint8),
        reference_corners=((20, 20), (80, 20), (80, 80), (20, 80)),
    )

    assert detected == ((20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0))


def test_calibration_alignment_rejects_camera_drift(monkeypatch) -> None:
    calibration = BoardCalibration(((100, 80), (500, 90), (510, 480), (90, 470)))
    frame = np.zeros((500, 700, 3), dtype=np.uint8)
    shifted = ((130, 80), (530, 90), (540, 480), (120, 470))
    monkeypatch.setattr("pc_app.calibration.detect_board_corners", lambda *_args, **_kwargs: shifted)

    result = check_calibration_alignment(frame, calibration)

    assert not result.aligned
    assert "相机被移动" in result.reason


def test_calibration_alignment_accepts_same_geometry(monkeypatch) -> None:
    calibration = BoardCalibration(((100, 80), (500, 90), (510, 480), (90, 470)))
    frame = np.zeros((500, 700, 3), dtype=np.uint8)
    monkeypatch.setattr("pc_app.calibration.detect_board_corners", lambda *_args, **_kwargs: calibration.corners)

    result = check_calibration_alignment(frame, calibration)

    assert result.aligned
    assert result.max_corner_error == 0.0


def test_calibration_alignment_rejects_unstable_corner_detection(monkeypatch) -> None:
    calibration = BoardCalibration(((100, 80), (500, 90), (510, 480), (90, 470)))
    frame = np.zeros((500, 700, 3), dtype=np.uint8)

    def unstable_detection(*_args, **_kwargs):
        raise ValueError("棋盘角点不唯一")

    monkeypatch.setattr("pc_app.calibration.detect_board_corners", unstable_detection)

    result = check_calibration_alignment(frame, calibration)

    assert not result.aligned
    assert "不稳定" in result.reason
