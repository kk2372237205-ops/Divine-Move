import cv2
import numpy as np
import pytest

from pc_app.board import Stone
from pc_app.calibration import BoardCalibration
from pc_app.vision import (
    BoardStoneRecognizer,
    BoardVisionResult,
    StableBoardTracker,
    StoneDetection,
    detect_board_obstructions,
    detect_gameplay_occlusion,
)


def identity_calibration() -> BoardCalibration:
    return BoardCalibration(((0, 0), (519, 0), (519, 519), (0, 519)))


def test_reference_recognition_detects_black_and_white_stones(tmp_path) -> None:
    calibration = identity_calibration()
    empty = np.full((520, 520, 3), 150, dtype=np.uint8)
    for index in range(13):
        point = round(index * 519 / 12)
        cv2.line(empty, (0, point), (519, point), (80, 80, 80), 1)
        cv2.line(empty, (point, 0), (point, 519), (80, 80, 80), 1)
    # The five reference dots are already present in both frames and must not
    # be treated as pieces.
    cv2.circle(empty, (round(3 * 519 / 12), round(3 * 519 / 12)), 3, (25, 25, 25), -1)

    recognizer = BoardStoneRecognizer(tmp_path / "empty_board.png")
    recognizer.capture_empty_board(empty, calibration)
    current = empty.copy()
    cv2.circle(current, (round(4 * 519 / 12), round(5 * 519 / 12)), 15, (25, 25, 25), -1)
    cv2.circle(current, (round(9 * 519 / 12), round(7 * 519 / 12)), 15, (245, 245, 245), -1)

    result = recognizer.detect(current, calibration)

    assert result.stone_at((4, 5)) == Stone.BLACK
    assert result.stone_at((9, 7)) == Stone.WHITE
    assert result.stone_at((3, 3)) == Stone.EMPTY


def test_reference_recognition_ignores_small_lighting_change(tmp_path) -> None:
    calibration = identity_calibration()
    empty = np.full((520, 520, 3), 150, dtype=np.uint8)
    recognizer = BoardStoneRecognizer(tmp_path / "empty_board.png")
    recognizer.capture_empty_board(empty, calibration)
    slightly_brighter = cv2.add(empty, np.full_like(empty, 5))

    assert recognizer.detect(slightly_brighter, calibration).stones == {}


def test_visual_measurement_reports_continuous_black_stone_offset(tmp_path) -> None:
    calibration = identity_calibration()
    empty = np.full((520, 520, 3), 165, dtype=np.uint8)
    recognizer = BoardStoneRecognizer(tmp_path / "empty_board.png")
    recognizer.capture_empty_board(empty, calibration)
    current = empty.copy()
    expected = calibration.board_pixel((6, 6))
    cv2.circle(current, (266, 250), 16, (20, 20, 20), -1)

    measurement = recognizer.measure_black_stone_centers(current, calibration, ((6, 6),))[0]

    assert measurement.move == (6, 6)
    assert measurement.observed_pixel[0] == pytest.approx(266, abs=1.0)
    assert measurement.observed_pixel[1] == pytest.approx(250, abs=1.0)
    assert measurement.offset_pixel[0] == pytest.approx(6.5, abs=1.0)
    assert measurement.offset_pixel[1] == pytest.approx(-9.5, abs=1.0)


def test_reference_can_be_saved_in_a_chinese_windows_path(tmp_path) -> None:
    calibration = identity_calibration()
    empty = np.full((520, 520, 3), 150, dtype=np.uint8)
    recognizer = BoardStoneRecognizer(tmp_path / "五子棋机器人" / "empty_board.png")

    saved = recognizer.capture_empty_board(empty, calibration)

    assert saved.exists()
    assert recognizer.detect(empty, calibration).stones == {}


def test_tracker_requires_three_consecutive_equal_frames() -> None:
    detection = StoneDetection((6, 6), Stone.BLACK, 0.9, 40.0)
    result = BoardVisionResult((detection,))
    tracker = StableBoardTracker(required_frames=3)

    assert tracker.update(result) is False
    assert tracker.update(result) is False
    assert tracker.update(result) is True
    assert tracker.stable_stones == {(6, 6): Stone.BLACK}
    assert tracker.update(result) is False


def test_board_clearance_detects_piece_and_large_foreign_object() -> None:
    calibration = identity_calibration()
    empty = np.full((520, 520, 3), 165, dtype=np.uint8)
    clear = detect_board_obstructions(empty, calibration)
    assert clear.clear

    with_piece = empty.copy()
    cv2.circle(with_piece, (260, 260), 18, (20, 20, 20), -1)
    assert not detect_board_obstructions(with_piece, calibration).clear

    with_card = empty.copy()
    cv2.rectangle(with_card, (190, 155), (330, 350), (250, 250, 250), -1)
    result = detect_board_obstructions(with_card, calibration)
    assert not result.clear
    assert result.obstructions[0].area > 1000


def test_gameplay_occlusion_gate_ignores_a_piece_but_freezes_for_an_arm_sized_object() -> None:
    calibration = identity_calibration()
    empty = np.full((520, 520, 3), 165, dtype=np.uint8)
    piece = empty.copy()
    cv2.circle(piece, (260, 260), 18, (20, 20, 20), -1)
    arm = empty.copy()
    cv2.rectangle(arm, (30, 140), (290, 340), (20, 20, 20), -1)

    assert detect_gameplay_occlusion(piece, calibration) is None
    occlusion = detect_gameplay_occlusion(arm, calibration)
    assert occlusion is not None
    assert occlusion.area >= 6_000


def test_gameplay_occlusion_gate_ignores_dense_legal_stone_cluster() -> None:
    calibration = identity_calibration()
    board = np.full((520, 520, 3), 165, dtype=np.uint8)
    # Neighbouring pieces are about seven pixels apart at this scale. The
    # clearance detector joins them, so a 4x4 cluster used to resemble an
    # arm-sized obstruction and freeze an otherwise valid game.
    for row in range(4, 8):
        for column in range(4, 8):
            center = calibration.board_pixel((column, row))
            cv2.circle(board, tuple(map(round, center)), 18, (20, 20, 20), -1)

    assert detect_gameplay_occlusion(board, calibration) is None


def test_board_clearance_detects_white_object_with_close_brightness() -> None:
    calibration = identity_calibration()
    # This simulates the photographed tan board: the card's grayscale value is
    # only 36 levels brighter, so detection needs the colour-distance path.
    board = np.full((520, 520, 3), (164, 205, 237), dtype=np.uint8)
    card = board.copy()
    cv2.rectangle(card, (180, 140), (340, 370), (201, 250, 255), -1)

    result = detect_board_obstructions(card, calibration)

    assert not result.clear
    assert len(result.obstructions) == 1
    assert result.obstructions[0].area > 1000


def test_board_clearance_ignores_fixed_corner_shadow() -> None:
    calibration = identity_calibration()
    board = np.full((520, 520, 3), (164, 205, 237), dtype=np.uint8)
    # Simulate the dark, fixed left-bottom board-frame shadow seen in preview.
    # It can extend slightly into the corrected corner cells as exposure
    # changes, so this is deliberately farther in than the masked 12px rim.
    cv2.rectangle(board, (22, 420), (123, 498), (70, 90, 95), -1)

    assert detect_board_obstructions(board, calibration).clear
