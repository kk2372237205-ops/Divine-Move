import cv2
import numpy as np
import pytest

from pc_app.calibration import BoardCalibration
from pc_app.labels import BoxOverride, draw_grid_annotations, parse_grid_coordinates, project_grid_annotations


def identity_calibration() -> BoardCalibration:
    return BoardCalibration(
        corners=((0, 0), (519, 0), (519, 519), (0, 519)),
        output_size=520,
        board_size=13,
    )


def test_parse_internal_and_competition_coordinates() -> None:
    assert parse_grid_coordinates("6,6 3,3; (9,3)") == ((6, 6), (3, 3), (9, 3))
    assert parse_grid_coordinates("7,7 13,1", coordinate_system="competition") == ((6, 6), (0, 0))


def test_parse_coordinates_rejects_duplicates_and_malformed_values() -> None:
    with pytest.raises(ValueError, match="重复"):
        parse_grid_coordinates("1,1 1,1")
    with pytest.raises(ValueError, match="解析"):
        parse_grid_coordinates("1x1")
    with pytest.raises(ValueError, match="越界"):
        parse_grid_coordinates("13,0")


def test_project_grid_annotations_normalizes_original_image_boxes() -> None:
    annotations = project_grid_annotations(identity_calibration(), (520, 520, 3), [(6, 6)], [(12, 0)], radius=10)

    center = annotations[0]
    assert center.class_id == 0
    assert center.move == (6, 6)
    assert center.yolo[1] == pytest.approx(259.5 / 520)
    assert center.yolo[2] == pytest.approx(259.5 / 520)
    assert center.yolo[3] == pytest.approx(20 / 520)
    assert center.yolo[4] == pytest.approx(20 / 520)
    assert annotations[1].class_id == 1
    assert all(0 <= value <= 1 for value in annotations[1].yolo[1:])


def test_project_rejects_overlap_and_oversized_boxes() -> None:
    calibration = identity_calibration()
    with pytest.raises(ValueError, match="不能重复"):
        project_grid_annotations(calibration, (520, 520, 3), [(1, 1)], [(1, 1)])
    with pytest.raises(ValueError, match="过大"):
        project_grid_annotations(calibration, (520, 520, 3), [(1, 1)], radius=22)


def test_project_uses_a_per_stone_box_override() -> None:
    annotations = project_grid_annotations(
        identity_calibration(),
        (520, 520, 3),
        [(6, 6)],
        radius=10,
        box_overrides=(BoxOverride(move=(6, 6), center=(270.0, 250.0), radius=12.0),),
    )

    item = annotations[0]
    assert item.yolo[1] == pytest.approx(270.0 / 520)
    assert item.yolo[2] == pytest.approx(250.0 / 520)
    assert item.yolo[3] == pytest.approx(24.0 / 520)
    assert item.yolo[4] == pytest.approx(24.0 / 520)


def test_preview_draws_projected_boxes() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    calibration = BoardCalibration(
        corners=((0, 0), (99, 0), (99, 99), (0, 99)), output_size=100, board_size=5
    )
    annotations = project_grid_annotations(calibration, image.shape, [(2, 2)], radius=8)
    preview = draw_grid_annotations(image, annotations)
    assert preview.shape == image.shape
    assert int(cv2.countNonZero(cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY))) > 0
