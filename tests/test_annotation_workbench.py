import json

import numpy as np
import pytest

import pc_app.annotation_workbench as workbench_module

from pc_app.annotation_workbench import (
    BLACK,
    BOARD_DISPLAY_MARGIN,
    EMPTY,
    WHITE,
    calibration_for_image,
    stabilize_board_corners,
    cycle_state,
    display_grid_geometry,
    exclude_image_family,
    display_point_to_move,
    load_reviewed_truths,
    states_to_moves,
)


def test_cycle_state_is_explicit_empty_black_white_cycle() -> None:
    assert [cycle_state(value) for value in (EMPTY, BLACK, WHITE)] == [BLACK, WHITE, EMPTY]


def test_cycle_state_rejects_unknown_state() -> None:
    with pytest.raises(ValueError):
        cycle_state(9)


def test_states_to_moves_returns_stable_row_major_order() -> None:
    black, white = states_to_moves({(3, 2): BLACK, (1, 0): WHITE, (0, 2): BLACK, (5, 1): WHITE})
    assert black == ((0, 2), (3, 2))
    assert white == ((1, 0), (5, 1))


def test_padded_display_maps_both_edge_intersections_without_clipping() -> None:
    margin, spacing = display_grid_geometry(13)
    assert margin == BOARD_DISPLAY_MARGIN
    assert display_point_to_move(margin, margin, 13) == (0, 0)
    assert display_point_to_move(margin + 12 * spacing, margin + 12 * spacing, 13) == (12, 12)
    assert display_point_to_move(0, 0, 13) is None


def test_load_reviewed_truths_tracks_latest_reviewed_record(tmp_path) -> None:
    truth_path = tmp_path / "truth.jsonl"
    truth_path.write_text(
        '{"image":"a.jpg","status":"reviewed","revision":1}\n'
        '{"image":"b.jpg","status":"draft","revision":1}\n'
        '{"image":"a.jpg","status":"reviewed","revision":2}\n',
        encoding="utf-8",
    )

    truths = load_reviewed_truths(truth_path)

    assert set(truths) == {"a.jpg"}
    assert truths["a.jpg"]["revision"] == 2


def test_calibration_for_image_uses_detected_corners(monkeypatch) -> None:
    fallback = workbench_module.BoardCalibration(((0, 0), (519, 0), (519, 519), (0, 519)))
    corners = ((10.0, 20.0), (500.0, 21.0), (498.0, 505.0), (11.0, 504.0))
    monkeypatch.setattr(workbench_module, "detect_board_corners", lambda _image: corners)

    calibration, detected = calibration_for_image(np.zeros((520, 520, 3), dtype=np.uint8), fallback)

    assert detected is True
    assert calibration.corners == corners
    assert calibration.output_size == fallback.output_size


def test_calibration_for_image_falls_back_when_board_is_not_found(monkeypatch) -> None:
    fallback = workbench_module.BoardCalibration(((0, 0), (519, 0), (519, 519), (0, 519)))
    monkeypatch.setattr(workbench_module, "detect_board_corners", lambda _image: None)

    calibration, detected = calibration_for_image(np.zeros((520, 520, 3), dtype=np.uint8), fallback)

    assert detected is False
    assert calibration is fallback


def test_stabilize_board_corners_repairs_occluded_lower_left_corner() -> None:
    corners = (
        (467.95, 113.68),
        (1263.76, 108.33),
        (1266.91, 973.01),
        (455.02, 865.04),  # false inner intersection behind the picker
    )

    repaired = stabilize_board_corners(corners)

    assert repaired[:3] == corners[:3]
    assert repaired[3][1] > 970
    assert 445 < repaired[3][0] < 460


def test_exclude_image_family_moves_source_variants_labels_and_preview(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    source = dataset_root / "raw" / "sample.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"raw")
    for suffix in ("brightness", "color", "flip", "blur"):
        augmented = dataset_root / "augmented" / f"sample_{suffix}.jpg"
        augmented.parent.mkdir(parents=True, exist_ok=True)
        augmented.write_bytes(suffix.encode())
    for stem in ("sample", "sample_brightness", "sample_color", "sample_flip", "sample_blur"):
        label = dataset_root / "labels" / f"{stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    preview = dataset_root / "annotation_previews" / "sample_reviewed.jpg"
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_bytes(b"preview")

    archive = exclude_image_family(dataset_root, source, reason="test")

    assert not source.exists()
    assert not (dataset_root / "augmented" / "sample_brightness.jpg").exists()
    assert not (dataset_root / "labels" / "sample.txt").exists()
    assert (archive / "raw" / "sample.jpg").read_bytes() == b"raw"
    assert (archive / "labels" / "sample_flip.txt").exists()
    record = json.loads((dataset_root / "annotations" / "exclusions.jsonl").read_text(encoding="utf-8"))
    assert record["source"] == "sample.jpg"
    assert record["reason"] == "test"
    assert len(record["files"]) == 11
