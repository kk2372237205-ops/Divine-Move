from pathlib import Path

import cv2
import numpy as np
import pytest

from pc_app.annotation import (
    build_truth,
    derive_flipped_yolo,
    archive_annotation_family,
    estimate_stone_radius,
    export_truth_family_labels,
    project_truth_annotations,
    propose_grid_stones,
    sha256_file,
    write_truth_record,
)
from pc_app.calibration import BoardCalibration
from pc_app.dataset import DatasetManager
from pc_app.labels import BoxOverride


def identity_calibration() -> BoardCalibration:
    return BoardCalibration(
        corners=((0, 0), (519, 0), (519, 519), (0, 519)),
        output_size=520,
        board_size=13,
    )


def test_flip_yolo_mirrors_only_center_x() -> None:
    assert derive_flipped_yolo(((0, 0.2, 0.3, 0.1, 0.2), (1, 0.8, 0.4, 0.2, 0.1))) == (
        (0, 0.8, 0.3, 0.1, 0.2),
        (1, 0.19999999999999996, 0.4, 0.2, 0.1),
    )


def test_truth_requires_review_before_export_and_keeps_hash(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    image = np.full((520, 520, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(image_path), image)
    truth = build_truth(
        image_path,
        identity_calibration(),
        [(6, 6)],
        [],
        radius=10,
        annotator="tester",
    )
    assert truth.source_sha256 == sha256_file(image_path)
    assert project_truth_annotations(truth, image_path, identity_calibration())[0][0] == 0
    log = tmp_path / "annotations.jsonl"
    write_truth_record(log, truth)
    assert log.read_text(encoding="utf-8").count("\n") == 1


def test_truth_rejects_changed_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(image_path), np.full((520, 520, 3), 120, dtype=np.uint8))
    truth = build_truth(image_path, identity_calibration(), [(6, 6)], [], radius=10, annotator="tester")
    image_path.write_bytes(b"changed")
    with pytest.raises((OSError, ValueError)):
        project_truth_annotations(truth, image_path, identity_calibration())


def test_truth_preserves_per_stone_box_override(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(image_path), np.full((520, 520, 3), 120, dtype=np.uint8))
    truth = build_truth(
        image_path,
        identity_calibration(),
        [(6, 6)],
        [],
        radius=10,
        annotator="tester",
        box_overrides=(BoxOverride(move=(6, 6), center=(270.0, 250.0), radius=12.0),),
    )

    assert truth.as_json()["box_overrides"] == [{"move": [6, 6], "center": [270.0, 250.0], "radius": 12.0}]
    projected = project_truth_annotations(truth, image_path, identity_calibration())
    assert projected[0][1:3] == pytest.approx((270.0 / 520, 250.0 / 520))


def test_reviewed_truth_exports_source_and_all_four_augmented_labels(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    manager = DatasetManager(dataset_root)
    image_path = dataset_root / "raw" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    cv2.imwrite(str(image_path), np.full((520, 520, 3), 120, dtype=np.uint8))
    manager.augment(image_path)
    truth = build_truth(
        image_path,
        identity_calibration(),
        [(6, 6)],
        [(12, 0)],
        radius=10,
        annotator="tester",
    )

    outputs = export_truth_family_labels(truth, image_path, identity_calibration(), dataset_root)

    assert len(outputs) == 5
    assert all(path.exists() for path in outputs)
    trace_rows = (dataset_root / "annotations" / "derived_labels.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(trace_rows) == 5
    assert '"variant": "flip"' in trace_rows[3]
    flip_rows = (dataset_root / "labels" / "sample_flip.txt").read_text(encoding="utf-8").splitlines()
    assert any(row.startswith("0 0.500962 0.499038") for row in flip_rows)
    with pytest.raises(FileExistsError):
        export_truth_family_labels(truth, image_path, identity_calibration(), dataset_root)


def test_archive_annotation_family_preserves_existing_files_without_overwrite(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    manager = DatasetManager(dataset_root)
    image_path = dataset_root / "raw" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    cv2.imwrite(str(image_path), np.full((520, 520, 3), 120, dtype=np.uint8))
    manager.augment(image_path)
    truth = build_truth(image_path, identity_calibration(), [(6, 6)], [], radius=10, annotator="tester")
    export_truth_family_labels(truth, image_path, identity_calibration(), dataset_root)
    preview_dir = dataset_root / "annotation_previews"
    preview_dir.mkdir(parents=True)
    preview = preview_dir / "sample_reviewed.jpg"
    preview.write_bytes(b"preview")
    before = (dataset_root / "labels" / "sample.txt").read_bytes()

    archive = archive_annotation_family(dataset_root, image_path, reason="test")

    assert archive is not None
    assert (archive / "labels" / "sample.txt").read_bytes() == before
    assert (archive / "annotation_previews" / "sample_reviewed.jpg").read_bytes() == b"preview"
    assert (archive / "archive_manifest.json").exists()


def test_proposal_rejects_empty_image() -> None:
    with pytest.raises(ValueError):
        propose_grid_stones(np.empty((0, 0, 3), dtype=np.uint8), identity_calibration())


def test_estimate_radius_uses_representative_real_images() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "dataset/raw/board_low_normal_20260829_001227_385540.jpg",
        root / "dataset/raw/board_medium_normal_20260828_234912_704741.jpg",
        root / "dataset/raw/board_high_normal_20260829_002205_224372.jpg",
    ]
    if not all(path.exists() for path in paths):
        pytest.skip("real capture set is not present")
    radius = estimate_stone_radius(paths, __import__("pc_app.calibration", fromlist=["BoardCalibration"]).BoardCalibration.from_json(root / "calibration/board.json"))
    assert 12 < radius < 21
