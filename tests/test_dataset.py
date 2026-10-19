import cv2
import numpy as np

from pc_app.dataset import DatasetManager


def test_dataset_saves_and_augments(tmp_path) -> None:
    manager = DatasetManager(tmp_path / "dataset")
    frame = np.full((24, 32, 3), 120, dtype=np.uint8)
    source = manager.save_frame(frame)
    outputs = manager.augment(source)
    assert source.suffix == ".jpg"
    assert len(outputs) == 4
    assert manager.counts() == (1, 4)
    assert all(cv2.imread(str(path)) is not None for path in outputs)


def test_dataset_tracks_competition_capture_categories(tmp_path) -> None:
    manager = DatasetManager(tmp_path / "dataset")
    frame = np.full((1080, 1920, 3), 120, dtype=np.uint8)
    manager.save_frame(frame, lighting="low", edge_scene=True, scene_id="scene_01", notes="edge layout")
    manager.save_frame(frame, lighting="high", edge_scene=False)

    stats = manager.stats()

    assert stats.raw == 2
    assert stats.low_light == 1
    assert stats.medium_light == 0
    assert stats.high_light == 1
    assert stats.edge_scenes == 1
    assert not stats.raw_ready
    records = manager._read_records()
    assert records[0].scene_id == "scene_01"
    assert records[0].notes == "edge layout"


def test_dataset_rejects_unknown_lighting_level(tmp_path) -> None:
    manager = DatasetManager(tmp_path / "dataset")
    frame = np.full((24, 32, 3), 120, dtype=np.uint8)

    try:
        manager.save_frame(frame, lighting="unknown")
    except ValueError as error:
        assert "光照档位" in str(error)
    else:
        raise AssertionError("unknown lighting levels must be rejected")


def test_dataset_augmentation_does_not_overwrite_existing_variant(tmp_path) -> None:
    manager = DatasetManager(tmp_path / "dataset")
    frame = np.full((24, 32, 3), 120, dtype=np.uint8)
    source = manager.save_frame(frame)
    first = manager.augment(source)
    marker = first[0].read_bytes()

    second = manager.augment(source)

    assert second == []
    assert first[0].read_bytes() == marker


def test_dataset_saves_valid_yolo_labels_without_overwrite(tmp_path) -> None:
    manager = DatasetManager(tmp_path / "dataset")
    frame = np.full((24, 32, 3), 120, dtype=np.uint8)
    source = manager.save_frame(frame)
    label = manager.save_yolo_label(source, [(0, 0.5, 0.5, 0.1, 0.1), (1, 0.2, 0.2, 0.08, 0.08)])

    assert label.read_text(encoding="utf-8").count("\n") == 2
    try:
        manager.save_yolo_label(source, [(0, 0.5, 0.5, 0.1, 0.1)])
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing labels must not be overwritten")
