"""Grid-aware annotation proposals and traceable YOLO label derivation.

The competition board is a fixed 13 x 13 grid.  This module deliberately
does not claim that an image-only proposal is ground truth: proposals are
reviewed in the annotation workbench before they are exported as labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable

import cv2
import numpy as np

from .board import BOARD_SIZE, Move
from .calibration import BoardCalibration
from .dataset import DatasetManager
from .image_files import read_color_image
from .labels import BoxOverride


@dataclass(frozen=True)
class GridProposal:
    """A candidate occupied grid point proposed for human review."""

    move: Move
    class_id: int
    confidence: float
    contrast: float
    center_value: float
    ring_value: float


@dataclass(frozen=True)
class AnnotationTruth:
    """Versioned human-accepted source annotation for one original image."""

    image: str
    source_sha256: str
    width: int
    height: int
    black_moves: tuple[Move, ...]
    white_moves: tuple[Move, ...]
    radius: float
    status: str
    quality: str
    annotator: str
    reviewed_at: str
    notes: str = ""
    revision: int = 1
    box_overrides: tuple[BoxOverride, ...] = ()

    def as_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["black_moves"] = [list(move) for move in self.black_moves]
        payload["white_moves"] = [list(move) for move in self.white_moves]
        payload["box_overrides"] = [item.as_json() for item in self.box_overrides]
        return payload


def sha256_file(path: Path) -> str:
    """Return a content hash so labels can be traced to immutable image bytes."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def propose_grid_stones(
    image: np.ndarray,
    calibration: BoardCalibration,
    *,
    minimum_contrast: float = 8.0,
) -> tuple[GridProposal, ...]:
    """Propose occupied cells from local grid-point contrast.

    The local annulus removes most illumination gradients.  A negative
    center-versus-ring contrast suggests black and a positive contrast
    suggests white.  The result is intentionally a proposal only; a human
    must accept or correct every image before label export.
    """

    if image is None or image.size == 0:
        raise ValueError("不能在空图像上生成标注候选")
    warped = calibration.warp(image)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    spacing = (calibration.output_size - 1) / (calibration.board_size - 1)
    yy, xx = np.ogrid[: calibration.output_size, : calibration.output_size]
    contrasts: list[tuple[Move, float, float, float]] = []
    for row in range(calibration.board_size):
        for column in range(calibration.board_size):
            center_x, center_y = calibration.board_pixel((column, row))
            distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
            center = gray[distance <= spacing * 0.25]
            ring = gray[(distance >= spacing * 0.39) & (distance <= spacing * 0.49)]
            if center.size == 0 or ring.size == 0:
                continue
            center_value = float(np.median(center))
            ring_value = float(np.median(ring))
            contrasts.append(((column, row), center_value - ring_value, center_value, ring_value))
    if not contrasts:
        return ()

    values = np.asarray([item[1] for item in contrasts], dtype=np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(float(minimum_contrast), 2.5 * mad)
    proposals: list[GridProposal] = []
    for move, contrast, center_value, ring_value in contrasts:
        distance = abs(contrast - median)
        if distance < threshold:
            continue
        class_id = 0 if contrast < median else 1
        confidence = min(1.0, max(0.0, (distance - threshold) / max(threshold, 1.0)))
        proposals.append(
            GridProposal(
                move=move,
                class_id=class_id,
                confidence=confidence,
                contrast=contrast,
                center_value=center_value,
                ring_value=ring_value,
            )
        )
    return tuple(sorted(proposals, key=lambda proposal: proposal.move[1:]))


def estimate_stone_radius(
    image_paths: Iterable[Path],
    calibration: BoardCalibration,
    *,
    min_radius: float = 12.0,
    max_radius: float = 21.0,
) -> float:
    """Estimate a common warped-image stone radius from representative images."""

    radii: list[float] = []
    for path in image_paths:
        image = read_color_image(path)
        if image is None:
            continue
        warped = calibration.warp(image)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=20.0,
            param1=80.0,
            param2=18.0,
            minRadius=int(min_radius),
            maxRadius=int(max_radius),
        )
        if circles is None:
            continue
        radii.extend(
            float(radius)
            for radius in circles[0, :, 2]
            if min_radius <= float(radius) <= max_radius
        )
    spacing = (calibration.output_size - 1) / (calibration.board_size - 1)
    if not radii:
        raise ValueError("代表图中未估计到棋子轮廓，请人工确认框半径")
    estimate = float(np.median(np.asarray(radii, dtype=np.float32)))
    if estimate * 2 >= spacing:
        raise ValueError("估计棋子半径已接近网格间距，拒绝生成可能重叠的标注框")
    return estimate


def build_truth(
    image_path: Path,
    calibration: BoardCalibration,
    black_moves: Iterable[Move],
    white_moves: Iterable[Move],
    *,
    radius: float,
    annotator: str,
    quality: str = "accepted",
    status: str = "reviewed",
    notes: str = "",
    revision: int = 1,
    box_overrides: Iterable[BoxOverride] = (),
) -> AnnotationTruth:
    """Create a checked source truth record without writing any files."""

    image = read_color_image(image_path)
    if image is None:
        raise OSError(f"无法读取待标注图像：{image_path}")
    black = tuple(sorted(set(black_moves), key=lambda move: (move[1], move[0])))
    white = tuple(sorted(set(white_moves), key=lambda move: (move[1], move[0])))
    all_moves = black + white
    if not all_moves:
        raise ValueError("至少需要一个黑棋或白棋坐标")
    if len(set(all_moves)) != len(all_moves):
        raise ValueError("黑棋和白棋坐标不能重复")
    for column, row in all_moves:
        if not (0 <= column < calibration.board_size and 0 <= row < calibration.board_size):
            raise ValueError(f"棋盘坐标越界：{(column, row)}")
    if radius <= 0 or radius * 2 >= (calibration.output_size - 1) / (calibration.board_size - 1):
        raise ValueError("标注框半径必须为正数且小于半个网格间距")
    overrides = tuple(box_overrides)
    if len({item.move for item in overrides}) != len(overrides):
        raise ValueError("同一棋子不能有多个框修正")
    if not set(item.move for item in overrides).issubset(set(all_moves)):
        raise ValueError("框修正必须对应当前已确认的棋子")
    return AnnotationTruth(
        image=image_path.name,
        source_sha256=sha256_file(image_path),
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        black_moves=black,
        white_moves=white,
        radius=float(radius),
        status=status,
        quality=quality,
        annotator=annotator,
        reviewed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=notes,
        revision=revision,
        box_overrides=overrides,
    )


def export_truth_label(
    truth: AnnotationTruth,
    image_path: Path,
    calibration: BoardCalibration,
    labels_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export one reviewed source truth to YOLO format."""

    if truth.status != "reviewed":
        raise ValueError("只有 reviewed 真值才能导出正式标签")
    manager = DatasetManager(labels_dir.parent)
    annotations = project_truth_annotations(truth, image_path, calibration)
    return manager.save_yolo_label(image_path, annotations, overwrite=overwrite)


def export_truth_family_labels(
    truth: AnnotationTruth,
    image_path: Path,
    calibration: BoardCalibration,
    dataset_root: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Export one reviewed source label and its four deterministic variants.

    All target images and label paths are checked before the first write.  This
    keeps a missing augmentation or an existing label from producing a silent
    half-exported family.  The source truth is always projected from the
    original image; only the horizontal flip changes its normalized x-center.
    """

    if truth.status != "reviewed":
        raise ValueError("只有 reviewed 真值才能导出标签族")
    image_path = Path(image_path)
    dataset = DatasetManager(dataset_root)
    source_annotations = project_truth_annotations(truth, image_path, calibration)
    variants = (
        ("source", image_path, source_annotations),
        ("brightness", dataset.augmented_dir / f"{image_path.stem}_brightness.jpg", source_annotations),
        ("color", dataset.augmented_dir / f"{image_path.stem}_color.jpg", source_annotations),
        ("flip", dataset.augmented_dir / f"{image_path.stem}_flip.jpg", derive_flipped_yolo(source_annotations)),
        ("blur", dataset.augmented_dir / f"{image_path.stem}_blur.jpg", source_annotations),
    )
    targets: list[tuple[str, Path, tuple[tuple[int, float, float, float, float], ...], Path]] = []
    for variant, image, annotations in variants:
        if not image.exists():
            raise FileNotFoundError(f"增强图不存在，不能导出完整标签族：{image}")
        label_path = dataset.labels_dir / f"{image.stem}.txt"
        if label_path.exists() and not overwrite:
            raise FileExistsError(f"标签已存在，为避免覆盖而拒绝写入：{label_path}")
        targets.append((variant, image, annotations, label_path))

    if sha256_file(image_path) != truth.source_sha256:
        raise ValueError("图像哈希与真值记录不一致，拒绝导出标签族")

    outputs: list[Path] = []
    trace_rows: list[dict[str, object]] = []
    for variant, image, annotations, _label_path in targets:
        label_path = dataset.save_yolo_label(image, annotations, overwrite=overwrite)
        outputs.append(label_path)
        trace_rows.append(
            {
                "source": image_path.name,
                "source_sha256": truth.source_sha256,
                "variant": variant,
                "image": image.name,
                "image_sha256": sha256_file(image),
                "label": label_path.name,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
    trace_path = dataset.root / "annotations" / "derived_labels.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as stream:
        for row in trace_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return tuple(outputs)


def project_truth_annotations(
    truth: AnnotationTruth,
    image_path: Path,
    calibration: BoardCalibration,
) -> tuple[tuple[int, float, float, float, float], ...]:
    """Project reviewed grid positions into original-image YOLO boxes."""

    from .labels import project_grid_annotations

    image = read_color_image(image_path)
    if image is None:
        raise OSError(f"无法读取图像：{image_path}")
    if sha256_file(image_path) != truth.source_sha256:
        raise ValueError("图像哈希与真值记录不一致，拒绝导出标签")
    projected = project_grid_annotations(
        calibration,
        image.shape,
        truth.black_moves,
        truth.white_moves,
        radius=truth.radius,
        box_overrides=truth.box_overrides,
    )
    return tuple(annotation.yolo for annotation in projected)


def derive_flipped_yolo(
    annotations: Iterable[tuple[int, float, float, float, float]],
) -> tuple[tuple[int, float, float, float, float], ...]:
    """Mirror YOLO boxes for a horizontal flip while preserving classes."""

    mirrored: list[tuple[int, float, float, float, float]] = []
    for class_id, center_x, center_y, width, height in annotations:
        center_x = 1.0 - float(center_x)
        width = float(width)
        # A clipped edge box can leave a tiny negative value after the
        # subtraction above (for example -3e-17).  Clamp the center to the
        # nearest legal position while preserving the box width.
        half_width = width / 2.0
        center_x = max(half_width, min(1.0 - half_width, center_x))
        mirrored.append((int(class_id), center_x, float(center_y), width, float(height)))
    return tuple(mirrored)


def write_truth_record(path: Path, truth: AnnotationTruth) -> None:
    """Append a truth revision; never overwrite an existing annotation log."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(truth.as_json(), ensure_ascii=False) + "\n")


def archive_annotation_family(
    dataset_root: Path,
    image_path: Path,
    *,
    reason: str = "",
) -> Path | None:
    """Copy active labels/previews aside before an explicit truth revision.

    Existing files are never overwritten by this operation. Missing files are
    simply absent from the archive manifest; the caller can still regenerate
    them from the new reviewed truth.
    """

    dataset_root = Path(dataset_root)
    image_path = Path(image_path)
    dataset = DatasetManager(dataset_root)
    stem = image_path.stem
    candidates = [
        dataset.labels_dir / f"{stem}.txt",
        *(dataset.labels_dir / f"{stem}_{suffix}.txt" for suffix in ("brightness", "color", "flip", "blur")),
    ]
    preview_dir = dataset.root / "annotation_previews"
    candidates.extend(sorted(preview_dir.glob(f"{stem}_reviewed*.jpg")))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None

    history_root = dataset.root / "annotations" / "history" / stem
    history_root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10000):
        archive_root = history_root / f"revision_backup_{index:04d}"
        if not archive_root.exists():
            break
    else:
        raise OSError(f"无法为标注修订创建不重复备份目录：{history_root}")

    manifest: list[dict[str, str | int]] = []
    for source in existing:
        relative = source.relative_to(dataset.root)
        destination = archive_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest.append(
            {
                "source": relative.as_posix(),
                "archived": destination.relative_to(archive_root).as_posix(),
                "size": int(source.stat().st_size),
                "sha256": sha256_file(source),
            }
        )
    payload = {
        "image": image_path.name,
        "reason": reason,
        "archived_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": manifest,
    }
    (archive_root / "archive_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return archive_root


__all__ = [
    "AnnotationTruth",
    "GridProposal",
    "archive_annotation_family",
    "build_truth",
    "derive_flipped_yolo",
    "estimate_stone_radius",
    "export_truth_family_labels",
    "export_truth_label",
    "project_truth_annotations",
    "propose_grid_stones",
    "sha256_file",
    "write_truth_record",
]
