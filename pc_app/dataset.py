"""Competition dataset capture and deterministic augmentation."""

from __future__ import annotations

from datetime import datetime
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .image_files import read_color_image, write_image


LIGHTING_LEVELS = ("low", "medium", "high")
YOLO_CLASS_NAMES = ("black", "white")


@dataclass(frozen=True)
class DatasetStats:
    raw: int
    augmented: int
    labeled: int
    low_light: int
    medium_light: int
    high_light: int
    edge_scenes: int

    @property
    def raw_ready(self) -> bool:
        return self.raw >= 200

    @property
    def augmented_ready(self) -> bool:
        return self.augmented >= 800

    @property
    def labeled_ready(self) -> bool:
        return self.labeled >= 20


@dataclass(frozen=True)
class CaptureRecord:
    file: str
    captured_at: str
    lighting: str
    edge_scene: bool
    width: int
    height: int
    scene_id: str = ""
    notes: str = ""


class DatasetManager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.raw_dir = root / "raw"
        self.augmented_dir = root / "augmented"
        self.labels_dir = root / "labels"
        self.manifest_path = root / "manifest.jsonl"

    def save_frame(
        self,
        frame: np.ndarray,
        prefix: str = "board",
        *,
        lighting: str = "medium",
        edge_scene: bool = False,
        scene_id: str = "",
        notes: str = "",
    ) -> Path:
        if frame is None or frame.size == 0:
            raise ValueError("不能保存空图像")
        if lighting not in LIGHTING_LEVELS:
            raise ValueError(f"光照档位必须是: {', '.join(LIGHTING_LEVELS)}")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        scene = "edge" if edge_scene else "normal"
        path = self.raw_dir / f"{prefix}_{lighting}_{scene}_{stamp}.jpg"
        suffix = 1
        while path.exists():
            path = self.raw_dir / f"{prefix}_{lighting}_{scene}_{stamp}_{suffix:02d}.jpg"
            suffix += 1
        write_image(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        height, width = frame.shape[:2]
        self._append_record(
            CaptureRecord(
                file=path.name,
                captured_at=datetime.now().isoformat(timespec="seconds"),
                lighting=lighting,
                edge_scene=edge_scene,
                width=int(width),
                height=int(height),
                scene_id=scene_id.strip(),
                notes=notes.strip(),
            )
        )
        return path

    def augment(self, source: Path, *, overwrite: bool = False) -> list[Path]:
        """Create the four required variants without overwriting by default."""
        image = read_color_image(source)
        if image is None:
            raise OSError(f"无法读取原图: {source}")
        stem = source.stem
        variants = {
            "brightness": self._brightness(image),
            "color": self._color(image),
            "flip": cv2.flip(image, 1),
            "blur": cv2.GaussianBlur(image, (5, 5), 0),
        }
        paths: list[Path] = []
        for suffix, variant in variants.items():
            path = self.augmented_dir / f"{stem}_{suffix}.jpg"
            if path.exists() and not overwrite:
                continue
            write_image(path, variant, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(path)
        return paths

    def augment_all_raw(self, *, overwrite: bool = False) -> list[Path]:
        outputs: list[Path] = []
        for source in sorted(self.raw_dir.glob("*.jpg")):
            outputs.extend(self.augment(source, overwrite=overwrite))
        return outputs

    def save_yolo_label(
        self,
        image: Path | str,
        annotations: Iterable[tuple[int, float, float, float, float]],
        *,
        overwrite: bool = False,
    ) -> Path:
        """Save ``class cx cy width height`` annotations for one image.

        Coordinates are normalized to ``[0, 1]`` as required by YOLO.  The
        label file is created only when it does not already exist unless the
        caller explicitly opts into replacement.
        """
        image_path = Path(image)
        if not image_path.is_absolute():
            image_path = self.raw_dir / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"原图不存在：{image_path}")
        rows: list[str] = []
        for annotation in annotations:
            if len(annotation) != 5:
                raise ValueError("标注必须是 class_id、中心 x、中心 y、宽度、高度")
            class_id, center_x, center_y, width, height = annotation
            if int(class_id) != class_id or int(class_id) not in range(len(YOLO_CLASS_NAMES)):
                raise ValueError(f"类别编号必须是 0（黑棋）或 1（白棋）：{class_id}")
            values = (float(center_x), float(center_y), float(width), float(height))
            if not all(0.0 <= value <= 1.0 for value in values[:2]) or not all(
                0.0 < value <= 1.0 for value in values[2:]
            ):
                raise ValueError("YOLO 中心坐标必须在 [0,1]，宽高必须在 (0,1]")
            if values[0] - values[2] / 2 < 0 or values[0] + values[2] / 2 > 1:
                raise ValueError("标注框横向越出图像范围")
            if values[1] - values[3] / 2 < 0 or values[1] + values[3] / 2 > 1:
                raise ValueError("标注框纵向越出图像范围")
            rows.append(f"{int(class_id)} " + " ".join(f"{value:.6f}" for value in values))
        if not rows:
            raise ValueError("至少需要一个棋子标注")
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        if label_path.exists() and not overwrite:
            raise FileExistsError(f"标注已存在，为避免覆盖而拒绝写入：{label_path}")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return label_path

    def counts(self) -> tuple[int, int]:
        stats = self.stats()
        return stats.raw, stats.augmented

    def stats(self) -> DatasetStats:
        raw_files = list(self.raw_dir.glob("*.jpg")) if self.raw_dir.exists() else []
        augmented = len(list(self.augmented_dir.glob("*.jpg"))) if self.augmented_dir.exists() else 0
        labeled = len(list(self.labels_dir.glob("*.txt"))) if self.labels_dir.exists() else 0
        records = self._read_records()
        known_files = {record.file for record in records}
        for path in raw_files:
            if path.name in known_files:
                continue
            lighting = next((level for level in LIGHTING_LEVELS if f"_{level}_" in path.name), "medium")
            records.append(
                CaptureRecord(
                    file=path.name,
                    captured_at="",
                    lighting=lighting,
                    edge_scene="_edge_" in path.name,
                    width=0,
                    height=0,
                )
            )
        return DatasetStats(
            raw=len(raw_files),
            augmented=augmented,
            labeled=labeled,
            low_light=sum(record.lighting == "low" for record in records),
            medium_light=sum(record.lighting == "medium" for record in records),
            high_light=sum(record.lighting == "high" for record in records),
            edge_scenes=sum(record.edge_scene for record in records),
        )

    def _append_record(self, record: CaptureRecord) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def _read_records(self) -> list[CaptureRecord]:
        if not self.manifest_path.exists():
            return []
        records: list[CaptureRecord] = []
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records.append(CaptureRecord(**payload))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return records

    @staticmethod
    def _brightness(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 28)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def _color(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 24)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
