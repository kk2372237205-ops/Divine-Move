"""Read-only validation for a teammate taking over offline annotation."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fail(message: str) -> None:
    print(f"[失败] {message}")
    raise SystemExit(1)


def main() -> int:
    if sys.version_info[:2] != (3, 10):
        fail(f"需要 Python 3.10，当前是 {sys.version.split()[0]}")

    try:
        import cv2
        import numpy as np
        from PIL import Image  # noqa: F401
        import serial  # noqa: F401
        import tkinter  # noqa: F401
    except ImportError as error:
        fail(f"依赖导入失败：{error}")

    from pc_app.annotation import propose_grid_stones
    from pc_app.calibration import BoardCalibration
    from pc_app.image_files import read_color_image

    dataset = ROOT / "dataset"
    raw_dir = dataset / "raw"
    augmented_dir = dataset / "augmented"
    labels_dir = dataset / "labels"
    truth_path = dataset / "annotations" / "truth.jsonl"
    exclusions_path = dataset / "annotations" / "exclusions.jsonl"
    calibration_path = ROOT / "calibration" / "board.json"

    required = (raw_dir, augmented_dir, labels_dir, calibration_path)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        fail("缺少项目内容：" + "、".join(missing))

    raw_paths = sorted(raw_dir.glob("*.jpg"))
    augmented_paths = sorted(augmented_dir.glob("*.jpg"))
    if len(raw_paths) < 200:
        fail(f"原图只有 {len(raw_paths)} 张，少于规程要求的 200 张")

    suffixes = ("brightness", "color", "flip", "blur")
    missing_variants: list[str] = []
    for source in raw_paths:
        for suffix in suffixes:
            target = augmented_dir / f"{source.stem}_{suffix}.jpg"
            if not target.exists():
                missing_variants.append(target.name)
    if missing_variants:
        sample = "、".join(missing_variants[:5])
        fail(f"缺少 {len(missing_variants)} 张增强图，例如：{sample}")

    calibration = BoardCalibration.from_json(calibration_path)
    sample = read_color_image(raw_paths[0])
    if sample is None:
        fail(f"无法读取代表原图：{raw_paths[0].name}")
    padded = calibration.warp_padded(sample, padding=24)
    if padded.shape != (calibration.output_size + 48, calibration.output_size + 48, 3):
        fail(f"校正预览尺寸异常：{padded.shape}")
    proposals = propose_grid_stones(sample, calibration)

    truths: dict[str, dict[str, object]] = {}
    if truth_path.exists():
        for line_number, line in enumerate(truth_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                image_name = str(payload["image"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                fail(f"truth.jsonl 第 {line_number} 行无效：{error}")
            truths[image_name] = payload

    # truth.jsonl is an append-only history and deliberately keeps the records of
    # samples that were later excluded.  Such a sample has already been archived
    # under dataset/excluded/, so its labels no longer exist by design.  Reading
    # exclusions.jsonl first is what keeps an intentional exclusion from being
    # reported as a missing-label failure, and lets the checks below still run
    # over every remaining reviewed image instead of stopping at the first one.
    excluded_stems: set[str] = set()
    if exclusions_path.exists():
        for line_number, line in enumerate(exclusions_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                fail(f"exclusions.jsonl 第 {line_number} 行无效：{error}")
            source = payload.get("source")
            if not source:
                fail(f"exclusions.jsonl 第 {line_number} 行缺少 source 字段")
            excluded_stems.add(Path(str(source)).stem)

    expected_labels = 0
    missing_labels: list[str] = []
    for image_name in truths:
        stem = Path(image_name).stem
        if stem in excluded_stems:
            continue
        expected_labels += 5
        names = [f"{stem}.txt", *(f"{stem}_{suffix}.txt" for suffix in suffixes)]
        missing_labels.extend(name for name in names if not (labels_dir / name).exists())
    if missing_labels:
        sample = "、".join(sorted(missing_labels)[:5])
        fail(f"已完成真值缺少 {len(missing_labels)} 个标签文件，例如：{sample}")

    label_count = len(list(labels_dir.glob("*.txt")))
    if label_count != expected_labels:
        fail(f"标签数量不一致：未排除真值 {expected_labels // 5} 张应有 {expected_labels} 份，实际 {label_count} 份")

    print(f"[通过] Python {sys.version.split()[0]}")
    print(f"[通过] OpenCV {cv2.__version__} / NumPy {np.__version__}")
    print(f"[通过] 原图 {len(raw_paths)} 张 / 增强图 {len(augmented_paths)} 张，四类增强完整")
    print(f"[通过] 棋盘标定可用，代表图候选 {len(proposals)} 个")
    print(f"[通过] 已完成真值 {len(truths)} 张（其中已排除归档 {len(excluded_stems)} 张）/ 标签 {label_count} 份")
    print("[安全] 本次检查没有打开摄像头、串口或机械臂")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
