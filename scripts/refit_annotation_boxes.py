"""Create reviewed label revisions whose boxes follow visible stone outlines.

The grid coordinates and black/white classes are never changed.  Before a
label family is replaced, the old labels and preview are archived by the same
revision mechanism used by the annotation workbench.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pc_app.annotation import (  # noqa: E402
    archive_annotation_family,
    build_truth,
    export_truth_family_labels,
    write_truth_record,
)
from pc_app.calibration import BoardCalibration  # noqa: E402
from pc_app.image_files import read_color_image, write_image  # noqa: E402
from pc_app.labels import draw_grid_annotations, fit_grid_box_overrides, project_grid_annotations  # noqa: E402


def load_latest_truths(path: Path) -> dict[str, dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            image = str(payload["image"])
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"{path.name} 第 {line_number} 行无效：{error}") from error
        if payload.get("status") == "reviewed":
            latest[image] = payload
    return latest


def main() -> int:
    parser = argparse.ArgumentParser(description="为已审核棋子标注生成按轮廓贴合的修订版本")
    parser.add_argument("--apply", action="store_true", help="实际写入修订；默认只统计")
    args = parser.parse_args()

    dataset = ROOT / "dataset"
    truth_path = dataset / "annotations" / "truth.jsonl"
    calibration = BoardCalibration.from_json(ROOT / "calibration" / "board.json")
    latest = load_latest_truths(truth_path)
    summary = {"truth_images": len(latest), "revised": 0, "fitted_boxes": 0, "unfitted_boxes": 0}

    for name, record in sorted(latest.items()):
        source = dataset / "raw" / name
        image = read_color_image(source)
        if image is None:
            raise OSError(f"无法读取原图：{source}")
        black = tuple(tuple(int(value) for value in move) for move in record.get("black_moves", []))
        white = tuple(tuple(int(value) for value in move) for move in record.get("white_moves", []))
        overrides = fit_grid_box_overrides(image, calibration, (*black, *white))
        summary["fitted_boxes"] += len(overrides)
        summary["unfitted_boxes"] += len(black) + len(white) - len(overrides)
        if not overrides:
            continue
        summary["revised"] += 1
        if not args.apply:
            continue
        truth = build_truth(
            source,
            calibration,
            black,
            white,
            radius=float(record["radius"]),
            annotator=str(record.get("annotator", "operator")),
            quality=str(record.get("quality", "accepted")),
            notes=(str(record.get("notes", "")).strip() + "; 自动按可见棋子轮廓贴合标注框").strip("; "),
            revision=int(record.get("revision", 1)) + 1,
            box_overrides=overrides,
        )
        archive_annotation_family(dataset, source, reason="自动按已确认棋子的可见轮廓贴合标注框")
        export_truth_family_labels(truth, source, calibration, dataset, overwrite=True)
        write_truth_record(truth_path, truth)
        annotations = project_grid_annotations(
            calibration,
            image.shape,
            black,
            white,
            radius=truth.radius,
            box_overrides=truth.box_overrides,
        )
        preview = draw_grid_annotations(image, annotations)
        write_image(dataset / "annotation_previews" / f"{source.stem}_reviewed.jpg", preview)

    print(json.dumps(summary, ensure_ascii=False))
    if not args.apply:
        print("未写入文件；如需生成修订，请追加 --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
