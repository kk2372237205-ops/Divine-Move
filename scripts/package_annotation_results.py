"""Create a non-overwriting ZIP containing only annotation return artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_files() -> list[Path]:
    patterns = (
        DATASET / "annotations" / "truth.jsonl",
        DATASET / "annotations" / "derived_labels.jsonl",
    )
    files = [path for path in patterns if path.exists()]
    files.extend(sorted((DATASET / "labels").glob("*.txt")))
    files.extend(sorted((DATASET / "annotation_previews").glob("*.jpg")))
    calibration = ROOT / "calibration" / "board.json"
    if calibration.exists():
        files.append(calibration)
    return files


def count_truths(path: Path) -> int:
    if not path.exists():
        return 0
    images: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            images.add(str(json.loads(line)["image"]))
    return len(images)


def main() -> int:
    truth_path = DATASET / "annotations" / "truth.jsonl"
    truth_count = count_truths(truth_path)
    if truth_count == 0:
        raise SystemExit("尚无已完成的真值记录，拒绝生成空成果包。")

    files = collect_files()
    label_count = sum(path.parent == DATASET / "labels" for path in files)
    if label_count != truth_count * 5:
        raise SystemExit(
            f"标签数量异常：{truth_count} 张真值应对应 {truth_count * 5} 份标签，实际 {label_count} 份。"
        )

    output_dir = ROOT / "handoff_exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = output_dir / f"annotation_results_{stamp}.zip"

    members: list[dict[str, object]] = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        members.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "format": "gobang-annotation-handoff-v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "truth_images": truth_count,
        "label_files": label_count,
        "preview_files": sum("annotation_previews" in str(path) for path in files),
        "members": members,
        "merge_warning": "Do not extract over another project copy. Verify and merge without overwriting.",
    }
    instructions = (
        "这是五子棋标注成果包。\r\n"
        "请把整个 ZIP 原样发回，不要拆分、改名或直接覆盖另一份项目。\r\n"
        "核心真值：dataset/annotations/truth.jsonl\r\n"
        "YOLO 标签：dataset/labels/*.txt\r\n"
        "核验预览：dataset/annotation_previews/*.jpg\r\n"
        "收到后应先核对清单与哈希，再以不覆盖方式合并。\r\n"
    )

    with zipfile.ZipFile(output, mode="x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr(
            "ANNOTATION_HANDOFF_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        archive.writestr("请先阅读.txt", instructions.encode("utf-8-sig"))

    print(f"已完成真值：{truth_count} 张")
    print(f"已打包标签：{label_count} 份")
    print(f"成果包：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
