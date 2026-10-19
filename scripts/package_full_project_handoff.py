"""Create a complete, reproducible project handoff archive.

The archive contains source code, dataset artifacts, calibration/firmware
references, documentation and recovery notes. Machine-local environments and
generated caches are deliberately excluded because they are not portable.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_REGULATION = Path(r"D:\金砖大赛\第三届人工智能算法实践与应用赛项赛竞赛规程.pdf")
EXCLUDED_DIRS = {".git", ".pytest_cache", "__pycache__", ".venv", ".venv-annotation", "handoff_exports"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_excluded(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    if parts & {item.lower() for item in EXCLUDED_DIRS}:
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    if path.name.lower().endswith((".tmp", ".exclude_")):
        return True
    return False


def collect_files() -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or is_excluded(path.relative_to(ROOT)):
            continue
        files.append((path, path.relative_to(ROOT).as_posix()))
    if EXTERNAL_REGULATION.exists():
        files.append((EXTERNAL_REGULATION, "reference/第三届人工智能算法实践与应用赛项赛竞赛规程.pdf"))
    return files


def count_jsonl_images(path: Path) -> int:
    if not path.exists():
        return 0
    images: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            images.add(str(json.loads(line)["image"]))
    return len(images)


def dataset_snapshot() -> dict[str, int]:
    dataset = ROOT / "dataset"
    history = dataset / "annotations" / "history"
    excluded = dataset / "excluded"
    return {
        "raw_images": len(list((dataset / "raw").glob("*.jpg"))),
        "augmented_images": len(list((dataset / "augmented").glob("*.jpg"))),
        "label_files": len(list((dataset / "labels").glob("*.txt"))),
        "truth_images": count_jsonl_images(dataset / "annotations" / "truth.jsonl"),
        "annotation_history_files": sum(path.is_file() for path in history.rglob("*")) if history.exists() else 0,
        "excluded_archive_files": sum(path.is_file() for path in excluded.rglob("*")) if excluded.exists() else 0,
    }


def main() -> int:
    files = collect_files()
    if not files:
        raise SystemExit("项目中没有可打包文件，拒绝生成空交接包。")

    snapshot = dataset_snapshot()
    if snapshot["raw_images"] < 200:
        raise SystemExit(f"原图只有 {snapshot['raw_images']} 张，拒绝生成不完整项目包。")
    if snapshot["augmented_images"] != snapshot["raw_images"] * 4:
        raise SystemExit("四类增强图数量不完整，拒绝生成项目包。")
    if snapshot["truth_images"] and snapshot["label_files"] != snapshot["truth_images"] * 5:
        raise SystemExit("真值与标签数量不一致，拒绝生成项目包。")

    output_dir = ROOT / "handoff_exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = output_dir / f"gobang_full_handoff_{stamp}.zip"

    members: list[dict[str, object]] = []
    for source, archive_path in files:
        members.append({"path": archive_path, "size": source.stat().st_size, "sha256": sha256_file(source)})

    manifest = {
        "format": "gobang-full-project-handoff-v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_snapshot": snapshot,
        "excluded_from_archive": sorted(EXCLUDED_DIRS | {"*.pyc", "*.pyo", "*.tmp"}),
        "members": members,
        "instructions": "先解压到新的项目目录，再阅读 HANDOFF_README.md；不要直接覆盖另一份项目或手动合并同名文件。",
    }
    readme = f"""# 五子棋机器人项目完整交接包

生成时间：{manifest['created_at']}

数据快照：原图 {snapshot['raw_images']} 张，增强图 {snapshot['augmented_images']} 张，标签 {snapshot['label_files']} 份，真值 {snapshot['truth_images']} 张。

## 接手顺序

1. 将整个 ZIP 解压到一个新的目录，不要覆盖已有项目。
2. 阅读 `标注任务交接说明.md`、`README.md` 和本文件。
3. 先双击 `01_首次安装标注环境.bat` 建立本机的 `.venv-annotation`，再运行 `.\\.venv-annotation\\Scripts\\python.exe scripts\\verify_project_handoff.py` 自动核对清单和 SHA-256。
4. 只做离线标注时，接着依次运行 `02_标注环境自检.bat`、`03_启动标注工作台.bat`。
5. 已完成图片若需改错，在工作台选中图片后点击“修改已完成标注”；不要手工删除标签或真值日志。
6. 需要把新增标注交回时运行 `04_打包并返回标注成果.bat`；需要再次交接整个项目时运行 `05_打包完整项目交接包.bat`。
7. 只有实际连接机器人时才运行 `run.bat`。机器人、摄像头和串口不包含在 ZIP 中；首次硬件动作前必须重新确认端口、相机机位、标定和安全区域。

本包不包含 `.venv`、`.venv-annotation`、缓存和旧 handoff ZIP；这些内容不能跨电脑可靠复用。新电脑应使用 01 脚本建立自己的独立环境。

文件完整性请核对 `PROJECT_HANDOFF_MANIFEST.json` 中每个成员的大小和 SHA-256。
"""

    with zipfile.ZipFile(output, mode="x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for source, archive_path in files:
            archive.write(source, archive_path)
        archive.writestr("PROJECT_HANDOFF_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        archive.writestr("HANDOFF_README.md", readme.encode("utf-8-sig"))

    print(f"完整项目交接包：{output}")
    print(f"包含文件：{len(files)} 个；数据快照：{json.dumps(snapshot, ensure_ascii=False)}")
    print("已排除：项目虚拟环境、Python 缓存、旧交接 ZIP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
