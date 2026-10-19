"""Verify an extracted full-project handoff against its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a complete Gobang handoff archive after extraction")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "PROJECT_HANDOFF_MANIFEST.json"
    if not manifest_path.exists():
        raise SystemExit(f"找不到项目清单：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        raise SystemExit("项目清单没有有效成员")

    failures: list[str] = []
    for row in members:
        if not isinstance(row, dict):
            failures.append(f"清单成员格式无效：{row!r}")
            continue
        relative = row.get("path")
        if not isinstance(relative, str):
            failures.append(f"清单路径无效：{row!r}")
            continue
        path = (root / Path(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            failures.append(f"清单路径越出项目目录：{relative}")
            continue
        if not path.is_file():
            failures.append(f"缺少文件：{relative}")
            continue
        expected_size = int(row.get("size", -1))
        expected_hash = str(row.get("sha256", ""))
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_size != expected_size:
            failures.append(f"大小不一致：{relative}（清单 {expected_size}，实际 {actual_size}）")
        if actual_hash != expected_hash:
            failures.append(f"SHA-256 不一致：{relative}")

    if failures:
        print("[失败] 交接包完整性校验未通过")
        for failure in failures[:20]:
            print(f"- {failure}")
        if len(failures) > 20:
            print(f"- 其余 {len(failures) - 20} 项省略")
        return 1
    print(f"[通过] 已核对 {len(members)} 个文件，大小和 SHA-256 全部一致")
    print(f"[数据] {json.dumps(manifest.get('dataset_snapshot', {}), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
