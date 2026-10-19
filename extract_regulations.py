from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


PDF = Path(r"C:\Users\23722\Desktop\金砖大赛\第三届人工智能算法实践与应用赛项赛竞赛规程.pdf")
OUT = Path(r"C:\Users\23722\Desktop\五子棋机器人\_analysis")


def main() -> None:
    reader = PdfReader(PDF)
    chunks = [f"页数：{len(reader.pages)}", ""]
    for number, page in enumerate(reader.pages, start=1):
        chunks.append(f"\n===== 第 {number} 页 =====\n")
        chunks.append(page.extract_text() or "[无可提取文字]")
    (OUT / "competition_regulations_extracted.txt").write_text("\n".join(chunks), encoding="utf-8")
    print(f"pages={len(reader.pages)}")


if __name__ == "__main__":
    main()
