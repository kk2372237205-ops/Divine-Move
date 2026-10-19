from __future__ import annotations

from pathlib import Path

from pdf2image import convert_from_path


SOURCE = Path(r"C:\Users\23722\Desktop\金砖大赛\第三届人工智能算法实践与应用赛项赛竞赛规程.pdf")
OUT = Path(r"C:\Users\23722\Desktop\五子棋机器人\_analysis\competition_pages")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(SOURCE, dpi=150, fmt="png", thread_count=2)
    for index, page in enumerate(pages, start=1):
        page.save(OUT / f"page-{index:02d}.png", "PNG")
    print(f"rendered={len(pages)}")


if __name__ == "__main__":
    main()
