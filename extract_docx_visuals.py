from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"C:\Users\23722\Desktop\五子棋机器人")
SOURCE = ROOT / "_materials_readonly" / "下棋机械臂资源-0701"
OUT = ROOT / "_analysis" / "docx_visuals"


def font(size: int):
    for candidate in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def make_sheet(doc_name: str, images: list[tuple[str, Image.Image]]) -> None:
    thumb_size = (350, 260)
    caption_height = 48
    cols = 3
    rows = max(1, (len(images) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * thumb_size[0], rows * (thumb_size[1] + caption_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (name, image) in enumerate(images):
        image = image.convert("RGB")
        image.thumbnail(thumb_size)
        left = (index % cols) * thumb_size[0] + (thumb_size[0] - image.width) // 2
        top = (index // cols) * (thumb_size[1] + caption_height) + (thumb_size[1] - image.height) // 2
        sheet.paste(image, (left, top))
        draw.text((index % cols * thumb_size[0] + 6, (index // cols + 1) * thumb_size[1] + (index // cols) * caption_height + 6), name, fill="black", font=font(14))
    safe = doc_name.replace("/", "_").replace("\\", "_")
    sheet.save(OUT / f"{safe}.jpg", quality=90)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for docx in sorted(SOURCE.rglob("*.docx")):
        display_images = []
        doc_out = OUT / docx.stem
        doc_out.mkdir(exist_ok=True)
        with zipfile.ZipFile(docx) as archive:
            media = sorted(name for name in archive.namelist() if name.startswith("word/media/"))
            for number, name in enumerate(media, start=1):
                raw = archive.read(name)
                suffix = Path(name).suffix
                image_path = doc_out / f"{number:02d}_{Path(name).name}"
                image_path.write_bytes(raw)
                try:
                    image = Image.open(BytesIO(raw))
                    image.load()
                    display_images.append((f"{number:02d} {Path(name).name} {image.width}x{image.height}", image))
                except Exception:
                    pass
        if display_images:
            make_sheet(docx.stem, display_images)
            print(f"{docx.name}: {len(display_images)} images")


if __name__ == "__main__":
    main()
