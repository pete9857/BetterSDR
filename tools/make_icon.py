"""Compile the logo into the Windows icon the packaged build wears.

Run by hand, like `tools/build_basemap.py`; the output is committed, so
nothing at runtime renders an SVG and nothing in the build depends on an
image library. PySide6 is already a dependency, so Qt does the rasterising
and the ICO container is written here - it is a header, a directory and a
run of images, and writing it directly is smaller than taking on Pillow for
one file that changes about once a year.

    .venv/Scripts/python.exe tools/make_icon.py

Two encodings, because Windows wants both. The small sizes go in as DIBs,
which is what every version of the shell has read since 1995; 128 and 256 go
in as PNG, which is what Vista introduced precisely so that a 256-pixel icon
is not a quarter of a megabyte of uncompressed BGRA. An icon that carries
only one of the two is either enormous or blurry in the places nobody thinks
to look - the Alt-Tab switcher, the taskbar at 200% scaling, the file
Properties dialog.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QBuffer, QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "BetterSDRLogo.svg"
TARGET = ROOT / "bettersdr" / "ui" / "assets" / "bettersdr.ico"

# Every size the Windows shell asks for. 16 is the title bar and the tree
# view, 32 the desktop, 48 the default Explorer view, 256 the preview pane.
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)

# Below this the artwork is drawn edge to edge; above it the shell is
# showing the icon large enough that a margin reads as deliberate.
PNG_FROM = 128

# The logo is taller than it is wide and an icon is square, so it is fitted
# inside with this much of the square left clear on the long axis.
MARGIN = 0.04


def render(renderer: QSvgRenderer, size: int) -> QImage:
    """The logo centred on a transparent square of `size` pixels."""
    image = QImage(QSize(size, size), QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)

    natural = renderer.defaultSize()
    usable = size * (1.0 - 2.0 * MARGIN)
    scale = min(usable / natural.width(), usable / natural.height())
    width = natural.width() * scale
    height = natural.height() * scale

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    renderer.render(
        painter, QRectF((size - width) / 2, (size - height) / 2, width, height)
    )
    painter.end()
    return image


def as_png(image: QImage) -> bytes:
    store = QByteArray()
    buffer = QBuffer(store)
    buffer.open(QBuffer.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    buffer.close()
    return bytes(store)


def as_dib(image: QImage) -> bytes:
    """A BITMAPINFOHEADER, the pixels bottom-up, then the 1-bit AND mask.

    The header claims twice the real height: the second half is the mask,
    which 32-bit icons no longer use for transparency but which the format
    still requires and which some shell paths still read. Rows of both
    bitmaps are padded to four bytes.
    """
    width = image.width()
    height = image.height()
    converted = image.convertToFormat(QImage.Format.Format_ARGB32)

    # Qt's ARGB32 is 0xAARRGGBB in native byte order, which on x86 is
    # already the B, G, R, A byte order a DIB wants.
    pointer = converted.constBits()
    argb = np.frombuffer(pointer, dtype=np.uint8, count=height * converted.bytesPerLine())
    argb = argb.reshape(height, converted.bytesPerLine())[:, : width * 4]
    pixels = argb[::-1].tobytes()

    # The AND mask marks the pixels to leave alone; on a 32-bit icon only
    # the fully transparent ones qualify. Rows are padded to four bytes.
    alpha = argb.reshape(height, width, 4)[::-1, :, 3]
    bits = np.packbits(alpha == 0, axis=1)
    stride = ((width + 31) // 32) * 4
    mask = np.zeros((height, stride), dtype=np.uint8)
    mask[:, : bits.shape[1]] = bits
    mask = mask.tobytes()

    header = struct.pack(
        "<IiiHHIIiiII",
        40,  # biSize
        width,
        height * 2,  # biHeight: XOR bitmap and AND mask together
        1,  # biPlanes
        32,  # biBitCount
        0,  # biCompression: BI_RGB
        len(pixels) + len(mask),
        0,
        0,
        0,
        0,
    )
    return header + pixels + mask


def build_ico(images: list[tuple[int, bytes]]) -> bytes:
    """Pack encoded images into an ICO. 256 is written as 0, per the format."""
    directory = bytearray(struct.pack("<HHH", 0, 1, len(images)))
    offset = 6 + 16 * len(images)
    body = bytearray()
    for size, data in images:
        directory += struct.pack(
            "<BBBBHHII",
            size if size < 256 else 0,
            size if size < 256 else 0,
            0,  # bColorCount: 0 for anything above 8-bit
            0,  # bReserved
            1,  # wPlanes
            32,  # wBitCount
            len(data),
            offset,
        )
        body += data
        offset += len(data)
    return bytes(directory) + bytes(body)


def main() -> int:
    if not SOURCE.exists():
        print(f"logo not found: {SOURCE}", file=sys.stderr)
        return 1

    app = QGuiApplication(sys.argv[:1])  # noqa: F841 - must outlive the renderer
    renderer = QSvgRenderer(str(SOURCE))
    if not renderer.isValid():
        print(f"could not read {SOURCE}", file=sys.stderr)
        return 1

    images: list[tuple[int, bytes]] = []
    for size in SIZES:
        image = render(renderer, size)
        images.append((size, as_png(image) if size >= PNG_FROM else as_dib(image)))

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(build_ico(images))
    print(f"{TARGET.relative_to(ROOT)}: {TARGET.stat().st_size:,} bytes")
    for size, data in images:
        kind = "png" if size >= PNG_FROM else "dib"
        print(f"  {size:>3} x {size:<3} {kind}  {len(data):>8,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
