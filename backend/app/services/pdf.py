"""PDF page rendering and embedded-image extraction (pypdfium2 + Pillow).

Pure functions: no database, no settings, no storage. PDFium is native C++ code parsing
untrusted files, so this module is only ever called from worker processes, never from the
web API process.
"""

from __future__ import annotations

import io
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
from PIL import Image

POINTS_PER_INCH = 72


class PdfFormatError(Exception):
    """The file cannot be read as a PDF (corrupt, truncated, encrypted...)."""


Bounds = tuple[float, float, float, float]  # (left, bottom, right, top) in PDF points


# ---------------------------------------------------------------------------------- opening
def open_pdf(path: Path | str) -> pdfium.PdfDocument:
    try:
        return pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as exc:
        raise PdfFormatError(
            "The file is not a readable PDF (it may be corrupt or password-protected)"
        ) from exc


# ---------------------------------------------------------------------------------- rendering
def render_scale(width_pt: float, height_pt: float, dpi: int, max_pixels: int) -> float:
    """Scale factor for the requested DPI, reduced if the page would exceed ``max_pixels``.

    The cap protects the worker from a hostile (or just enormous, e.g. a poster-sized) page
    requesting gigabytes of bitmap memory.
    """
    scale = dpi / POINTS_PER_INCH
    pixels = (width_pt * scale) * (height_pt * scale)
    if pixels > max_pixels:
        scale *= math.sqrt(max_pixels / pixels)
    return scale


@dataclass
class RenderedPage:
    image: Image.Image  # RGB
    _posconv: object  # pdfium's own page->bitmap coordinate converter

    def pixel_bbox(self, bounds: Bounds) -> tuple[float, float, float, float]:
        """Convert PDF-space bounds to (x0, y0, x1, y1) pixels of ``image`` (origin top-left).

        Uses PDFium's converter, which accounts for page rotation and CropBox.
        """
        left, bottom, right, top = bounds
        points = [
            self._posconv.to_bitmap(x, y)  # type: ignore[attr-defined]
            for x, y in ((left, bottom), (right, bottom), (left, top), (right, top))
        ]
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        return (min(xs), min(ys), max(xs), max(ys))


def render_page(page: pdfium.PdfPage, scale: float) -> RenderedPage:
    bitmap = page.render(scale=scale)
    try:
        image = bitmap.to_pil().convert("RGB")
        return RenderedPage(image=image, _posconv=bitmap.get_posconv(page))
    finally:
        bitmap.close()


# ---------------------------------------------------------------------------------- images
def to_rgb(image: Image.Image) -> Image.Image:
    """Normalise any PDF image mode to plain RGB (transparency is flattened onto white)."""
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")  # L, 1, CMYK, P, I;16 ...


def encode_jpeg(image: Image.Image, quality: int = 90) -> bytes:
    buffer = io.BytesIO()
    to_rgb(image).save(buffer, "JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def make_thumbnail(image: Image.Image, max_px: int) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    return thumb


def dhash(image: Image.Image, size: int = 8) -> str:
    """64-bit difference hash as 16 hex chars. Visually similar images differ by few bits.

    Stored on source images so near-duplicate detection (Phase 6) is a cheap Hamming lookup.
    """
    gray = to_rgb(image).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            left = pixels[row * (size + 1) + col]
            right = pixels[row * (size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return f"{bits:0{size * size // 4}x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")


@dataclass
class EmbeddedImage:
    image: Image.Image  # RGB, at the image's own native resolution
    # Position on the rendered page image in pixels, or None when it cannot be determined
    # reliably (images nested inside form XObjects report bounds in the form's own space).
    bbox: tuple[float, float, float, float] | None


def iter_embedded_images(
    page: pdfium.PdfPage, rendered: RenderedPage, *, min_px: int
) -> Iterator[EmbeddedImage]:
    """Yield the raster images embedded in the page, at native resolution.

    Native extraction is preferable to cropping the rendered page: it returns the original
    pixels (no resampling, no overlapping text baked in) and needs no ML at all. Images
    smaller than ``min_px`` on either side (icons, bullets, rules) are skipped.
    """
    for obj in page.get_objects(filter=(pdfium_raw.FPDF_PAGEOBJ_IMAGE,)):
        try:
            width, height = obj.get_px_size()
            if min(width, height) < min_px:
                continue
            bitmap = obj.get_bitmap(render=False)
            try:
                image = to_rgb(bitmap.to_pil())
            finally:
                bitmap.close()
            bbox = rendered.pixel_bbox(obj.get_bounds()) if obj.level == 0 else None
            yield EmbeddedImage(image=image, bbox=bbox)
        except pdfium.PdfiumError:
            continue  # one undecodable image must not lose the rest of the page
