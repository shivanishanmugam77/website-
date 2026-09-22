"""Builds tiny, valid PDFs with exactly-known contents for tests (no external files needed).

Each image is a solid colour, so a test can assert both *where* an embedded image ends up
and *what* pixels it contains.
"""

from __future__ import annotations

import io
import zlib
from dataclasses import dataclass, field

from PIL import Image


@dataclass
class ImageSpec:
    x: float  # lower-left corner in PDF points (origin bottom-left)
    y: float
    w: float  # displayed size in points
    h: float
    px: int = 32  # the image's own pixel size (square)
    color: tuple[int, int, int] = (255, 0, 0)
    codec: str = "flate"  # "flate" (raw RGB, lossless) or "jpeg" (DCTDecode, like real catalogues)


@dataclass
class PageSpec:
    width: float = 595
    height: float = 842
    rotate: int = 0
    crop: tuple[float, float, float, float] | None = None  # (left, bottom, right, top)
    images: list[ImageSpec] = field(default_factory=list)


def _image_object(spec: ImageSpec) -> bytes:
    if spec.codec == "jpeg":
        buffer = io.BytesIO()
        Image.new("RGB", (spec.px, spec.px), spec.color).save(buffer, "JPEG", quality=95)
        data, filter_name = buffer.getvalue(), "DCTDecode"
    else:
        data, filter_name = zlib.compress(bytes(spec.color) * (spec.px * spec.px)), "FlateDecode"
    head = (
        f"<< /Type /XObject /Subtype /Image /Width {spec.px} /Height {spec.px} "
        f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /{filter_name} "
        f"/Length {len(data)} >>\nstream\n"
    )
    return head.encode() + data + b"\nendstream"


def build_pdf(pages: list[PageSpec] | None = None) -> bytes:
    pages = pages or [PageSpec()]
    bodies: dict[int, bytes] = {}
    next_id = 3
    kids: list[int] = []
    for spec in pages:
        page_id, content_id = next_id, next_id + 1
        image_ids = list(range(next_id + 2, next_id + 2 + len(spec.images)))
        next_id += 2 + len(spec.images)
        kids.append(page_id)

        content = "".join(
            f"q {img.w} 0 0 {img.h} {img.x} {img.y} cm /Im{i} Do Q\n"
            for i, img in enumerate(spec.images)
        ).encode()
        xobjects = " ".join(f"/Im{i} {oid} 0 R" for i, oid in enumerate(image_ids))
        extras = f" /Rotate {spec.rotate}" if spec.rotate else ""
        if spec.crop:
            extras += " /CropBox [{} {} {} {}]".format(*spec.crop)
        bodies[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {spec.width} {spec.height}]{extras} "
            f"/Contents {content_id} 0 R /Resources << /XObject << {xobjects} >> >> >>"
        ).encode()
        content_head = f"<< /Length {len(content)} >>\nstream\n".encode()
        bodies[content_id] = content_head + content + b"endstream"
        for oid, img in zip(image_ids, spec.images, strict=True):
            bodies[oid] = _image_object(img)

    bodies[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kid_refs = " ".join(f"{k} 0 R" for k in kids)
    bodies[2] = f"<< /Type /Pages /Kids [{kid_refs}] /Count {len(kids)} >>".encode()

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for oid in sorted(bodies):
        offsets[oid] = len(out)
        out += f"{oid} 0 obj\n".encode() + bodies[oid] + b"\nendobj\n"
    xref_at = len(out)
    count = max(bodies) + 1
    out += f"xref\n0 {count}\n0000000000 65535 f \n".encode()
    for oid in range(1, count):
        out += f"{offsets[oid]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(out)


def blank_pages(count: int, **kwargs) -> bytes:  # noqa: ANN003
    return build_pdf([PageSpec(**kwargs) for _ in range(count)])


def build_nested_form_pdf() -> bytes:
    """One page whose only image lives inside a Form XObject that is itself translated.

    On the page the red image occupies x 110..170, y 60..100 (PDF points) but PDFium reports
    its bounds in the form's own coordinate space, which is why nested images get no bbox.
    """
    image = zlib.compress(bytes((255, 0, 0)) * (16 * 16))
    page_content = b"q 1 0 0 1 100 50 cm /Fm0 Do Q\n"
    form_content = b"q 60 0 0 40 10 10 cm /Im0 Do Q\n"
    bodies = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R "
            b"/Resources << /XObject << /Fm0 5 0 R >> >> >>"
        ),
        4: b"<< /Length %d >>\nstream\n" % len(page_content) + page_content + b"endstream",
        5: (
            b"<< /Type /XObject /Subtype /Form /BBox [0 0 200 150] "
            b"/Resources << /XObject << /Im0 6 0 R >> >> /Length %d >>\nstream\n"
            % len(form_content)
            + form_content
            + b"endstream"
        ),
        6: (
            b"<< /Type /XObject /Subtype /Image /Width 16 /Height 16 /ColorSpace /DeviceRGB "
            b"/BitsPerComponent 8 /Filter /FlateDecode /Length %d >>\nstream\n" % len(image)
            + image
            + b"\nendstream"
        ),
    }
    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for oid in sorted(bodies):
        offsets[oid] = len(out)
        out += f"{oid} 0 obj\n".encode() + bodies[oid] + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 7\n0000000000 65535 f \n"
    for oid in range(1, 7):
        out += f"{offsets[oid]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(out)
