"""PDF rendering and native image extraction, verified against real PDFium output."""

from __future__ import annotations

import pytest
from PIL import Image

from app.services.pdf import (
    PdfFormatError,
    dhash,
    encode_jpeg,
    hamming_distance,
    iter_embedded_images,
    make_thumbnail,
    open_pdf,
    render_page,
    render_scale,
    to_rgb,
)
from tests.pdf_factory import ImageSpec, PageSpec, blank_pages, build_nested_form_pdf, build_pdf


def open_bytes(tmp_path, data: bytes):  # noqa: ANN001, ANN201
    path = tmp_path / "test.pdf"
    path.write_bytes(data)
    return open_pdf(path)


# ------------------------------------------------------------------------------ opening
@pytest.mark.parametrize(
    "data",
    [b"", b"this is not a pdf", b"%PDF-1.4\ntruncated garbage", build_pdf()[:150]],
    ids=["empty", "text", "bad-body", "truncated"],
)
def test_unreadable_files_raise_a_clean_error(tmp_path, data: bytes) -> None:  # noqa: ANN001
    with pytest.raises(PdfFormatError):
        open_bytes(tmp_path, data)


def test_page_count(tmp_path) -> None:  # noqa: ANN001
    assert len(open_bytes(tmp_path, blank_pages(3))) == 3


# ------------------------------------------------------------------------------ rendering
def test_render_scale_follows_dpi() -> None:
    assert render_scale(595, 842, dpi=72, max_pixels=10**9) == pytest.approx(1.0)
    assert render_scale(595, 842, dpi=150, max_pixels=10**9) == pytest.approx(150 / 72)


def test_render_scale_is_capped_for_enormous_pages() -> None:
    width_pt = height_pt = 14400  # a 200-inch square "page"
    scale = render_scale(width_pt, height_pt, dpi=150, max_pixels=40_000_000)
    assert (width_pt * scale) * (height_pt * scale) <= 40_000_000 * 1.001


def test_render_page_produces_an_rgb_image_of_the_expected_size(tmp_path) -> None:  # noqa: ANN001
    page = open_bytes(tmp_path, blank_pages(1, width=300, height=200))[0]
    rendered = render_page(page, scale=2.0)
    assert rendered.image.mode == "RGB"
    assert rendered.image.size == (600, 400)
    assert rendered.image.getpixel((10, 10)) == (255, 255, 255)  # blank page is white


# ------------------------------------------------------------------------------ extraction
def test_embedded_images_come_out_at_native_resolution_in_the_right_place(tmp_path) -> None:  # noqa: ANN001
    data = build_pdf(
        [
            PageSpec(
                images=[
                    ImageSpec(x=50, y=600, w=200, h=150, px=64, color=(255, 0, 0)),
                    ImageSpec(x=300, y=100, w=250, h=250, px=80, color=(0, 0, 255)),
                ]
            )
        ]
    )
    page = open_bytes(tmp_path, data)[0]
    scale = 150 / 72
    found = list(iter_embedded_images(page, render_page(page, scale), min_px=16))
    assert len(found) == 2

    red, blue = found
    assert red.image.size == (64, 64) and red.image.getpixel((5, 5)) == (255, 0, 0)
    assert blue.image.size == (80, 80) and blue.image.getpixel((5, 5)) == (0, 0, 255)

    # PDF y-axis points up; the page image's points down.
    expected_red = (50 * scale, (842 - 750) * scale, 250 * scale, (842 - 600) * scale)
    assert red.bbox == pytest.approx(expected_red, abs=1.5)


def test_tiny_decorative_images_are_skipped(tmp_path) -> None:  # noqa: ANN001
    data = build_pdf(
        [PageSpec(images=[ImageSpec(10, 10, 20, 20, px=4), ImageSpec(100, 100, 100, 100, px=64)])]
    )
    page = open_bytes(tmp_path, data)[0]
    found = list(iter_embedded_images(page, render_page(page, 1.0), min_px=32))
    assert [f.image.size for f in found] == [(64, 64)]


def test_jpeg_encoded_images_are_decoded(tmp_path) -> None:  # noqa: ANN001
    """Real catalogues embed JPEGs (DCTDecode), not raw pixels."""
    green = ImageSpec(50, 50, 100, 100, px=64, color=(0, 200, 0), codec="jpeg")
    data = build_pdf([PageSpec(images=[green])])
    page = open_bytes(tmp_path, data)[0]
    (found,) = iter_embedded_images(page, render_page(page, 1.0), min_px=16)
    red, green, blue = found.image.getpixel((32, 32))
    assert green > 180 and red < 60 and blue < 60


@pytest.mark.parametrize("rotate", [0, 90, 180, 270])
@pytest.mark.parametrize("crop", [None, (50, 40, 250, 180)], ids=["full", "cropped"])
def test_bbox_lands_on_the_image_whatever_the_page_rotation_or_cropbox(
    tmp_path, rotate: int, crop  # noqa: ANN001
) -> None:
    """The decisive check: sample the *rendered page* at the reported bbox and expect the
    image's colour there. If the coordinate conversion were wrong, this would fail."""
    spec = PageSpec(
        300, 200, rotate=rotate, crop=crop, images=[ImageSpec(100, 60, 60, 40, color=(255, 0, 0))]
    )
    page = open_bytes(tmp_path, build_pdf([spec]))[0]
    rendered = render_page(page, scale=1.0)
    (found,) = iter_embedded_images(page, rendered, min_px=8)
    assert found.bbox is not None
    x0, y0, x1, y1 = found.bbox
    centre = rendered.image.getpixel((round((x0 + x1) / 2), round((y0 + y1) / 2)))
    assert centre == (255, 0, 0)
    assert 1 <= (x1 - x0) <= rendered.image.size[0] and 1 <= (y1 - y0) <= rendered.image.size[1]


def test_images_nested_in_forms_are_extracted_but_never_given_a_wrong_position(tmp_path) -> None:  # noqa: ANN001
    page = open_bytes(tmp_path, build_nested_form_pdf())[0]
    (found,) = iter_embedded_images(page, render_page(page, 1.0), min_px=8)
    assert found.image.size == (16, 16)
    assert found.bbox is None  # PDFium reports form-space bounds, so we must not guess


# ------------------------------------------------------------------------------ image helpers
def test_to_rgb_handles_every_common_pdf_image_mode() -> None:
    for mode in ("L", "1", "CMYK", "RGBA", "LA", "P"):
        assert to_rgb(Image.new(mode, (4, 4))).mode == "RGB"


def test_transparency_is_flattened_onto_white() -> None:
    transparent = Image.new("RGBA", (2, 2), (255, 0, 0, 0))
    assert to_rgb(transparent).getpixel((0, 0)) == (255, 255, 255)


def test_encode_jpeg_returns_a_real_jpeg() -> None:
    data = encode_jpeg(Image.new("RGBA", (10, 10), (10, 20, 30, 255)))
    assert data[:3] == b"\xff\xd8\xff"
    assert Image.open(__import__("io").BytesIO(data)).size == (10, 10)


def test_thumbnail_keeps_aspect_ratio_and_never_upscales() -> None:
    big = make_thumbnail(Image.new("RGB", (1000, 500)), 200)
    assert big.size == (200, 100)
    small = make_thumbnail(Image.new("RGB", (50, 40)), 200)
    assert small.size == (50, 40)


def test_perceptual_hash_finds_near_duplicates_and_separates_different_images() -> None:
    def gradient(brightness: int = 0, flip: bool = False) -> Image.Image:
        image = Image.new("RGB", (64, 64))
        for x in range(64):
            for y in range(64):
                value = min(255, x * 4 + brightness)
                image.putpixel((x, y), (value, value, value))
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if flip else image

    original = dhash(gradient())
    assert len(original) == 16
    assert dhash(gradient()) == original  # deterministic
    assert hamming_distance(original, dhash(gradient(brightness=15))) <= 4  # same picture, brighter
    assert hamming_distance(original, dhash(gradient(flip=True))) > 20  # genuinely different
