from __future__ import annotations

import pytest

from app.utils.files import (
    DEFAULT_FILENAME,
    MAX_FILENAME_LENGTH,
    default_catalogue_name,
    has_pdf_signature,
    sanitize_filename,
)


def test_pdf_signature_detection() -> None:
    assert has_pdf_signature(b"%PDF-1.7\n...")
    assert has_pdf_signature(b"\xef\xbb\xbf junk before %PDF-1.4")  # spec allows leading junk
    assert not has_pdf_signature(b"")
    assert not has_pdf_signature(b"PK\x03\x04 a zip file")
    assert not has_pdf_signature(b"<html>%PDF-</html>".rjust(3000, b" ")[1500:])  # too far in


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BOGAO 2026.pdf", "BOGAO 2026.pdf"),
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\Desktop\\catalogue.pdf", "catalogue.pdf"),
        ("evil\x00name\x1b[31m.pdf", "evilname[31m.pdf"),
        ("  spaced   out  .pdf ", "spaced out .pdf"),
        ("...", DEFAULT_FILENAME),
        ("", DEFAULT_FILENAME),
        (None, DEFAULT_FILENAME),
        ("办公椅目录.pdf", "办公椅目录.pdf"),
    ],
)
def test_sanitize_filename(raw: str | None, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_overlong_filenames_are_truncated_but_keep_their_extension() -> None:
    name = sanitize_filename("a" * 1000 + ".pdf")
    assert len(name) <= MAX_FILENAME_LENGTH and name.endswith(".pdf")


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("BOGAO_2026 (v2).pdf", "BOGAO_2026 (v2)"),
        ("catalogue.PDF", "catalogue"),
        (".pdf", "Untitled catalogue"),
    ],
)
def test_default_catalogue_name(filename: str, expected: str) -> None:
    assert default_catalogue_name(filename) == expected
