"""Small, dependency-free helpers for handling uploaded files safely."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

PDF_MAGIC = b"%PDF-"
_PDF_HEADER_WINDOW = 1024  # the PDF spec allows junk before the header, but only this far in
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")
MAX_FILENAME_LENGTH = 255
DEFAULT_FILENAME = "catalogue.pdf"


def has_pdf_signature(head: bytes) -> bool:
    """Cheap content sniff: does this look like the start of a PDF?

    This is *not* validation (the real parse happens in the worker); it only rejects
    obvious non-PDFs before they are written to storage.
    """
    return PDF_MAGIC in head[:_PDF_HEADER_WINDOW]


def sanitize_filename(raw: str | None) -> str:
    """A safe display name for an uploaded file.

    The result is only ever *shown* (storage keys never contain it), but it is still
    stripped of directories, control characters and absurd lengths so it cannot confuse
    logs, terminals or downstream UIs.
    """
    if not raw:
        return DEFAULT_FILENAME
    name = PurePosixPath(raw.replace("\\", "/")).name  # drop any directory components
    name = _WHITESPACE.sub(" ", _CONTROL_CHARS.sub("", name)).strip(" .")
    if not name:
        return DEFAULT_FILENAME
    if len(name) > MAX_FILENAME_LENGTH:
        stem, dot, suffix = name.rpartition(".")
        keep = MAX_FILENAME_LENGTH - len(suffix) - 1
        if dot and 0 < len(suffix) <= 10:
            name = f"{stem[:keep]}.{suffix}"
        else:
            name = name[:MAX_FILENAME_LENGTH]
    return name


def default_catalogue_name(filename: str) -> str:
    """Human name derived from a file name: 'BOGAO_2026 (v2).pdf' -> 'BOGAO_2026 (v2)'."""
    stem = filename.rsplit(".", 1)[0] if filename.lower().endswith(".pdf") else filename
    return stem.strip() or "Untitled catalogue"
