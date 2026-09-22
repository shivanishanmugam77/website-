"""Tiny helpers that build valid rows with unique values."""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.passwords import hash_password
from app.models import (
    Catalogue,
    CataloguePage,
    Category,
    DetectedObject,
    ProcessingJob,
    Product,
    ProductCandidate,
    SegmentationResult,
    SourceImage,
    Supplier,
    User,
)
from app.models.enums import (
    JobStatus,
    JobType,
    SegmentationStatus,
    SourceImageKind,
)
from app.services.pipeline import original_pdf_key
from app.services.storage import Storage


TEST_PASSWORD = "correct-horse-battery-staple"


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def make_user(session: Session, password: str | None = None, **overrides) -> User:  # noqa: ANN003
    """``password`` (if given) is properly hashed so the user can actually log in."""
    values = {"email": f"{_unique('user')}@example.com", "password_hash": "not-a-real-hash"}
    if password is not None:
        values["password_hash"] = hash_password(password)
    user = User(**{**values, **overrides})
    session.add(user)
    session.flush()
    return user


def make_supplier(session: Session, **overrides) -> Supplier:  # noqa: ANN003
    slug = _unique("supplier")
    supplier = Supplier(**{"name": slug.title(), "slug": slug, **overrides})
    session.add(supplier)
    session.flush()
    return supplier


def make_category(session: Session, **overrides) -> Category:  # noqa: ANN003
    slug = _unique("category")
    category = Category(**{"name": slug.title(), "slug": slug, **overrides})
    session.add(category)
    session.flush()
    return category


def make_catalogue(session: Session, **overrides) -> Catalogue:  # noqa: ANN003
    values = {
        "name": "Sample catalogue",
        "original_filename": "sample.pdf",
        "storage_key": f"catalogues/{uuid.uuid4()}/original.pdf",
        "file_size_bytes": 1024,
        "sha256": uuid.uuid4().hex + uuid.uuid4().hex,
    }
    catalogue = Catalogue(**{**values, **overrides})
    session.add(catalogue)
    session.flush()
    return catalogue


def make_page(session: Session, catalogue: Catalogue, number: int = 1) -> CataloguePage:
    page = CataloguePage(catalogue=catalogue, page_number=number)
    session.add(page)
    session.flush()
    return page


def make_source_image(session: Session, page: CataloguePage) -> SourceImage:
    image = SourceImage(
        page=page,
        kind=SourceImageKind.REGION,
        storage_key=f"catalogues/x/source-images/{uuid.uuid4()}.jpg",
        width_px=800,
        height_px=600,
    )
    session.add(image)
    session.flush()
    return image


def make_detected_object(session: Session, image: SourceImage) -> DetectedObject:
    obj = DetectedObject(
        source_image=image,
        label="desk",
        prompt="desk",
        confidence=0.9,
        bbox_x0=10,
        bbox_y0=20,
        bbox_x1=300,
        bbox_y1=400,
        model_name="test-detector",
    )
    session.add(obj)
    session.flush()
    return obj


def make_segmentation(session: Session, obj: DetectedObject) -> SegmentationResult:
    result = SegmentationResult(
        detected_object=obj,
        status=SegmentationStatus.SUCCEEDED,
        confidence=0.88,
        model_name="test-segmenter",
    )
    session.add(result)
    session.flush()
    return result


def make_candidate(session: Session, catalogue: Catalogue, **overrides) -> ProductCandidate:  # noqa: ANN003
    candidate = ProductCandidate(catalogue=catalogue, **overrides)
    session.add(candidate)
    session.flush()
    return candidate


def make_product(session: Session, **overrides) -> Product:  # noqa: ANN003
    slug = _unique("product")
    product = Product(**{"name": "Test product", "slug": slug, **overrides})
    session.add(product)
    session.flush()
    return product


def settings_with(**overrides) -> Settings:  # noqa: ANN003
    """The real settings with some values replaced (for pipeline / upload limits)."""
    return get_settings().model_copy(update=overrides)


def make_stored_catalogue(
    session: Session, storage: Storage, data: bytes, **overrides  # noqa: ANN003
) -> tuple[Catalogue, ProcessingJob]:
    """A catalogue whose PDF is really in storage, plus its queued job."""
    catalogue_id = uuid.uuid4()
    key = original_pdf_key(catalogue_id)
    stored = storage.put_bytes(key, data)
    values = {
        "name": "Stored catalogue",
        "original_filename": "stored.pdf",
        "file_size_bytes": stored.size,
        "sha256": stored.sha256,
    }
    catalogue = Catalogue(id=catalogue_id, storage_key=key, **{**values, **overrides})
    session.add(catalogue)
    session.flush()
    job = ProcessingJob(
        catalogue_id=catalogue_id, job_type=JobType.CATALOGUE_PROCESS, status=JobStatus.QUEUED
    )
    session.add(job)
    session.flush()
    return catalogue, job
