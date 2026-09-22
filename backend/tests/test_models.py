"""Schema-level behaviour: tables, relationships, cascades, constraints, JSONB, pgvector."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import Engine, delete, inspect, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.constants import EMBEDDING_DIM
from app.models import (
    AppSetting,
    AuditLog,
    Base,
    Catalogue,
    CataloguePage,
    Category,
    CorrectionRecord,
    DetectedObject,
    DuplicateFlag,
    Enquiry,
    ImageEmbedding,
    OCRResult,
    ProcessingJob,
    Product,
    ProductCandidate,
    ProductImage,
    RefreshToken,
    SegmentationResult,
    SourceImage,
    Supplier,
    WishlistItem,
)
from app.models.enums import (
    AuditAction,
    CandidateStatus,
    CatalogueStatus,
    CorrectionType,
    DataOrigin,
    DuplicateSignal,
    JobType,
    ProductImageType,
    ProductStatus,
    UserRole,
)
from tests import factories as f

EXPECTED_TABLES = {
    "users",
    "suppliers",
    "categories",
    "app_settings",
    "catalogues",
    "catalogue_pages",
    "source_images",
    "ocr_results",
    "detected_objects",
    "segmentation_results",
    "image_embeddings",
    "product_candidates",
    "candidate_pages",
    "candidate_images",
    "candidate_objects",
    "products",
    "product_images",
    "processing_jobs",
    "duplicate_flags",
    "enquiries",
    "wishlist_items",
    "audit_logs",
    "correction_records",
    "refresh_tokens",
    "panel_edits",
}


def count(session: Session, model: type) -> int:
    return len(session.scalars(select(model)).all())


@contextmanager
def rejected(
    session: Session,
    error: type[Exception] | tuple[type[Exception], ...] = IntegrityError,
) -> Iterator[None]:
    """Assert the enclosed block is rejected by the database, without losing earlier work.

    The block runs in a SAVEPOINT that is rolled back on failure, so objects created
    before it remain usable (a plain ``session.rollback()`` would discard them all).
    """
    with pytest.raises(error), session.begin_nested():
        yield


# ----------------------------------------------------------------------------- structure
def test_every_expected_table_exists(engine: Engine) -> None:
    actual = set(inspect(engine).get_table_names()) - {"alembic_version"}
    assert actual == EXPECTED_TABLES
    assert set(Base.metadata.tables) == EXPECTED_TABLES


def test_required_postgres_extensions_are_installed(db_session: Session) -> None:
    installed = set(db_session.scalars(text("SELECT extname FROM pg_extension")))
    assert {"vector", "pg_trgm"} <= installed


def test_no_identifier_was_silently_truncated_by_postgres(db_session: Session) -> None:
    """PostgreSQL truncates identifiers to 63 chars; a name of exactly that length is
    almost certainly a truncated one and would desync the models from the database."""
    rows = db_session.execute(
        text(
            "SELECT name FROM ("
            " SELECT conname AS name FROM pg_constraint"
            "  WHERE connamespace = 'public'::regnamespace"
            " UNION ALL SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"
            ") names WHERE length(name) >= 63"
        )
    ).all()
    assert rows == []


# ----------------------------------------------------------------------------- traceability
def build_chain(session: Session) -> dict:
    """Supplier -> Catalogue -> Page -> Image -> Object -> Segmentation -> Candidate -> Product."""
    supplier = f.make_supplier(session)
    catalogue = f.make_catalogue(session, supplier=supplier)
    page = f.make_page(session, catalogue, number=7)
    image = f.make_source_image(session, page)
    obj = f.make_detected_object(session, image)
    seg = f.make_segmentation(session, obj)
    candidate = f.make_candidate(session, catalogue)
    candidate.pages.append(page)
    candidate.source_images.append(image)
    candidate.detected_objects.append(obj)
    category = f.make_category(session)
    product = f.make_product(
        session,
        candidate=candidate,
        catalogue=catalogue,
        source_page=page,
        supplier=supplier,
        category=category,
        origin=DataOrigin.AI,
    )
    product.images.append(
        ProductImage(
            image_type=ProductImageType.HERO,
            storage_key="products/x/masked/hero.png",
            source_image=image,
            detected_object=obj,
            segmentation_result=seg,
        )
    )
    session.flush()
    return {
        "supplier": supplier,
        "catalogue": catalogue,
        "page": page,
        "image": image,
        "obj": obj,
        "seg": seg,
        "candidate": candidate,
        "product": product,
        "category": category,
    }


def test_product_is_traceable_back_to_the_exact_page_and_object(db_session: Session) -> None:
    chain = build_chain(db_session)
    db_session.expire_all()
    product = db_session.get(Product, chain["product"].id)
    assert product is not None
    assert product.supplier.id == chain["supplier"].id
    assert product.catalogue.id == chain["catalogue"].id
    assert product.source_page.page_number == 7
    hero = product.images[0]
    assert hero.source_image.page.catalogue_id == chain["catalogue"].id
    assert hero.detected_object.id == chain["obj"].id
    assert hero.segmentation_result.id == chain["seg"].id
    assert product.candidate.pages[0].page_number == 7


def test_one_candidate_can_span_many_pages_images_and_objects(db_session: Session) -> None:
    catalogue = f.make_catalogue(db_session)
    candidate = f.make_candidate(db_session, catalogue)
    for number in (1, 2, 3):
        page = f.make_page(db_session, catalogue, number)
        image = f.make_source_image(db_session, page)
        candidate.pages.append(page)
        candidate.source_images.append(image)
        candidate.detected_objects.append(f.make_detected_object(db_session, image))
    db_session.flush()
    db_session.expire_all()
    reloaded = db_session.get(ProductCandidate, candidate.id)
    assert len(reloaded.pages) == len(reloaded.source_images) == len(reloaded.detected_objects) == 3


def test_deleting_a_catalogue_cascades_through_the_pipeline_artefacts(db_session: Session) -> None:
    chain = build_chain(db_session)
    catalogue_id = chain["catalogue"].id
    db_session.add(
        OCRResult(
            page=chain["page"],
            line_index=0,
            text="YY-11",
            bbox_x0=0,
            bbox_y0=0,
            bbox_x1=10,
            bbox_y1=10,
            engine="test",
        )
    )
    db_session.add(ProcessingJob(catalogue_id=catalogue_id, job_type=JobType.CATALOGUE_PROCESS))
    db_session.flush()

    db_session.delete(chain["catalogue"])
    db_session.flush()
    db_session.expire_all()

    assert count(db_session, CataloguePage) == 0
    assert count(db_session, SourceImage) == 0
    assert count(db_session, DetectedObject) == 0
    assert count(db_session, SegmentationResult) == 0
    assert count(db_session, OCRResult) == 0
    assert count(db_session, ProductCandidate) == 0
    assert count(db_session, ProcessingJob) == 0


def test_published_product_survives_deletion_of_its_source_catalogue(db_session: Session) -> None:
    chain = build_chain(db_session)
    product_id = chain["product"].id
    db_session.delete(chain["catalogue"])
    db_session.flush()
    db_session.expire_all()

    product = db_session.get(Product, product_id)
    assert product is not None, "approved products must outlive their source catalogue"
    assert product.candidate_id is None
    assert product.catalogue_id is None
    assert product.source_page_id is None
    assert product.images[0].source_image_id is None
    assert product.images[0].segmentation_result_id is None


def test_deleting_a_product_removes_its_images(db_session: Session) -> None:
    chain = build_chain(db_session)
    db_session.delete(chain["product"])
    db_session.flush()
    assert count(db_session, ProductImage) == 0


# ----------------------------------------------------------------------------- constraints
def test_duplicate_email_is_rejected(db_session: Session) -> None:
    user = f.make_user(db_session)
    with rejected(db_session):
        f.make_user(db_session, email=user.email)


def test_page_numbers_are_unique_per_catalogue_only(db_session: Session) -> None:
    first = f.make_catalogue(db_session)
    second = f.make_catalogue(db_session)
    f.make_page(db_session, first, 1)
    f.make_page(db_session, second, 1)  # same number, different catalogue: fine
    with rejected(db_session):
        f.make_page(db_session, first, 1)


def test_page_number_must_be_positive(db_session: Session) -> None:
    catalogue = f.make_catalogue(db_session)
    with rejected(db_session):
        f.make_page(db_session, catalogue, 0)


def test_segmentation_is_one_to_one_with_detected_object(db_session: Session) -> None:
    chain = build_chain(db_session)
    with rejected(db_session):
        # By FK (not via the relationship, which would orphan-delete the first result).
        db_session.add(
            SegmentationResult(
                detected_object_id=chain["obj"].id, status="FAILED", model_name="another"
            )
        )
        db_session.flush()


@pytest.mark.parametrize("bad_value", [-0.01, 1.01])
def test_candidate_confidence_must_be_between_zero_and_one(
    db_session: Session, bad_value: float
) -> None:
    catalogue = f.make_catalogue(db_session)
    with rejected(db_session):
        f.make_candidate(db_session, catalogue, overall_confidence=bad_value)


def test_candidate_confidence_may_be_null_or_valid(db_session: Session) -> None:
    catalogue = f.make_catalogue(db_session)
    f.make_candidate(db_session, catalogue)
    f.make_candidate(db_session, catalogue, overall_confidence=0.0)
    f.make_candidate(db_session, catalogue, overall_confidence=1.0)


def test_catalogue_progress_is_bounded(db_session: Session) -> None:
    with rejected(db_session):
        f.make_catalogue(db_session, processing_progress=101)


def test_price_is_optional_and_never_negative(db_session: Session) -> None:
    f.make_product(db_session)  # no price at all is the normal case
    f.make_product(db_session, price=Decimal("199.50"), currency="INR")
    with rejected(db_session):
        f.make_product(db_session, price=Decimal("-1"), currency="INR")


def test_price_without_currency_is_rejected(db_session: Session) -> None:
    with rejected(db_session):
        f.make_product(db_session, price=Decimal("10"))


def test_duplicate_flag_needs_exactly_one_target(db_session: Session) -> None:
    catalogue = f.make_catalogue(db_session)
    candidate = f.make_candidate(db_session, catalogue)
    other = f.make_candidate(db_session, catalogue)
    product = f.make_product(db_session)
    base = {"candidate_id": candidate.id, "signal": DuplicateSignal.SKU, "score": 0.9}

    db_session.add(DuplicateFlag(**base, existing_product_id=product.id))
    db_session.flush()  # one target: fine

    with rejected(db_session):  # no target
        db_session.add(DuplicateFlag(**base))
        db_session.flush()
    with rejected(db_session):  # two targets
        db_session.add(
            DuplicateFlag(**base, existing_product_id=product.id, other_candidate_id=other.id)
        )
        db_session.flush()


def test_invalid_enum_value_is_rejected_by_the_orm(db_session: Session) -> None:
    with rejected(db_session, SQLAlchemyError):
        f.make_catalogue(db_session, status="NOT_A_STATUS")


def test_enum_values_round_trip_as_plain_strings(db_session: Session) -> None:
    catalogue = f.make_catalogue(db_session, status=CatalogueStatus.REVIEW_REQUIRED)
    raw = db_session.execute(
        text("SELECT status FROM catalogues WHERE id = :id"), {"id": catalogue.id}
    ).scalar_one()
    assert raw == "REVIEW_REQUIRED"


def test_database_side_defaults_apply_to_raw_inserts(db_session: Session) -> None:
    """Bypass the ORM so only the *database* defaults are exercised."""
    user = db_session.execute(
        text(
            "INSERT INTO users (email, password_hash) VALUES ('raw@example.com', 'x') "
            "RETURNING id, role, is_active, created_at, updated_at,"
            " failed_login_attempts, locked_until"
        )
    ).one()
    assert user.id is not None
    assert (user.role, user.is_active) == ("CUSTOMER", True)
    assert (user.failed_login_attempts, user.locked_until) == (0, None)
    assert user.created_at is not None and user.updated_at is not None

    catalogue = db_session.execute(
        text(
            "INSERT INTO catalogues (name, original_filename, storage_key, file_size_bytes, sha256)"
            " VALUES ('c', 'c.pdf', 'k', 1, 'abc') RETURNING id, status, processing_progress"
        )
    ).one()
    assert (catalogue.status, catalogue.processing_progress) == ("UPLOADED", 0)

    candidate = db_session.execute(
        text(
            "INSERT INTO product_candidates (catalogue_id) VALUES (:cid) "
            "RETURNING status, origin, fields, flags, reasoning"
        ),
        {"cid": catalogue.id},
    ).one()
    assert (candidate.status, candidate.origin) == ("NEEDS_REVIEW", "AI")
    assert candidate.fields == {} and candidate.flags == [] and candidate.reasoning == []

    product = db_session.execute(
        text(
            "INSERT INTO products (slug, name) VALUES ('raw-slug', 'raw') "
            "RETURNING status, origin, is_featured, specifications, field_provenance"
        )
    ).one()
    assert (product.status, product.origin, product.is_featured) == ("DRAFT", "HUMAN", False)
    assert product.specifications == {} and product.field_provenance == {}


def test_orm_defaults_apply(db_session: Session) -> None:
    catalogue = f.make_catalogue(db_session)
    candidate = f.make_candidate(db_session, catalogue)
    assert catalogue.status is CatalogueStatus.UPLOADED
    assert candidate.status is CandidateStatus.NEEDS_REVIEW
    assert candidate.origin is DataOrigin.AI
    assert f.make_product(db_session).status is ProductStatus.DRAFT
    assert f.make_user(db_session).role is UserRole.CUSTOMER


# ----------------------------------------------------------------------------- restrict rules
def test_category_with_children_cannot_be_deleted(db_session: Session) -> None:
    parent = f.make_category(db_session)
    f.make_category(db_session, parent=parent)
    with rejected(db_session):
        db_session.delete(parent)
        db_session.flush()


def test_category_in_use_by_a_product_cannot_be_deleted(db_session: Session) -> None:
    category = f.make_category(db_session)
    f.make_product(db_session, category=category)
    with rejected(db_session):
        db_session.execute(delete(Category).where(Category.id == category.id))


def test_supplier_in_use_cannot_be_deleted(db_session: Session) -> None:
    supplier = f.make_supplier(db_session)
    f.make_catalogue(db_session, supplier=supplier)
    with rejected(db_session):
        db_session.execute(delete(Supplier).where(Supplier.id == supplier.id))


def test_refresh_tokens_are_unique_and_deleted_with_their_user(db_session: Session) -> None:
    user = f.make_user(db_session)
    family = uuid.uuid4()
    expires = datetime.now(UTC)
    db_session.add(
        RefreshToken(user_id=user.id, family_id=family, token_hash="a" * 64, expires_at=expires)
    )
    db_session.flush()
    with rejected(db_session):  # the same digest twice
        db_session.add(
            RefreshToken(
                user_id=user.id, family_id=family, token_hash="a" * 64, expires_at=expires
            )
        )
        db_session.flush()
    db_session.delete(user)
    db_session.flush()
    assert count(db_session, RefreshToken) == 0


# ----------------------------------------------------------------------------- audit + people
def test_audit_log_survives_deletion_of_the_actor(db_session: Session) -> None:
    admin = f.make_user(db_session, role=UserRole.ADMIN, email="admin@example.com")
    entry = AuditLog(
        actor_id=admin.id,
        actor_email=admin.email,
        action=AuditAction.APPROVED_PRODUCT,
        entity_type="product",
        entity_id="123",
        details={"from": "NEEDS_REVIEW"},
    )
    db_session.add(entry)
    db_session.flush()
    db_session.delete(admin)
    db_session.flush()
    db_session.expire_all()
    kept = db_session.get(AuditLog, entry.id)
    assert kept.actor_id is None
    assert kept.actor_email == "admin@example.com"
    assert kept.details == {"from": "NEEDS_REVIEW"}


def test_guest_enquiry_and_wishlist_rules(db_session: Session) -> None:
    product = f.make_product(db_session)
    guest = Enquiry(
        name="Guest", email="g@example.com", message="Price please?", product_id=product.id
    )
    db_session.add(guest)
    db_session.flush()
    assert guest.user_id is None

    user = f.make_user(db_session)
    db_session.add(WishlistItem(user_id=user.id, product_id=product.id))
    db_session.flush()
    with rejected(db_session):
        db_session.add(WishlistItem(user_id=user.id, product_id=product.id))
        db_session.flush()


def test_corrections_outlive_their_candidate(db_session: Session) -> None:
    catalogue = f.make_catalogue(db_session)
    candidate = f.make_candidate(db_session, catalogue)
    record = CorrectionRecord(
        candidate_id=candidate.id,
        correction_type=CorrectionType.FIELD_EDIT,
        field_name="name",
        ai_value={"value": "YY-1l", "source": "OCR"},
        human_value={"value": "YY-11"},
        context={"catalogue_id": str(catalogue.id), "page": 7},
    )
    db_session.add(record)
    db_session.flush()
    db_session.delete(candidate)
    db_session.flush()
    db_session.expire_all()
    kept = db_session.get(CorrectionRecord, record.id)
    assert kept.candidate_id is None
    assert kept.human_value == {"value": "YY-11"}
    assert kept.context["page"] == 7


def test_null_json_values_are_stored_as_sql_null(db_session: Session) -> None:
    record = CorrectionRecord(correction_type=CorrectionType.FIELD_EDIT, ai_value=None)
    db_session.add(record)
    db_session.flush()
    is_null = db_session.execute(
        text("SELECT ai_value IS NULL FROM correction_records WHERE id = :id"), {"id": record.id}
    ).scalar_one()
    assert is_null is True


# ----------------------------------------------------------------------------- JSONB
def test_candidate_provenance_fields_round_trip(db_session: Session) -> None:
    catalogue = f.make_catalogue(db_session)
    fields = {
        "name": {"value": "YY-11", "source": "OCR", "confidence": 0.97},
        "dimensions": None,
        "material": {"value": "官方 Oak", "source": "OCR", "confidence": 0.71},
    }
    candidate = f.make_candidate(
        db_session, catalogue, fields=fields, flags=["POSSIBLE_DUPLICATE"], reasoning=["same SKU"]
    )
    db_session.expire_all()
    assert db_session.get(ProductCandidate, candidate.id).fields == fields
    found = db_session.scalars(
        select(ProductCandidate).where(ProductCandidate.fields["name"]["source"].astext == "OCR")
    ).all()
    assert [c.id for c in found] == [candidate.id]


# ----------------------------------------------------------------------------- pgvector
def _vec(hot_index: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[hot_index] = 1.0
    return vector


def test_embeddings_support_cosine_nearest_neighbour_search(db_session: Session) -> None:
    catalogue = f.make_catalogue(db_session)
    image = f.make_source_image(db_session, f.make_page(db_session, catalogue))
    for index in (0, 1, 2):
        db_session.add(
            ImageEmbedding(source_image_id=image.id, model_name="t", embedding=_vec(index))
        )
    db_session.flush()

    query = _vec(1)
    nearest = db_session.scalars(
        select(ImageEmbedding)
        .where(ImageEmbedding.source_image_id == image.id)
        .order_by(ImageEmbedding.embedding.cosine_distance(query))
        .limit(1)
    ).one()
    assert nearest.embedding[1] == pytest.approx(1.0)
    assert len(nearest.embedding) == EMBEDDING_DIM


def test_embedding_of_wrong_dimension_is_rejected(db_session: Session) -> None:
    catalogue = f.make_catalogue(db_session)
    image = f.make_source_image(db_session, f.make_page(db_session, catalogue))
    with rejected(db_session, (SQLAlchemyError, ValueError)):
        db_session.add(
            ImageEmbedding(source_image_id=image.id, model_name="t", embedding=[0.1, 0.2])
        )
        db_session.flush()


# ----------------------------------------------------------------------------- settings/categories
def test_app_setting_stores_json_scalars(db_session: Session) -> None:
    db_session.add(AppSetting(key="test.threshold", value=0.85))
    db_session.flush()
    db_session.expire_all()
    assert db_session.get(AppSetting, "test.threshold").value == 0.85


def test_category_tree_navigation(db_session: Session) -> None:
    root = f.make_category(db_session)
    child = f.make_category(db_session, parent=root)
    db_session.expire_all()
    assert child.parent.id == root.id
    assert [c.id for c in root.children] == [child.id]
