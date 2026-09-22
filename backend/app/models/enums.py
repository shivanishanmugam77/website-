"""Enumerations stored as VARCHAR (not native PG enums) so adding a value never
requires a blocking ``ALTER TYPE`` migration."""

from __future__ import annotations

from enum import StrEnum


class UserRole(StrEnum):
    ADMIN = "ADMIN"
    CUSTOMER = "CUSTOMER"


class CatalogueStatus(StrEnum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    COMPLETED = "COMPLETED"
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"
    FAILED = "FAILED"


class PageStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    RENDERED = "RENDERED"  # page image and embedded images extracted; later stages pending
    TEXT_READ = "TEXT_READ"  # text recognised and the page type assigned (Phase 4)
    PANELS_FOUND = "PANELS_FOUND"  # the page's photo panels cut out (Phase 5a)
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PageType(StrEnum):
    COVER = "COVER"
    BRAND = "BRAND"
    MARKETING = "MARKETING"
    PRODUCT = "PRODUCT"
    LIFESTYLE = "LIFESTYLE"
    PRODUCT_DETAIL = "PRODUCT_DETAIL"
    PRODUCT_CONFIGURATION = "PRODUCT_CONFIGURATION"
    SPECIFICATION = "SPECIFICATION"
    COLLAGE = "COLLAGE"
    REFERENCE = "REFERENCE"
    UNKNOWN = "UNKNOWN"


class SourceImageKind(StrEnum):
    PAGE = "PAGE"  # the full rendered page treated as an image
    EMBEDDED = "EMBEDDED"  # image object extracted natively from the PDF
    REGION = "REGION"  # region detected on the rendered page


class SegmentationStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    FAILED = "FAILED"


class CandidateStatus(StrEnum):
    NEEDS_REVIEW = "NEEDS_REVIEW"
    READY = "READY"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MERGED = "MERGED"


class DataOrigin(StrEnum):
    """Who produced a record. ``AI_HUMAN_REVIEW`` = AI output edited by an admin."""

    AI = "AI"
    AI_HUMAN_REVIEW = "AI_HUMAN_REVIEW"
    HUMAN = "HUMAN"


class ProductStatus(StrEnum):
    DRAFT = "DRAFT"
    PUBLISHED = "PUBLISHED"
    UNPUBLISHED = "UNPUBLISHED"


class ProductImageType(StrEnum):
    HERO = "HERO"
    ORIGINAL = "ORIGINAL"
    MASKED = "MASKED"
    DETAIL = "DETAIL"
    LIFESTYLE = "LIFESTYLE"
    CONFIGURATION = "CONFIGURATION"
    THUMBNAIL = "THUMBNAIL"
    SOURCE = "SOURCE"


class JobType(StrEnum):
    CATALOGUE_PROCESS = "CATALOGUE_PROCESS"
    PAGE_RETRY = "PAGE_RETRY"
    CANDIDATE_REPROCESS = "CANDIDATE_REPROCESS"


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class EnquiryStatus(StrEnum):
    NEW = "NEW"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"


class AuditAction(StrEnum):
    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILED = "LOGIN_FAILED"
    LOGOUT = "LOGOUT"
    USER_REGISTERED = "USER_REGISTERED"
    ADMIN_CREATED = "ADMIN_CREATED"
    PASSWORD_CHANGED = "PASSWORD_CHANGED"
    PASSWORD_CHANGE_FAILED = "PASSWORD_CHANGE_FAILED"
    TOKEN_REUSE_DETECTED = "TOKEN_REUSE_DETECTED"
    UPLOAD_CATALOGUE = "UPLOAD_CATALOGUE"
    REPROCESS_CATALOGUE = "REPROCESS_CATALOGUE"
    DELETE_CATALOGUE = "DELETE_CATALOGUE"
    CREATE_PRODUCT = "CREATE_PRODUCT"
    EDIT_PRODUCT = "EDIT_PRODUCT"
    DELETE_PRODUCT = "DELETE_PRODUCT"
    APPROVED_PRODUCT = "APPROVED_PRODUCT"
    REJECT_PRODUCT = "REJECT_PRODUCT"
    PUBLISH_PRODUCT = "PUBLISH_PRODUCT"
    UNPUBLISH_PRODUCT = "UNPUBLISH_PRODUCT"
    REPROCESS_PRODUCT = "REPROCESS_PRODUCT"
    MERGE_CANDIDATES = "MERGE_CANDIDATES"
    SPLIT_CANDIDATE = "SPLIT_CANDIDATE"
    MANAGE_CATEGORY = "MANAGE_CATEGORY"
    MANAGE_SUPPLIER = "MANAGE_SUPPLIER"
    PANEL_CREATED = "PANEL_CREATED"
    PANEL_UPDATED = "PANEL_UPDATED"
    PANEL_DELETED = "PANEL_DELETED"
    PANEL_EDIT_UNDONE = "PANEL_EDIT_UNDONE"
    CORRECTIONS_DISCARDED = "CORRECTIONS_DISCARDED"
    CANDIDATES_ASSEMBLED = "CANDIDATES_ASSEMBLED"


class PanelEditAction(StrEnum):
    """What an admin did to a photo panel (one row per change in ``panel_edits``)."""

    CREATE = "CREATE"  # drew a panel the system missed
    UPDATE = "UPDATE"  # moved or resized one
    DELETE = "DELETE"  # removed one that was not needed


class CorrectionType(StrEnum):
    FIELD_EDIT = "FIELD_EDIT"
    IMAGE_REPLACED = "IMAGE_REPLACED"
    MASK_CORRECTED = "MASK_CORRECTED"
    CROP_CORRECTED = "CROP_CORRECTED"
    MERGE = "MERGE"
    SPLIT = "SPLIT"
    REJECTION = "REJECTION"


class DuplicateSignal(StrEnum):
    SKU = "SKU"
    IMAGE_HASH = "IMAGE_HASH"
    OCR_TEXT = "OCR_TEXT"
    EMBEDDING = "EMBEDDING"


class DuplicateStatus(StrEnum):
    OPEN = "OPEN"
    DISMISSED = "DISMISSED"
    CONFIRMED = "CONFIRMED"
