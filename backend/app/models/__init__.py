"""Importing this package registers every model on ``Base.metadata`` (Alembic relies on it)."""

from app.models.app_setting import AppSetting
from app.models.audit import AuditLog, CorrectionRecord
from app.models.base import Base
from app.models.candidate import (
    ProductCandidate,
    candidate_detected_objects,
    candidate_pages,
    candidate_source_images,
)
from app.models.catalogue import Catalogue, CataloguePage
from app.models.category import Category
from app.models.duplicate import DuplicateFlag
from app.models.engagement import Enquiry, WishlistItem
from app.models.job import ProcessingJob
from app.models.media import (
    DetectedObject,
    ImageEmbedding,
    OCRResult,
    SegmentationResult,
    SourceImage,
)
from app.models.panel_edit import PanelEdit
from app.models.product import Product, ProductImage
from app.models.refresh_token import RefreshToken
from app.models.supplier import Supplier
from app.models.user import User

__all__ = [
    "AppSetting",
    "AuditLog",
    "Base",
    "Catalogue",
    "CataloguePage",
    "Category",
    "CorrectionRecord",
    "DetectedObject",
    "DuplicateFlag",
    "Enquiry",
    "ImageEmbedding",
    "OCRResult",
    "PanelEdit",
    "ProcessingJob",
    "Product",
    "ProductCandidate",
    "ProductImage",
    "RefreshToken",
    "SegmentationResult",
    "SourceImage",
    "Supplier",
    "User",
    "WishlistItem",
    "candidate_detected_objects",
    "candidate_pages",
    "candidate_source_images",
]
