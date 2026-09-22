from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.api.deps import AdminUser, ClientInfoDep, DbSession, require_admin
from app.models import Supplier
from app.models.enums import AuditAction
from app.schemas.catalogue import SupplierCreate, SupplierOut
from app.services.audit import record_audit
from app.utils.slugs import slugify

router = APIRouter(
    prefix="/admin/suppliers", tags=["suppliers"], dependencies=[Depends(require_admin)]
)


@router.get("", response_model=list[SupplierOut])
def list_suppliers(db: DbSession) -> list[Supplier]:
    return list(db.scalars(select(Supplier).order_by(Supplier.name)))


@router.post("", response_model=SupplierOut, status_code=status.HTTP_201_CREATED)
def create_supplier(
    payload: SupplierCreate, db: DbSession, admin: AdminUser, client: ClientInfoDep
) -> Supplier:
    name = payload.name.strip()
    # Names in scripts without a Latin transliteration slugify to "": fall back to an id.
    slug = slugify(name) or f"supplier-{uuid.uuid4().hex[:8]}"
    supplier = Supplier(
        name=name,
        slug=slug,
        website=payload.website,
        contact_email=payload.contact_email,
        description=payload.description,
    )
    try:
        with db.begin_nested():
            db.add(supplier)
    except IntegrityError:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="A supplier with this (or a very similar) name already exists",
        ) from None
    record_audit(
        db,
        AuditAction.MANAGE_SUPPLIER,
        actor=admin,
        entity_type="supplier",
        entity_id=str(supplier.id),
        details={"action": "create", "name": name},
        ip_address=client.ip,
    )
    db.commit()
    return supplier
