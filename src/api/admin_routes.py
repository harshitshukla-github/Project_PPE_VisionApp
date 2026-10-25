"""The admin API, as specified in docs/admin-api.md.

Every endpoint needs a dashboard session, and every write its CSRF token
(both enforced by :func:`require_user`). Admin-only endpoints say so with
``require_user(Role.ADMIN)``; the rest admit supervisors too, and the
modules in :mod:`src.admin` scope what a supervisor sees to their own site.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Body, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.admin import audit as audit_log
from src.admin import jobs, settings_api, sites, users, workers
from src.admin.common import applies_within_s, page
from src.admin.metrics import load_ap50
from src.admin.views import audit_view, user_view, worker_view
from src.api.deps import get_detector
from src.auth.deps import client_ip, db_session, require_user
from src.db.models import AuthSession, PpeItem, Role
from src.inference.detector import PPEDetector

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

ANY_USER = Depends(require_user())
ADMIN = Depends(require_user(Role.ADMIN))


# ---------------------------------------------------------------------------
# Request bodies. Optional fields are optional so a PATCH can name only what
# changes; which were actually sent is read from model_fields_set.
# ---------------------------------------------------------------------------


class SiteCreate(BaseModel):
    key: str = Field(max_length=200)
    display_name: str = Field(max_length=500)


class SitePatch(BaseModel):
    display_name: str | None = Field(None, max_length=500)
    is_active: bool | None = None


class RequirementIn(BaseModel):
    item: str = Field(max_length=200)
    mode: str = Field(max_length=50)


class JobCreate(BaseModel):
    key: str = Field(max_length=200)
    display_name: str = Field(max_length=500)
    requirements: list[RequirementIn] = Field(max_length=100)


class JobPatch(BaseModel):
    version: int
    display_name: str | None = Field(None, max_length=500)
    is_active: bool | None = None


class RequirementsPut(BaseModel):
    version: int
    requirements: list[RequirementIn] = Field(max_length=100)


class JobOrder(BaseModel):
    keys: list[str] = Field(max_length=500)


class UserCreate(BaseModel):
    email: str = Field(max_length=500)
    display_name: str = Field(max_length=500)
    role: Literal["admin", "supervisor"]
    site: str | None = Field(None, max_length=200)


class UserPatch(BaseModel):
    display_name: str | None = Field(None, max_length=500)
    role: Literal["admin", "supervisor"] | None = None
    site: str | None = Field(None, max_length=200)
    is_active: bool | None = None


class WorkerCreate(BaseModel):
    worker_id: str = Field(max_length=200)
    display_name: str = Field(max_length=500)
    site: str | None = Field(None, max_length=200)


class WorkerPatch(BaseModel):
    display_name: str | None = Field(None, max_length=500)
    site: str | None = Field(None, max_length=200)
    is_active: bool | None = None


def _sent(body: BaseModel, *exclude: str) -> dict[str, Any]:
    """The fields the client actually sent, `null` included."""
    return {name: getattr(body, name) for name in body.model_fields_set if name not in exclude}


def _reqs(body_reqs: list[RequirementIn]) -> list[dict[str, str]]:
    return [{"item": r.item, "mode": r.mode} for r in body_reqs]


# ---------------------------------------------------------------------------
# Sites (§3)
# ---------------------------------------------------------------------------


@router.get("/sites")
def list_sites(record: AuthSession = ANY_USER, db: Session = Depends(db_session)) -> list[dict]:
    """Every site for an admin; only their own for a supervisor."""
    return sites.list_sites(db, record.user)


@router.post("/sites", status_code=201)
def create_site(body: SiteCreate, request: Request, record: AuthSession = ADMIN,
                db: Session = Depends(db_session)) -> dict:
    site = sites.create_site(db, body.key, body.display_name, record.user.email,
                             client_ip(request))
    return {"site": sites.view(db, site)}


@router.patch("/sites/{key}")
def update_site(key: str, body: SitePatch, request: Request, record: AuthSession = ADMIN,
                db: Session = Depends(db_session)) -> dict:
    site = sites.update_site(db, key, _sent(body), record.user.email, client_ip(request))
    return {"site": sites.view(db, site)}


# ---------------------------------------------------------------------------
# Items (§4)
# ---------------------------------------------------------------------------


@router.get("/items")
def list_items(record: AuthSession = ANY_USER, db: Session = Depends(db_session),
               detector: PPEDetector = Depends(get_detector)) -> list[dict]:
    """The catalogue, with the loaded model's AP50 per item where it has one."""
    ap50 = load_ap50(detector.weights_path)
    return [
        {"key": item.key, "display_name": item.display_name, "detectable": item.detectable,
         "detector_ap50": ap50.get(item.key) if item.detectable else None}
        for item in db.query(PpeItem).order_by(PpeItem.key)
    ]


# ---------------------------------------------------------------------------
# Jobs and the matrix (§5)
# ---------------------------------------------------------------------------


@router.get("/jobs")
def list_jobs(record: AuthSession = ANY_USER, db: Session = Depends(db_session)) -> list[dict]:
    return [jobs.view(job) for job in jobs.list_jobs(db)]


@router.post("/jobs", status_code=201)
def create_job(body: JobCreate, request: Request, record: AuthSession = ADMIN,
               db: Session = Depends(db_session)) -> dict:
    job, warnings = jobs.create_job(db, body.key, body.display_name, _reqs(body.requirements),
                                    record.user.email, client_ip(request))
    return {"job": jobs.view(job), "warnings": warnings, "applies_within_s": applies_within_s()}


@router.patch("/jobs/{key}")
def update_job(key: str, body: JobPatch, request: Request, record: AuthSession = ADMIN,
               db: Session = Depends(db_session)) -> dict:
    job = jobs.update_job(db, key, body.version, _sent(body, "version"), record.user.email,
                          client_ip(request))
    return {"job": jobs.view(job), "applies_within_s": applies_within_s()}


@router.put("/jobs/{key}/requirements")
def replace_requirements(key: str, body: RequirementsPut, request: Request,
                         record: AuthSession = ADMIN,
                         db: Session = Depends(db_session)) -> dict:
    job, warnings = jobs.replace_requirements(db, key, body.version, _reqs(body.requirements),
                                              record.user.email, client_ip(request))
    return {"job": jobs.view(job), "warnings": warnings, "applies_within_s": applies_within_s()}


@router.put("/job-order")
def reorder_jobs(body: JobOrder, request: Request, record: AuthSession = ADMIN,
                 db: Session = Depends(db_session)) -> list[dict]:
    return [jobs.view(job) for job in jobs.reorder(db, body.keys, record.user.email,
                                                    client_ip(request))]


# ---------------------------------------------------------------------------
# Runtime settings (§6)
# ---------------------------------------------------------------------------


@router.get("/settings")
def list_settings(record: AuthSession = ADMIN, db: Session = Depends(db_session)) -> list[dict]:
    return settings_api.list_settings(db)


@router.patch("/settings")
def patch_settings(request: Request, patch: dict[str, Any] = Body(...),
                   record: AuthSession = ADMIN, db: Session = Depends(db_session)) -> dict:
    """An RFC 7396 merge patch of setting values, applied all or nothing."""
    warnings = settings_api.patch_settings(db, patch, record.user.email, client_ip(request))
    return {"settings": settings_api.list_settings(db), "warnings": warnings,
            "applies_within_s": applies_within_s()}


# ---------------------------------------------------------------------------
# Dashboard users (§7)
# ---------------------------------------------------------------------------


@router.get("/users")
def list_users(record: AuthSession = ADMIN, db: Session = Depends(db_session)) -> list[dict]:
    return [user_view(user) for user in users.list_users(db)]


@router.post("/users", status_code=201)
def create_user(body: UserCreate, request: Request, record: AuthSession = ADMIN,
                db: Session = Depends(db_session)) -> dict:
    user, password = users.create_user(db, body.model_dump(), record.user.email,
                                       client_ip(request))
    return {"user": user_view(users.get_user(db, user.id)), "temporary_password": password}


@router.patch("/users/{user_id}")
def update_user(user_id: int, body: UserPatch, request: Request, record: AuthSession = ADMIN,
                db: Session = Depends(db_session)) -> dict:
    user = users.update_user(db, user_id, _sent(body), record.user, record.user.email,
                             client_ip(request))
    return {"user": user_view(user)}


@router.post("/users/{user_id}/reset-password")
def reset_password(user_id: int, request: Request, record: AuthSession = ADMIN,
                   db: Session = Depends(db_session)) -> dict:
    return {"temporary_password": users.reset_password(db, user_id, record.user,
                                                       record.user.email, client_ip(request))}


# ---------------------------------------------------------------------------
# Workers (§8)
# ---------------------------------------------------------------------------


@router.get("/workers")
def list_workers(
    search: str | None = Query(None, max_length=100),
    site: str | None = Query(None, max_length=200),
    active: bool | None = None,
    locked: bool | None = None,
    limit: int = Query(50, ge=1, le=200),
    cursor: str | None = Query(None, max_length=500),
    record: AuthSession = ANY_USER,
    db: Session = Depends(db_session),
) -> dict:
    rows, next_cursor, total = workers.list_workers(
        db, record.user, search=search, site=site, active=active, locked=locked, limit=limit,
        cursor=cursor)
    return page([worker_view(w) for w in rows], next_cursor, total)


@router.get("/workers/{worker_id}")
def get_worker(worker_id: str, record: AuthSession = ANY_USER,
               db: Session = Depends(db_session)) -> dict:
    return worker_view(workers.get_worker(db, record.user, worker_id))


@router.post("/workers", status_code=201)
def create_worker(body: WorkerCreate, request: Request, record: AuthSession = ANY_USER,
                  db: Session = Depends(db_session)) -> dict:
    worker, pin = workers.create_worker(db, record.user, _sent(body), record.user.email,
                                        client_ip(request))
    return {"worker": worker_view(worker), "one_time_pin": pin}


@router.patch("/workers/{worker_id}")
def update_worker(worker_id: str, body: WorkerPatch, request: Request,
                  record: AuthSession = ANY_USER, db: Session = Depends(db_session)) -> dict:
    worker = workers.update_worker(db, record.user, worker_id, _sent(body), record.user.email,
                                   client_ip(request))
    return {"worker": worker_view(worker)}


@router.post("/workers/{worker_id}/reset-pin")
def reset_pin(worker_id: str, request: Request, record: AuthSession = ANY_USER,
              db: Session = Depends(db_session)) -> dict:
    return {"one_time_pin": workers.reset_pin(db, record.user, worker_id, record.user.email,
                                              client_ip(request))}


# ---------------------------------------------------------------------------
# Audit log (§9)
# ---------------------------------------------------------------------------


@router.get("/audit")
def list_audit(
    action: str | None = Query(None, max_length=64),
    target: str | None = Query(None, max_length=160),
    actor: str | None = Query(None, max_length=254),
    site: str | None = Query(None, max_length=64),
    from_: datetime | None = Query(None, alias="from"),
    to: datetime | None = None,
    limit: int = Query(50, ge=1, le=200),
    cursor: str | None = Query(None, max_length=500),
    record: AuthSession = ADMIN,
    db: Session = Depends(db_session),
) -> dict:
    rows, next_cursor, total = audit_log.list_entries(
        db, action=action, target=target, actor=actor, site=site, start=from_, end=to,
        limit=limit, cursor=cursor)
    return page([audit_view(e) for e in rows], next_cursor, total)
