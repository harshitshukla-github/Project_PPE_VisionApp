"""Login logic: verify credentials, lock out guessing, issue and check sessions.

Every function takes a SQLAlchemy ``Session`` and commits what it changes
itself, including a failed attempt that is about to be reported as an error:
a lockout counter that rolled back with the error would never lock anything.

The behaviour implemented here is specified in docs/auth-api.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.auth.crypto import (
    generate_pin,
    hash_secret,
    is_valid_email,
    is_valid_worker_code,
    needs_rehash,
    new_csrf_token,
    new_token,
    normalize_email,
    normalize_worker_code,
    password_problem,
    pin_problem,
    spend_verification_time,
    token_hash,
    verify_secret,
)
from src.auth.errors import AuthError, human_duration
from src.auth.ratelimit import ip_limiter
from src.db.models import AuthSession, Role, User, Worker
from src.settings import settings

MAX_FAILED_ATTEMPTS = 5
LOCKOUT = timedelta(minutes=15)
# last_seen_at is written at most this often, not on every request.
SEEN_WRITE_INTERVAL = timedelta(seconds=60)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class IssuedSession:
    """A new session. ``token`` exists only here; the database keeps its hash."""

    token: str
    record: AuthSession


# ---------------------------------------------------------------------------
# Signing in
# ---------------------------------------------------------------------------


def login_worker(
    db: Session, worker_id: str, pin: str, ip: str | None, user_agent: str | None
) -> tuple[IssuedSession, Worker]:
    """Check a worker ID and PIN and start a worker session.

    Raises:
        AuthError: 401 ``invalid_credentials``, 403 ``account_disabled``,
            423 ``account_locked`` or 429 ``too_many_attempts``.
    """
    _refuse_if_ip_limited(ip)
    now = utcnow()
    worker = db.scalar(
        select(Worker)
        .where(Worker.worker_code == normalize_worker_code(worker_id))
        .with_for_update()
    )
    wrong = AuthError(401, "invalid_credentials", "Wrong worker ID or PIN.")
    if worker is None:
        spend_verification_time(pin)
        db.rollback()
        _record_ip_failure(ip)
        raise wrong
    _refuse_if_locked(db, worker, now, "PIN")
    if not verify_secret(worker.pin_hash, pin):
        raise _count_failure(db, worker, now, ip, wrong, "PIN")
    if not worker.is_active:
        db.rollback()
        raise AuthError(403, "account_disabled",
                        "This worker account is disabled. Ask your supervisor.")

    _clear_failures(worker, now)
    if needs_rehash(worker.pin_hash):
        worker.pin_hash = hash_secret(pin)
    issued = _issue(db, worker=worker, lifetime=timedelta(hours=settings.ppb_worker_session_hours),
                    ip=ip, user_agent=user_agent, now=now)
    db.commit()
    return issued, worker


def login_user(
    db: Session, email: str, password: str, ip: str | None, user_agent: str | None
) -> tuple[IssuedSession, User]:
    """Check an email and password and start a dashboard session.

    Raises:
        AuthError: as :func:`login_worker`.
    """
    _refuse_if_ip_limited(ip)
    now = utcnow()
    user = db.scalar(
        select(User).where(User.email == normalize_email(email)).with_for_update()
    )
    wrong = AuthError(401, "invalid_credentials", "Wrong email or password.")
    if user is None:
        spend_verification_time(password)
        db.rollback()
        _record_ip_failure(ip)
        raise wrong
    _refuse_if_locked(db, user, now, "password")
    if not verify_secret(user.password_hash, password):
        raise _count_failure(db, user, now, ip, wrong, "password")
    if not user.is_active:
        db.rollback()
        raise AuthError(403, "account_disabled", "This account is disabled. Ask an admin.")

    _clear_failures(user, now)
    if needs_rehash(user.password_hash):
        user.password_hash = hash_secret(password)
    issued = _issue(db, user=user, lifetime=timedelta(hours=settings.ppb_user_session_hours),
                    ip=ip, user_agent=user_agent, now=now, csrf=True)
    db.commit()
    return issued, user


# ---------------------------------------------------------------------------
# Using a session
# ---------------------------------------------------------------------------


def resolve_worker_session(db: Session, token: str) -> AuthSession | None:
    """The live worker session for ``token``, or ``None``."""
    record = _live_session(db, token)
    if record is None or record.worker_id is None or not record.worker.is_active:
        return None
    _touch(db, record)
    return record


def resolve_user_session(db: Session, token: str) -> AuthSession | None:
    """The live dashboard session for ``token``, or ``None``.

    Dashboard sessions also end after ``PPB_USER_IDLE_MINUTES`` without a
    request.
    """
    record = _live_session(db, token)
    if record is None or record.user_id is None or not record.user.is_active:
        return None
    idle_limit = timedelta(minutes=settings.ppb_user_idle_minutes)
    if record.last_seen_at + idle_limit <= utcnow():
        return None
    _touch(db, record)
    return record


def user_session_expiry(record: AuthSession) -> datetime:
    """When a dashboard session ends if it stays idle from now."""
    idle_end = record.last_seen_at + timedelta(minutes=settings.ppb_user_idle_minutes)
    return min(record.expires_at, idle_end)


def revoke(db: Session, record: AuthSession) -> None:
    """End one session."""
    if record.revoked_at is None:
        record.revoked_at = utcnow()
    db.commit()


# ---------------------------------------------------------------------------
# Changing a PIN or password
# ---------------------------------------------------------------------------


def change_worker_pin(db: Session, record: AuthSession, current_pin: str, new_pin: str,
                      ip: str | None) -> None:
    """Replace a worker's PIN and end their other sessions.

    The current PIN is checked like a login, lockout included, so a stolen
    token can't be used to guess it.

    Raises:
        AuthError: 401 ``invalid_credentials``, 422 ``weak_pin``, 423 ``account_locked``.
    """
    now = utcnow()
    worker = db.scalar(select(Worker).where(Worker.id == record.worker_id).with_for_update())
    _refuse_if_locked(db, worker, now, "PIN")
    if not verify_secret(worker.pin_hash, current_pin):
        raise _count_failure(db, worker, now, ip,
                             AuthError(401, "invalid_credentials", "Your current PIN is wrong."),
                             "PIN")
    problem = pin_problem(new_pin, worker.pin_hash)
    if problem:
        db.rollback()
        raise AuthError(422, "weak_pin", problem)
    worker.pin_hash = hash_secret(new_pin)
    worker.must_change_pin = False
    worker.failed_attempts = 0
    _revoke_others(db, AuthSession.worker_id == worker.id, keep=record.id, now=now)
    db.commit()


def change_user_password(db: Session, record: AuthSession, current_password: str,
                         new_password: str, ip: str | None) -> None:
    """Replace a dashboard user's password and end their other sessions.

    Raises:
        AuthError: 401 ``invalid_credentials``, 422 ``weak_password``,
            423 ``account_locked``.
    """
    now = utcnow()
    user = db.scalar(select(User).where(User.id == record.user_id).with_for_update())
    _refuse_if_locked(db, user, now, "password")
    if not verify_secret(user.password_hash, current_password):
        raise _count_failure(db, user, now, ip,
                             AuthError(401, "invalid_credentials",
                                       "Your current password is wrong."),
                             "password")
    problem = password_problem(new_password, user.email, user.password_hash)
    if problem:
        db.rollback()
        raise AuthError(422, "weak_password", problem)
    user.password_hash = hash_secret(new_password)
    user.must_change_password = False
    user.failed_attempts = 0
    _revoke_others(db, AuthSession.user_id == user.id, keep=record.id, now=now)
    db.commit()


# ---------------------------------------------------------------------------
# Creating accounts (CLI now, admin API in Phase 3)
# ---------------------------------------------------------------------------


def create_user(db: Session, email: str, display_name: str, role: Role, password: str,
                must_change_password: bool = False) -> User:
    """Create a dashboard account.

    Raises:
        ValueError: The email, name or password is unusable, or the email is taken.
    """
    email = normalize_email(email)
    if not is_valid_email(email):
        raise ValueError(f"'{email}' is not an email address.")
    name = _clean_name(display_name)
    problem = password_problem(password, email)
    if problem:
        raise ValueError(problem)
    user = User(email=email, display_name=name, role=Role(role).value,
                password_hash=hash_secret(password), must_change_password=must_change_password)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise ValueError(f"An account for {email} already exists.") from None
    return user


def create_worker(db: Session, worker_id: str, display_name: str,
                  created_by: int | None = None) -> tuple[Worker, str]:
    """Create a worker with a one-time PIN, which must be changed at first login.

    Returns:
        The worker, and the PIN to hand over. It is not stored anywhere.

    Raises:
        ValueError: The worker ID or name is unusable, or the ID is taken.
    """
    code = normalize_worker_code(worker_id)
    if not is_valid_worker_code(code):
        raise ValueError(
            f"'{worker_id}' is not a valid worker ID: use 2-32 letters, digits and "
            "hyphens, not starting with a hyphen."
        )
    pin = generate_pin()
    worker = Worker(worker_code=code, display_name=_clean_name(display_name),
                    pin_hash=hash_secret(pin), must_change_pin=True, created_by=created_by)
    db.add(worker)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise ValueError(f"Worker {code} already exists.") from None
    return worker, pin


def reset_worker_pin(db: Session, worker_id: str) -> tuple[Worker, str]:
    """Issue a new one-time PIN, unlock the worker and end all their sessions.

    Raises:
        ValueError: No such worker.
    """
    code = normalize_worker_code(worker_id)
    worker = db.scalar(select(Worker).where(Worker.worker_code == code).with_for_update())
    if worker is None:
        db.rollback()
        raise ValueError(f"No worker with ID {code}.")
    pin = generate_pin()
    now = utcnow()
    worker.pin_hash = hash_secret(pin)
    worker.must_change_pin = True
    worker.failed_attempts = 0
    worker.locked_until = None
    _revoke_others(db, AuthSession.worker_id == worker.id, keep=None, now=now)
    db.commit()
    return worker, pin


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _issue(db: Session, *, lifetime: timedelta, ip: str | None, user_agent: str | None,
           now: datetime, user: User | None = None, worker: Worker | None = None,
           csrf: bool = False) -> IssuedSession:
    token = new_token()
    record = AuthSession(
        token_hash=token_hash(token),
        csrf_token=new_csrf_token() if csrf else None,
        user=user,
        worker=worker,
        created_at=now,
        last_seen_at=now,
        expires_at=now + lifetime,
        ip=(ip or "")[:64] or None,
        user_agent=(user_agent or "")[:255] or None,
    )
    db.add(record)
    db.flush()
    return IssuedSession(token=token, record=record)


def _live_session(db: Session, token: str) -> AuthSession | None:
    if not token:
        return None
    record = db.scalar(select(AuthSession).where(AuthSession.token_hash == token_hash(token)))
    if record is None or record.revoked_at is not None or record.expires_at <= utcnow():
        return None
    return record


def _touch(db: Session, record: AuthSession) -> None:
    now = utcnow()
    if now - record.last_seen_at >= SEEN_WRITE_INTERVAL:
        record.last_seen_at = now
        db.commit()


def _revoke_others(db: Session, owner_clause, keep: int | None, now: datetime) -> None:
    statement = update(AuthSession).where(owner_clause, AuthSession.revoked_at.is_(None))
    if keep is not None:
        statement = statement.where(AuthSession.id != keep)
    db.execute(statement.values(revoked_at=now))


def _refuse_if_locked(db: Session, account: User | Worker, now: datetime, what: str) -> None:
    if account.locked_until is not None and account.locked_until > now:
        db.rollback()
        wait = (account.locked_until - now).total_seconds()
        raise AuthError(423, "account_locked",
                        f"Too many wrong {what}s. Try again in {human_duration(wait)}.", wait)


def _count_failure(db: Session, account: User | Worker, now: datetime, ip: str | None,
                   wrong: AuthError, what: str) -> AuthError:
    """Record a wrong secret; return the error to raise (locked on the last try)."""
    account.failed_attempts += 1
    error = wrong
    if account.failed_attempts >= MAX_FAILED_ATTEMPTS:
        account.locked_until = now + LOCKOUT
        account.failed_attempts = 0
        wait = LOCKOUT.total_seconds()
        error = AuthError(423, "account_locked",
                          f"Too many wrong {what}s. Try again in {human_duration(wait)}.", wait)
    db.commit()
    _record_ip_failure(ip)
    return error


def _clear_failures(account: User | Worker, now: datetime) -> None:
    account.failed_attempts = 0
    account.locked_until = None
    account.last_login_at = now


def _refuse_if_ip_limited(ip: str | None) -> None:
    wait = ip_limiter.retry_after(ip or "unknown")
    if wait is not None:
        raise AuthError(429, "too_many_attempts",
                        f"Too many sign-in attempts from this network. "
                        f"Try again in {human_duration(wait)}.", wait)


def _record_ip_failure(ip: str | None) -> None:
    ip_limiter.record_failure(ip or "unknown")


def _clean_name(display_name: str) -> str:
    name = " ".join((display_name or "").split())
    if not name or len(name) > 100:
        raise ValueError("A name is needed, of at most 100 characters.")
    return name
