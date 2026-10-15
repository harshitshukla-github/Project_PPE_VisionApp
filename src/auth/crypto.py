"""Hashing and rules for passwords, PINs and session tokens.

Passwords and PINs are hashed with Argon2id (argon2-cffi's defaults: 3
passes over 64 MiB). Session tokens are random and long, so a plain SHA-256
is enough to store them: the point is only that the database never holds a
token anyone could use.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from functools import lru_cache

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

_hasher = PasswordHasher()

PIN_LENGTH = 6
PASSWORD_MIN = 12
PASSWORD_MAX = 128
WORKER_CODE_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9-]{1,31}$")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def hash_secret(secret: str) -> str:
    """Hash a password or PIN for storage."""
    return _hasher.hash(secret)


def verify_secret(stored_hash: str, secret: str) -> bool:
    """Return whether ``secret`` matches ``stored_hash``. Never raises."""
    try:
        return _hasher.verify(stored_hash, secret)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(stored_hash: str) -> bool:
    """Whether a hash was made with weaker parameters than today's defaults."""
    return _hasher.check_needs_rehash(stored_hash)


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    return hash_secret("no account has this secret")


def spend_verification_time(secret: str) -> None:
    """Do the work of a real check, for an account that doesn't exist.

    Without this, "no such account" answers measurably faster than "wrong
    PIN", and the login form would reveal which accounts exist.
    """
    verify_secret(_dummy_hash(), secret)


def new_token() -> str:
    """A session token: 32 random bytes, URL-safe, 43 characters."""
    return secrets.token_urlsafe(32)


def new_csrf_token() -> str:
    """A CSRF token for a dashboard session."""
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    """The form a session token is stored and looked up in."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def tokens_match(expected: str | None, given: str | None) -> bool:
    """Compare two tokens in constant time; ``None`` never matches."""
    if not expected or not given:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), given.encode("utf-8"))


# ---------------------------------------------------------------------------
# Rules (docs/auth-api.md §4 and §6)
# ---------------------------------------------------------------------------


def pin_problem(pin: str, current_pin_hash: str | None = None) -> str | None:
    """Explain why ``pin`` can't be used, or return ``None`` if it can.

    Args:
        pin: The proposed PIN.
        current_pin_hash: The worker's current PIN hash, to refuse reusing it.

    Returns:
        A sentence for the worker, or ``None``.
    """
    if not re.fullmatch(rf"\d{{{PIN_LENGTH}}}", pin or ""):
        return f"A PIN must be exactly {PIN_LENGTH} digits."
    if len(set(pin)) == 1:
        return "A PIN can't be one digit repeated."
    steps = {int(b) - int(a) for a, b in zip(pin, pin[1:])}
    if steps in ({1}, {-1}):
        return "A PIN can't be a straight run like 123456 or 654321."
    if current_pin_hash and verify_secret(current_pin_hash, pin):
        return "Choose a PIN different from your current one."
    return None


def generate_pin() -> str:
    """A random PIN that passes :func:`pin_problem`, for a supervisor to issue."""
    while True:
        pin = f"{secrets.randbelow(10**PIN_LENGTH):0{PIN_LENGTH}d}"
        if pin_problem(pin) is None:
            return pin


def password_problem(
    password: str, email: str, current_password_hash: str | None = None
) -> str | None:
    """Explain why ``password`` can't be used, or return ``None`` if it can."""
    if len(password or "") < PASSWORD_MIN:
        return f"A password needs at least {PASSWORD_MIN} characters."
    if len(password) > PASSWORD_MAX:
        return f"A password can have at most {PASSWORD_MAX} characters."
    if password.strip().lower() == email.strip().lower():
        return "A password can't be your email address."
    if current_password_hash and verify_secret(current_password_hash, password):
        return "Choose a password different from your current one."
    return None


def normalize_worker_code(raw: str) -> str:
    """Worker IDs are matched without regard to case: ``w-0042`` → ``W-0042``."""
    return (raw or "").strip().upper()


def is_valid_worker_code(code: str) -> bool:
    """2–32 letters, digits and hyphens, not starting with a hyphen (upper case)."""
    return bool(WORKER_CODE_PATTERN.fullmatch(code))


def normalize_email(raw: str) -> str:
    """Emails are stored and matched in lower case."""
    return (raw or "").strip().lower()


def is_valid_email(email: str) -> bool:
    """A loose shape check: something@something.something, at most 254 characters."""
    return len(email) <= 254 and bool(EMAIL_PATTERN.fullmatch(email))
