"""Login rules that need no database: PINs, passwords, IDs, tokens, limits.

The behaviour is specified in docs/auth-api.md.
"""

from __future__ import annotations

import json

import pytest

from src.api.errors import ApiError
from src.auth.crypto import (
    generate_pin,
    hash_secret,
    is_valid_email,
    is_valid_worker_code,
    new_token,
    normalize_email,
    normalize_worker_code,
    password_problem,
    pin_problem,
    token_hash,
    tokens_match,
    verify_secret,
)
from src.auth.errors import AuthError, human_duration
from src.auth.ratelimit import FailureLimiter


class TestPins:
    @pytest.mark.parametrize("pin", ["482913", "730154", "100200", "909091"])
    def test_acceptable(self, pin: str) -> None:
        assert pin_problem(pin) is None

    @pytest.mark.parametrize(
        ("pin", "why"),
        [
            ("12345", "exactly 6 digits"),
            ("1234567", "exactly 6 digits"),
            ("12a456", "exactly 6 digits"),
            ("", "exactly 6 digits"),
            ("111111", "one digit repeated"),
            ("123456", "straight run"),
            ("654321", "straight run"),
            ("012345", "straight run"),
            ("456789", "straight run"),
        ],
    )
    def test_refused(self, pin: str, why: str) -> None:
        assert why in pin_problem(pin)

    def test_current_pin_cannot_be_reused(self) -> None:
        current = hash_secret("482913")
        assert "different" in pin_problem("482913", current)
        assert pin_problem("730154", current) is None

    def test_generated_pins_always_pass_the_rules(self) -> None:
        pins = [generate_pin() for _ in range(300)]
        assert all(pin_problem(p) is None for p in pins)
        assert len(set(pins)) > 250  # random, not a fixed sequence


class TestPasswords:
    def test_acceptable(self) -> None:
        assert password_problem("correct horse battery", "a@example.com") is None

    @pytest.mark.parametrize(
        ("password", "why"),
        [
            ("short", "at least 12"),
            ("x" * 129, "at most 128"),
            ("Admin@Example.com", "email address"),
        ],
    )
    def test_refused(self, password: str, why: str) -> None:
        assert why in password_problem(password, "admin@example.com")

    def test_current_password_cannot_be_reused(self) -> None:
        current = hash_secret("correct horse battery")
        assert "different" in password_problem("correct horse battery", "a@example.com", current)


class TestIdentifiers:
    def test_worker_ids_are_upper_cased(self) -> None:
        assert normalize_worker_code("  w-0042 ") == "W-0042"

    @pytest.mark.parametrize("code", ["W-0042", "AB", "123", "SITE7-CREW2-04", "A" * 32])
    def test_valid_worker_ids(self, code: str) -> None:
        assert is_valid_worker_code(code)

    @pytest.mark.parametrize("code", ["W", "-W42", "W 42", "W_42", "A" * 33, "w-0042", ""])
    def test_invalid_worker_ids(self, code: str) -> None:
        assert not is_valid_worker_code(code)

    def test_emails_are_lower_cased(self) -> None:
        assert normalize_email("  Admin@Example.COM ") == "admin@example.com"

    @pytest.mark.parametrize("email", ["a@b.co", "first.last@site.example.com"])
    def test_valid_emails(self, email: str) -> None:
        assert is_valid_email(email)

    @pytest.mark.parametrize("email", ["admin", "a@b", "a b@c.com", "@b.com", "a@@b.com"])
    def test_invalid_emails(self, email: str) -> None:
        assert not is_valid_email(email)


class TestHashingAndTokens:
    def test_hash_verifies_only_the_right_secret(self) -> None:
        stored = hash_secret("482913")
        assert stored.startswith("$argon2id$")
        assert verify_secret(stored, "482913")
        assert not verify_secret(stored, "482914")

    def test_garbage_hash_is_a_mismatch_not_a_crash(self) -> None:
        assert not verify_secret("not-a-hash", "482913")

    def test_tokens_are_long_random_and_stored_as_sha256(self) -> None:
        token = new_token()
        assert len(token) == 43
        assert new_token() != token
        assert len(token_hash(token)) == 64
        assert token_hash(token) == token_hash(token)
        assert token not in token_hash(token)

    def test_token_comparison(self) -> None:
        assert tokens_match("abc", "abc")
        assert not tokens_match("abc", "abd")
        assert not tokens_match(None, "abc")
        assert not tokens_match("abc", None)
        assert not tokens_match("", "")


class TestFailureLimiter:
    def _limiter(self) -> tuple[FailureLimiter, list[float]]:
        now = [0.0]
        return FailureLimiter(limit=3, window_s=60, clock=lambda: now[0]), now

    def test_blocks_after_the_limit_within_the_window(self) -> None:
        limiter, now = self._limiter()
        for _ in range(3):
            assert limiter.retry_after("1.2.3.4") is None
            limiter.record_failure("1.2.3.4")
        now[0] = 10
        assert limiter.retry_after("1.2.3.4") == pytest.approx(50)

    def test_other_addresses_are_not_affected(self) -> None:
        limiter, _ = self._limiter()
        for _ in range(3):
            limiter.record_failure("1.2.3.4")
        assert limiter.retry_after("5.6.7.8") is None

    def test_old_failures_age_out(self) -> None:
        limiter, now = self._limiter()
        for _ in range(3):
            limiter.record_failure("1.2.3.4")
        now[0] = 61
        assert limiter.retry_after("1.2.3.4") is None


class TestErrors:
    def test_field_is_included_only_when_set(self) -> None:
        with_field = json.loads(ApiError(422, "unknown_job", "x", field="job_category")
                                .to_response().body)
        assert with_field == {"detail": "x", "code": "unknown_job", "field": "job_category"}
        assert "field" not in json.loads(ApiError(404, "not_found", "x").to_response().body)

    def test_auth_errors_are_api_errors(self) -> None:
        assert isinstance(AuthError(401, "not_authenticated", "x"), ApiError)

    def test_body_keeps_detail_a_string_and_adds_code(self) -> None:
        response = AuthError(423, "account_locked", "Locked.", retry_after_s=839.2).to_response()
        body = json.loads(response.body)
        assert body == {"detail": "Locked.", "code": "account_locked", "retry_after_s": 840}
        assert response.headers["Retry-After"] == "840"

    def test_bearer_401_says_how_to_authenticate(self) -> None:
        response = AuthError(401, "not_authenticated", "x", bearer=True).to_response()
        assert response.headers["WWW-Authenticate"] == "Bearer"
        cookie_401 = AuthError(401, "not_authenticated", "x").to_response()
        assert "WWW-Authenticate" not in cookie_401.headers

    @pytest.mark.parametrize(("seconds", "text"), [(40, "40 seconds"), (900, "15 minutes"),
                                                   (839.5, "14 minutes")])
    def test_human_duration(self, seconds: float, text: str) -> None:
        assert human_duration(seconds) == text
