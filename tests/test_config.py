"""app/core/config.py::Settings -- a secret pasted into Render's env var UI
(or a .env file saved with a trailing newline) can silently carry a trailing
"\\n" into the value. That's invisible everywhere except when later sent as a
raw HTTP header, where h11 rejects it outright
(httpx.LocalProtocolError: "Illegal header value ...\\n").

Found live: SUPABASE_SERVICE_ROLE_KEY had a trailing newline on Render,
breaking every real storage upload with exactly that error -- the request
never even reached Supabase.
"""

import os

import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.config import Settings


def test_settings_strips_whitespace_configured():
    """Pin the actual mechanism: str_strip_whitespace must stay enabled on
    Settings, or this whole class of bug comes back silently."""
    assert Settings.model_config.get("str_strip_whitespace") is True


def test_trailing_newline_in_env_var_is_stripped(monkeypatch):
    """Reproduces the real bug with a minimal Settings-shaped model (the real
    Settings requires every env var in .env.example, which is impractical to
    fully stub here) -- proves str_strip_whitespace actually removes a
    trailing newline the way Render's env var UI can introduce one."""

    class Probe(BaseSettings):
        model_config = SettingsConfigDict(str_strip_whitespace=True)
        supabase_service_role_key: str

    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "fake-test-key-not-a-real-secret\n")

    probe = Probe()

    assert probe.supabase_service_role_key == "fake-test-key-not-a-real-secret"
    assert "\n" not in probe.supabase_service_role_key


# --------------------------------------------------------------------------- #
# Storage backend selection. Every Settings field the real app needs is
# supplied, and `_env_file=None` keeps the local .env out of it.
# --------------------------------------------------------------------------- #

_BASE_ENV = {
    "APP_NAME": "t", "ENV": "development", "HOST": "h", "PORT": "1", "CORS_ORIGINS": "x",
    "FRONTEND_URL": "x", "DATABASE_URL": "postgresql://x", "REDIS_URL": "redis://x",
    "CACHE_CLEAR_SECRET": "s", "FIREBASE_CREDENTIALS_PATH": "f", "SMTP_HOST": "h",
    "SMTP_PORT": "1", "SMTP_USER": "u", "SMTP_PASSWORD": "p", "EMAIL_FROM_NAME": "n",
    "EMAIL_FROM_ADDRESS": "a@b.c", "TRUST_PROXY": "false", "RAZORPAY_KEY_ID": "k",
    "RAZORPAY_KEY_SECRET": "s", "RAZORPAY_WEBHOOK_SECRET": "w", "FAL_KEY": "k",
    "PUBLIC_BACKEND_URL": "https://x", "FAL_PER_TEAM_CONCURRENCY_LIMIT": "5",
}
_R2_ENV = {
    "R2_ACCOUNT_ID": "acct", "R2_ACCESS_KEY_ID": "ak", "R2_SECRET_ACCESS_KEY": "sk",
    "R2_BUCKET": "bkt", "R2_PUBLIC_BASE_URL": "https://cdn.example.com",
}
_SUPABASE_ENV = {"SUPABASE_URL": "https://p.supabase.co", "SUPABASE_SERVICE_ROLE_KEY": "key"}
_STORAGE_VARS = (
    "STORAGE_BACKEND", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_PUBLIC_BASE_URL", "R2_KEY_PREFIX",
)


def _settings(monkeypatch, **env):
    for name in _STORAGE_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in {**_BASE_ENV, **env}.items():
        monkeypatch.setenv(name, value)
    return Settings(_env_file=None)


def test_backend_defaults_to_r2_and_needs_r2_vars(monkeypatch):
    with pytest.raises(Exception, match="R2_ACCOUNT_ID is required when STORAGE_BACKEND=r2"):
        _settings(monkeypatch)
    s = _settings(monkeypatch, **_R2_ENV)
    assert s.storage_backend == "r2"
    assert s.supabase_url is None and s.supabase_service_role_key is None


@pytest.mark.parametrize("missing", list(_R2_ENV))
def test_each_missing_r2_var_is_named_in_the_error(monkeypatch, missing):
    env = {k: v for k, v in _R2_ENV.items() if k != missing}
    with pytest.raises(Exception, match=f"{missing} is required"):
        _settings(monkeypatch, STORAGE_BACKEND="r2", **env)


def test_empty_r2_var_counts_as_missing(monkeypatch):
    with pytest.raises(Exception, match="R2_BUCKET is required"):
        _settings(monkeypatch, **{**_R2_ENV, "R2_BUCKET": ""})


def test_supabase_backend_needs_only_supabase_vars(monkeypatch):
    s = _settings(monkeypatch, STORAGE_BACKEND="supabase", **_SUPABASE_ENV)
    assert s.storage_backend == "supabase" and s.r2_bucket is None


@pytest.mark.parametrize("missing", list(_SUPABASE_ENV))
def test_supabase_backend_names_missing_var(monkeypatch, missing):
    env = {k: v for k, v in _SUPABASE_ENV.items() if k != missing}
    with pytest.raises(Exception, match=f"{missing} is required when STORAGE_BACKEND=supabase"):
        _settings(monkeypatch, STORAGE_BACKEND="supabase", **env)


def test_invalid_storage_backend_fails_startup(monkeypatch):
    with pytest.raises(Exception, match="storage_backend"):
        _settings(monkeypatch, STORAGE_BACKEND="s3", **_R2_ENV, **_SUPABASE_ENV)


def test_backend_is_not_inferred_from_environment_name(monkeypatch):
    assert _settings(monkeypatch, ENV="production", **_R2_ENV).storage_backend == "r2"
    s = _settings(monkeypatch, ENV="development", STORAGE_BACKEND="supabase", **_SUPABASE_ENV)
    assert s.storage_backend == "supabase"


@pytest.mark.parametrize("raw,expected", [
    ("", ""), ("/", ""), ("dev", "dev/"), ("/dev", "dev/"), ("dev/", "dev/"),
    ("//dev//", "dev/"), ("a/b", "a/b/"),
])
def test_key_prefix_normalised(monkeypatch, raw, expected):
    assert _settings(monkeypatch, R2_KEY_PREFIX=raw, **_R2_ENV).r2_key_prefix == expected


def test_key_prefix_defaults_to_empty(monkeypatch):
    assert _settings(monkeypatch, **_R2_ENV).r2_key_prefix == ""


def test_public_base_url_trailing_slashes_stripped(monkeypatch):
    s = _settings(monkeypatch, **{**_R2_ENV, "R2_PUBLIC_BASE_URL": "https://cdn.example.com//"})
    assert s.r2_public_base_url == "https://cdn.example.com"
