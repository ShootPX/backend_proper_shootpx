# --------------------------------------------------------------------------- #
# Database isolation. This block MUST run before anything imports `app`:
# app.core.config builds `settings` (and app.core.database its engine) at import.
#
# The suite never reads DATABASE_URL. The process's DATABASE_URL is forcibly
# replaced -- with TEST_DATABASE_URL when that is set and passes db_guard, and
# otherwise with an address that cannot connect. A pydantic-settings env var
# beats the .env file, so the real database in .env is unreachable from any
# test, marked or not: an unmarked test that touches the DB fails loudly rather
# than silently reading or writing real data. Tests that genuinely need
# Postgres carry @pytest.mark.real_db and are skipped unless TEST_DATABASE_URL
# is configured.
# --------------------------------------------------------------------------- #
import os
from pathlib import Path


def _test_setting(name: str) -> str:
    """A TEST_DATABASE_* value from the environment, else from .env. Scans for
    that one key only, so no other line of .env (notably DATABASE_URL) is read
    into anything."""
    if name in os.environ:            # an explicit empty value disables the setting
        return os.environ[name].strip()
    env_file = Path(__file__).resolve().parent.parent / ".env"
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == name:
                return value.strip().strip("'\"")
    except OSError:
        pass
    return ""


from tests.db_guard import UNREACHABLE_URL, problems_with_test_database_url

_TEST_DB_URL = _test_setting("TEST_DATABASE_URL")
_GUARD_ARGS = dict(
    allowed_hosts=_test_setting("TEST_DATABASE_ALLOWED_HOSTS"),
    deny_substrings=_test_setting("TEST_DATABASE_DENY_SUBSTRINGS"),
)
# The dev opt-in only means anything together with a URL.
_ALLOW_DEV = bool(_TEST_DB_URL) and _test_setting("TEST_DATABASE_ALLOW_DEV") == "1"
_TEST_DB_PROBLEMS = (
    problems_with_test_database_url(_TEST_DB_URL, allow_dev=_ALLOW_DEV, **_GUARD_ARGS)
    if _TEST_DB_URL
    else []
)
_USABLE_TEST_DB_URL = _TEST_DB_URL if _TEST_DB_URL and not _TEST_DB_PROBLEMS else ""
# True when the URL is only acceptable BECAUSE of the dev opt-in, i.e. it is a hosted
# project whose schema the tests do not own (see _test_database).
_IS_DEV_OPT_IN = bool(_USABLE_TEST_DB_URL) and bool(
    problems_with_test_database_url(_USABLE_TEST_DB_URL, allow_dev=False, **_GUARD_ARGS)
)
os.environ["DATABASE_URL"] = _USABLE_TEST_DB_URL or UNREACHABLE_URL

from unittest.mock import MagicMock

import pytest

from app.core.limiter import limiter
from app import worker
from app.services import generation as generation_svc
from app.services import tool_definitions as tool_definitions_svc


@pytest.fixture(autouse=True)
def _disable_rate_limiter():
    """Rate limiting uses a shared Redis store; disable it in tests so counters
    from one test (or a previous run) don't make another test flaky."""
    limiter.enabled = False
    yield
    limiter.enabled = True


@pytest.fixture(autouse=True)
def _mock_fal_slot_by_default(monkeypatch):
    """release_fal_slot / try_reserve_fal_slot talk to real Redis. Almost no
    test in this suite is actually exercising the fal-concurrency-limit
    feature itself, so auto-mock every caller's own imported reference to
    them by default -- otherwise any test that reaches a webhook, sweep, or
    submit code path corrupts the real dev Redis inflight counters with
    unmatched increments/decrements. Found live (twice): this exact gap
    drove the global counter to -14.

    A test that wants the real Redis-backed behavior (e.g.
    test_fal_slot_cap_real_redis_global_and_per_team) imports
    try_reserve_fal_slot/release_fal_slot directly from
    app.services.generation_lock and calls them directly, bypassing this.
    A test that wants to assert on a call (e.g. "released exactly once with
    this team_id") sets its own explicit monkeypatch, which simply overrides
    this default for that test."""
    monkeypatch.setattr(generation_svc, "release_fal_slot", MagicMock())
    monkeypatch.setattr(worker, "release_fal_slot", MagicMock())
    monkeypatch.setattr(worker, "try_reserve_fal_slot", MagicMock(return_value=True))


@pytest.fixture(autouse=True)
def _disable_tool_definition_cache(monkeypatch):
    """get_tool_definition() caches ToolDefinition rows in real Redis
    (60s TTL, key "tooldef:<feature_type>"). Almost every test in this suite
    mocks db.query(...) directly for a ToolDefinition lookup and expects that
    mock to be hit every time -- a real cache hit would skip the DB entirely
    and return whatever a PREVIOUS test (or real dev traffic) last cached
    under the same feature_type, and a real cache set would try to
    json.dumps() a MagicMock's attributes into actual dev Redis. Same class
    of bug already hit once with the fal-slot counters above -- force every
    lookup here to behave as a permanent cache miss with no writes."""
    monkeypatch.setattr(tool_definitions_svc, "get_cached", lambda key: None)
    monkeypatch.setattr(tool_definitions_svc, "set_cached", lambda key, value, ttl=None: None)


class _BlockedSMTP:
    """Stand-in for smtplib.SMTP that refuses to connect and records the attempt."""
    attempts: list = []

    def __init__(self, host=None, port=None, *a, **k):
        _BlockedSMTP.attempts.append((host, port))
        raise ConnectionRefusedError(
            "the test suite attempted to open a REAL SMTP connection -- stub "
            "send_email (or the function that calls it) in this test"
        )


@pytest.fixture(autouse=True)
def _never_send_real_email(monkeypatch):
    """Hard guarantee that running the tests can never email a real person.

    Several tests run against the real dev database, whose users are real
    people with real inboxes, and the app sends genuine email through the
    configured SMTP account. One such test (a real-DB team delete/restore
    check) did exactly that -- every suite run emailed real users a
    "team restored" message. Stubbing send_email in each test is what should
    happen, but "every test remembers" is precisely how that leaked, so
    this enforces it: any attempt to open an SMTP connection is refused and
    then FAILS the test at teardown, even if the calling code swallows the
    error (notify_team_restored and friends deliberately do), pointing at
    the test that needs a stub.
    """
    from app.core import email as email_module

    _BlockedSMTP.attempts = []
    monkeypatch.setattr(email_module.smtplib, "SMTP", _BlockedSMTP)
    yield
    attempts, _BlockedSMTP.attempts = _BlockedSMTP.attempts, []
    assert not attempts, (
        f"this test tried to send real email ({len(attempts)} SMTP connection attempt(s) "
        f"to {attempts[0][0]}:{attempts[0][1]}) -- stub send_email in it"
    )


# --------------------------------------------------------------------------- #
# real_db tests
# --------------------------------------------------------------------------- #

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_db: needs a real Postgres (INSERTs/DELETEs rows). Runs only against "
        "TEST_DATABASE_URL, which must pass tests/db_guard.py; skipped when unset.",
    )
    if _TEST_DB_PROBLEMS:
        # Set but unsafe: refuse to run ANY test, not just the real_db ones -- a
        # misconfigured URL should be fixed, not silently tolerated.
        pytest.exit(
            "Refusing to run: TEST_DATABASE_URL is not safe for tests:\n  - "
            + "\n  - ".join(_TEST_DB_PROBLEMS)
            + "\nUse a dedicated, local or explicitly allowed database whose name contains "
            "'test' (see tests/db_guard.py). A hosted dev/test project can opt in with "
            "TEST_DATABASE_ALLOW_DEV=1 -- that never overrides a production-looking name "
            "or TEST_DATABASE_DENY_SUBSTRINGS.",
            returncode=2,
        )


def pytest_report_header(config):
    if not _USABLE_TEST_DB_URL:
        return "real_db tests: SKIPPED (no usable TEST_DATABASE_URL)"
    from sqlalchemy import make_url

    url = make_url(_USABLE_TEST_DB_URL)
    mode = "DEV OPT-IN (schema is NOT created or altered by the tests)" if _IS_DEV_OPT_IN else "local test database"
    lines = [f"real_db tests: {url.host}:{url.port or 5432}/{url.database} -- {mode}"]
    if _IS_DEV_OPT_IN and not _GUARD_ARGS["deny_substrings"]:
        lines.append("  warning: TEST_DATABASE_DENY_SUBSTRINGS is empty -- add your production project ref/host")
    return lines


def pytest_collection_modifyitems(config, items):
    if _USABLE_TEST_DB_URL:
        return
    skip = pytest.mark.skip(reason="real_db test: set TEST_DATABASE_URL to a dedicated test database")
    for item in items:
        if item.get_closest_marker("real_db"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def _test_database():
    """Create the schema in the (guard-approved) test database and make sure two
    users exist -- several real_db tests borrow existing users for FKs instead of
    inserting their own. Idempotent, and only ever run against TEST_DATABASE_URL."""
    from sqlalchemy import make_url

    from app.core.database import Base, SessionLocal, engine
    import app.models  # noqa: F401 -- registers every table on Base.metadata
    from app.models.user import User

    target = make_url(_USABLE_TEST_DB_URL)
    if (engine.url.host, engine.url.port, engine.url.database) != (target.host, target.port, target.database):
        pytest.exit("engine is not pointed at TEST_DATABASE_URL -- refusing to touch it", returncode=2)

    if _IS_DEV_OPT_IN:
        # A hosted dev project's schema belongs to its migrations. create_all would
        # quietly create any table the models have that the migrations have not
        # been applied for yet -- exactly what "write the migration, don't run it"
        # rules out -- and seeded users would land in a shared database.
        return

    Base.metadata.create_all(engine)
    db = SessionLocal()
    try:
        for n in range(db.query(User).count(), 2):
            db.add(User(firebase_uid=f"test-seed-{n}", email=f"test-seed-{n}@example.invalid", name=f"Test seed {n}"))
        db.commit()
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _real_db_ready(request):
    if request.node.get_closest_marker("real_db"):
        request.getfixturevalue("_test_database")
    yield
