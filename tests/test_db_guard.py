"""The safety net around the real-database tests (tests/db_guard.py + conftest.py)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.config import settings
from tests.db_guard import UNREACHABLE_URL, problems_with_test_database_url

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("url", [
    "postgresql://postgres:pw@localhost:5432/shootpx_test",
    "postgresql://postgres:pw@127.0.0.1/test",
    "postgresql+psycopg2://u:p@[::1]:5433/app-test",
    "postgresql://u:p@host.docker.internal/shootpx_tests",
    "postgresql://u:p@devbox.local/ci_testing",
])
def test_local_test_databases_are_accepted(url):
    assert problems_with_test_database_url(url) == []


@pytest.mark.parametrize("url,why", [
    # production-looking names anywhere in host / user / database
    ("postgresql://u:p@localhost/shootpx_prod_test", "looks like production"),
    ("postgresql://u:p@prod-db.internal.local/shootpx_test", "looks like production"),
    ("postgresql://production:p@localhost/shootpx_test", "looks like production"),
    ("postgresql://u:p@localhost/live_test", "looks like production"),
    # the actual shape of the .env database: shared Supabase pooler host, default db name
    ("postgresql://postgres.abcdefgh:p@aws-1-ap-northeast-1.pooler.supabase.com:5432/postgres", "remote"),
    # a remote host is never trusted by default, even with "test" in the name
    ("postgresql://u:p@db.example.com/shootpx_test", "remote"),
    # local but not obviously a test database
    ("postgresql://u:p@localhost/postgres", "does not contain the word 'test'"),
    ("postgresql://u:p@localhost/shootpx", "does not contain the word 'test'"),
    ("postgresql://u:p@localhost/contest", "does not contain the word 'test'"),   # substring is not a word
    # malformed
    ("mysql://u:p@localhost/shootpx_test", "scheme"),
    ("postgresql://u:p@/shootpx_test", "no host"),
    ("postgresql://u:p@localhost", "no database"),
])
def test_unsafe_urls_are_rejected(url, why):
    problems = problems_with_test_database_url(url)
    assert problems, f"{url} should have been refused"
    assert any(why in p for p in problems), problems


def test_a_remote_host_needs_an_exact_allow_list_entry():
    url = "postgresql://u:p@test-db.example.com/shootpx_test"
    assert problems_with_test_database_url(url)
    assert problems_with_test_database_url(url, allowed_hosts="other.example.com")
    assert problems_with_test_database_url(url, allowed_hosts="test-db.example.com") == []
    assert problems_with_test_database_url(url, allowed_hosts="A.example.com, TEST-DB.example.com") == []


def test_allow_listing_a_host_does_not_bypass_the_production_words_or_the_test_name():
    host = "db.example.com"
    assert problems_with_test_database_url(f"postgresql://u:p@{host}/shootpx_prod_test", allowed_hosts=host)
    assert problems_with_test_database_url(f"postgresql://u:p@{host}/postgres", allowed_hosts=host)


def test_the_deny_list_catches_a_production_project_ref():
    url = "postgresql://postgres.abcdefgh:p@localhost/shootpx_test"
    assert problems_with_test_database_url(url) == []
    assert problems_with_test_database_url(url, deny_substrings="zzz, ABCDEFGH")


def test_the_process_never_holds_a_real_database_url():
    """conftest replaced DATABASE_URL before `app` was imported. Whatever the app
    is configured with is either the unreachable placeholder or a URL the guard
    approved -- so the .env database (a remote host whose db is named "postgres")
    can never be it."""
    in_use = settings.database_url
    assert in_use == os.environ["DATABASE_URL"]
    assert in_use == UNREACHABLE_URL or problems_with_test_database_url(
        in_use,
        allowed_hosts=os.environ.get("TEST_DATABASE_ALLOWED_HOSTS", ""),
    ) == []


def _run_pytest_collection(extra_env):
    env = dict(os.environ)
    env.update({"TEST_DATABASE_URL": "", "TEST_DATABASE_ALLOWED_HOSTS": "", "TEST_DATABASE_DENY_SUBSTRINGS": ""})
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_db_guard.py", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )


def test_pytest_refuses_to_run_at_all_when_test_database_url_looks_like_production():
    res = _run_pytest_collection({
        "TEST_DATABASE_URL": "postgresql://postgres.abcdefgh:p@aws-1-ap-northeast-1.pooler.supabase.com:5432/postgres",
    })

    assert res.returncode == 2, res.stdout + res.stderr
    assert "Refusing to run" in res.stdout + res.stderr


def test_pytest_starts_normally_with_a_safe_test_database_url():
    res = _run_pytest_collection({"TEST_DATABASE_URL": "postgresql://u:p@localhost:5432/shootpx_test"})

    assert res.returncode == 0, res.stdout + res.stderr


def test_real_db_tests_are_skipped_when_no_test_database_is_configured():
    env = dict(os.environ)
    env.update({"TEST_DATABASE_URL": "", "TEST_DATABASE_ALLOWED_HOSTS": "", "TEST_DATABASE_DENY_SUBSTRINGS": ""})
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_billing.py", "-q", "-rs", "-p", "no:cacheprovider"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )

    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert "skipped" in out and "TEST_DATABASE_URL" in out


# --------------------------------------------------------------------------- #
# TEST_DATABASE_ALLOW_DEV -- the hosted dev/test project opt-in
# --------------------------------------------------------------------------- #

SUPABASE_DEV_URL = "postgresql://postgres.devprojref:p@aws-1-ap-northeast-1.pooler.supabase.com:5432/postgres"


def test_the_dev_opt_in_accepts_a_hosted_project_that_fails_the_strict_rules():
    assert problems_with_test_database_url(SUPABASE_DEV_URL)                      # refused as-is
    assert problems_with_test_database_url(SUPABASE_DEV_URL, allow_dev=True) == []


def test_the_dev_opt_in_never_overrides_a_production_looking_name():
    for url in (
        "postgresql://postgres.abc:p@aws-1.pooler.supabase.com:5432/postgres_prod",
        "postgresql://prod:p@aws-1.pooler.supabase.com:5432/postgres",
        "postgresql://postgres.abc:p@prod-db.example.com:5432/postgres",
    ):
        assert any("looks like production" in p for p in problems_with_test_database_url(url, allow_dev=True))


def test_the_deny_list_wins_over_the_dev_opt_in_for_host_and_project_ref():
    assert problems_with_test_database_url(SUPABASE_DEV_URL, allow_dev=True, deny_substrings="devprojref")
    assert problems_with_test_database_url(SUPABASE_DEV_URL, allow_dev=True, deny_substrings="pooler.supabase.com")
    assert problems_with_test_database_url(SUPABASE_DEV_URL, allow_dev=True, deny_substrings="other, DEVPROJREF")
    assert problems_with_test_database_url(SUPABASE_DEV_URL, allow_dev=True, deny_substrings="somethingelse") == []


def _collect(extra_env, quiet=True):
    env = dict(os.environ)
    env.update({
        "TEST_DATABASE_URL": "", "TEST_DATABASE_ALLOWED_HOSTS": "",
        "TEST_DATABASE_DENY_SUBSTRINGS": "", "TEST_DATABASE_ALLOW_DEV": "",
    })
    env.update(extra_env)
    cmd = [sys.executable, "-m", "pytest", "tests/test_db_guard.py", "--collect-only", "-p", "no:cacheprovider"]
    return subprocess.run(cmd + (["-q"] if quiet else []), cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=120)


def test_a_hosted_url_without_the_opt_in_is_refused():
    res = _collect({"TEST_DATABASE_URL": SUPABASE_DEV_URL})
    assert res.returncode == 2
    assert "TEST_DATABASE_ALLOW_DEV=1" in res.stdout + res.stderr      # tells you how to opt in


def test_the_opt_in_without_a_url_does_nothing_and_real_db_tests_stay_skipped():
    res = _collect({"TEST_DATABASE_ALLOW_DEV": "1"}, quiet=False)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "real_db tests: SKIPPED" in res.stdout


def test_url_plus_opt_in_starts_and_says_so_in_the_header():
    res = _collect({"TEST_DATABASE_URL": SUPABASE_DEV_URL, "TEST_DATABASE_ALLOW_DEV": "1"}, quiet=False)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "DEV OPT-IN" in res.stdout
    header = [line for line in res.stdout.splitlines() if "real_db tests:" in line]
    assert header and "postgres.devprojref" not in header[0] and ":p@" not in header[0]   # no credentials echoed


def test_url_plus_opt_in_is_refused_when_the_deny_list_matches_host_or_ref():
    for needle in ("devprojref", "pooler.supabase.com"):
        res = _collect({
            "TEST_DATABASE_URL": SUPABASE_DEV_URL, "TEST_DATABASE_ALLOW_DEV": "1",
            "TEST_DATABASE_DENY_SUBSTRINGS": needle,
        })
        assert res.returncode == 2, (needle, res.stdout + res.stderr)
        assert "TEST_DATABASE_DENY_SUBSTRINGS" in res.stdout + res.stderr


def test_the_opt_in_never_lets_a_production_looking_url_through():
    res = _collect({
        "TEST_DATABASE_URL": "postgresql://postgres.abc:p@aws-1.pooler.supabase.com:5432/postgres_prod",
        "TEST_DATABASE_ALLOW_DEV": "1",
    })
    assert res.returncode == 2
