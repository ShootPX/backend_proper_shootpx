"""Decides whether TEST_DATABASE_URL points somewhere it is safe to run the
real-database tests (which INSERT and DELETE rows).

The check FAILS CLOSED and looks only at the test URL itself -- it never reads
DATABASE_URL (see conftest.py). A URL is accepted only if ALL of these hold:

  1. it is a postgres URL with a host and a database name;
  2. nothing that identifies the server (host, user, database name) contains a
     production-looking word: prod / production / prd / live / primary;
  3. the host is local (localhost, 127.0.0.1, ::1, *.localhost, *.local,
     host.docker.internal) OR is listed, exactly, in TEST_DATABASE_ALLOWED_HOSTS
     (comma-separated) -- a remote host is never trusted by default;
  4. the database name contains the word "test" (shootpx_test, test, app-test...).

Rule 4 matters for hosted Postgres. Supabase's pooler host is shared by every
project and its default database is literally named "postgres", so neither the
host nor the default database name can say "this is not production". Requiring a
dedicated database whose own name says "test" can.

Dev opt-in (TEST_DATABASE_ALLOW_DEV=1, together with TEST_DATABASE_URL): relaxes rules
3 and 4 ONLY, for a hosted dev/test project that cannot satisfy them (shared pooler
host, database named "postgres"). Rules 1 and 2 and the deny-list below still apply
unchanged, so a production-looking name is refused with or without the opt-in.

Extra deny-list: TEST_DATABASE_DENY_SUBSTRINGS (comma-separated, case-insensitive)
rejects any URL containing one of the given substrings -- put your production
project ref or host in there so a copy-paste mix-up is caught by name.
"""

import re
from urllib.parse import unquote, urlsplit

# Never touched: a URL that fails to connect fast, so a test that reaches the
# database without being marked real_db fails loudly instead of reaching anything.
UNREACHABLE_URL = "postgresql://tests_have_no_database:x@127.0.0.1:1/tests_have_no_database"

_PRODUCTION_WORDS = re.compile(r"(?<![a-z0-9])(prod|production|prd|live|primary)(?![a-z0-9])", re.IGNORECASE)
_TEST_WORD = re.compile(r"(?<![a-z0-9])test(s|ing)?(?![a-z0-9])", re.IGNORECASE)
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}


def _is_local(host: str) -> bool:
    host = host.lower()
    return host in _LOCAL_HOSTS or host.endswith(".localhost") or host.endswith(".local")


def problems_with_test_database_url(
    url: str,
    allowed_hosts: str = "",
    deny_substrings: str = "",
    allow_dev: bool = False,
) -> list[str]:
    """Reasons `url` must not be used for real-DB tests. Empty list = acceptable."""
    problems: list[str] = []
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        user = unquote(parts.username or "")
        database = unquote(parts.path.lstrip("/"))
        scheme = parts.scheme.lower()
    except ValueError:
        return ["it is not a parseable URL"]

    if not scheme.startswith("postgres"):
        problems.append(f"scheme must be postgres/postgresql, got {scheme!r}")
    if not host:
        problems.append("it has no host")
    if not database:
        problems.append("it names no database")
    if problems:
        return problems

    for label, value in (("host", host), ("user", user), ("database name", database)):
        if _PRODUCTION_WORDS.search(value):
            problems.append(f"the {label} {value!r} looks like production")

    if not allow_dev:
        allowed = {h.strip().lower() for h in allowed_hosts.split(",") if h.strip()}
        if not _is_local(host) and host not in allowed:
            problems.append(
                f"host {host!r} is remote and not listed in TEST_DATABASE_ALLOWED_HOSTS "
                "(a hosted dev/test project can opt in with TEST_DATABASE_ALLOW_DEV=1)"
            )

        if not _TEST_WORD.search(database):
            problems.append(f"the database name {database!r} does not contain the word 'test'")

    lowered = url.lower()
    for needle in (s.strip().lower() for s in deny_substrings.split(",")):
        if needle and needle in lowered:
            problems.append("it contains an entry from TEST_DATABASE_DENY_SUBSTRINGS")
            break

    return problems
