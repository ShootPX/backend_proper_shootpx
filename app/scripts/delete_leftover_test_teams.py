"""Deletes the four leftover test teams that an interrupted run of
test_only_an_owner_can_delete_or_restore_a_team_real_db left in the dev database
on 2026-09-29 (all named "Owner-only delete test", two members each, nothing else).

Dry run by default: it only READS and prints what it would delete. Nothing is
deleted unless BOTH flags are given:

    python -m app.scripts.delete_leftover_test_teams                 # dry run
    python -m app.scripts.delete_leftover_test_teams --execute --confirm-host <db host>

`--confirm-host` must equal the host of the database the script connected to (it
is printed by the dry run), so a stale .env can never make this run against a
database you did not mean. The connection comes from the app's normal settings
(.env / environment); no credentials are in this file.

The migrations must be applied first: purge_team also clears credit_ledger, so on a
database without that table the script stops with exit code 3 before doing anything.

Deletion is all-or-nothing and checked first. Each team must:
  * be one of the four ids below (nothing else can be deleted with this script),
  * still be named exactly "Owner-only delete test",
  * have at most 2 members and NO generation jobs, subscriptions, transactions,
    invites or ledger entries -- i.e. look exactly like what the test leaves.
If any team fails a check the script deletes nothing and exits 1. A team that is
already gone is fine, so re-running after a success is harmless. It removes only
these teams and their membership rows: users are never touched.
"""

import argparse
import sys
import uuid

from sqlalchemy import func, inspect

import app.models  # noqa: F401 -- registers every table before any query
from app.core.database import SessionLocal
from app.models.billing_transaction import BillingTransaction
from app.models.credit_ledger import CreditLedger
from app.models.generation_job import GenerationJob
from app.models.team import Team
from app.models.team_invite import TeamInvite
from app.models.team_member import TeamMember
from app.models.team_subscription import TeamSubscription
from app.services.teams import purge_team

LEFTOVER_TEAM_IDS = (
    uuid.UUID("b8064ca8-de73-497a-affd-214c0990469f"),
    uuid.UUID("6369799a-1a1e-4d03-a3e9-2dc1a3728c08"),
    uuid.UUID("202763be-d1cb-49d0-af26-8f474a77629e"),
    uuid.UUID("b76827b6-0e91-403c-8cff-9ccb68476050"),
)
EXPECTED_NAME = "Owner-only delete test"
MAX_MEMBERS = 2

# Tables that must hold NOTHING for a team before it may be deleted.
_MUST_BE_EMPTY = (
    ("generation_jobs", GenerationJob),
    ("team_subscriptions", TeamSubscription),
    ("billing_transactions", BillingTransaction),
    ("team_invites", TeamInvite),
    ("credit_ledger", CreditLedger),
)


def _count(db, model, team_id) -> int:
    return db.query(func.count()).select_from(model).filter(model.team_id == team_id).scalar() or 0


def inspect_teams(db, team_ids=LEFTOVER_TEAM_IDS) -> list[dict]:
    """One report per id: whether it exists and, if it does, every reason it may
    NOT be deleted. Read-only."""
    reports = []
    for team_id in team_ids:
        team = db.query(Team).filter(Team.id == team_id).first()
        if team is None:
            reports.append({"id": team_id, "exists": False, "problems": []})
            continue
        problems = []
        if team.name != EXPECTED_NAME:
            problems.append(f"name is {team.name!r}, expected {EXPECTED_NAME!r}")
        members = _count(db, TeamMember, team_id)
        if members > MAX_MEMBERS:
            problems.append(f"has {members} members (a test team has at most {MAX_MEMBERS})")
        for label, model in _MUST_BE_EMPTY:
            n = _count(db, model, team_id)
            if n:
                problems.append(f"has {n} {label} row(s)")
        reports.append({
            "id": team_id, "exists": True, "name": team.name, "members": members,
            "created_at": team.created_at, "problems": problems,
        })
    return reports


def delete_teams(db, reports: list[dict]) -> list:
    """Purge every existing team in `reports`. The caller has already checked there
    are no problems. Returns the ids deleted."""
    deleted = []
    for report in reports:
        if not report["exists"]:
            continue
        purge_team(db, report["id"])       # its own commit; removes members, never users
        deleted.append(report["id"])
    return deleted


def _describe(db) -> tuple[str, str]:
    """(host, "host:port/database") of the database this session is connected to."""
    url = db.get_bind().url
    host = url.host or ""
    return host, f"{host}:{url.port or 5432}/{url.database}"


def main(argv=None, session_factory=SessionLocal, out=print) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--execute", action="store_true", help="actually delete (default: dry run)")
    parser.add_argument("--confirm-host", default="", help="host of the database you intend to delete from")
    args = parser.parse_args(argv)

    db = session_factory()
    try:
        host, label = _describe(db)
        out(f"database: {label}")

        if not inspect(db.get_bind()).has_table("credit_ledger"):
            out("stopping: this database has no credit_ledger table. Apply "
                "migrations/2026-09-29_credit_ledger_and_reconciliation.sql first (purge_team clears it).")
            return 3

        reports = inspect_teams(db)
        for r in reports:
            if not r["exists"]:
                out(f"  {r['id']}  already gone")
            else:
                out(f"  {r['id']}  {r['name']!r}  members={r['members']}  created={r['created_at']}"
                    + (f"  REFUSED: {'; '.join(r['problems'])}" if r["problems"] else ""))

        if any(r["problems"] for r in reports):
            out("nothing deleted: at least one team no longer looks like a leftover test team")
            return 1

        present = [r for r in reports if r["exists"]]
        if not args.execute:
            out(f"dry run: would delete {len(present)} team(s) and their membership rows. "
                f"Re-run with --execute --confirm-host {host} to do it.")
            return 0

        if not host or args.confirm_host != host:
            out(f"refusing to delete: --confirm-host must equal the database host ({host!r}), "
                f"got {args.confirm_host!r}")
            return 2

        deleted = delete_teams(db, reports)
        remaining = [r["id"] for r in inspect_teams(db) if r["exists"]]
        out(f"deleted {len(deleted)} team(s); {len(remaining)} of the listed ids still exist")
        return 0 if not remaining else 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
