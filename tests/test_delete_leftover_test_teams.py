"""app/scripts/delete_leftover_test_teams.py, against an in-memory SQLite copy of
just the tables purge_team touches. The script itself is never pointed at a real
database by this file."""

import uuid

import pytest

from app.scripts import delete_leftover_test_teams as script


@pytest.fixture
def db_factory():
    from sqlalchemy import create_engine
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.core.database import Base
    from app.models.billing_transaction import BillingTransaction
    from app.models.credit_ledger import CreditLedger
    from app.models.generation_job import GenerationJob
    from app.models.team import Team
    from app.models.team_invite import TeamInvite
    from app.models.team_member import TeamMember
    from app.models.team_subscription import TeamSubscription
    from app.models.user import User

    @compiles(JSONB, "sqlite")
    def _jsonb(type_, compiler, **kw):
        return "JSON"

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine, tables=[
        User.__table__, Team.__table__, TeamMember.__table__, TeamInvite.__table__,
        TeamSubscription.__table__, BillingTransaction.__table__, CreditLedger.__table__,
        GenerationJob.__table__,
    ])
    factory = sessionmaker(bind=engine)
    yield factory
    engine.dispose()


def _seed(factory, ids=script.LEFTOVER_TEAM_IDS):
    from app.models.team import Team
    from app.models.team_member import TeamMember
    from app.models.user import User

    db = factory()
    users = [User(id=uuid.uuid4(), firebase_uid=f"u{i}", email=f"u{i}@example.invalid") for i in range(2)]
    db.add_all(users)
    for team_id in ids:
        db.add(Team(id=team_id, name=script.EXPECTED_NAME))
        db.flush()
        db.add_all([TeamMember(team_id=team_id, user_id=users[0].id, role="owner"),
                    TeamMember(team_id=team_id, user_id=users[1].id, role="editor")])
    # bystanders that must survive: a real-looking team, and a same-named team with another id
    real = Team(id=uuid.uuid4(), name="Acme Studio")
    lookalike = Team(id=uuid.uuid4(), name=script.EXPECTED_NAME)
    db.add_all([real, lookalike])
    db.flush()
    db.add(TeamMember(team_id=real.id, user_id=users[0].id, role="owner"))
    db.commit()
    survivors = {real.id, lookalike.id}
    db.close()
    return survivors


HOST = "db.dev.example.com"


@pytest.fixture(autouse=True)
def _pretend_the_database_has_a_host(monkeypatch):
    """SQLite has no host; the script's --confirm-host check needs one."""
    monkeypatch.setattr(script, "_describe", lambda db: (HOST, f"{HOST}:5432/postgres"))


def _run(factory, *argv):
    lines = []
    code = script.main(list(argv), session_factory=factory, out=lines.append)
    return code, "\n".join(lines)


def _team_ids(factory):
    from app.models.team import Team
    db = factory()
    try:
        return {t.id for t in db.query(Team).all()}
    finally:
        db.close()


def _counts(factory):
    from app.models.team_member import TeamMember
    from app.models.user import User
    db = factory()
    try:
        return db.query(User).count(), db.query(TeamMember).count()
    finally:
        db.close()


def test_the_four_ids_are_the_ones_the_select_reported():
    assert {str(i) for i in script.LEFTOVER_TEAM_IDS} == {
        "b8064ca8-de73-497a-affd-214c0990469f", "6369799a-1a1e-4d03-a3e9-2dc1a3728c08",
        "202763be-d1cb-49d0-af26-8f474a77629e", "b76827b6-0e91-403c-8cff-9ccb68476050",
    }


def test_a_dry_run_reads_and_deletes_nothing(db_factory):
    _seed(db_factory)
    before = (_team_ids(db_factory), _counts(db_factory))

    code, out = _run(db_factory)

    assert code == 0
    assert "dry run: would delete 4 team(s)" in out
    assert "--execute --confirm-host" in out
    assert (_team_ids(db_factory), _counts(db_factory)) == before


def test_execute_without_the_host_confirmation_is_refused(db_factory):
    _seed(db_factory)
    before = _team_ids(db_factory)

    code, out = _run(db_factory, "--execute")

    assert code == 2 and "refusing to delete" in out
    assert _team_ids(db_factory) == before


def test_execute_with_the_wrong_host_is_refused(db_factory):
    _seed(db_factory)
    before = _team_ids(db_factory)

    code, out = _run(db_factory, "--execute", "--confirm-host", "some-other-host.example.com")

    assert code == 2
    assert _team_ids(db_factory) == before


def _host(factory):
    return HOST


def test_execute_with_the_right_host_deletes_exactly_the_four_and_only_their_memberships(db_factory):
    survivors = _seed(db_factory)
    users_before, members_before = _counts(db_factory)

    code, out = _run(db_factory, "--execute", "--confirm-host", _host(db_factory))

    assert code == 0 and "deleted 4 team(s); 0 of the listed ids still exist" in out
    assert _team_ids(db_factory) == survivors               # bystanders (incl. the same-named one) untouched
    users_after, members_after = _counts(db_factory)
    assert users_after == users_before                      # users are never deleted
    assert members_before - members_after == 8              # 4 teams x 2 memberships


def test_running_it_again_after_success_is_harmless(db_factory):
    survivors = _seed(db_factory)
    _run(db_factory, "--execute", "--confirm-host", _host(db_factory))

    code, out = _run(db_factory, "--execute", "--confirm-host", _host(db_factory))

    assert code == 0 and "already gone" in out and "deleted 0 team(s)" in out
    assert _team_ids(db_factory) == survivors


def test_a_partly_deleted_set_is_finished(db_factory):
    from app.services.teams import purge_team

    survivors = _seed(db_factory)
    db = db_factory()
    purge_team(db, script.LEFTOVER_TEAM_IDS[0])
    db.close()

    code, out = _run(db_factory, "--execute", "--confirm-host", _host(db_factory))

    assert code == 0 and "deleted 3 team(s)" in out
    assert _team_ids(db_factory) == survivors


@pytest.mark.parametrize("tamper,expected", [
    ("rename", "name is"),
    ("members", "has 3 members"),
    ("job", "generation_jobs"),
    ("transaction", "billing_transactions"),
    ("subscription", "team_subscriptions"),
    ("invite", "team_invites"),
    ("ledger", "credit_ledger"),
])
def test_a_team_that_no_longer_looks_like_a_leftover_blocks_everything(db_factory, tamper, expected):
    from app.models.billing_transaction import BillingTransaction
    from app.models.credit_ledger import CreditLedger
    from app.models.generation_job import GenerationJob
    from app.models.team import Team
    from app.models.team_invite import TeamInvite
    from app.models.team_member import TeamMember
    from app.models.team_subscription import TeamSubscription
    from app.models.user import User

    _seed(db_factory)
    victim = script.LEFTOVER_TEAM_IDS[1]
    db = db_factory()
    user = db.query(User).first()
    if tamper == "rename":
        db.query(Team).filter(Team.id == victim).update({"name": "Acme Studio"})
    elif tamper == "members":
        extra = User(id=uuid.uuid4(), firebase_uid="extra", email="extra@example.invalid")
        db.add(extra)
        db.flush()
        db.add(TeamMember(team_id=victim, user_id=extra.id, role="editor"))
    elif tamper == "job":
        db.add(GenerationJob(team_id=victim, user_id=user.id, feature_type="recolor", status="completed",
                             input_params={}, credits_charged=1))
    elif tamper == "transaction":
        db.add(BillingTransaction(team_id=victim, type="credit_pack", razorpay_payment_id="pay_x",
                                  amount=1, credits_added=1))
    elif tamper == "subscription":
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        db.add(TeamSubscription(team_id=victim, subscription_id=uuid.uuid4(), credits_per_refill=1,
                                next_refill_at=now, current_period_end=now))
    elif tamper == "invite":
        from datetime import datetime, timedelta, timezone
        db.add(TeamInvite(team_id=victim, email="x@example.invalid", role="editor",
                          token=str(uuid.uuid4()), invited_by=user.id,
                          expires_at=datetime.now(timezone.utc) + timedelta(days=1)))
    elif tamper == "ledger":
        db.add(CreditLedger(team_id=victim, pool="topup", entry_type="topup_purchase", amount=1,
                            idempotency_key="k", source="webhook", entry_metadata={}))
    db.commit()
    db.close()
    before = _team_ids(db_factory)

    code, out = _run(db_factory, "--execute", "--confirm-host", _host(db_factory))

    assert code == 1
    assert "REFUSED" in out and expected in out and "nothing deleted" in out
    assert _team_ids(db_factory) == before                  # not even the three clean ones


def test_it_only_ever_targets_the_four_ids(db_factory):
    """Other teams with the very same name are never candidates."""
    survivors = _seed(db_factory)
    _run(db_factory, "--execute", "--confirm-host", _host(db_factory))
    assert survivors <= _team_ids(db_factory)


def test_an_empty_confirm_host_never_passes_even_against_a_hostless_database(db_factory, monkeypatch):
    monkeypatch.setattr(script, "_describe", lambda db: ("", ":5432/x"))    # e.g. a unix-socket connection
    _seed(db_factory)
    before = _team_ids(db_factory)

    code, out = _run(db_factory, "--execute")

    assert code == 2 and "refusing to delete" in out
    assert _team_ids(db_factory) == before


def test_a_database_without_the_ledger_table_stops_cleanly_before_touching_anything(db_factory):
    from app.models.credit_ledger import CreditLedger

    _seed(db_factory)
    before = _team_ids(db_factory)
    CreditLedger.__table__.drop(db_factory.kw["bind"])         # a database the migration has not reached

    for argv in ([], ["--execute", "--confirm-host", HOST]):
        code, out = _run(db_factory, *argv)
        assert code == 3 and "credit_ledger" in out and "migrations/" in out
    assert _team_ids(db_factory) == before
