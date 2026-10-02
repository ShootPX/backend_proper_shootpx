"""Abandoned (never-paid) subscription checkouts.

Found live: closing Razorpay without paying left a `pending` row on the team
(plan + placeholder renewal date), which then (a) showed up in the UI as the
team's plan, (b) blocked every further checkout with "This team already has an
active subscription", and (c) was only cleaned up once a day -- by a cron that
would also have deleted a paying customer whose renewal payment had failed
(status "pending" too, created_at = original signup date).
"""
import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app import worker
from app.services import billing as billing_svc
from app.services import teams as teams_svc
from app.services.billing import is_unpaid_checkout

TEAM_ID = uuid.uuid4()
PLAN_A = uuid.uuid4()
PLAN_B = uuid.uuid4()


def _row(status="pending", credits_per_refill=0, subscription_id=PLAN_A, razorpay_id="sub_old"):
    return SimpleNamespace(
        team_id=TEAM_ID, status=status, credits_per_refill=credits_per_refill,
        subscription_id=subscription_id, razorpay_subscription_id=razorpay_id,
        current_period_end=datetime(2026, 9, 20, tzinfo=timezone.utc), next_refill_at=None,
        renewal_notice_sent_at=None, pending_razorpay_subscription_id=None,
        pending_subscription_id=None, pending_switch_id=None, pending_switch_expires_at=None,
    )


def _plan(plan_id=PLAN_A, razorpay_plan_id="plan_x"):
    return SimpleNamespace(id=plan_id, razorpay_plan_id=razorpay_plan_id, period_label="month", slug="monthly")


class Rzp:
    """Records every call made to Razorpay's subscription API."""

    def __init__(self, fetch_status="created", create_id="sub_new", cancel_error=None, create_error=None):
        self.calls = []
        self.fetch_status, self.create_id = fetch_status, create_id
        self.cancel_error, self.create_error = cancel_error, create_error

    def fetch(self, sid):
        self.calls.append(("fetch", sid))
        return {"id": sid, "status": self.fetch_status}

    def create(self, payload):
        self.calls.append(("create", payload["plan_id"]))
        if self.create_error:
            raise self.create_error
        return {"id": self.create_id}

    def cancel(self, sid):
        self.calls.append(("cancel", sid))
        if self.cancel_error:
            raise self.cancel_error
        return {"id": sid, "status": "cancelled"}

    def names(self):
        return [c[0] for c in self.calls]


def _db(plan, row):
    db = MagicMock()

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        target = plan if name == "Subscription" else row
        q.filter.return_value.first.return_value = target
        q.filter.return_value.with_for_update.return_value.first.return_value = target
        return q

    db.query.side_effect = query
    return db


@pytest.fixture
def rzp(monkeypatch):
    fake = Rzp()
    monkeypatch.setattr(billing_svc.razorpay_client, "subscription", fake)
    return fake


# --------------------------------------------------------------------------- #
# what counts as an unpaid attempt
# --------------------------------------------------------------------------- #

def test_only_a_never_activated_pending_row_is_an_unpaid_checkout():
    assert is_unpaid_checkout(_row("pending", credits_per_refill=0)) is True
    # the OTHER meaning of "pending": a real, paying subscription whose renewal failed
    assert is_unpaid_checkout(_row("pending", credits_per_refill=350)) is False
    assert is_unpaid_checkout(_row("active", credits_per_refill=350)) is False
    assert is_unpaid_checkout(_row("cancelled", credits_per_refill=0)) is False
    assert is_unpaid_checkout(_row("halted", credits_per_refill=350)) is False
    assert is_unpaid_checkout(None) is False


# --------------------------------------------------------------------------- #
# checkout no longer blocked by an abandoned attempt
# --------------------------------------------------------------------------- #

def test_retrying_the_same_plan_reuses_the_payable_razorpay_subscription(rzp):
    """"Retry payment" / a double click: hand back the SAME subscription, do not
    create a second one or cancel the one a checkout window may have open."""
    db = _db(_plan(PLAN_A), _row(subscription_id=PLAN_A, razorpay_id="sub_old"))

    out = billing_svc.create_subscription_checkout(db, TEAM_ID, PLAN_A)

    assert out["razorpay_subscription_id"] == "sub_old"
    assert rzp.names() == ["fetch"]          # no create, no cancel


def test_a_dead_old_attempt_is_replaced(rzp):
    rzp.fetch_status = "cancelled"
    row = _row(subscription_id=PLAN_A, razorpay_id="sub_old")
    db = _db(_plan(PLAN_A), row)

    out = billing_svc.create_subscription_checkout(db, TEAM_ID, PLAN_A)

    assert out["razorpay_subscription_id"] == "sub_new"
    assert row.razorpay_subscription_id == "sub_new"
    assert row.status == "pending" and row.credits_per_refill == 0
    assert ("cancel", "sub_old") in rzp.calls


def test_choosing_a_different_plan_replaces_the_unpaid_attempt(rzp):
    row = _row(subscription_id=PLAN_A, razorpay_id="sub_old")
    db = _db(_plan(PLAN_B), row)

    out = billing_svc.create_subscription_checkout(db, TEAM_ID, PLAN_B)

    assert out["razorpay_subscription_id"] == "sub_new"
    assert row.subscription_id == PLAN_B
    assert rzp.names() == ["create", "cancel"]           # new one made FIRST, old cancelled after
    assert ("cancel", "sub_old") in rzp.calls


def test_the_old_attempt_is_only_cancelled_after_the_new_one_exists(rzp):
    """If creating the replacement fails, the user's previous (still payable)
    attempt must be left alone."""
    rzp.create_error = RuntimeError("razorpay down")
    row = _row(subscription_id=PLAN_A, razorpay_id="sub_old")
    db = _db(_plan(PLAN_B), row)

    with pytest.raises(RuntimeError):
        billing_svc.create_subscription_checkout(db, TEAM_ID, PLAN_B)

    assert "cancel" not in rzp.names()
    db.rollback.assert_called()


def test_failing_to_cancel_the_old_attempt_never_fails_the_new_checkout(rzp):
    rzp.cancel_error = RuntimeError("cannot cancel")
    db = _db(_plan(PLAN_B), _row(subscription_id=PLAN_A))

    out = billing_svc.create_subscription_checkout(db, TEAM_ID, PLAN_B)

    assert out["razorpay_subscription_id"] == "sub_new"


@pytest.mark.parametrize("row", [
    _row("active", credits_per_refill=350),
    # a real customer whose renewal payment failed -- must NOT be treated as abandoned
    _row("pending", credits_per_refill=350),
])
def test_a_real_subscription_still_blocks_checkout(rzp, row):
    db = _db(_plan(PLAN_B), row)

    with pytest.raises(ValueError, match="already has an active subscription"):
        billing_svc.create_subscription_checkout(db, TEAM_ID, PLAN_B)

    assert rzp.calls == []


# --------------------------------------------------------------------------- #
# cancel / switch on an unpaid attempt
# --------------------------------------------------------------------------- #

def test_cancelling_an_unpaid_attempt_works_even_if_razorpay_refuses(rzp):
    """Nothing was ever billed, so a Razorpay error must not leave the user
    stuck with an attempt they are trying to get rid of."""
    rzp.cancel_error = RuntimeError("razorpay says no")
    row = _row()
    db = _db(_plan(), row)

    result = billing_svc.cancel_subscription(db, TEAM_ID)

    assert result.status == "cancelled"
    db.commit.assert_called_once()


def test_cancelling_a_real_subscription_still_requires_razorpay_to_succeed(rzp):
    rzp.cancel_error = RuntimeError("razorpay down")
    row = _row("active", credits_per_refill=350)
    db = _db(_plan(), row)

    with pytest.raises(billing_svc.RazorpayCancelError):
        billing_svc.cancel_subscription(db, TEAM_ID)

    assert row.status == "active"          # never show cancelled while still being billed


def test_switching_away_from_an_unpaid_attempt_is_just_a_fresh_checkout(rzp):
    row = _row(subscription_id=PLAN_A, razorpay_id="sub_old")
    team = SimpleNamespace(subscription_credits_remaining=40, topup_credits_balance=10)
    db = _db(_plan(PLAN_B), row)

    out = billing_svc.switch_subscription(db, TEAM_ID, PLAN_B)

    assert out["razorpay_subscription_id"] == "sub_new"
    assert row.subscription_id == PLAN_B
    # no credits were carried over (an unpaid attempt never granted any)
    assert (team.subscription_credits_remaining, team.topup_credits_balance) == (40, 10)


# --------------------------------------------------------------------------- #
# GET /teams/{id}/billing: what the frontend is told
# --------------------------------------------------------------------------- #

_DEFAULT_PLAN = SimpleNamespace(slug="monthly")


def _billing_db(sub, plan=_DEFAULT_PLAN):
    team = SimpleNamespace(subscription_credits_remaining=0, topup_credits_balance=5)
    db = MagicMock()
    db.query.return_value.outerjoin.return_value.outerjoin.return_value.filter.return_value.first.return_value = (
        team, sub, plan,
    )
    return db


def test_billing_reports_created_and_no_renewal_date_for_an_unpaid_attempt():
    out = teams_svc.get_team_billing(_billing_db(_row("pending", credits_per_refill=0)), TEAM_ID)

    assert out["subscription_status"] == "created"
    assert out["current_period_end"] is None      # the placeholder date is never shown
    assert out["plan"] == "monthly"               # kept so the UI can offer "Retry payment" on the right plan


def test_billing_reports_active_normally():
    out = teams_svc.get_team_billing(_billing_db(_row("active", credits_per_refill=350)), TEAM_ID)

    assert out["subscription_status"] == "active"
    assert out["current_period_end"] == "2026-09-20T00:00:00+00:00"


def test_billing_keeps_pending_for_a_failed_renewal_distinct_from_created():
    """A paying customer in a payment-retry window is NOT 'payment not
    completed' -- their plan is live."""
    out = teams_svc.get_team_billing(_billing_db(_row("pending", credits_per_refill=350)), TEAM_ID)

    assert out["subscription_status"] == "pending"
    assert out["current_period_end"] is not None


def test_billing_with_no_subscription_at_all():
    out = teams_svc.get_team_billing(_billing_db(None, plan=None), TEAM_ID)
    assert out["subscription_status"] is None and out["plan"] is None and out["current_period_end"] is None


# --------------------------------------------------------------------------- #
# the cleanup cron -- against an isolated DB (it deletes rows, and it once
# would have deleted a paying customer's)
# --------------------------------------------------------------------------- #

def test_cleanup_only_removes_stale_never_paid_attempts(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.core.database import Base
    from app.models.team_subscription import TeamSubscription

    @compiles(JSONB, "sqlite")
    def _jsonb(element, compiler, **kw):
        return "JSON"

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine, tables=[TeamSubscription.__table__])
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(worker, "SessionLocal", Session)

    now = datetime.now(timezone.utc)
    db = Session()

    def row(status, credits, age_minutes):
        r = TeamSubscription(
            team_id=uuid.uuid4(), subscription_id=uuid.uuid4(), status=status,
            credits_per_refill=credits, created_at=now - timedelta(minutes=age_minutes),
            next_refill_at=now, current_period_end=now, last_paid_count=0,
        )
        db.add(r)
        db.flush()
        return r.team_id

    stale_unpaid = row("pending", 0, 120)          # abandoned checkout, long ago  -> DELETE
    young_unpaid = row("pending", 0, 5)            # user may still be paying      -> keep
    renewal_failed = row("pending", 350, 60 * 24 * 90)   # PAYING customer, signed up 90 days ago -> KEEP
    active = row("active", 350, 60 * 24 * 90)      # -> keep
    cancelled = row("cancelled", 0, 120)           # -> keep
    db.commit()

    asyncio.run(worker.cleanup_stale_pending_subscriptions({}))

    db.expire_all()
    remaining = {r.team_id for r in db.query(TeamSubscription).all()}
    assert stale_unpaid not in remaining
    assert remaining == {young_unpaid, renewal_failed, active, cancelled}
    db.close()


def test_cleanup_runs_every_fifteen_minutes_not_once_a_day():
    """It used to run only at 04:00, so an abandoned attempt could linger ~24h."""
    job = next(j for j in worker.WorkerSettings.cron_jobs if j.coroutine is worker.cleanup_stale_pending_subscriptions)
    assert job.hour is None                     # not pinned to one hour of the day
    assert job.minute == {7, 22, 37, 52}
