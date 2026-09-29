"""services/reconciliation.py -- the hourly comparison of subscriptions with
Razorpay, and the `reconcile_subscriptions` cron that runs it.

The repair tests use the filter-aware fakes (tests/billing_fakes.py) and drive the
REAL webhook code, because the guarantee that matters is that a repair and the
original webhook can never both apply. Candidate selection is plain SQL, so it is
tested against an in-memory SQLite table.
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from app.services import reconciliation as recon
from app.services import webhooks as webhooks_svc
from tests.billing_fakes import (
    FakeRazorpay, ledger_entries, make_db, make_plan, make_team, make_team_sub,
)

TEAM_ID = uuid.uuid4()
SUB_ID = "sub_live"
NOW = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)


def ts(dt):
    return int(dt.timestamp())


class World:
    """A team on an active monthly plan, three cycles paid, renewal overdue."""

    def __init__(self, monkeypatch):
        self.plan = make_plan("month", slug="monthly", credits=350, price=99900)
        self.team = make_team(TEAM_ID, subscription_credits=40, topup=10)
        self.row = make_team_sub(
            team_id=TEAM_ID, subscription_id=self.plan.id, razorpay_subscription_id=SUB_ID,
            credits_per_refill=350, last_paid_count=3,
            current_period_end=NOW - timedelta(hours=3), next_refill_at=NOW - timedelta(hours=3),
        )
        self.db = make_db(team_subs=[self.row], plans=[self.plan], teams=[self.team])
        self.rzp = FakeRazorpay().install(monkeypatch, webhooks_svc)
        self.rzp.statuses[SUB_ID] = "active"
        self.new_end = NOW + timedelta(days=30)
        self.rzp.entities[SUB_ID] = {"paid_count": 4, "current_end": ts(self.new_end)}
        self.alerts = []
        monkeypatch.setattr(recon, "send_alert", lambda msg: self.alerts.append(msg))
        monkeypatch.setattr(webhooks_svc, "send_renewal_notice_email", lambda *a, **k: None)

    def reconcile(self):
        return recon.reconcile_subscription(self.db, self.row.id, now=NOW)

    def charged_webhook(self):
        webhooks_svc.handle_subscription_charged(
            self.db, {"payload": {"subscription": {"entity": {"id": SUB_ID}}}},
        )


@pytest.fixture
def w(monkeypatch):
    return World(monkeypatch)


# --------------------------------------------------------------------------- #
# a missed `charged` webhook
# --------------------------------------------------------------------------- #

def test_a_missed_renewal_is_granted_recorded_and_the_period_advanced(w):
    result = w.reconcile()

    assert result["outcome"] == "repaired"
    assert result["repairs"] == ["missed_charge:paid_count 3->4"]
    assert w.team.subscription_credits_remaining == 350          # refill replaces the pool
    assert w.row.last_paid_count == 4
    assert w.row.current_period_end == w.new_end                 # Razorpay's own cycle end
    assert w.row.next_refill_at > NOW - timedelta(hours=1)
    assert w.row.last_reconciled_at == NOW
    w.db.commit.assert_called()

    (entry,) = ledger_entries(w.db)
    assert entry.idempotency_key == f"sub:{SUB_ID}:cycle:4"
    assert entry.source == "reconcile"
    assert (entry.pool, entry.entry_type) == ("subscription", "subscription_grant")
    assert entry.amount == 350 - 40 and entry.balance_after == 350
    assert entry.entry_metadata == {"plan": "monthly", "period": "month", "cycle": 4, "price": 99900}


def test_a_repair_is_logged_at_error_level_and_alerted(w, caplog):
    import logging
    with caplog.at_level(logging.ERROR, logger="app.services.reconciliation"):
        w.reconcile()
    rec = next(r for r in caplog.records if "reconciliation repaired" in r.getMessage())
    assert rec.levelno == logging.ERROR
    assert f"team_id={TEAM_ID}" in rec.getMessage() and SUB_ID in rec.getMessage()


def test_running_twice_repairs_once(w):
    w.reconcile()
    w.team.subscription_credits_remaining = 120           # spent some since the repair
    ledger_before = len(ledger_entries(w.db))

    second = w.reconcile()

    assert second == {"outcome": "ok", "repairs": []}
    assert w.team.subscription_credits_remaining == 120   # NOT refilled again
    assert len(ledger_entries(w.db)) == ledger_before


def test_the_late_webhook_after_a_repair_is_a_no_op(w):
    w.reconcile()
    w.team.subscription_credits_remaining = 120

    w.charged_webhook()                                    # the webhook finally arrives

    assert w.team.subscription_credits_remaining == 120
    assert len(ledger_entries(w.db)) == 1


def test_the_webhook_then_reconcile_grants_only_once(w):
    w.charged_webhook()                                    # the webhook did arrive
    w.team.subscription_credits_remaining = 120

    result = w.reconcile()

    assert result["repairs"] == []
    assert w.team.subscription_credits_remaining == 120
    assert len(ledger_entries(w.db)) == 1
    assert ledger_entries(w.db)[0].source == "webhook"


def test_several_missed_cycles_produce_one_grant_for_the_latest(w):
    w.rzp.entities[SUB_ID]["paid_count"] = 6

    result = w.reconcile()

    assert result["repairs"] == ["missed_charge:paid_count 3->6"]
    assert w.team.subscription_credits_remaining == 350
    (entry,) = ledger_entries(w.db)
    assert entry.idempotency_key == f"sub:{SUB_ID}:cycle:6"
    assert w.row.last_paid_count == 6


# --------------------------------------------------------------------------- #
# status / period drift
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("remote", ["cancelled", "expired"])
def test_an_ended_razorpay_subscription_cancels_the_row(w, remote):
    w.rzp.statuses[SUB_ID] = remote

    result = w.reconcile()

    assert w.row.status == "cancelled"
    assert result["outcome"] == "repaired"
    assert w.team.subscription_credits_remaining == 40     # nothing granted or taken away


def test_completed_cancels_only_an_active_row(w):
    w.rzp.statuses[SUB_ID] = "completed"
    w.reconcile()
    assert w.row.status == "cancelled"

    w.row.status = "pending"                              # a row mid renewal-retry is left alone
    w.row.last_reconciled_at = None
    w.reconcile()
    assert w.row.status == "pending"


def test_halted_at_razorpay_halts_the_row(w):
    w.rzp.statuses[SUB_ID] = "halted"
    w.reconcile()
    assert w.row.status == "halted"


def test_a_recovered_subscription_becomes_active_again(w):
    w.row.status = "pending"                               # renewal-failed state
    w.row.last_paid_count = 4                              # nothing new to charge
    w.reconcile()
    assert w.row.status == "active"


def test_period_end_drift_over_an_hour_is_corrected_but_small_drift_is_not(w):
    w.row.last_paid_count = 4                              # no new charge, only drift
    w.row.current_period_end = w.new_end + timedelta(days=2)
    result = w.reconcile()
    assert w.row.current_period_end == w.new_end
    assert result["repairs"][0].startswith("period_end:")

    w.row.current_period_end = w.new_end + timedelta(minutes=30)
    result = w.reconcile()
    assert result == {"outcome": "ok", "repairs": []}
    assert w.row.current_period_end == w.new_end + timedelta(minutes=30)


def test_an_in_sync_subscription_only_gets_its_timestamp_updated(w):
    w.row.last_paid_count = 4
    w.row.current_period_end = w.new_end

    result = w.reconcile()

    assert result == {"outcome": "ok", "repairs": []}
    assert w.row.last_reconciled_at == NOW
    assert ledger_entries(w.db) == []


# --------------------------------------------------------------------------- #
# missed activations
# --------------------------------------------------------------------------- #

def _unpaid_checkout(w):
    w.row.status = "pending"
    w.row.credits_per_refill = 0
    w.row.last_paid_count = 0
    w.rzp.entities[SUB_ID] = {
        "plan_id": w.plan.razorpay_plan_id,
        "notes": {"team_id": str(TEAM_ID), "subscription_id": str(w.plan.id)},
        "paid_count": 1,
    }


def test_a_paid_checkout_whose_activation_was_missed_is_activated(w):
    _unpaid_checkout(w)

    result = w.reconcile()

    assert result["repairs"] == ["missed_activation"]
    assert w.row.status == "active"
    assert w.row.credits_per_refill == 350
    assert w.team.subscription_credits_remaining == 350
    (entry,) = ledger_entries(w.db)
    assert entry.idempotency_key == f"sub:{SUB_ID}:cycle:1"


def test_an_unpaid_checkout_that_is_still_just_created_is_left_alone(w):
    _unpaid_checkout(w)
    w.rzp.statuses[SUB_ID] = "created"

    result = w.reconcile()

    assert result == {"outcome": "ok", "repairs": []}
    assert w.row.status == "pending" and w.row.credits_per_refill == 0
    assert w.team.subscription_credits_remaining == 40


def test_a_paid_upgrade_whose_activation_was_missed_is_promoted(w):
    yearly = make_plan("year", slug="yearly", credits=1200)
    w.db = make_db(team_subs=[w.row], plans=[w.plan, yearly], teams=[w.team])
    switch_id = uuid.uuid4()
    w.row.last_paid_count = 4
    w.row.current_period_end = w.new_end
    w.row.pending_subscription_id = yearly.id
    w.row.pending_razorpay_subscription_id = "sub_pending_y"
    w.row.pending_switch_id = switch_id
    w.row.pending_switch_expires_at = NOW + timedelta(hours=20)
    w.rzp.statuses["sub_pending_y"] = "active"
    w.rzp.entities["sub_pending_y"] = {
        "plan_id": yearly.razorpay_plan_id,
        "notes": {"team_id": str(TEAM_ID), "subscription_id": str(yearly.id), "switch_id": str(switch_id)},
    }

    result = w.reconcile()

    assert result["repairs"] == ["missed_switch_activation"]
    assert w.row.razorpay_subscription_id == "sub_pending_y"
    assert w.row.subscription_id == yearly.id
    assert w.row.pending_razorpay_subscription_id is None
    assert SUB_ID in w.rzp.cancelled                       # old plan cancelled
    assert w.team.topup_credits_balance == 10 + 40


def test_an_unpaid_pending_switch_is_left_for_the_expiry_job(w):
    w.row.last_paid_count = 4
    w.row.current_period_end = w.new_end
    w.row.pending_subscription_id = uuid.uuid4()
    w.row.pending_razorpay_subscription_id = "sub_pending_y"
    w.row.pending_switch_id = uuid.uuid4()
    w.row.pending_switch_expires_at = NOW + timedelta(hours=20)
    w.rzp.statuses["sub_pending_y"] = "created"

    result = w.reconcile()

    assert result == {"outcome": "ok", "repairs": []}
    assert w.row.pending_razorpay_subscription_id == "sub_pending_y"
    assert w.rzp.cancelled == []


# --------------------------------------------------------------------------- #
# safety
# --------------------------------------------------------------------------- #

def test_a_razorpay_outage_changes_nothing(w):
    w.rzp.fail_fetch = True

    result = w.reconcile()

    assert result == {"outcome": "error", "repairs": []}
    assert w.row.last_reconciled_at is None                # retried next run
    assert w.team.subscription_credits_remaining == 40
    assert ledger_entries(w.db) == []
    w.db.commit.assert_not_called()
    w.db.rollback.assert_called()


def test_a_failure_while_repairing_rolls_back_and_reports_an_error(w, monkeypatch):
    monkeypatch.setattr(
        webhooks_svc, "refill_subscription_credits",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")),
    )

    result = w.reconcile()

    assert result == {"outcome": "error", "repairs": []}
    w.db.rollback.assert_called()
    assert w.row.last_reconciled_at is None


@pytest.mark.parametrize("mutate", [
    lambda row: setattr(row, "status", "cancelled"),
    lambda row: setattr(row, "razorpay_subscription_id", None),
])
def test_rows_that_are_not_live_are_skipped_untouched(w, mutate):
    mutate(w.row)

    assert w.reconcile() == {"outcome": "skipped", "repairs": []}
    assert w.rzp.cancelled == [] and ledger_entries(w.db) == []


def test_reconcile_never_takes_credits_away_when_razorpay_says_cancelled(w):
    w.rzp.statuses[SUB_ID] = "cancelled"
    w.reconcile()
    assert w.team.subscription_credits_remaining == 40
    assert w.team.topup_credits_balance == 10


# --------------------------------------------------------------------------- #
# the batch
# --------------------------------------------------------------------------- #

def test_the_batch_survives_one_failure_counts_outcomes_and_alerts_once(w, monkeypatch):
    ids = [uuid.uuid4() for _ in range(4)]
    monkeypatch.setattr(recon, "find_reconcile_candidates", lambda db, now, limit: ids)
    outcomes = {
        ids[0]: {"outcome": "repaired", "repairs": ["x"]},
        ids[1]: RuntimeError("boom"),
        ids[2]: {"outcome": "ok", "repairs": []},
        ids[3]: {"outcome": "error", "repairs": []},
    }

    def fake_reconcile(db, row_id, now):
        result = outcomes[row_id]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(recon, "reconcile_subscription", fake_reconcile)

    summary = recon.reconcile_due_subscriptions(w.db, now=NOW)

    assert summary == {"checked": 2, "repaired": 1, "errors": 2, "skipped": 0}
    assert len(w.alerts) == 1 and "RECONCILE_REPAIRED" in w.alerts[0]


def test_the_batch_sends_no_alert_when_nothing_was_repaired(w, monkeypatch):
    monkeypatch.setattr(recon, "find_reconcile_candidates", lambda db, now, limit: [uuid.uuid4()])
    monkeypatch.setattr(recon, "reconcile_subscription", lambda db, row_id, now: {"outcome": "ok", "repairs": []})

    recon.reconcile_due_subscriptions(w.db, now=NOW)

    assert w.alerts == []


# --------------------------------------------------------------------------- #
# candidate selection (real SQL, SQLite)
# --------------------------------------------------------------------------- #

@pytest.fixture
def sqlite_session():
    from sqlalchemy import create_engine
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.core.database import Base
    from app.models.team_subscription import TeamSubscription

    @compiles(JSONB, "sqlite")
    def _jsonb(type_, compiler, **kw):
        return "JSON"

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine, tables=[TeamSubscription.__table__])
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


def _row(session, name, **kw):
    from app.models.team_subscription import TeamSubscription

    defaults = dict(
        id=uuid.uuid4(), team_id=uuid.uuid4(), subscription_id=uuid.uuid4(),
        razorpay_subscription_id=f"sub_{name}", status="active", credits_per_refill=100,
        last_paid_count=1, next_refill_at=NOW, current_period_end=NOW + timedelta(days=10),
        last_reconciled_at=NOW - timedelta(minutes=10),
    )
    defaults.update(kw)
    row = TeamSubscription(**defaults)
    session.add(row)
    session.commit()
    return row.id


def test_candidate_selection_picks_exactly_the_rows_worth_checking(sqlite_session):
    s = sqlite_session
    overdue = _row(s, "overdue", current_period_end=NOW - timedelta(hours=3),
                   last_reconciled_at=NOW - timedelta(hours=2))
    overdue_just_checked = _row(s, "overdue_checked", current_period_end=NOW - timedelta(hours=3),
                                last_reconciled_at=NOW - timedelta(minutes=10))
    barely_due = _row(s, "barely_due", current_period_end=NOW - timedelta(minutes=30),
                      last_reconciled_at=NOW - timedelta(hours=2))
    never_checked = _row(s, "never", last_reconciled_at=None)
    stale_audit = _row(s, "stale", last_reconciled_at=NOW - timedelta(hours=25))
    fresh_healthy = _row(s, "healthy")
    cancelled = _row(s, "cancelled", status="cancelled", last_reconciled_at=None)
    no_razorpay = _row(s, "none", razorpay_subscription_id=None, last_reconciled_at=None)
    halted = _row(s, "halted", status="halted", last_reconciled_at=None)
    old_switch = _row(s, "old_switch", pending_switch_expires_at=NOW + timedelta(hours=23),   # made ~1h ago
                      pending_subscription_id=uuid.uuid4(), pending_razorpay_subscription_id="sub_p1",
                      pending_switch_id=uuid.uuid4(), last_reconciled_at=NOW - timedelta(hours=2))
    young_switch = _row(s, "young_switch", pending_switch_expires_at=NOW + timedelta(hours=23, minutes=50),
                        pending_subscription_id=uuid.uuid4(), pending_razorpay_subscription_id="sub_p2",
                        pending_switch_id=uuid.uuid4(), last_reconciled_at=NOW - timedelta(hours=2))

    picked = recon.find_reconcile_candidates(s, NOW, limit=50)

    assert set(picked) == {overdue, never_checked, stale_audit, halted, old_switch}
    assert not {overdue_just_checked, barely_due, fresh_healthy, cancelled, no_razorpay, young_switch} & set(picked)


def test_candidates_are_never_checked_first_then_oldest_first_and_limited(sqlite_session):
    s = sqlite_session
    older = _row(s, "older", last_reconciled_at=NOW - timedelta(hours=48))
    old = _row(s, "old", last_reconciled_at=NOW - timedelta(hours=30))
    never = _row(s, "never", last_reconciled_at=None)

    assert recon.find_reconcile_candidates(s, NOW, limit=50) == [never, older, old]
    assert recon.find_reconcile_candidates(s, NOW, limit=2) == [never, older]


# --------------------------------------------------------------------------- #
# the cron
# --------------------------------------------------------------------------- #

def test_the_cron_runs_the_batch_off_the_event_loop_and_closes_the_session(monkeypatch):
    import threading

    from app import worker

    db = MagicMock()
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    seen = {}

    def fake_batch(session):
        seen["db"] = session
        seen["thread"] = threading.current_thread()
        return {"checked": 0, "repaired": 0, "errors": 0, "skipped": 0}

    monkeypatch.setattr(recon, "reconcile_due_subscriptions", fake_batch)

    asyncio.run(worker.reconcile_subscriptions({}))

    assert seen["db"] is db
    assert seen["thread"] is not threading.main_thread()      # not blocking the loop
    db.close.assert_called_once()


def test_a_failing_run_is_swallowed_and_still_closes_the_session(monkeypatch):
    from app import worker

    db = MagicMock()
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    monkeypatch.setattr(recon, "reconcile_due_subscriptions",
                        lambda session: (_ for _ in ()).throw(RuntimeError("boom")))

    asyncio.run(worker.reconcile_subscriptions({}))       # must not raise

    db.close.assert_called_once()


def test_the_cron_is_scheduled_hourly():
    from app import worker

    job = next(j for j in worker.WorkerSettings.cron_jobs if j.coroutine is worker.reconcile_subscriptions)
    assert job.minute == {11}
    assert job.hour is None                                # every hour
    assert worker.reconcile_subscriptions in worker.WorkerSettings.functions
