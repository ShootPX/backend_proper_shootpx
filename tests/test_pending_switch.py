"""The second half of a plan switch: promotion on `subscription.activated`, expiry
of unpaid switches, and everything else that has to know a pending switch exists
(cancel, resubscribe, the team billing status, the worker cron).

These use the filter-aware fakes in billing_fakes.py, because the point is which
Razorpay id matches which column -- something the return-anything MagicMocks used
elsewhere cannot show.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.services import billing as billing_svc
from app.services import teams as teams_svc
from app.services import webhooks as webhooks_svc
from tests.billing_fakes import FakeRazorpay, make_db, make_plan, make_team, make_team_sub

TEAM_ID = uuid.uuid4()
OLD_ID = "sub_old"
NEW_ID = "sub_new_pending"


class World:
    """Team on a weekly plan with an unpaid yearly upgrade (NEW_ID) pending."""

    def __init__(self, monkeypatch):
        self.old_plan = make_plan("week", slug="weekly", credits=100)
        self.new_plan = make_plan("year", slug="yearly", credits=1200)
        self.team = make_team(TEAM_ID, subscription_credits=40, topup=10)
        self.switch_id = uuid.uuid4()
        self.row = make_team_sub(
            team_id=TEAM_ID,
            subscription_id=self.old_plan.id,
            razorpay_subscription_id=OLD_ID,
            credits_per_refill=100,
            pending_subscription_id=self.new_plan.id,
            pending_razorpay_subscription_id=NEW_ID,
            pending_switch_id=self.switch_id,
            pending_switch_expires_at=datetime.now(timezone.utc) + timedelta(hours=10),
        )
        self.db = make_db(
            team_subs=[self.row], plans=[self.old_plan, self.new_plan], teams=[self.team],
        )
        self.rzp = FakeRazorpay().install(monkeypatch, webhooks_svc)
        # what Razorpay knows about the replacement, as set at switch time
        self.rzp.entities[NEW_ID] = {
            "plan_id": self.new_plan.razorpay_plan_id,
            "notes": {
                "team_id": str(TEAM_ID),
                "subscription_id": str(self.new_plan.id),
                "switch_id": str(self.switch_id),
            },
        }
        self.rzp.statuses[OLD_ID] = "active"
        self.rzp.statuses[NEW_ID] = "active"   # the moment `activated` fires

    def activated(self, rzp_id=NEW_ID):
        webhooks_svc.handle_subscription_activated(
            self.db, {"payload": {"subscription": {"entity": {"id": rzp_id}}}},
        )

    def event(self, rzp_id):
        return {"payload": {"subscription": {"entity": {"id": rzp_id}}}}

    def assert_not_promoted(self):
        assert self.row.razorpay_subscription_id == OLD_ID
        assert self.row.subscription_id == self.old_plan.id
        assert self.row.pending_razorpay_subscription_id == NEW_ID     # still waiting
        assert self.team.subscription_credits_remaining == 40
        assert self.team.topup_credits_balance == 10


@pytest.fixture
def world(monkeypatch):
    return World(monkeypatch)


# --------------------------------------------------------------------------- #
# promotion on subscription.activated
# --------------------------------------------------------------------------- #

def test_activation_promotes_the_pending_switch_in_one_commit(world):
    world.activated()

    row = world.row
    # old plan cancelled at Razorpay, new plan installed
    assert world.rzp.cancelled == [OLD_ID]
    assert row.razorpay_subscription_id == NEW_ID
    assert row.subscription_id == world.new_plan.id
    assert row.status == "active"
    assert row.credits_per_refill == 100            # 1200 / 12 monthly slices for a yearly plan
    assert row.last_paid_count == 0
    assert row.current_period_end > datetime.now(timezone.utc) + timedelta(days=364)
    # leftover old-plan credits kept as top-up, new plan's first slice granted
    assert world.team.topup_credits_balance == 10 + 40
    assert world.team.subscription_credits_remaining == 100
    # pending switch cleared
    assert row.pending_subscription_id is None
    assert row.pending_razorpay_subscription_id is None
    assert row.pending_switch_id is None
    assert row.pending_switch_expires_at is None
    world.db.commit.assert_called_once()


def test_activation_redelivery_is_a_no_op(world):
    world.activated()
    snapshot = (
        world.team.topup_credits_balance, world.team.subscription_credits_remaining,
        world.row.razorpay_subscription_id, world.row.current_period_end,
    )
    cancelled_before = list(world.rzp.cancelled)
    world.db.commit.reset_mock()

    world.activated()          # Razorpay redelivers the same webhook

    assert (
        world.team.topup_credits_balance, world.team.subscription_credits_remaining,
        world.row.razorpay_subscription_id, world.row.current_period_end,
    ) == snapshot                                  # credits not moved or granted twice
    assert world.rzp.cancelled == cancelled_before  # old plan not cancelled twice
    world.db.commit.assert_not_called()


def test_old_cancel_failure_raises_logs_switch_id_and_leaves_everything_pending(world, caplog):
    world.rzp.fail_cancel.add(OLD_ID)

    with caplog.at_level(logging.ERROR, logger="app.services.webhooks"):
        with pytest.raises(RuntimeError):
            world.activated()

    world.assert_not_promoted()
    world.db.commit.assert_not_called()
    rec = next(r for r in caplog.records if "could not cancel the OLD subscription" in r.getMessage())
    msg = rec.getMessage()
    assert rec.levelno == logging.ERROR
    assert f"switch_id={world.switch_id}" in msg
    assert f"team_id={TEAM_ID}" in msg
    assert f"old_razorpay_subscription_id={OLD_ID}" in msg
    assert f"new_razorpay_subscription_id={NEW_ID}" in msg


def test_a_webhook_retry_after_the_old_cancel_failed_completes_the_switch(world):
    world.rzp.fail_cancel.add(OLD_ID)
    with pytest.raises(RuntimeError):
        world.activated()

    world.rzp.fail_cancel.clear()      # Razorpay's retry finds the API healthy again
    world.activated()

    assert world.row.razorpay_subscription_id == NEW_ID
    assert world.row.pending_razorpay_subscription_id is None
    assert world.team.topup_credits_balance == 50
    assert world.team.subscription_credits_remaining == 100


def test_old_plan_already_ended_counts_as_cancelled(world):
    """A previous delivery cancelled it and then failed to commit: cancel() now
    errors, but Razorpay says it is already gone -- carry on."""
    world.rzp.fail_cancel.add(OLD_ID)
    world.rzp.statuses[OLD_ID] = "cancelled"

    world.activated()

    assert world.row.razorpay_subscription_id == NEW_ID
    assert world.row.pending_razorpay_subscription_id is None


def test_commit_failure_after_the_old_cancel_logs_and_raises_then_the_retry_succeeds(world, caplog):
    # a real rollback discards the uncommitted changes; the fake DB has to be told to
    row_before, team_before = dict(vars(world.row)), dict(vars(world.team))

    def rollback():
        vars(world.row).update(row_before)
        vars(world.team).update(team_before)

    world.db.rollback.side_effect = rollback
    world.db.commit.side_effect = RuntimeError("db down")
    with caplog.at_level(logging.ERROR, logger="app.services.webhooks"):
        with pytest.raises(RuntimeError, match="db down"):
            world.activated()

    rec = next(r for r in caplog.records if "commit failed AFTER the old plan was cancelled" in r.getMessage())
    assert f"switch_id={world.switch_id}" in rec.getMessage()
    assert world.rzp.cancelled == [OLD_ID]
    world.db.rollback.assert_called()
    world.assert_not_promoted()                 # the failed attempt changed nothing durable

    # retry: the old plan is already cancelled at Razorpay, the DB is healthy again
    world.db.commit.side_effect = None
    world.rzp.fail_cancel.add(OLD_ID)           # cancelling an ended plan errors -- must be tolerated
    world.rzp.statuses[OLD_ID] = "cancelled"
    world.activated()

    assert world.row.razorpay_subscription_id == NEW_ID
    assert world.row.pending_razorpay_subscription_id is None
    assert world.team.topup_credits_balance == 50       # moved exactly once


def test_activation_promotes_even_if_the_old_plan_already_ended(world):
    """The old plan completed while the upgrade was unpaid; its cancel errors."""
    world.row.status = "cancelled"
    world.rzp.fail_cancel.add(OLD_ID)
    world.rzp.statuses[OLD_ID] = "completed"
    world.team.subscription_credits_remaining = 0

    world.activated()

    assert world.row.status == "active"
    assert world.row.razorpay_subscription_id == NEW_ID
    assert world.team.subscription_credits_remaining == 100


def test_activation_for_a_different_plan_or_switch_is_refused(world):
    world.rzp.entities[NEW_ID]["plan_id"] = "plan_tampered"

    world.activated()

    world.assert_not_promoted()
    assert world.rzp.cancelled == []
    world.db.commit.assert_not_called()


def test_activation_with_a_foreign_switch_id_is_refused(world):
    world.rzp.entities[NEW_ID]["notes"]["switch_id"] = str(uuid.uuid4())

    world.activated()

    world.assert_not_promoted()
    assert world.rzp.cancelled == []


def test_activation_fetch_failure_raises_so_razorpay_retries(world):
    world.rzp.fail_fetch = True
    with pytest.raises(RuntimeError):
        world.activated()
    world.assert_not_promoted()


# --------------------------------------------------------------------------- #
# the other webhooks while a switch is pending
# --------------------------------------------------------------------------- #

def test_late_cancelled_for_the_old_id_after_promotion_leaves_the_new_plan_alone(world):
    world.activated()

    webhooks_svc.handle_subscription_cancelled(world.db, world.event(OLD_ID))

    assert world.row.status == "active"
    assert world.row.razorpay_subscription_id == NEW_ID


def test_cancelled_for_the_pending_id_never_touches_the_live_plan(world):
    """We trigger this ourselves when replacing/expiring a pending switch."""
    webhooks_svc.handle_subscription_cancelled(world.db, world.event(NEW_ID))

    assert world.row.status == "active"
    world.assert_not_promoted()


@pytest.mark.parametrize("handler", [
    webhooks_svc.handle_subscription_halted,
    webhooks_svc.handle_subscription_completed,
    webhooks_svc.handle_subscription_pending,
])
def test_other_events_for_the_pending_id_never_touch_the_live_plan(world, handler):
    handler(world.db, world.event(NEW_ID))

    assert world.row.status == "active"
    world.assert_not_promoted()


def test_charged_for_the_old_plan_keeps_renewing_it_and_keeps_the_pending_switch(world):
    world.rzp.entities[OLD_ID] = {"paid_count": 4}
    world.old_plan.total_count = None
    world.row.last_paid_count = 3
    world.team.subscription_credits_remaining = 5

    webhooks_svc.handle_subscription_charged(world.db, world.event(OLD_ID))

    assert world.row.last_paid_count == 4
    assert world.row.pending_razorpay_subscription_id == NEW_ID
    assert world.row.razorpay_subscription_id == OLD_ID


def test_charged_for_the_pending_id_before_activation_is_ignored(world):
    world.rzp.entities[NEW_ID]["paid_count"] = 1

    webhooks_svc.handle_subscription_charged(world.db, world.event(NEW_ID))

    world.assert_not_promoted()


# --------------------------------------------------------------------------- #
# expiry
# --------------------------------------------------------------------------- #

def _expire(world, *, hours_past=1):
    world.row.pending_switch_expires_at = datetime.now(timezone.utc) - timedelta(hours=hours_past)
    return billing_svc.expire_pending_switch(world.db, world.row.id)


def test_expiry_cancels_the_unpaid_replacement_and_clears_the_pending_switch(world):
    world.rzp.statuses[NEW_ID] = "created"

    assert _expire(world) == "cancelled"

    assert world.rzp.cancelled == [NEW_ID]
    assert world.row.pending_razorpay_subscription_id is None
    assert world.row.pending_subscription_id is None
    assert world.row.pending_switch_id is None
    assert world.row.pending_switch_expires_at is None
    # the live plan was never touched
    assert world.row.status == "active"
    assert world.row.razorpay_subscription_id == OLD_ID
    assert world.team.subscription_credits_remaining == 40
    world.db.commit.assert_called_once()


def test_expiry_leaves_a_switch_that_was_actually_paid_alone(world, caplog):
    world.rzp.statuses[NEW_ID] = "authenticated"
    before = world.row.pending_switch_id

    with caplog.at_level(logging.ERROR, logger="app.services.billing"):
        assert _expire(world) == "in_flight"

    assert world.rzp.cancelled == []                       # money already taken
    assert world.row.pending_razorpay_subscription_id == NEW_ID
    assert world.row.pending_switch_id == before
    assert world.row.pending_switch_expires_at > datetime.now(timezone.utc)      # deadline pushed out
    rec = next(r for r in caplog.records if "NOT cancelling" in r.getMessage())
    assert rec.levelno == logging.ERROR and str(before) in rec.getMessage()


@pytest.mark.parametrize("status", ["cancelled", "expired"])
def test_expiry_just_clears_an_already_dead_replacement(world, status):
    world.rzp.statuses[NEW_ID] = status

    assert _expire(world) == "cleared"

    assert world.rzp.cancelled == []                       # nothing to cancel
    assert world.row.pending_razorpay_subscription_id is None


def test_expiry_retries_later_when_razorpay_cannot_be_reached(world):
    world.rzp.fail_fetch = True

    assert _expire(world) == "error"

    assert world.row.pending_razorpay_subscription_id == NEW_ID     # kept for the next run
    world.db.commit.assert_not_called()


def test_expiry_keeps_the_switch_when_the_cancel_fails(world):
    world.rzp.statuses[NEW_ID] = "created"
    world.rzp.fail_cancel.add(NEW_ID)

    assert _expire(world) == "error"

    assert world.row.pending_razorpay_subscription_id == NEW_ID
    world.db.commit.assert_not_called()


def test_expiry_skips_a_switch_that_is_not_due_yet(world):
    world.row.pending_switch_expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

    assert billing_svc.expire_pending_switch(world.db, world.row.id) == "skipped"

    assert world.rzp.cancelled == []
    assert world.row.pending_razorpay_subscription_id == NEW_ID


def test_expiry_skips_a_row_with_no_pending_switch(world):
    world.row.pending_razorpay_subscription_id = None
    world.row.pending_switch_expires_at = None

    assert billing_svc.expire_pending_switch(world.db, world.row.id) == "skipped"
    assert world.rzp.cancelled == []


def test_expiry_cron_processes_every_due_row_and_survives_one_failure(monkeypatch):
    import asyncio
    from unittest.mock import MagicMock

    from app import worker

    ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [(i,) for i in ids]
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)

    seen = []

    def fake_expire(session, row_id):
        seen.append(row_id)
        if row_id == ids[1]:
            raise RuntimeError("boom")
        return "cancelled"

    monkeypatch.setattr(billing_svc, "expire_pending_switch", fake_expire)

    asyncio.run(worker.expire_pending_switches({}))

    assert seen == ids                       # the failing row did not stop the rest
    db.rollback.assert_called()              # ...and its transaction was rolled back
    db.close.assert_called_once()
    assert any(
        getattr(job, "coroutine", None) is worker.expire_pending_switches
        for job in worker.WorkerSettings.cron_jobs
    ), "expire_pending_switches must be scheduled"
    assert worker.expire_pending_switches in worker.WorkerSettings.functions


# --------------------------------------------------------------------------- #
# cancel / resubscribe / status know about a pending switch
# --------------------------------------------------------------------------- #

def test_cancelling_the_plan_also_cancels_the_unpaid_upgrade(world):
    world.rzp.statuses[NEW_ID] = "created"
    order = []
    world.db.commit.side_effect = lambda: order.append("commit")
    real_cancel = world.rzp.cancel
    world.rzp.cancel = lambda sid: (order.append(f"cancel:{sid}"), real_cancel(sid))[1]
    billing_svc.razorpay_client.subscription.cancel = world.rzp.cancel

    billing_svc.cancel_subscription(world.db, TEAM_ID)

    assert world.row.status == "cancelled"
    assert world.row.pending_razorpay_subscription_id is None
    assert world.row.pending_subscription_id is None
    # live plan cancelled first, unpaid upgrade only after the commit
    assert order == [f"cancel:{OLD_ID}", "commit", f"cancel:{NEW_ID}"]


def test_resubscribing_drops_a_leftover_pending_switch(world):
    """The old plan ended naturally while the upgrade sat unpaid; a fresh checkout
    reuses the row, and the old pending switch must not survive onto it."""
    world.row.status = "cancelled"
    third = make_plan("month", slug="monthly")
    world.db = make_db(team_subs=[world.row], plans=[world.old_plan, world.new_plan, third], teams=[world.team])

    billing_svc.create_subscription_checkout(world.db, TEAM_ID, third.id)

    assert world.row.pending_razorpay_subscription_id is None
    assert world.row.pending_switch_id is None
    assert world.row.subscription_id == third.id
    assert NEW_ID in world.rzp.cancelled                   # the orphaned unpaid upgrade is retired


def test_team_billing_status_reports_the_pending_switch(world):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    db = MagicMock()
    db.query.return_value.outerjoin.return_value.outerjoin.return_value.filter.return_value.first.return_value = (
        world.team, world.row, world.old_plan,
    )
    db.query.return_value.filter.return_value.first.return_value = SimpleNamespace(slug="yearly")

    out = teams_svc.get_team_billing(db, TEAM_ID)

    assert out["plan"] == "weekly"                          # the live plan is still reported
    assert out["subscription_status"] == "active"
    assert out["pending_switch"] == {
        "plan": "yearly",
        "expires_at": world.row.pending_switch_expires_at.isoformat(),
    }


def test_team_billing_status_has_no_pending_switch_normally(world):
    from unittest.mock import MagicMock

    world.row.pending_subscription_id = None
    db = MagicMock()
    db.query.return_value.outerjoin.return_value.outerjoin.return_value.filter.return_value.first.return_value = (
        world.team, world.row, world.old_plan,
    )

    assert teams_svc.get_team_billing(db, TEAM_ID)["pending_switch"] is None
