"""switch_subscription + POST /teams/{team_id}/subscriptions/{new_subscription_id}/switch

A switch (upgrade) no longer touches the current plan. It creates the replacement
Razorpay subscription and stores it as a PENDING switch on the team's row; the old
plan stays live, and is cancelled (with credits moved) only when the replacement's
`subscription.activated` webhook arrives -- that half is in test_pending_switch.py.

These tests cover the initiating half: validation ordering, the tier guard, that
nothing about the live plan changes, one-pending-switch-per-team (same plan is
reused, a different plan replaces), and every failure path.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.database import get_db
from app.deps import get_current_user
from app.main import app
from app.services import billing as billing_svc
from tests.billing_fakes import FakeRazorpay, make_db, make_plan, make_team, make_team_sub

TEAM_ID = uuid.uuid4()
OLD_RAZORPAY_ID = "sub_live_old"


def _url(new_sub_id):
    return f"/billing/teams/{TEAM_ID}/subscriptions/{new_sub_id}/switch"


class World:
    """One team on a weekly plan, a yearly plan to upgrade to, and a fake Razorpay."""

    def __init__(self, monkeypatch):
        self.old_plan = make_plan("week", slug="weekly")
        self.new_plan = make_plan("year", slug="yearly")
        self.team = make_team(TEAM_ID, subscription_credits=120, topup=30)
        self.row = make_team_sub(
            team_id=TEAM_ID,
            subscription_id=self.old_plan.id,
            razorpay_subscription_id=OLD_RAZORPAY_ID,
            current_period_end=datetime.now(timezone.utc) + timedelta(days=3),
        )
        self.rzp = FakeRazorpay().install(monkeypatch, billing_svc)
        self.plans = [self.old_plan, self.new_plan]
        self.rebuild_db()

    def rebuild_db(self):
        self.db = make_db(team_subs=[self.row], plans=self.plans, teams=[self.team])
        return self.db

    def switch(self, plan=None):
        return billing_svc.switch_subscription(self.db, TEAM_ID, (plan or self.new_plan).id)

    def assert_live_plan_untouched(self):
        assert self.row.status == "active"
        assert self.row.razorpay_subscription_id == OLD_RAZORPAY_ID
        assert self.row.subscription_id == self.old_plan.id
        assert self.team.subscription_credits_remaining == 120
        assert self.team.topup_credits_balance == 30
        assert OLD_RAZORPAY_ID not in self.rzp.cancelled


@pytest.fixture
def world(monkeypatch):
    return World(monkeypatch)


@pytest.fixture
def client(world, monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: MagicMock(id=uuid.uuid4())
    app.dependency_overrides[get_db] = lambda: world.db
    monkeypatch.setattr("app.routes.checkout.is_team_owner", lambda db, tid, uid: True)
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def _post(client, plan):
    return client.post(_url(plan.id), headers={"Authorization": "Bearer x"})


# --------------------------------------------------------------------------- #
# route contract
# --------------------------------------------------------------------------- #

def test_switch_requires_owner(world, client, monkeypatch):
    monkeypatch.setattr("app.routes.checkout.is_team_owner", lambda db, tid, uid: False)
    res = _post(client, world.new_plan)
    assert res.status_code == 403
    assert world.rzp.created == [] and world.rzp.cancelled == []


def test_switch_ignores_injected_request_body(world, client):
    """Both ids come from the path; nothing in a request body should matter."""
    res = client.post(
        _url(world.new_plan.id),
        headers={"Authorization": "Bearer x"},
        json={"subscription_id": "attacker", "credits": 999999},
    )
    assert res.status_code == 200
    assert world.rzp.created[0]["plan_id"] == world.new_plan.razorpay_plan_id


# --------------------------------------------------------------------------- #
# 1. initiating a switch leaves the current plan completely alone
# --------------------------------------------------------------------------- #

def test_switch_stores_a_pending_switch_and_leaves_the_live_plan_untouched(world, client):
    res = _post(client, world.new_plan)

    assert res.status_code == 200
    body = res.json()
    assert body["razorpay_subscription_id"] == "sub_new_1"
    assert body["pending_switch"] is True
    assert body["pending_switch_expires_at"] == world.row.pending_switch_expires_at.isoformat()

    world.assert_live_plan_untouched()          # no cancel, no credit move, same plan
    assert world.row.pending_subscription_id == world.new_plan.id
    assert world.row.pending_razorpay_subscription_id == "sub_new_1"
    assert world.row.pending_switch_id is not None
    world.db.commit.assert_called_once()


def test_switch_creates_replacement_with_reference_notes_and_total_count(world):
    world.switch()

    sent = world.rzp.created[0]
    assert sent["plan_id"] == world.new_plan.razorpay_plan_id
    assert sent["total_count"] == 1                          # "year" -> 1
    assert sent["notes"]["team_id"] == str(TEAM_ID)
    assert sent["notes"]["subscription_id"] == str(world.new_plan.id)
    assert sent["notes"]["switch_id"] == str(world.row.pending_switch_id)


def test_pending_switch_expiry_uses_the_configured_ttl(world, monkeypatch):
    monkeypatch.setattr(billing_svc.settings, "pending_switch_ttl_hours", 5)
    before = datetime.now(timezone.utc)

    world.switch()

    expires = world.row.pending_switch_expires_at
    assert before + timedelta(hours=5) <= expires <= datetime.now(timezone.utc) + timedelta(hours=5)


def test_replacement_is_created_with_expire_by_equal_to_the_stored_deadline(world):
    world.switch()

    sent = world.rzp.created[0]
    stored = world.row.pending_switch_expires_at
    assert isinstance(sent["expire_by"], int)                  # Razorpay wants a Unix timestamp
    assert sent["expire_by"] == int(stored.timestamp())        # the same instant as our own expiry
    assert sent["expire_by"] > int(datetime.now(timezone.utc).timestamp())


def test_expire_by_follows_the_configured_ttl(world, monkeypatch):
    monkeypatch.setattr(billing_svc.settings, "pending_switch_ttl_hours", 3)
    before = int((datetime.now(timezone.utc) + timedelta(hours=3)).timestamp())

    world.switch()

    sent = world.rzp.created[0]["expire_by"]
    assert before <= sent <= before + 5                        # ~3h out, allowing test runtime


def test_default_ttl_is_24_hours():
    assert billing_svc.settings.pending_switch_ttl_hours == 24


# --------------------------------------------------------------------------- #
# 2. validation -- fails before touching anything
# --------------------------------------------------------------------------- #

def test_switch_to_unconfigured_plan_fails_before_touching_razorpay(world, client):
    world.new_plan.razorpay_plan_id = None

    res = _post(client, world.new_plan)

    assert res.status_code == 400
    assert "not configured" in res.json()["detail"].lower()
    assert world.rzp.created == [] and world.rzp.cancelled == []
    assert world.row.pending_razorpay_subscription_id is None


def test_switch_with_no_active_subscription_fails_cleanly(world, client):
    world.row.status = "cancelled"

    res = _post(client, world.new_plan)

    assert res.status_code == 400
    assert "no active subscription" in res.json()["detail"].lower()
    assert world.rzp.created == []


# --------------------------------------------------------------------------- #
# 3. failure paths
# --------------------------------------------------------------------------- #

def test_switch_when_razorpay_create_fails_stores_nothing(world, client):
    world.rzp.fail_create = True

    res = _post(client, world.new_plan)

    assert res.status_code == 500
    world.assert_live_plan_untouched()
    assert world.row.pending_razorpay_subscription_id is None
    world.db.commit.assert_not_called()


def test_switch_commit_failure_logs_ids_and_cancels_the_unpaid_replacement(world, caplog):
    world.db.commit.side_effect = RuntimeError("db down")

    with caplog.at_level(logging.ERROR, logger="app.services.billing"):
        with pytest.raises(RuntimeError, match="db down"):
            world.switch()

    rec = next(r for r in caplog.records if "pending switch commit failed" in r.getMessage())
    msg = rec.getMessage()
    assert rec.levelno == logging.ERROR
    assert f"team_id={TEAM_ID}" in msg
    assert "new_razorpay_subscription_id=sub_new_1" in msg
    assert "switch_id=" in msg and "switch_id=None" not in msg
    world.db.rollback.assert_called()
    # never stored on a row and never paid -> retired, and the live plan untouched
    assert world.rzp.cancelled == ["sub_new_1"]


# --------------------------------------------------------------------------- #
# 4. one pending switch per team
# --------------------------------------------------------------------------- #

def _with_pending(world, plan, rzp_id="sub_pending_a", hours_left=10):
    world.row.pending_subscription_id = plan.id
    world.row.pending_razorpay_subscription_id = rzp_id
    world.row.pending_switch_id = uuid.uuid4()
    world.row.pending_switch_expires_at = datetime.now(timezone.utc) + timedelta(hours=hours_left)


def test_same_upgrade_again_reuses_the_existing_unpaid_subscription(world, client):
    _with_pending(world, world.new_plan)
    world.rzp.statuses["sub_pending_a"] = "created"

    res = _post(client, world.new_plan)

    assert res.status_code == 200
    assert res.json()["razorpay_subscription_id"] == "sub_pending_a"
    assert world.rzp.created == [] and world.rzp.cancelled == []     # nothing new, nothing cancelled
    assert world.row.pending_razorpay_subscription_id == "sub_pending_a"
    world.db.commit.assert_not_called()


def test_a_different_upgrade_replaces_the_pending_one(world, client, monkeypatch):
    month = make_plan("month", slug="monthly")
    world.plans.append(month)
    world.rebuild_db()
    _with_pending(world, world.new_plan)            # pending yearly
    old_switch_id = world.row.pending_switch_id

    order = []
    world.db.commit.side_effect = lambda: order.append("commit")
    real_cancel = world.rzp.cancel

    def recording_cancel(sid):
        order.append(f"cancel:{sid}")
        return real_cancel(sid)

    monkeypatch.setattr(billing_svc.razorpay_client.subscription, "cancel", recording_cancel)

    res = _post(client, month)

    assert res.status_code == 200
    assert res.json()["razorpay_subscription_id"] == "sub_new_1"
    assert world.row.pending_subscription_id == month.id
    assert world.row.pending_razorpay_subscription_id == "sub_new_1"
    assert world.row.pending_switch_id != old_switch_id
    # exactly one pending switch remains, and the old unpaid one is cancelled only
    # AFTER the replacement is durably stored
    assert order == ["commit", "cancel:sub_pending_a"]
    world.assert_live_plan_untouched()


def test_a_dead_or_expired_pending_checkout_is_replaced_even_for_the_same_plan(world, client):
    _with_pending(world, world.new_plan)
    world.rzp.statuses["sub_pending_a"] = "expired"

    res = _post(client, world.new_plan)

    assert res.status_code == 200
    assert res.json()["razorpay_subscription_id"] == "sub_new_1"


def test_a_pending_checkout_past_its_deadline_is_replaced_even_if_still_created(world, client):
    _with_pending(world, world.new_plan, hours_left=-1)
    world.rzp.statuses["sub_pending_a"] = "created"

    res = _post(client, world.new_plan)

    assert res.json()["razorpay_subscription_id"] == "sub_new_1"
    assert "sub_pending_a" in world.rzp.cancelled


@pytest.mark.parametrize("status", ["authenticated", "active", "pending"])
def test_replacing_is_refused_while_the_pending_payment_is_being_processed(world, client, status):
    """Cancelling a subscription the user JUST paid for would destroy their
    payment -- wait for its activated webhook instead."""
    month = make_plan("month")
    world.plans.append(month)
    world.rebuild_db()
    _with_pending(world, world.new_plan)
    world.rzp.statuses["sub_pending_a"] = status

    res = _post(client, month)

    assert res.status_code == 400
    assert "still being processed" in res.json()["detail"]
    assert world.rzp.created == [] and world.rzp.cancelled == []
    assert world.row.pending_razorpay_subscription_id == "sub_pending_a"


def test_replacing_is_refused_when_the_pending_one_cannot_be_checked(world, client):
    month = make_plan("month")
    world.plans.append(month)
    world.rebuild_db()
    _with_pending(world, world.new_plan)
    world.rzp.fail_fetch = True

    res = _post(client, month)

    assert res.status_code == 400
    assert world.rzp.created == [] and world.rzp.cancelled == []


def test_failure_creating_the_replacement_keeps_the_previous_pending_one(world, client):
    month = make_plan("month")
    world.plans.append(month)
    world.rebuild_db()
    _with_pending(world, world.new_plan)
    world.rzp.fail_create = True

    res = _post(client, month)

    assert res.status_code == 500
    assert world.row.pending_razorpay_subscription_id == "sub_pending_a"
    assert world.rzp.cancelled == []


# --------------------------------------------------------------------------- #
# 5. tier guard (unchanged behaviour)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("current,new,code", [
    ("year", "week", "PLAN_DOWNGRADE_BLOCKED"),
    ("year", "month", "PLAN_DOWNGRADE_BLOCKED"),
    ("month", "week", "PLAN_DOWNGRADE_BLOCKED"),
    ("year", "year", "PLAN_ALREADY_ACTIVE"),
])
def test_non_upgrade_mid_period_is_rejected_before_anything_changes(world, client, current, new, code):
    world.old_plan.period_label = current
    world.new_plan.period_label = new
    period_end = world.row.current_period_end

    res = _post(client, world.new_plan)

    assert res.status_code == 409
    detail = res.json()["detail"]
    assert detail["code"] == code
    assert detail["current_period_end"] == period_end.isoformat()
    assert world.rzp.created == [] and world.rzp.cancelled == []
    assert world.row.pending_razorpay_subscription_id is None
    world.assert_live_plan_untouched()


@pytest.mark.parametrize("current,new", [("week", "month"), ("week", "year"), ("month", "year")])
def test_upgrade_mid_period_still_allowed(world, client, current, new):
    world.old_plan.period_label = current
    world.new_plan.period_label = new

    res = _post(client, world.new_plan)

    assert res.status_code == 200
    assert world.row.pending_razorpay_subscription_id == "sub_new_1"


def test_lower_tier_switch_allowed_once_period_has_ended(world, client):
    world.old_plan.period_label = "year"
    world.new_plan.period_label = "week"
    world.row.current_period_end = datetime.now(timezone.utc) - timedelta(minutes=1)

    res = _post(client, world.new_plan)

    assert res.status_code == 200
    assert world.row.pending_razorpay_subscription_id == "sub_new_1"
