"""Price-tampering safety + correctness for subscription checkout & webhooks.

Same trust rules as the credit-pack flow:
  * the checkout route takes no request body — only `subscription_id` in the path
  * the plan_id (and therefore the amount Razorpay charges) comes only from the
    catalog row, never from the caller
  * handle_subscription_activated fetches the subscription server-to-server and
    never trusts the webhook payload's own notes
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
from app.services import webhooks as webhooks_svc

SUB_ID = uuid.uuid4()
TEAM_ID = uuid.uuid4()
REAL_PLAN_ID = "plan_real_monthly"


class FakePlan:
    id = SUB_ID
    slug = "monthly"
    period_label = "month"
    credits = 350
    razorpay_plan_id = REAL_PLAN_ID
    is_active = True
    # Mirrors the hardcoded {"week": 52, "month": 12, "year": 1}["month"] that
    # create_subscription_checkout actually sends to Razorpay. Nothing in the
    # codebase enforces this match — see test_charged_renewal_notice tests.
    total_count = 12


class FakePlanUnconfigured(FakePlan):
    razorpay_plan_id = None


class FakePlanRetired(FakePlan):
    is_active = False


# --------------------------------------------------------------------------- #
# checkout endpoint
# --------------------------------------------------------------------------- #

def _url():
    return f"/billing/teams/{TEAM_ID}/subscriptions/{SUB_ID}/checkout"


@pytest.fixture
def sub_client(monkeypatch):
    fake_user = MagicMock(id=uuid.uuid4())
    state = {"plan": FakePlan(), "existing_team_sub": None, "created": []}

    db = MagicMock()

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "Subscription":
            q.filter.return_value.first.return_value = state["plan"]
        else:  # TeamSubscription — checkout locks it with .with_for_update().first()
            q.filter.return_value.with_for_update.return_value.first.return_value = (
                state["existing_team_sub"]
            )
            q.filter.return_value.first.return_value = state["existing_team_sub"]
        return q

    db.query.side_effect = query

    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr("app.routes.checkout.is_team_owner", lambda db, tid, uid: True)

    def fake_create(payload):
        state["created"].append(payload)
        return {"id": "sub_test_1"}

    monkeypatch.setattr(billing_svc.razorpay_client.subscription, "create", fake_create)

    yield TestClient(app, raise_server_exceptions=False), state, db
    app.dependency_overrides.clear()


def test_subscription_checkout_uses_db_plan_ignoring_injected_fields(sub_client):
    client, state, db = sub_client
    res = client.post(
        _url(),
        headers={"Authorization": "Bearer x"},
        json={"plan_id": "plan_ATTACKER", "amount": 1, "total_count": 999, "credits": 999999},
    )
    assert res.status_code == 200
    assert res.json()["razorpay_subscription_id"] == "sub_test_1"

    assert len(state["created"]) == 1
    sent = state["created"][0]
    assert sent["plan_id"] == REAL_PLAN_ID          # from DB, not "plan_ATTACKER"
    assert sent["total_count"] == 12                # month -> 12, not 999
    assert sent["notes"] == {"team_id": str(TEAM_ID), "subscription_id": str(SUB_ID)}
    assert "amount" not in sent
    assert 1 not in sent.values()


def test_subscription_checkout_blocks_when_active_row_exists(sub_client):
    client, state, db = sub_client
    state["existing_team_sub"] = MagicMock(status="active")
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 400
    assert "already" in res.json()["detail"].lower()
    assert state["created"] == []


def test_subscription_checkout_blocks_when_pending_row_exists(sub_client):
    """The second of two rapid clicks: the first left a `pending` row behind."""
    client, state, db = sub_client
    state["existing_team_sub"] = MagicMock(status="pending")
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 400
    assert state["created"] == []          # no second Razorpay subscription


def test_subscription_checkout_integrity_error_becomes_conflict(sub_client):
    """The DB-level race: two requests both pass the check, the loser's INSERT
    trips UNIQUE(team_id)."""
    from sqlalchemy.exc import IntegrityError

    client, state, db = sub_client
    state["existing_team_sub"] = None
    db.flush.side_effect = IntegrityError("dup", {}, Exception())
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 400
    assert "already" in res.json()["detail"].lower()
    assert state["created"] == []
    db.rollback.assert_called()


def test_subscription_checkout_rolls_back_pending_on_razorpay_failure(sub_client, monkeypatch):
    """A Razorpay error must not leave a stuck `pending` row."""
    client, state, db = sub_client
    state["existing_team_sub"] = None

    def boom(payload):
        raise RuntimeError("razorpay down")

    monkeypatch.setattr(billing_svc.razorpay_client.subscription, "create", boom)
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 500
    db.rollback.assert_called()
    db.commit.assert_not_called()


def test_subscription_checkout_reuses_cancelled_row_on_resubscribe(sub_client):
    client, state, db = sub_client
    cancelled = MagicMock(status="cancelled")
    state["existing_team_sub"] = cancelled
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 200
    assert len(state["created"]) == 1          # a fresh Razorpay subscription
    assert cancelled.status == "pending"       # row reused, not a new insert
    assert cancelled.razorpay_subscription_id == "sub_test_1"  # real id stored, not None
    db.add.assert_not_called()
    db.commit.assert_called()


def test_two_rapid_checkouts_create_only_one_subscription(sub_client):
    """End-to-end: click 1 lands a pending row + one Razorpay subscription;
    click 2 sees that row and is rejected — no second subscription."""
    client, state, db = sub_client

    res1 = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res1.status_code == 200

    # click 1 left a pending row behind
    state["existing_team_sub"] = MagicMock(status="pending")

    res2 = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res2.status_code == 400
    assert len(state["created"]) == 1          # exactly one Razorpay subscription


def test_subscription_checkout_rejects_unconfigured_plan(sub_client):
    client, state, db = sub_client
    state["plan"] = FakePlanUnconfigured()
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 400
    assert state["created"] == []


def test_subscription_checkout_currently_allows_retired_plan(sub_client):
    """KNOWN GAP, not a pass/fail expectation: create_subscription_checkout only
    checks the plan exists and has a razorpay_plan_id. It never checks
    is_active, so someone who already has a retired plan's UUID (an old
    bookmark, a stale frontend cache, a direct API call) can still start a
    real checkout against it even though GET /landing/billing no longer lists
    it for new visitors. This test documents today's actual behaviour
    (succeeds) rather than asserting it's correct — whether retired plans
    should be checkout-blockable is a product decision, not made here."""
    client, state, db = sub_client
    state["plan"] = FakePlanRetired()
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 200
    assert len(state["created"]) == 1


def test_subscription_checkout_requires_owner(sub_client, monkeypatch):
    client, state, db = sub_client
    monkeypatch.setattr("app.routes.checkout.is_team_owner", lambda db, tid, uid: False)
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 403


# --------------------------------------------------------------------------- #
# subscription.activated webhook
# --------------------------------------------------------------------------- #

def _activated_db(plan, locked=None, team_sub_by_team=None):
    """`locked` = the row found (and locked) by razorpay_subscription_id — the
    normal case, since checkout now stores the id. `team_sub_by_team` is only
    consulted when `locked` is None (legacy / safety fallback)."""
    db = MagicMock()
    seen = {"ts": 0}

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "TeamSubscription":
            seen["ts"] += 1
            # query 1 is the pending-switch lookup, query 2 the normal lookup by
            # razorpay id (both resolve to `locked`), query 3 the team_id fallback
            val = locked if seen["ts"] <= 2 else team_sub_by_team
            q.filter.return_value.with_for_update.return_value.first.return_value = val
            q.filter.return_value.first.return_value = val
        elif name == "Subscription":
            q.filter.return_value.first.return_value = plan
        return q

    db.query.side_effect = query
    return db


def _sub_event(kind, sub_id="sub_test_1"):
    return {"event": kind, "payload": {"subscription": {"entity": {"id": sub_id}}}}


def test_activated_trusts_fetched_notes_not_webhook_payload(monkeypatch):
    refills = []
    monkeypatch.setattr(
        webhooks_svc, "refill_subscription_credits",
        lambda db, team_id, amount: refills.append((team_id, amount)),
    )
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "plan_id": REAL_PLAN_ID,
                     "notes": {"team_id": str(TEAM_ID), "subscription_id": str(SUB_ID)}},
    )
    db = _activated_db(FakePlan())

    # the webhook body itself carries nothing we trust — only the id is used
    webhooks_svc.handle_subscription_activated(db, _sub_event("subscription.activated"))

    assert refills == [(str(TEAM_ID), 350)]  # month plan -> full credits, fetched team
    db.add.assert_called_once()
    db.commit.assert_called()


def test_activated_rejects_plan_id_mismatch(monkeypatch):
    refills = []
    monkeypatch.setattr(
        webhooks_svc, "refill_subscription_credits",
        lambda *a, **k: refills.append(a),
    )
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "plan_id": "plan_TAMPERED",
                     "notes": {"team_id": str(TEAM_ID), "subscription_id": str(SUB_ID)}},
    )
    db = _activated_db(FakePlan())

    webhooks_svc.handle_subscription_activated(db, _sub_event("subscription.activated"))

    assert refills == []            # nothing granted
    db.add.assert_not_called()


def test_activated_is_idempotent_when_row_already_active(monkeypatch):
    fetch = MagicMock()
    monkeypatch.setattr(webhooks_svc.razorpay_client.subscription, "fetch", fetch)
    monkeypatch.setattr(webhooks_svc, "refill_subscription_credits", lambda *a, **k: None)

    active = MagicMock(status="active", razorpay_subscription_id="sub_new")
    db = _activated_db(FakePlan(), locked=active)
    webhooks_svc.handle_subscription_activated(db, _sub_event("subscription.activated", "sub_new"))

    fetch.assert_not_called()          # short-circuited before any work
    db.add.assert_not_called()
    db.commit.assert_not_called()


def test_activated_does_not_resurrect_a_cancelled_row(monkeypatch):
    """The row was cancelled (during the pending window). A late webhook for that
    same id must not reactivate it or grant credits."""
    refills = []
    fetch = MagicMock()
    monkeypatch.setattr(webhooks_svc, "refill_subscription_credits",
                        lambda *a, **k: refills.append(a))
    monkeypatch.setattr(webhooks_svc.razorpay_client.subscription, "fetch", fetch)

    cancelled = MagicMock(status="cancelled", razorpay_subscription_id="sub_new")
    db = _activated_db(FakePlan(), locked=cancelled)
    webhooks_svc.handle_subscription_activated(db, _sub_event("subscription.activated", "sub_new"))

    assert refills == []
    assert cancelled.status == "cancelled"   # unchanged
    fetch.assert_not_called()
    db.add.assert_not_called()
    db.commit.assert_not_called()


def test_activated_promotes_pending_row_from_checkout(monkeypatch):
    """The normal flow: checkout left a `pending` row carrying the real id;
    activation promotes it in place — no duplicate insert."""
    refills = []
    monkeypatch.setattr(
        webhooks_svc, "refill_subscription_credits",
        lambda db, team_id, amount: refills.append((team_id, amount)),
    )
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "plan_id": REAL_PLAN_ID,
                     "notes": {"team_id": str(TEAM_ID), "subscription_id": str(SUB_ID)}},
    )
    pending = MagicMock(status="pending", razorpay_subscription_id="sub_new")
    db = _activated_db(FakePlan(), locked=pending)

    webhooks_svc.handle_subscription_activated(db, _sub_event("subscription.activated", "sub_new"))

    db.add.assert_not_called()
    assert pending.status == "active"
    assert pending.razorpay_subscription_id == "sub_new"
    assert pending.credits_per_refill == 350
    assert refills == [(str(TEAM_ID), 350)]


def test_activated_inserts_when_id_not_on_any_row(monkeypatch):
    """Fallback: the id isn't stored anywhere (legacy / race) and the team has no
    row -> insert a fresh active row."""
    refills = []
    monkeypatch.setattr(
        webhooks_svc, "refill_subscription_credits",
        lambda db, team_id, amount: refills.append((team_id, amount)),
    )
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "plan_id": REAL_PLAN_ID,
                     "notes": {"team_id": str(TEAM_ID), "subscription_id": str(SUB_ID)}},
    )
    db = _activated_db(FakePlan(), locked=None, team_sub_by_team=None)

    webhooks_svc.handle_subscription_activated(db, _sub_event("subscription.activated", "sub_new"))

    db.add.assert_called_once()
    assert refills == [(str(TEAM_ID), 350)]


def test_activated_does_not_clobber_a_different_active_subscription(monkeypatch):
    refills = []
    monkeypatch.setattr(
        webhooks_svc, "refill_subscription_credits", lambda *a, **k: refills.append(a),
    )
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "plan_id": REAL_PLAN_ID,
                     "notes": {"team_id": str(TEAM_ID), "subscription_id": str(SUB_ID)}},
    )
    other = MagicMock(status="active", razorpay_subscription_id="sub_OTHER")
    db = _activated_db(FakePlan(), locked=None, team_sub_by_team=other)

    webhooks_svc.handle_subscription_activated(db, _sub_event("subscription.activated", "sub_new"))

    assert refills == []
    assert other.razorpay_subscription_id == "sub_OTHER"   # untouched
    db.commit.assert_not_called()


# --------------------------------------------------------------------------- #
# subscription.charged webhook
# --------------------------------------------------------------------------- #

def _charged_db(team_sub, plan):
    db = MagicMock()

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "TeamSubscription":
            q.filter.return_value.first.return_value = team_sub
            q.filter.return_value.with_for_update.return_value.first.return_value = team_sub
        elif name == "Subscription":
            q.filter.return_value.first.return_value = plan
        return q

    db.query.side_effect = query
    return db


def test_charged_first_payment_does_not_double_refill(monkeypatch):
    refills = []
    monkeypatch.setattr(
        webhooks_svc, "refill_subscription_credits",
        lambda *a, **k: refills.append(a),
    )
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "paid_count": 1},  # the initial charge
    )
    team_sub = MagicMock(
        team_id=str(TEAM_ID), subscription_id=str(SUB_ID),
        credits_per_refill=350, last_paid_count=0,
    )
    db = _charged_db(team_sub, FakePlan())

    webhooks_svc.handle_subscription_charged(db, _sub_event("subscription.charged"))

    assert refills == []              # activation already granted this period
    assert team_sub.status == "active"


def test_charged_renewal_refills(monkeypatch):
    refills = []
    monkeypatch.setattr(
        webhooks_svc, "refill_subscription_credits",
        lambda db, team_id, amount: refills.append((team_id, amount)),
    )
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "paid_count": 2},  # a renewal
    )
    team_sub = MagicMock(
        team_id=str(TEAM_ID), subscription_id=str(SUB_ID),
        credits_per_refill=350, last_paid_count=0,
    )
    db = _charged_db(team_sub, FakePlan())

    webhooks_svc.handle_subscription_charged(db, _sub_event("subscription.charged"))

    assert refills == [(str(TEAM_ID), 350)]


def test_charged_redelivered_webhook_does_not_double_refill_or_advance(monkeypatch):
    """Razorpay can redeliver the same subscription.charged event. The second
    delivery reports the SAME paid_count as the first, so it must be a no-op:
    refill_subscription_credits fires only once, and next_refill_at / the
    last_paid_count marker only advance once."""
    refills = []
    monkeypatch.setattr(
        webhooks_svc, "refill_subscription_credits",
        lambda db, team_id, amount: refills.append((team_id, amount)),
    )
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "paid_count": 2},  # a renewal, unchanged across both deliveries
    )
    team_sub = MagicMock(
        team_id=str(TEAM_ID), subscription_id=str(SUB_ID),
        credits_per_refill=350, last_paid_count=1,
    )
    db = _charged_db(team_sub, FakePlan())
    event = _sub_event("subscription.charged")

    webhooks_svc.handle_subscription_charged(db, event)   # first delivery
    first_next_refill_at = team_sub.next_refill_at
    webhooks_svc.handle_subscription_charged(db, event)   # redelivered

    assert refills == [(str(TEAM_ID), 350)]        # only the first delivery refilled
    assert team_sub.last_paid_count == 2           # advanced once, not twice
    assert team_sub.next_refill_at == first_next_refill_at  # not pushed out again


def test_halted_and_cancelled_set_status(monkeypatch):
    for kind, expected in [("subscription.halted", "halted"), ("subscription.cancelled", "cancelled")]:
        team_sub = MagicMock()
        db = _charged_db(team_sub, FakePlan())
        handler = {
            "subscription.halted": webhooks_svc.handle_subscription_halted,
            "subscription.cancelled": webhooks_svc.handle_subscription_cancelled,
        }[kind]
        handler(db, _sub_event(kind))
        assert team_sub.status == expected
        db.commit.assert_called()


# --------------------------------------------------------------------------- #
# subscription.completed webhook (Case 2 — plan retirement / natural end)
# --------------------------------------------------------------------------- #

def test_completed_transitions_active_to_cancelled():
    team_sub = MagicMock(status="active")
    db = _charged_db(team_sub, FakePlan())
    webhooks_svc.handle_subscription_completed(db, _sub_event("subscription.completed"))
    assert team_sub.status == "cancelled"
    db.commit.assert_called_once()


def test_completed_noops_on_any_non_active_status():
    for status in ("pending", "cancelled", "halted"):
        team_sub = MagicMock(status=status)
        db = _charged_db(team_sub, FakePlan())
        webhooks_svc.handle_subscription_completed(db, _sub_event("subscription.completed"))
        assert team_sub.status == status           # unchanged
        db.commit.assert_not_called()


def test_completed_noop_when_no_row():
    db = _charged_db(None, FakePlan())
    webhooks_svc.handle_subscription_completed(db, _sub_event("subscription.completed"))
    db.commit.assert_not_called()


# --------------------------------------------------------------------------- #
# renewal-notice email inside handle_subscription_charged
# --------------------------------------------------------------------------- #

def _charged_db_with_team(team_sub, plan, owner_membership=None, owner_user=None):
    db = MagicMock()

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "TeamSubscription":
            q.filter.return_value.first.return_value = team_sub
            q.filter.return_value.with_for_update.return_value.first.return_value = team_sub
        elif name == "Subscription":
            q.filter.return_value.first.return_value = plan
        elif name == "TeamMember":
            q.filter.return_value.first.return_value = owner_membership
        elif name == "User":
            q.filter.return_value.first.return_value = owner_user
        return q

    db.query.side_effect = query
    return db


def test_renewal_notice_fires_on_second_to_last_charge(monkeypatch):
    sent = []
    monkeypatch.setattr(webhooks_svc, "send_email", lambda **kw: sent.append(kw))
    monkeypatch.setattr(webhooks_svc, "refill_subscription_credits", lambda *a, **k: None)
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "paid_count": 11},   # plan.total_count(12) - 1
    )
    plan = FakePlan()
    team_sub = MagicMock(team_id=str(TEAM_ID), subscription_id=str(SUB_ID),
                         credits_per_refill=350, last_paid_count=10, status="active")
    membership = MagicMock(user_id="user-1")
    owner = MagicMock(email="owner@example.com")
    db = _charged_db_with_team(team_sub, plan, owner_membership=membership, owner_user=owner)

    webhooks_svc.handle_subscription_charged(db, _sub_event("subscription.charged"))

    assert len(sent) == 1
    assert sent[0]["to"] == "owner@example.com"
    assert team_sub.last_paid_count == 11


def test_renewal_notice_does_not_fire_on_other_paid_counts(monkeypatch):
    sent = []
    monkeypatch.setattr(webhooks_svc, "send_email", lambda **kw: sent.append(kw))
    monkeypatch.setattr(webhooks_svc, "refill_subscription_credits", lambda *a, **k: None)
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "paid_count": 5},
    )
    plan = FakePlan()
    team_sub = MagicMock(team_id=str(TEAM_ID), subscription_id=str(SUB_ID),
                         credits_per_refill=350, last_paid_count=4, status="active")
    membership = MagicMock(user_id="user-1")
    owner = MagicMock(email="owner@example.com")
    db = _charged_db_with_team(team_sub, plan, owner_membership=membership, owner_user=owner)

    webhooks_svc.handle_subscription_charged(db, _sub_event("subscription.charged"))

    assert sent == []


def test_renewal_notice_does_not_refire_on_redelivered_webhook(monkeypatch):
    """Nested inside the same paid_count > last_paid_count guard that protects
    the refill: a redelivery of the exact event that already sent the notice
    must not send it a second time."""
    sent = []
    monkeypatch.setattr(webhooks_svc, "send_email", lambda **kw: sent.append(kw))
    monkeypatch.setattr(webhooks_svc, "refill_subscription_credits", lambda *a, **k: None)
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "paid_count": 11},
    )
    plan = FakePlan()
    team_sub = MagicMock(team_id=str(TEAM_ID), subscription_id=str(SUB_ID),
                         credits_per_refill=350, last_paid_count=10, status="active")
    membership = MagicMock(user_id="user-1")
    owner = MagicMock(email="owner@example.com")
    db = _charged_db_with_team(team_sub, plan, owner_membership=membership, owner_user=owner)
    event = _sub_event("subscription.charged")

    webhooks_svc.handle_subscription_charged(db, event)   # first delivery -> sends
    webhooks_svc.handle_subscription_charged(db, event)   # redelivered -> must not resend

    assert len(sent) == 1


def test_renewal_notice_skips_gracefully_when_no_owner_found(monkeypatch):
    sent = []
    monkeypatch.setattr(webhooks_svc, "send_email", lambda **kw: sent.append(kw))
    monkeypatch.setattr(webhooks_svc, "refill_subscription_credits", lambda *a, **k: None)
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "paid_count": 11},
    )
    plan = FakePlan()
    team_sub = MagicMock(team_id=str(TEAM_ID), subscription_id=str(SUB_ID),
                         credits_per_refill=350, last_paid_count=10, status="active")
    db = _charged_db_with_team(team_sub, plan, owner_membership=None, owner_user=None)

    # must not raise even though no TeamMember/User row exists
    webhooks_svc.handle_subscription_charged(db, _sub_event("subscription.charged"))

    assert sent == []
    assert team_sub.last_paid_count == 11    # the rest of the charge still processed


def test_renewal_notice_never_fires_for_yearly_plans(monkeypatch):
    """Yearly renewals never enter the `period_label != 'year'` branch at all
    (that's what gates BOTH the refill and the notice), so the notice can
    never fire for a yearly plan through this code path — confirmed safe (no
    crash, no misfire), though it does mean yearly owners get no 'ending soon'
    notice via this mechanism at all."""
    sent = []
    monkeypatch.setattr(webhooks_svc, "send_email", lambda **kw: sent.append(kw))
    refills = []
    monkeypatch.setattr(webhooks_svc, "refill_subscription_credits",
                        lambda *a, **k: refills.append(a))
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "paid_count": 1},   # yearly's only ever charge
    )
    plan = FakePlan()
    plan.period_label = "year"
    plan.total_count = 1
    team_sub = MagicMock(team_id=str(TEAM_ID), subscription_id=str(SUB_ID),
                         credits_per_refill=1000, last_paid_count=0, status="active")
    db = _charged_db_with_team(team_sub, plan)

    webhooks_svc.handle_subscription_charged(db, _sub_event("subscription.charged"))

    assert sent == []
    assert refills == []                  # yearly refills are the scheduler's job, not this webhook
    assert team_sub.last_paid_count == 1  # idempotency counter still advances correctly


def test_renewal_notice_smtp_failure_does_not_fail_the_whole_webhook(monkeypatch, caplog):
    """Real gap found live: send_email() previously wasn't wrapped here, so an
    SMTP hiccup would raise straight through handle_subscription_charged,
    surfacing as an unhandled 500 to Razorpay for a webhook that actually
    succeeded (last_paid_count already advanced by this point) -- and since
    Razorpay's retry would see paid_count <= last_paid_count on redelivery,
    the notice would never be attempted again. Must be caught, logged, and
    the rest of the charge (refill, last_paid_count, commit) must still go
    through."""
    monkeypatch.setattr(
        webhooks_svc, "send_email",
        MagicMock(side_effect=Exception("smtp connection refused")),
    )
    refilled = []
    monkeypatch.setattr(webhooks_svc, "refill_subscription_credits",
                         lambda *a, **k: refilled.append(a))
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch",
        lambda sid: {"id": sid, "paid_count": 11},   # plan.total_count(12) - 1
    )
    plan = FakePlan()
    team_sub = MagicMock(team_id=str(TEAM_ID), subscription_id=str(SUB_ID),
                         credits_per_refill=350, last_paid_count=10, status="active")
    membership = MagicMock(user_id="user-1")
    owner = MagicMock(id="user-1", email="owner@example.com")
    db = _charged_db_with_team(team_sub, plan, owner_membership=membership, owner_user=owner)

    with caplog.at_level(logging.ERROR):
        webhooks_svc.handle_subscription_charged(db, _sub_event("subscription.charged"))  # must not raise

    assert len(refilled) == 1                # the actual credit refill still happened
    assert team_sub.last_paid_count == 11     # idempotency counter still advanced
    db.commit.assert_called_once()            # the webhook still completes and commits
    assert "renewal-notice" in caplog.text.lower()


# --------------------------------------------------------------------------- #
# subscription.charged advances current_period_end
# --------------------------------------------------------------------------- #

def _charged_setup(monkeypatch, razor_sub, *, last_paid_count, status="active"):
    monkeypatch.setattr(webhooks_svc, "refill_subscription_credits", lambda *a, **k: None)
    monkeypatch.setattr(
        webhooks_svc.razorpay_client.subscription, "fetch", lambda sid: {"id": sid, **razor_sub},
    )
    old_end = datetime(2026, 1, 1, tzinfo=timezone.utc)
    team_sub = MagicMock(
        team_id=str(TEAM_ID), subscription_id=str(SUB_ID), status=status,
        credits_per_refill=350, last_paid_count=last_paid_count, current_period_end=old_end,
    )
    return team_sub, old_end


def test_charged_renewal_sets_period_end_to_razorpays_cycle_end(monkeypatch):
    cycle_end = int(datetime(2026, 8, 31, tzinfo=timezone.utc).timestamp())  # a 31-day month
    team_sub, _ = _charged_setup(monkeypatch, {"paid_count": 2, "current_end": cycle_end}, last_paid_count=1)

    webhooks_svc.handle_subscription_charged(_charged_db(team_sub, FakePlan()), _sub_event("subscription.charged"))

    assert team_sub.current_period_end == datetime.fromtimestamp(cycle_end, tz=timezone.utc)


def test_charged_renewal_falls_back_to_now_plus_period_without_current_end(monkeypatch):
    team_sub, old_end = _charged_setup(monkeypatch, {"paid_count": 2}, last_paid_count=1)
    before = datetime.now(timezone.utc)

    webhooks_svc.handle_subscription_charged(_charged_db(team_sub, FakePlan()), _sub_event("subscription.charged"))

    assert team_sub.current_period_end >= before + timedelta(days=30)
    assert team_sub.current_period_end > old_end


def test_charged_redelivery_does_not_push_period_end_out_again(monkeypatch):
    cycle_end = int(datetime(2026, 8, 31, tzinfo=timezone.utc).timestamp())
    team_sub, _ = _charged_setup(monkeypatch, {"paid_count": 2, "current_end": cycle_end}, last_paid_count=1)
    db = _charged_db(team_sub, FakePlan())

    webhooks_svc.handle_subscription_charged(db, _sub_event("subscription.charged"))
    first = team_sub.current_period_end
    webhooks_svc.handle_subscription_charged(db, _sub_event("subscription.charged"))  # redelivered

    assert team_sub.current_period_end == first


# --------------------------------------------------------------------------- #
# plans API: is_popular comes from the row, not from a plan name
# --------------------------------------------------------------------------- #

def test_subscription_out_exposes_is_popular_from_the_row():
    from types import SimpleNamespace
    from app.schemas.billing import SubscriptionOut

    def out(slug, popular):
        row = SimpleNamespace(
            id=uuid.uuid4(), slug=slug, name=slug, price=1, billing_period_days=30,
            period_label="month", credits=1, info=[], tag=None, is_popular=popular, sort_order=0,
        )
        return SubscriptionOut.model_validate(row).model_dump(by_alias=True)

    assert out("monthly", True)["isPopular"] is True
    assert out("monthly", False)["isPopular"] is False      # same slug, flag decides
