"""Credit ledger, phase 1 (money-in): what gets recorded, that it is idempotent, and
that it always lands in the same transaction as the balance change it describes.
"""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services import ledger as ledger_svc
from app.services import webhooks as webhooks_svc
from tests.billing_fakes import (
    FakeRazorpay, ledger_entries, make_db, make_plan, make_team, make_team_sub,
)

TEAM_ID = uuid.uuid4()


def _add(db, **overrides):
    args = dict(
        team_id=TEAM_ID, pool="topup", entry_type=ledger_svc.TOPUP_PURCHASE, amount=100,
        balance_after=100, idempotency_key="payment:pay_1", source="webhook",
    )
    args.update(overrides)
    return ledger_svc.record_credit_entry(db, **args)


# --------------------------------------------------------------------------- #
# the primitive
# --------------------------------------------------------------------------- #

def test_an_entry_is_added_once_per_idempotency_key():
    db = make_db()

    assert _add(db) is True
    assert _add(db) is False                                  # same key: ignored
    assert _add(db, idempotency_key="payment:pay_2") is True  # different key: recorded

    assert [e.idempotency_key for e in ledger_entries(db)] == ["payment:pay_1", "payment:pay_2"]


def test_recording_never_commits_or_flushes_on_its_own():
    db = make_db()
    _add(db)
    db.commit.assert_not_called()
    db.rollback.assert_not_called()


def test_entries_carry_every_field_and_metadata():
    db = make_db()
    switch_id = uuid.uuid4()
    _add(db, pool="subscription", entry_type=ledger_svc.SWITCH_TRANSFER, amount=-40, balance_after=0,
         razorpay_subscription_id="sub_1", razorpay_payment_id="pay_9", switch_id=switch_id,
         metadata={"k": "v"}, source="reconcile")

    (e,) = ledger_entries(db)
    assert (e.team_id, e.pool, e.entry_type, e.amount, e.balance_after) == (
        TEAM_ID, "subscription", "switch_transfer", -40, 0)
    assert (e.razorpay_subscription_id, e.razorpay_payment_id, e.switch_id, e.source) == (
        "sub_1", "pay_9", switch_id, "reconcile")
    assert e.entry_metadata == {"k": "v"}


def test_a_grant_records_the_change_the_replacing_refill_made():
    db = make_db()
    ledger_svc.record_subscription_grant(
        db, team_id=TEAM_ID, balance_before=40, balance_after=350,
        idempotency_key="sub:s:cycle:2", source="webhook")
    ledger_svc.record_subscription_grant(       # unused credits lapsed past a smaller grant
        db, team_id=TEAM_ID, balance_before=500, balance_after=350,
        idempotency_key="sub:s:cycle:3", source="webhook")

    assert [(e.amount, e.balance_after) for e in ledger_entries(db)] == [(310, 350), (-150, 350)]


def test_a_grant_is_skipped_when_the_old_balance_could_not_be_read():
    db = make_db()
    assert ledger_svc.record_subscription_grant(
        db, team_id=TEAM_ID, balance_before=None, balance_after=350,
        idempotency_key="k", source="webhook") is False
    assert ledger_entries(db) == []


def test_plan_metadata_is_json_safe_and_only_carries_price_for_a_real_charge():
    plan = make_plan("month", slug="monthly", price=99900)
    assert ledger_svc.plan_metadata(plan, cycle=2, charged=True) == {
        "plan": "monthly", "period": "month", "cycle": 2, "price": 99900}
    assert "price" not in ledger_svc.plan_metadata(plan, cycle=2)
    assert ledger_svc.plan_metadata(MagicMock()) == {}          # never leaks non-JSON values


# --------------------------------------------------------------------------- #
# activation
# --------------------------------------------------------------------------- #

def _activation_world(monkeypatch, credits=350, leftover=0):
    plan = make_plan("month", slug="monthly", credits=credits, price=99900)
    team = make_team(TEAM_ID, subscription_credits=leftover, topup=10)
    row = make_team_sub(
        team_id=TEAM_ID, subscription_id=plan.id, razorpay_subscription_id="sub_a",
        status="pending", credits_per_refill=0, last_paid_count=0,
    )
    db = make_db(team_subs=[row], plans=[plan], teams=[team])
    rzp = FakeRazorpay().install(monkeypatch, webhooks_svc)
    rzp.entities["sub_a"] = {
        "plan_id": plan.razorpay_plan_id,
        "notes": {"team_id": str(TEAM_ID), "subscription_id": str(plan.id)},
    }
    return SimpleNamespace(plan=plan, team=team, row=row, db=db, rzp=rzp)


def _activate(w, rzp_id="sub_a"):
    webhooks_svc.handle_subscription_activated(
        w.db, {"payload": {"subscription": {"entity": {"id": rzp_id}}}})


def test_activation_records_the_first_grant_in_the_same_commit(monkeypatch):
    w = _activation_world(monkeypatch, leftover=12)
    at_commit = []
    w.db.commit.side_effect = lambda: at_commit.append(len(ledger_entries(w.db)))

    _activate(w)

    (e,) = ledger_entries(w.db)
    assert e.idempotency_key == "sub:sub_a:cycle:1" and e.source == "webhook"
    assert (e.pool, e.entry_type, e.amount, e.balance_after) == ("subscription", "subscription_grant", 350 - 12, 350)
    assert e.entry_metadata["plan"] == "monthly" and e.entry_metadata["price"] == 99900
    assert w.team.subscription_credits_remaining == 350
    assert at_commit and all(n == 1 for n in at_commit)       # entry existed at EVERY commit: same transaction


def test_a_redelivered_activation_records_nothing_more(monkeypatch):
    w = _activation_world(monkeypatch)
    _activate(w)
    _activate(w)
    assert len(ledger_entries(w.db)) == 1


# --------------------------------------------------------------------------- #
# renewals (the `charged` webhook)
# --------------------------------------------------------------------------- #

def _charged(w, paid_count, current_end=None):
    w.rzp.entities["sub_a"] = {**w.rzp.entities.get("sub_a", {}), "paid_count": paid_count,
                               **({"current_end": current_end} if current_end else {})}
    webhooks_svc.handle_subscription_charged(
        w.db, {"payload": {"subscription": {"entity": {"id": "sub_a"}}}})


def test_a_monthly_renewal_is_recorded_once_even_if_the_webhook_repeats(monkeypatch):
    monkeypatch.setattr(webhooks_svc, "send_renewal_notice_email", lambda *a, **k: None)
    w = _activation_world(monkeypatch)
    _activate(w)
    w.team.subscription_credits_remaining = 100                # spent some of the first grant

    _charged(w, 2)
    _charged(w, 2)                                             # redelivery

    keys = [e.idempotency_key for e in ledger_entries(w.db)]
    assert keys == ["sub:sub_a:cycle:1", "sub:sub_a:cycle:2"]
    renewal = ledger_entries(w.db)[1]
    assert (renewal.amount, renewal.balance_after, renewal.source) == (250, 350, "webhook")


def test_the_initial_charge_is_not_a_second_grant(monkeypatch):
    w = _activation_world(monkeypatch)
    _activate(w)
    _charged(w, 1)                                             # fires alongside activation
    assert [e.idempotency_key for e in ledger_entries(w.db)] == ["sub:sub_a:cycle:1"]


def test_a_yearly_charge_records_no_grant_because_credits_come_monthly_from_the_worker(monkeypatch):
    w = _activation_world(monkeypatch)
    w.plan.period_label = "year"
    w.plan.credits = 1200
    _activate(w)
    before = len(ledger_entries(w.db))

    _charged(w, 2)

    assert len(ledger_entries(w.db)) == before


# --------------------------------------------------------------------------- #
# plan switch
# --------------------------------------------------------------------------- #

def _switch_world(monkeypatch, leftover):
    old_plan = make_plan("week", slug="weekly", credits=100)
    new_plan = make_plan("year", slug="yearly", credits=1200, price=999000)
    team = make_team(TEAM_ID, subscription_credits=leftover, topup=10)
    switch_id = uuid.uuid4()
    row = make_team_sub(
        team_id=TEAM_ID, subscription_id=old_plan.id, razorpay_subscription_id="sub_old",
        pending_subscription_id=new_plan.id, pending_razorpay_subscription_id="sub_new",
        pending_switch_id=switch_id, pending_switch_expires_at=datetime.now(timezone.utc) + timedelta(hours=5),
    )
    db = make_db(team_subs=[row], plans=[old_plan, new_plan], teams=[team])
    rzp = FakeRazorpay().install(monkeypatch, webhooks_svc)
    rzp.entities["sub_new"] = {
        "plan_id": new_plan.razorpay_plan_id,
        "notes": {"team_id": str(TEAM_ID), "subscription_id": str(new_plan.id), "switch_id": str(switch_id)},
    }
    return SimpleNamespace(team=team, row=row, db=db, rzp=rzp, switch_id=switch_id)


def test_a_switch_records_the_transfer_out_the_transfer_in_and_the_new_grant(monkeypatch):
    w = _switch_world(monkeypatch, leftover=40)
    at_commit = []
    w.db.commit.side_effect = lambda: at_commit.append(len(ledger_entries(w.db)))

    webhooks_svc.handle_subscription_activated(
        w.db, {"payload": {"subscription": {"entity": {"id": "sub_new"}}}})

    rows = {e.idempotency_key: e for e in ledger_entries(w.db)}
    assert set(rows) == {f"switch:{w.switch_id}:out", f"switch:{w.switch_id}:in", "sub:sub_new:cycle:1"}
    out, into, grant = (rows[f"switch:{w.switch_id}:out"], rows[f"switch:{w.switch_id}:in"], rows["sub:sub_new:cycle:1"])
    assert (out.pool, out.entry_type, out.amount, out.balance_after) == ("subscription", "switch_transfer", -40, 0)
    assert (into.pool, into.entry_type, into.amount, into.balance_after) == ("topup", "switch_transfer", 40, 50)
    assert (grant.pool, grant.entry_type, grant.amount, grant.balance_after) == (
        "subscription", "subscription_grant", 1200 // 12, 1200 // 12)
    assert all(e.switch_id == w.switch_id for e in (out, into, grant))
    assert at_commit == [3]                                   # all three, in the one commit


def test_a_switch_with_nothing_left_over_records_no_transfer(monkeypatch):
    w = _switch_world(monkeypatch, leftover=0)

    webhooks_svc.handle_subscription_activated(
        w.db, {"payload": {"subscription": {"entity": {"id": "sub_new"}}}})

    assert [e.entry_type for e in ledger_entries(w.db)] == ["subscription_grant"]


def test_a_redelivered_switch_activation_records_nothing_more(monkeypatch):
    w = _switch_world(monkeypatch, leftover=40)
    event = {"payload": {"subscription": {"entity": {"id": "sub_new"}}}}
    webhooks_svc.handle_subscription_activated(w.db, event)
    count = len(ledger_entries(w.db))

    webhooks_svc.handle_subscription_activated(w.db, event)

    assert len(ledger_entries(w.db)) == count


# --------------------------------------------------------------------------- #
# top-ups
# --------------------------------------------------------------------------- #

def test_a_paid_topup_is_recorded_with_the_amount_paid(monkeypatch):
    pack = SimpleNamespace(id=uuid.uuid4(), price=50000, credits=500, slug="pack-500")
    team = make_team(TEAM_ID, subscription_credits=0, topup=10)
    team.deleted_at = None
    db = make_db(teams=[team], packs=[pack])
    monkeypatch.setattr(webhooks_svc.razorpay_client.order, "fetch", lambda oid: {
        "amount": 50000, "notes": {"team_id": str(TEAM_ID), "credit_pack_id": str(pack.id)}})
    event = {"payload": {"payment": {"entity": {
        "id": "pay_77", "order_id": "order_1", "amount": 50000, "notes": {}}}}}
    at_commit = []
    db.commit.side_effect = lambda: at_commit.append(len(ledger_entries(db)))

    webhooks_svc.handle_payment_captured(db, event)

    assert team.topup_credits_balance == 510
    (e,) = ledger_entries(db)
    assert e.idempotency_key == "payment:pay_77"
    assert (e.pool, e.entry_type, e.amount, e.balance_after) == ("topup", "topup_purchase", 500, 510)
    assert e.razorpay_payment_id == "pay_77"
    assert e.entry_metadata == {"amount": 50000, "pack": "pack-500"}
    assert at_commit == [1]


def test_a_redelivered_payment_webhook_records_nothing_more(monkeypatch):
    pack = SimpleNamespace(id=uuid.uuid4(), price=50000, credits=500, slug="pack-500")
    team = make_team(TEAM_ID, subscription_credits=0, topup=0)
    team.deleted_at = None
    db = make_db(teams=[team], packs=[pack])
    monkeypatch.setattr(webhooks_svc.razorpay_client.order, "fetch", lambda oid: {
        "amount": 50000, "notes": {"team_id": str(TEAM_ID), "credit_pack_id": str(pack.id)}})
    event = {"payload": {"payment": {"entity": {
        "id": "pay_77", "order_id": "order_1", "amount": 50000, "notes": {}}}}}

    webhooks_svc.handle_payment_captured(db, event)
    webhooks_svc.handle_payment_captured(db, event)         # the BillingTransaction claim short-circuits it

    assert len(ledger_entries(db)) == 1
    assert team.topup_credits_balance == 500


def test_a_held_payment_for_a_deleted_team_records_no_credits(monkeypatch):
    pack = SimpleNamespace(id=uuid.uuid4(), price=50000, credits=500, slug="pack-500")
    team = make_team(TEAM_ID, subscription_credits=0, topup=0)
    team.deleted_at = datetime.now(timezone.utc)
    db = make_db(teams=[team], packs=[pack])
    monkeypatch.setattr(webhooks_svc.razorpay_client.order, "fetch", lambda oid: {
        "amount": 50000, "notes": {"team_id": str(TEAM_ID), "credit_pack_id": str(pack.id)}})
    event = {"payload": {"payment": {"entity": {
        "id": "pay_78", "order_id": "order_1", "amount": 50000, "notes": {}}}}}

    webhooks_svc.handle_payment_captured(db, event)

    assert ledger_entries(db) == []                          # nothing came in
    assert team.topup_credits_balance == 0


# --------------------------------------------------------------------------- #
# the scheduled monthly slices of a yearly plan
# --------------------------------------------------------------------------- #

def test_a_worker_refill_records_its_slice_keyed_by_the_due_time(monkeypatch):
    from app import worker

    due = datetime(2026, 9, 1, tzinfo=timezone.utc)
    row_id = uuid.uuid4()
    ts = MagicMock(id=row_id, team_id=TEAM_ID, subscription_id="s", status="active",
                   credits_per_refill=100, next_refill_at=due, razorpay_subscription_id="sub_y")
    plan = make_plan("year", slug="yearly", credits=1200)
    team = make_team(TEAM_ID, subscription_credits=30, topup=0)

    db = MagicMock()
    seen = {"ts": 0}

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "TeamSubscription":
            seen["ts"] += 1
            if seen["ts"] == 1:
                q.filter.return_value.all.return_value = [ts]
            else:
                q.filter.return_value.populate_existing.return_value.with_for_update.return_value.first.return_value = ts
            q.join.return_value.filter.return_value.all.return_value = []
        elif name == "Subscription":
            q.filter.return_value.first.return_value = plan
        elif name == "Team":
            q.filter.return_value.with_for_update.return_value.first.return_value = team
        elif name == "CreditLedger":
            q.filter.return_value.first.return_value = None
        return q

    db.query.side_effect = query
    monkeypatch.setattr(worker, "SessionLocal", lambda: db)
    order = []
    db.add.side_effect = lambda obj: order.append(("ledger", obj))
    monkeypatch.setattr(worker, "refill_subscription_credits", lambda *a, **k: order.append(("refill", a)))
    import asyncio

    asyncio.run(worker.refill_due_subscriptions({}))

    assert [kind for kind, _ in order] == ["ledger", "refill"]     # entry staged before the refill
    entry = order[0][1]
    assert entry.idempotency_key == f"refill:{row_id}:{due.isoformat()}"
    assert (entry.source, entry.amount, entry.balance_after) == ("worker", 70, 100)
    assert entry.razorpay_subscription_id == "sub_y"
