"""Filter-aware in-memory stand-ins for the billing tables.

The MagicMock fixtures elsewhere return the same row for ANY filter, which cannot
show that a webhook for one Razorpay id ignores a row holding a different id. The
pending-switch design depends on exactly that routing, so these fakes evaluate
the `column == value` / `column.in_([...])` clauses the code passes to .filter().
"""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock


def _matches(row, clause) -> bool:
    key = clause.left.key
    wanted = clause.right.value
    have = getattr(row, key)
    if isinstance(wanted, (list, tuple)):
        return have in wanted
    return str(have) == str(wanted) if have is not None else wanted is None


class _Query:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *clauses):
        return _Query([r for r in self._rows if all(_matches(r, c) for c in clauses)])

    def with_for_update(self, **_):
        return self

    def populate_existing(self):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


def make_team_sub(**overrides):
    base = dict(
        id=uuid.uuid4(),
        team_id=uuid.uuid4(),
        subscription_id=uuid.uuid4(),
        razorpay_subscription_id="sub_old",
        status="active",
        credits_per_refill=100,
        last_paid_count=3,
        next_refill_at=datetime.now(timezone.utc) + timedelta(days=2),
        current_period_end=datetime.now(timezone.utc) + timedelta(days=3),
        renewal_notice_sent_at=None,
        pending_subscription_id=None,
        pending_razorpay_subscription_id=None,
        pending_switch_id=None,
        pending_switch_expires_at=None,
        last_reconciled_at=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_plan(period_label, plan_id=None, credits=1200, razorpay_plan_id=None, slug=None, price=99900):
    return SimpleNamespace(
        id=plan_id or uuid.uuid4(),
        period_label=period_label,
        credits=credits,
        razorpay_plan_id=razorpay_plan_id or f"plan_{period_label}",
        slug=slug or period_label,
        price=price,
        total_count=None,
    )


def make_team(team_id, subscription_credits=40, topup=10):
    return SimpleNamespace(
        id=team_id,
        subscription_credits_remaining=subscription_credits,
        topup_credits_balance=topup,
    )


def make_db(*, team_subs=(), plans=(), teams=(), packs=()):
    """A Session stand-in whose query() honours .filter() clauses. Whatever the code
    passes to db.add() lands in the table named after its class, so `db.tables`
    (and ledger_entries(db)) show what would have been written."""
    tables = {
        "TeamSubscription": list(team_subs),
        "Subscription": list(plans),
        "Team": list(teams),
        "Credit": list(packs),
        "CreditLedger": [],
        "BillingTransaction": [],
    }
    db = MagicMock()

    def query(model):
        return _Query(tables.get(getattr(model, "__name__", ""), []))

    db.query.side_effect = query
    db.add.side_effect = lambda obj: tables.setdefault(type(obj).__name__, []).append(obj)
    db.tables = tables
    return db


def ledger_entries(db):
    return list(db.tables["CreditLedger"])


class FakeRazorpay:
    """Records every subscription call. `statuses` maps a Razorpay id to the status
    fetch() reports (default "created"); `fail_cancel` / `fail_create` /
    `fail_fetch` make those calls raise."""

    def __init__(self):
        self.created = []
        self.cancelled = []
        self.statuses = {}
        self.fail_create = False
        self.fail_fetch = False
        self.fail_cancel = set()      # ids whose cancel() raises
        self.entities = {}            # id -> extra fields returned by fetch()
        self._next = 1

    def create(self, payload):
        if self.fail_create:
            raise RuntimeError("razorpay create down")
        self.created.append(payload)
        new_id = f"sub_new_{self._next}"
        self._next += 1
        self.entities[new_id] = {"plan_id": payload["plan_id"], "notes": payload["notes"]}
        return {"id": new_id}

    def cancel(self, sid):
        self.cancelled.append(sid)
        if sid in self.fail_cancel:
            raise RuntimeError(f"razorpay cancel down for {sid}")
        self.statuses[sid] = "cancelled"
        return {}

    def fetch(self, sid):
        if self.fail_fetch:
            raise RuntimeError("razorpay fetch down")
        return {"id": sid, "status": self.statuses.get(sid, "created"), **self.entities.get(sid, {})}

    def install(self, monkeypatch, module):
        sub_api = module.razorpay_client.subscription
        monkeypatch.setattr(sub_api, "create", self.create)
        monkeypatch.setattr(sub_api, "cancel", self.cancel)
        monkeypatch.setattr(sub_api, "fetch", self.fetch)
        return self
