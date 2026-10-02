"""Credit ledger, phase 1: money-in only.

Every function here runs inside the CALLER's transaction and never commits, so a
ledger entry always commits or rolls back together with the balance change it
describes. See app/models/credit_ledger.py.
"""

import base64
import json
import logging
from datetime import datetime
from uuid import UUID

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app.models.credit_ledger import CreditLedger
from app.models.team import Team

logger = logging.getLogger(__name__)

SUBSCRIPTION_GRANT = "subscription_grant"
SWITCH_TRANSFER = "switch_transfer"
TOPUP_PURCHASE = "topup_purchase"

MONEY_IN_ENTRY_TYPES = (SUBSCRIPTION_GRANT, SWITCH_TRANSFER, TOPUP_PURCHASE)


def record_credit_entry(
    db: Session,
    *,
    team_id,
    pool: str,
    entry_type: str,
    amount: int,
    balance_after: int | None,
    idempotency_key: str,
    source: str,
    razorpay_subscription_id: str | None = None,
    razorpay_payment_id: str | None = None,
    switch_id=None,
    metadata: dict | None = None,
) -> bool:
    """Append one entry unless `idempotency_key` already has one. Returns True if
    it was added.

    The check-then-insert is safe because every caller already holds the team's
    subscription row lock (or, for top-ups, the unique payment-id claim); the
    UNIQUE constraint on idempotency_key is the backstop that turns any race
    that slips through into a failed transaction instead of a duplicate entry.
    """
    if db.query(CreditLedger).filter(CreditLedger.idempotency_key == idempotency_key).first() is not None:
        return False
    db.add(CreditLedger(
        team_id=team_id,
        pool=pool,
        entry_type=entry_type,
        amount=amount,
        balance_after=balance_after,
        idempotency_key=idempotency_key,
        source=source,
        razorpay_subscription_id=razorpay_subscription_id,
        razorpay_payment_id=razorpay_payment_id,
        switch_id=switch_id,
        entry_metadata=metadata or {},
    ))
    return True


def subscription_pool_balance(db: Session, team_id) -> int | None:
    """The team's current subscription-pool balance, read under the team row lock
    (the refill that follows takes the same lock), or None if it can't be read."""
    team = db.query(Team).filter(Team.id == team_id).with_for_update().first()
    value = getattr(team, "subscription_credits_remaining", None)
    return value if isinstance(value, int) else None


def record_subscription_grant(
    db: Session,
    *,
    team_id,
    balance_before: int | None,
    balance_after: int,
    idempotency_key: str,
    source: str,
    razorpay_subscription_id: str | None = None,
    metadata: dict | None = None,
) -> bool:
    """A subscription refill REPLACES the pool, so the entry's amount is the change
    it made (new - old): negative if unused credits lapsed past the new grant."""
    if balance_before is None:
        return False
    return record_credit_entry(
        db,
        team_id=team_id,
        pool="subscription",
        entry_type=SUBSCRIPTION_GRANT,
        amount=balance_after - balance_before,
        balance_after=balance_after,
        idempotency_key=idempotency_key,
        source=source,
        razorpay_subscription_id=razorpay_subscription_id,
        metadata=metadata,
    )


def plan_metadata(plan, cycle=None, charged: bool = False) -> dict:
    """Small, JSON-safe description of the plan a grant belongs to. `price` is
    only included for a grant that came from an actual charge."""
    meta = {}
    slug = getattr(plan, "slug", None)
    if isinstance(slug, str):
        meta["plan"] = slug
    period = getattr(plan, "period_label", None)
    if isinstance(period, str):
        meta["period"] = period
    if isinstance(cycle, int):
        meta["cycle"] = cycle
    price = getattr(plan, "price", None)
    if charged and isinstance(price, int):
        meta["price"] = price
    return meta


# --------------------------------------------------------------------------- #
# billing history (read side)
# --------------------------------------------------------------------------- #

DEFAULT_HISTORY_PAGE_SIZE = 20
MAX_HISTORY_PAGE_SIZE = 50


def _encode_cursor(created_at: datetime, entry_id) -> str:
    raw = json.dumps({"t": created_at.isoformat(), "i": str(entry_id)}).encode()
    return base64.urlsafe_b64encode(raw).decode()


def _decode_cursor(cursor: str) -> tuple[datetime, UUID]:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        return datetime.fromisoformat(payload["t"]), UUID(payload["i"])
    except Exception as exc:
        raise ValueError("Invalid cursor") from exc


def list_billing_history(db: Session, team_id, limit: int = DEFAULT_HISTORY_PAGE_SIZE, cursor: str | None = None) -> dict:
    """A team's money-in history, newest first, as a keyset-paginated page.

    Keyset (created_at, id), not OFFSET: a new entry landing between two page
    requests can't shift rows across the page boundary. `id` breaks ties -- the
    entries of one plan switch are written in a single transaction and share a
    timestamp. Raises ValueError for a malformed cursor.
    """
    limit = max(1, min(limit, MAX_HISTORY_PAGE_SIZE))
    query = db.query(CreditLedger).filter(
        CreditLedger.team_id == team_id,
        CreditLedger.entry_type.in_(MONEY_IN_ENTRY_TYPES),
    )
    if cursor:
        cursor_time, cursor_id = _decode_cursor(cursor)
        query = query.filter(or_(
            CreditLedger.created_at < cursor_time,
            and_(CreditLedger.created_at == cursor_time, CreditLedger.id < cursor_id),
        ))

    rows = query.order_by(CreditLedger.created_at.desc(), CreditLedger.id.desc()).limit(limit + 1).all()
    page, has_more = rows[:limit], len(rows) > limit

    return {
        "items": [_history_item(row) for row in page],
        "next_cursor": _encode_cursor(page[-1].created_at, page[-1].id) if has_more else None,
    }


def _history_item(row: CreditLedger) -> dict:
    meta = row.entry_metadata or {}
    # `amount` is what a top-up cost, `price` what a subscription charge cost (both
    # in Razorpay's smallest currency unit); free slices and transfers have neither.
    paid = meta.get("amount") if row.entry_type == TOPUP_PURCHASE else meta.get("price")
    return {
        "id": str(row.id),
        "type": row.entry_type,
        "pool": row.pool,
        "credits": row.amount,
        "balance_after": row.balance_after,
        "plan": meta.get("plan"),
        "period": meta.get("period"),
        "cycle": meta.get("cycle"),
        "amount_paid": paid if isinstance(paid, int) else None,
        # the payment id of a top-up, for support conversations; subscription ids stay internal
        "reference": row.razorpay_payment_id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
