"""Hourly reconciliation of subscriptions against Razorpay.

Webhooks can be lost or fail. Left alone, that means a paid renewal whose credits
never arrive, a paid upgrade that never takes effect, or a status that never
changes. This job re-derives the truth from Razorpay and repairs the difference,
using the SAME code paths the webhooks use (apply_subscription_charge,
handle_subscription_activated), so a repair and a late webhook can never both
apply: they serialise on the subscription row lock and the paid_count / pending
switch guards make the second one a no-op.

Repairs, per subscription:
  * unpaid checkout that Razorpay says is `active`   -> the `activated` webhook was missed
  * `active` with paid_count beyond last_paid_count  -> a `charged` webhook was missed
  * terminal / halted / recovered Razorpay status    -> local status updated to match
  * current_end drifting from current_period_end     -> the period end corrected
  * a pending switch whose new subscription is paid  -> its `activated` was missed

Never repaired: anything when Razorpay cannot be reached (nothing changes, retried
next run), and no direction of repair ever takes credits away.
"""

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.razorpay_client import razorpay_client
from app.core.watchdog import send_alert
from app.models.subscription import Subscription
from app.models.team_subscription import TeamSubscription
from app.services import webhooks as webhooks_svc
from app.services.billing import is_unpaid_checkout

logger = logging.getLogger(__name__)

# A renewal is only "overdue" this long after current_period_end: Razorpay charges
# at the boundary and its webhook can lag, so this is the slack before we look.
OVERDUE_GRACE = timedelta(hours=2)
# Overdue rows and paid-but-unpromoted switches are re-checked at most this often.
RECHECK_INTERVAL = timedelta(hours=1)
# Every live subscription is compared at least this often even if nothing looks wrong.
AUDIT_INTERVAL = timedelta(hours=24)
# How old a pending switch must be before we ask Razorpay whether it was paid.
PENDING_SWITCH_CHECK_AFTER = timedelta(minutes=30)
# Ignore differences smaller than this between Razorpay's current_end and ours.
PERIOD_DRIFT_TOLERANCE = timedelta(hours=1)
# Bounded per run: each row costs a Razorpay call (or two).
BATCH_LIMIT = 100

_LIVE_STATUSES = ("active", "pending", "halted")
_ENDED_RAZORPAY_STATUSES = ("cancelled", "expired")


def find_reconcile_candidates(db: Session, now: datetime | None = None, limit: int = BATCH_LIMIT) -> list:
    """Ids of the subscriptions worth comparing with Razorpay right now, never-checked
    first and then oldest-checked first."""
    now = now or datetime.now(timezone.utc)
    # A pending switch stores expires_at = created + TTL, so "created more than
    # PENDING_SWITCH_CHECK_AFTER ago" is "expires_at <= now + TTL - that".
    switch_cutoff = now + timedelta(hours=settings.pending_switch_ttl_hours) - PENDING_SWITCH_CHECK_AFTER
    not_checked_lately = or_(
        TeamSubscription.last_reconciled_at.is_(None),
        TeamSubscription.last_reconciled_at <= now - RECHECK_INTERVAL,
    )

    rows = (
        db.query(TeamSubscription.id)
        .filter(
            TeamSubscription.razorpay_subscription_id.isnot(None),
            TeamSubscription.status.in_(_LIVE_STATUSES),
            or_(
                TeamSubscription.last_reconciled_at.is_(None),
                TeamSubscription.last_reconciled_at <= now - AUDIT_INTERVAL,
                and_(TeamSubscription.current_period_end <= now - OVERDUE_GRACE, not_checked_lately),
                and_(
                    TeamSubscription.pending_switch_expires_at.isnot(None),
                    TeamSubscription.pending_switch_expires_at <= switch_cutoff,
                    not_checked_lately,
                ),
            ),
        )
        .order_by(
            TeamSubscription.last_reconciled_at.is_(None).desc(),
            TeamSubscription.last_reconciled_at,
            TeamSubscription.current_period_end,
        )
        .limit(limit)
        .all()
    )
    return [row_id for (row_id,) in rows]


def _activated_event(razorpay_subscription_id: str) -> dict:
    return {"payload": {"subscription": {"entity": {"id": razorpay_subscription_id}}}}


def _as_datetime(unix_ts) -> datetime | None:
    if isinstance(unix_ts, int) and unix_ts > 0:
        return datetime.fromtimestamp(unix_ts, tz=timezone.utc)
    return None


def reconcile_subscription(db: Session, team_subscription_id, now: datetime | None = None) -> dict:
    """Compare one subscription with Razorpay and repair any difference.

    Returns {"outcome": "ok" | "repaired" | "skipped" | "error", "repairs": [...]}.
    """
    now = now or datetime.now(timezone.utc)
    row = (
        db.query(TeamSubscription)
        .filter(TeamSubscription.id == team_subscription_id)
        .populate_existing()
        .with_for_update(skip_locked=True)
        .first()
    )
    if row is None or not row.razorpay_subscription_id or row.status not in _LIVE_STATUSES:
        db.rollback()
        return {"outcome": "skipped", "repairs": []}

    razorpay_id = row.razorpay_subscription_id
    team_id = row.team_id
    local_status = row.status
    try:
        remote = razorpay_client.subscription.fetch(razorpay_id)
    except Exception:
        db.rollback()
        logger.warning("reconcile: could not fetch %s -- leaving it for the next run", razorpay_id, exc_info=True)
        return {"outcome": "error", "repairs": []}

    remote_status = remote.get("status")
    repairs: list[str] = []

    try:
        if is_unpaid_checkout(row):
            if remote_status == "active":
                # Paid, but our `activated` never landed: run the webhook's own handler.
                repairs.append("missed_activation")
                webhooks_svc.handle_subscription_activated(db, _activated_event(razorpay_id))
        else:
            plan = db.query(Subscription).filter(Subscription.id == row.subscription_id).first()
            paid_count = remote.get("paid_count", 0)

            if remote_status in _ENDED_RAZORPAY_STATUSES:
                row.status = "cancelled"
                repairs.append(f"status:{local_status}->cancelled")
            elif remote_status == "completed" and local_status == "active":
                row.status = "cancelled"          # mirrors handle_subscription_completed
                repairs.append("status:active->cancelled(completed)")
            elif remote_status == "halted" and local_status != "halted":
                row.status = "halted"
                repairs.append(f"status:{local_status}->halted")
            elif remote_status == "active":
                if plan is not None and isinstance(paid_count, int) and paid_count > row.last_paid_count:
                    previous = row.last_paid_count
                    webhooks_svc.apply_subscription_charge(db, row, plan, remote, source="reconcile")
                    repairs.append(f"missed_charge:paid_count {previous}->{paid_count}")
                else:
                    if local_status != "active":
                        row.status = "active"
                        repairs.append(f"status:{local_status}->active")
                    remote_end = _as_datetime(remote.get("current_end"))
                    if remote_end and abs(remote_end - row.current_period_end) > PERIOD_DRIFT_TOLERANCE:
                        repairs.append(f"period_end:{row.current_period_end.isoformat()}->{remote_end.isoformat()}")
                        row.current_period_end = remote_end
    except Exception:
        db.rollback()
        logger.exception("reconcile: repair of %s failed -- rolled back, retrying next run", razorpay_id)
        return {"outcome": "error", "repairs": []}

    # A paid upgrade whose `activated` never arrived. The handler is idempotent and
    # routes on the pending column, exactly like the webhook.
    pending_id = row.pending_razorpay_subscription_id
    if pending_id:
        try:
            if razorpay_client.subscription.fetch(pending_id).get("status") == "active":
                repairs.append("missed_switch_activation")
                webhooks_svc.handle_subscription_activated(db, _activated_event(pending_id))
        except Exception:
            db.rollback()
            logger.exception("reconcile: could not check pending switch %s -- retrying next run", pending_id)
            return {"outcome": "error", "repairs": []}

    row.last_reconciled_at = now
    db.commit()

    if repairs:
        logger.error(
            "reconciliation repaired a subscription that webhooks missed: team_id=%s "
            "razorpay_subscription_id=%s repairs=%s local_status=%s razorpay_status=%s",
            team_id, razorpay_id, repairs, local_status, remote_status,
        )
    return {"outcome": "repaired" if repairs else "ok", "repairs": repairs}


def reconcile_due_subscriptions(db: Session, now: datetime | None = None, limit: int = BATCH_LIMIT) -> dict:
    """One hourly run. One subscription's failure never stops the rest."""
    now = now or datetime.now(timezone.utc)
    ids = find_reconcile_candidates(db, now, limit)
    db.rollback()  # end the read transaction before the per-row locking work

    summary = {"checked": 0, "repaired": 0, "errors": 0, "skipped": 0}
    for row_id in ids:
        try:
            result = reconcile_subscription(db, row_id, now)
        except Exception:
            db.rollback()
            logger.exception("reconcile: unexpected failure for subscription row %s", row_id)
            summary["errors"] += 1
            continue
        outcome = result["outcome"]
        if outcome == "error":
            summary["errors"] += 1
        elif outcome == "skipped":
            summary["skipped"] += 1
        else:
            summary["checked"] += 1
            if outcome == "repaired":
                summary["repaired"] += 1

    logger.info("reconcile run finished: %s", summary)
    if summary["repaired"]:
        # A repair means at least one webhook was lost or failed: worth a human's eyes.
        send_alert(
            f"RECONCILE_REPAIRED: {summary['repaired']} subscription(s) were out of sync with Razorpay "
            f"and were repaired ({summary}). Check webhook delivery in the Razorpay dashboard."
        )
    return summary
