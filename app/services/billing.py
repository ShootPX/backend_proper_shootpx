import logging
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.core.razorpay_client import razorpay_client
from app.models.credit import Credit
from app.models.subscription import Subscription
from app.models.team_subscription import TeamSubscription
from app.core.config import settings
from app.models.team import Team

logger = logging.getLogger(__name__)


class RazorpayCancelError(RuntimeError):
    """Razorpay rejected or failed the subscription-cancel call. Distinct from
    ValueError so callers can tell 'nothing to cancel' apart from 'the provider
    call failed' (delete_team logs-and-continues on the latter)."""


# Tier rank per billing period (plan `period_label`). A higher rank is a bigger
# commitment; switch_subscription only allows moving UP while a paid period is
# still running.
PLAN_TIER_RANK = {"week": 1, "month": 2, "year": 3}


class PlanSwitchBlocked(ValueError):
    """A switch to a same/lower-tier plan was refused because the current paid
    period has not ended. Carries a machine-readable `code` and the date the
    period ends so the client can offer 'switch after <date>'."""

    def __init__(self, code: str, message: str, current_period_end: datetime):
        super().__init__(message)
        self.code = code
        self.current_period_end = current_period_end


def is_unpaid_checkout(team_sub) -> bool:
    """
    True for a subscription row that exists only because a checkout was STARTED
    and never paid for: status "pending" with credits_per_refill still 0
    (activation is the only thing that sets it above 0 -- see
    webhooks.handle_subscription_activated).

    "pending" is deliberately not enough on its own, because it also means
    something completely different: handle_subscription_pending marks a
    genuinely ACTIVE, paying subscription "pending" when a renewal payment
    fails and Razorpay retries for days. Treating that customer's row as an
    abandoned checkout would let it be replaced or, in the old cleanup cron,
    deleted outright.
    """
    return team_sub is not None and team_sub.status == "pending" and (team_sub.credits_per_refill or 0) == 0


def _cancel_abandoned_razorpay_subscription(razorpay_subscription_id: str) -> None:
    """Best-effort. A never-paid ("created") Razorpay subscription can never
    charge anyone, so failing to cancel it must not block the user's retry --
    it is logged and left to expire."""
    try:
        razorpay_client.subscription.cancel(razorpay_subscription_id)
    except Exception:
        logger.warning(
            "could not cancel abandoned Razorpay subscription %s -- harmless (never paid), left as is",
            razorpay_subscription_id, exc_info=True,
        )




def clear_pending_switch(team_sub) -> str | None:
    """Clear a pending plan switch off `team_sub` and return the unpaid
    replacement's Razorpay id (or None) so the caller can cancel it AFTER its own
    commit -- an external call must never run before the local state that stops
    referencing it is durable."""
    pending_id = team_sub.pending_razorpay_subscription_id
    team_sub.pending_subscription_id = None
    team_sub.pending_razorpay_subscription_id = None
    team_sub.pending_switch_id = None
    team_sub.pending_switch_expires_at = None
    return pending_id


def create_credit_pack_checkout(db: Session, team_id, pack_id) -> dict:
    # The charge amount and the credit count are ALWAYS taken from the catalog row
    # here — never from anything the caller passes in. The checkout route accepts
    # no request body at all; `pack_id` is the only client input and it is just a
    # lookup key.
    pack = db.query(Credit).filter(Credit.id == pack_id).first()
    if not pack:
        raise ValueError("Credit pack not found")

    order = razorpay_client.order.create({
        "amount": pack.price,       # already in paise, from the DB
        "currency": "INR",
        "notes": {
            # Razorpay note values must be strings. `team_id` / `credit_pack_id`
            # are what the webhook uses to identify the purchase; `credits` is
            # informational only (the webhook re-reads it from the DB, never here).
            "team_id": str(team_id),
            "credit_pack_id": str(pack_id),
            "credits": str(pack.credits),
        },
    })

    return {
        "order_id": order["id"],
        "amount": order["amount"],
        "currency": order["currency"],
        "key_id": settings.razorpay_key_id,
    }

def _razorpay_total_count(plan) -> int:
    try:
        return {"week": 52, "month": 12, "year": 1}[plan.period_label]
    except KeyError:
        raise ValueError(f"Unsupported billing period: {plan.period_label!r}")


def create_subscription_checkout(db: Session, team_id, subscription_id) -> dict:
    plan = db.query(Subscription).filter(Subscription.id == subscription_id).first()
    if not plan:
        raise ValueError("Plan not found")
    if not plan.razorpay_plan_id:
        raise ValueError("This plan is not configured for payment yet")

    total_count = _razorpay_total_count(plan)

    now = datetime.now(timezone.utc)

    # Lock this team's subscription row (if it has one) for the whole checkout so
    # two rapid "Subscribe" clicks are serialised: the second waits here, then
    # sees the pending/active row the first one wrote and is rejected.
    row = (
        db.query(TeamSubscription)
        .filter(TeamSubscription.team_id == team_id)
        .with_for_update()
        .first()
    )
    replaced_razorpay_id = None
    if row is not None and row.status in ("active", "pending"):
        if not is_unpaid_checkout(row):
            # a real subscription: active, or paying but mid renewal-retry
            raise ValueError("This team already has an active subscription")

        # An abandoned, never-paid checkout (the user closed Razorpay). This
        # must NOT block them -- it was never a subscription.
        if row.subscription_id == subscription_id and row.razorpay_subscription_id:
            # Same plan again ("Retry payment", or a double click): hand back
            # the SAME Razorpay subscription if it is still payable, instead of
            # creating a second one and cancelling the first out from under a
            # checkout window the user may have open.
            try:
                existing = razorpay_client.subscription.fetch(row.razorpay_subscription_id)
            except Exception:
                existing = None
            if existing and existing.get("status") == "created":
                return {
                    "razorpay_subscription_id": row.razorpay_subscription_id,
                    "key_id": settings.razorpay_key_id,
                }
        # Different plan, or the old one is no longer payable: replace it.
        replaced_razorpay_id = row.razorpay_subscription_id

    # Claim a `pending` row BEFORE calling Razorpay — no external subscription is
    # created until this claim is secured. A brand-new team has no row, so two
    # concurrent requests race to INSERT and UNIQUE(team_id) rejects the loser; a
    # team resubscribing after cancel/halt reuses its existing row (a fresh
    # lifecycle) and the FOR UPDATE lock above serialises the requests.
    if row is None:
        row = TeamSubscription(team_id=team_id)
        db.add(row)
    row.subscription_id = subscription_id
    row.razorpay_subscription_id = None
    row.status = "pending"
    row.credits_per_refill = 0
    row.next_refill_at = now
    row.current_period_end = now
    # A fresh lifecycle needs its own renewal notice -- without resetting
    # this, a team resubscribing after a yearly plan completed (see
    # worker.py's send_yearly_renewal_notices) would carry over the OLD
    # cycle's sent_at and never get warned before the new cycle also ends.
    row.renewal_notice_sent_at = None
    # A pending switch belongs to the lifecycle being replaced (e.g. the old plan
    # ended naturally while an upgrade was unpaid). Left in place, its late
    # `activated` webhook would promote onto this brand-new row.
    stale_pending_id = clear_pending_switch(row)

    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise ValueError("This team already has an active subscription")

    # If Razorpay itself fails, roll the claim back so a stuck `pending` row never
    # blocks future checkout attempts.
    try:
        razor_sub = razorpay_client.subscription.create({
            "plan_id": plan.razorpay_plan_id,
            "total_count": total_count,
            "notes": {
                "team_id": str(team_id),
                "subscription_id": str(subscription_id),
            },
        })
    except Exception:
        db.rollback()
        raise

    # Store the real id NOW (not None) so a cancel during the pending window can
    # actually reach Razorpay, and so a cancelled row can't be resurrected by a
    # late subscription.activated webhook.
    row.razorpay_subscription_id = razor_sub["id"]
    db.commit()

    # Only AFTER the replacement is committed, so a failure creating it leaves
    # the user's previous (still payable) attempt untouched.
    if replaced_razorpay_id:
        _cancel_abandoned_razorpay_subscription(replaced_razorpay_id)
    if stale_pending_id:
        _cancel_abandoned_razorpay_subscription(stale_pending_id)

    return {
        "razorpay_subscription_id": razor_sub["id"],
        "key_id": settings.razorpay_key_id,
    }

def cancel_subscription(db: Session, team_id) -> TeamSubscription:
    team_sub = db.query(TeamSubscription).filter(
        TeamSubscription.team_id == team_id,
        TeamSubscription.status.in_(["active", "pending"]),
    ).with_for_update().first()

    if not team_sub:
        raise ValueError("This team has no active subscription to cancel")

    if is_unpaid_checkout(team_sub):
        # Never paid, so there is no billing to stop and nothing that can go
        # wrong by cancelling locally. Razorpay is told on a best-effort basis
        # only: an error there must not leave the user stuck with an "already
        # in progress" attempt they are trying to get rid of.
        if team_sub.razorpay_subscription_id:
            _cancel_abandoned_razorpay_subscription(team_sub.razorpay_subscription_id)
        team_sub.status = "cancelled"
        db.commit()
        return team_sub

    # Tell Razorpay to stop billing BEFORE we touch local state. If this fails we
    # leave status untouched — never show 'cancelled' while the card is still
    # being charged.
    if team_sub.razorpay_subscription_id:
        try:
            razorpay_client.subscription.cancel(team_sub.razorpay_subscription_id)
        except Exception as e:
            logger.exception(
                "Failed to cancel Razorpay subscription %s for team %s",
                team_sub.razorpay_subscription_id, team_id,
            )
            raise RazorpayCancelError(
                "Could not cancel the subscription with Razorpay. Please try again."
            ) from e

    team_sub.status = "cancelled"
    # Cancelling the plan also abandons any unpaid upgrade waiting on payment.
    pending_id = clear_pending_switch(team_sub)
    db.commit()
    if pending_id:
        _cancel_abandoned_razorpay_subscription(pending_id)
    return team_sub

def _reject_non_upgrade_mid_period(db: Session, team_sub, new_plan) -> None:
    """Raise PlanSwitchBlocked if `team_sub` is a live paid subscription whose
    period has not ended and `new_plan` is the same or a lower tier. Upgrades,
    and anything after current_period_end, are left to the normal switch flow."""
    if team_sub is None or team_sub.status not in ("active", "pending"):
        return
    period_end = team_sub.current_period_end
    if period_end is None or period_end <= datetime.now(timezone.utc):
        return

    current_plan = db.query(Subscription).filter(Subscription.id == team_sub.subscription_id).first()
    current_rank = PLAN_TIER_RANK.get(current_plan.period_label) if current_plan else None
    new_rank = PLAN_TIER_RANK.get(new_plan.period_label)
    if current_rank is None or new_rank is None or new_rank > current_rank:
        # unknown tier: don't block on data we can't rank -- behave as before
        return

    if new_rank == current_rank:
        raise PlanSwitchBlocked(
            "PLAN_ALREADY_ACTIVE",
            "This team is already on this plan tier.",
            period_end,
        )
    raise PlanSwitchBlocked(
        "PLAN_DOWNGRADE_BLOCKED",
        "You can switch to a lower plan after your current plan ends.",
        period_end,
    )


def switch_subscription(db: Session, team_id, new_subscription_id) -> dict:
    new_plan = db.query(Subscription).filter(Subscription.id == new_subscription_id).first()
    if not new_plan:
        raise ValueError("New plan not found")
    if not new_plan.razorpay_plan_id:
        raise ValueError("This plan is not configured for payment yet")

    existing = db.query(TeamSubscription).filter(TeamSubscription.team_id == team_id).first()
    if is_unpaid_checkout(existing):
        # "Switching" away from a checkout that was never paid is just starting
        # a fresh checkout: nothing to cancel, and no leftover credits to carry
        # over (a never-activated subscription never granted any).
        return create_subscription_checkout(db, team_id, new_subscription_id)

    _reject_non_upgrade_mid_period(db, existing, new_plan)

    total_count = _razorpay_total_count(new_plan)
    now = datetime.now(timezone.utc)

    # Lock the row for the whole operation: it serialises rapid clicks, and the
    # replacement's webhooks (which take the same lock) wait behind us.
    row = (
        db.query(TeamSubscription)
        .filter(
            TeamSubscription.team_id == team_id,
            TeamSubscription.status.in_(["active", "pending"]),
        )
        .with_for_update()
        .first()
    )
    if not row:
        raise ValueError("This team has no active subscription to switch")

    # The current plan is NOT touched here. This only starts a payment: the old
    # plan stays live, and is cancelled (and credits moved) when the new
    # subscription's `subscription.activated` webhook arrives -- see
    # webhooks._promote_pending_switch. Abandoning the payment page therefore
    # never leaves the team without a plan.
    replaced_pending_id = row.pending_razorpay_subscription_id
    if replaced_pending_id:
        try:
            pending_sub = razorpay_client.subscription.fetch(replaced_pending_id)
        except Exception:
            db.rollback()
            logger.exception(
                "could not check pending switch %s for team %s", replaced_pending_id, team_id,
            )
            raise ValueError("Could not check your pending upgrade. Please try again.")
        pending_status = pending_sub.get("status")

        still_valid = row.pending_switch_expires_at is None or row.pending_switch_expires_at > now
        if pending_status == "created" and still_valid and row.pending_subscription_id == new_subscription_id:
            # Same upgrade again (double click / "Retry payment"): hand back the
            # SAME unpaid subscription instead of creating a second one.
            db.rollback()
            return {
                "razorpay_subscription_id": replaced_pending_id,
                "key_id": settings.razorpay_key_id,
                "pending_switch": True,
                "pending_switch_expires_at": row.pending_switch_expires_at.isoformat()
                if row.pending_switch_expires_at else None,
            }
        if pending_status not in ("created", "cancelled", "expired"):
            # Already paid or being paid: cancelling it would cancel a payment the
            # user just made. Let its `activated` webhook land first.
            db.rollback()
            raise ValueError(
                "Your previous upgrade payment is still being processed. "
                "Please wait a minute before changing plans."
            )
        # otherwise: a different target plan, or a stale/dead checkout -> replace it

    switch_id = uuid.uuid4()
    expires_at = now + timedelta(hours=settings.pending_switch_ttl_hours)
    try:
        new_razor_sub = razorpay_client.subscription.create({
            "plan_id": new_plan.razorpay_plan_id,
            "total_count": total_count,
            # Razorpay itself stops accepting the authorisation payment at the same
            # deadline we store below, so a checkout window left open past the
            # expiry can't be paid after expire_pending_switch has retired it.
            "expire_by": int(expires_at.timestamp()),
            "notes": {
                "team_id": str(team_id),
                "subscription_id": str(new_subscription_id),
                "switch_id": str(switch_id),
            },
        })
    except Exception:
        db.rollback()
        raise

    row.pending_subscription_id = new_subscription_id
    row.pending_razorpay_subscription_id = new_razor_sub["id"]
    row.pending_switch_id = switch_id
    row.pending_switch_expires_at = expires_at
    try:
        db.commit()
    except Exception:
        logger.error(
            "pending switch commit failed after the Razorpay subscription was created: "
            "team_id=%s switch_id=%s new_razorpay_subscription_id=%s",
            team_id, switch_id, new_razor_sub["id"], exc_info=True,
        )
        db.rollback()
        # never stored on a row and never paid, so nothing else can reference it
        _cancel_abandoned_razorpay_subscription(new_razor_sub["id"])
        raise

    # Only AFTER the replacement is stored, so a failure above leaves the user's
    # previous pending attempt untouched.
    if replaced_pending_id:
        _cancel_abandoned_razorpay_subscription(replaced_pending_id)

    return {
        "razorpay_subscription_id": new_razor_sub["id"],
        "key_id": settings.razorpay_key_id,
        "pending_switch": True,
        "pending_switch_expires_at": expires_at.isoformat(),
    }


def expire_pending_switch(db: Session, team_subscription_id, now: datetime | None = None) -> str:
    """Expire one stale pending switch. Returns what happened:
    "skipped" (nothing to do / locked by someone else / not yet due),
    "cancelled" (unpaid replacement cancelled and cleared),
    "cleared" (Razorpay already considered it dead; just cleared),
    "in_flight" (it was actually paid -- left for its `activated` webhook),
    "error" (Razorpay unreachable; retried next run).
    """
    now = now or datetime.now(timezone.utc)
    row = (
        db.query(TeamSubscription)
        .filter(TeamSubscription.id == team_subscription_id)
        .populate_existing()
        .with_for_update(skip_locked=True)
        .first()
    )
    if (
        row is None
        or not row.pending_razorpay_subscription_id
        or row.pending_switch_expires_at is None
        or row.pending_switch_expires_at > now
    ):
        db.rollback()
        return "skipped"

    pending_id = row.pending_razorpay_subscription_id
    try:
        status = razorpay_client.subscription.fetch(pending_id).get("status")
    except Exception:
        db.rollback()
        logger.exception("could not check expired pending switch %s -- will retry", pending_id)
        return "error"

    if status not in ("created", "cancelled", "expired"):
        # Paid (authenticated/active/...). Cancelling would cancel money already
        # taken; its `activated` webhook should promote it. Push the deadline out
        # and shout: a paid switch still un-promoted after the TTL means the
        # webhook was lost or is failing.
        logger.error(
            "pending switch past its expiry but Razorpay reports %r -- NOT cancelling: "
            "team_id=%s switch_id=%s pending_razorpay_subscription_id=%s",
            status, row.team_id, row.pending_switch_id, pending_id,
        )
        row.pending_switch_expires_at = now + timedelta(hours=1)
        db.commit()
        return "in_flight"

    if status == "created":
        try:
            razorpay_client.subscription.cancel(pending_id)
        except Exception:
            db.rollback()
            logger.warning(
                "could not cancel expired pending switch %s -- will retry", pending_id, exc_info=True,
            )
            return "error"

    logger.info(
        "expired pending switch: team_id=%s switch_id=%s razorpay=%s (was %r)",
        row.team_id, row.pending_switch_id, pending_id, status,
    )
    clear_pending_switch(row)
    db.commit()
    return "cancelled" if status == "created" else "cleared"
