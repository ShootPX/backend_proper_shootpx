import hashlib
import hmac
import logging

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.core.config import settings
from app.core.razorpay_client import razorpay_client
from app.models.billing_transaction import BillingTransaction
from app.models.credit import Credit
from app.models.team import Team
from app.services.credits import add_topup_credits
from datetime import datetime, timedelta, timezone
from app.models.team_subscription import TeamSubscription
from app.models.subscription import Subscription
from app.services.credits import refill_subscription_credits
from app.services.billing import clear_pending_switch
from app.services import ledger as ledger_svc
from app.models.team_member import TeamMember
from app.models.user import User
from app.core.email import send_email
from app.core.config import settings

logger = logging.getLogger(__name__)


def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    expected = hmac.new(
        settings.razorpay_webhook_secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def send_renewal_notice_email(db: Session, team_sub: TeamSubscription) -> None:
    """
    Best-effort courtesy email warning a team's owner that their subscription
    is about to complete. Shared by handle_subscription_charged's own
    paid_count-based trigger (weekly/monthly plans -- fires one cycle before
    total_count is reached) and worker.py's send_yearly_renewal_notices cron
    (yearly plans -- fires a fixed number of days before current_period_end,
    since a yearly plan has no intermediate "charged" webhook to key off).

    Never raises: an SMTP failure here must not fail the caller's own
    transaction (a webhook that otherwise succeeded, or a scheduled job
    processing other teams too) -- logged and swallowed instead.
    """
    owner_membership = db.query(TeamMember).filter(
        TeamMember.team_id == team_sub.team_id,
        TeamMember.role == "owner",
    ).first()
    if not owner_membership:
        return
    owner = db.query(User).filter(User.id == owner_membership.user_id).first()
    if not owner:
        return

    try:
        send_email(
            to=owner.email,
            subject="Your ShootPX plan is ending soon",
            html=(
                f"Your current plan will complete after your next billing cycle. "
                f"Renew to keep your credits flowing: {settings.frontend_url}/billing"
            ),
        )
    except Exception:
        logger.error(
            "Failed to send renewal-notice email for team %s (owner %s)",
            team_sub.team_id, owner.id, exc_info=True,
        )


def handle_payment_captured(db: Session, event: dict) -> None:
    payment = event["payload"]["payment"]["entity"]
    razorpay_payment_id = payment["id"]
    amount_paid = payment["amount"]
    order_id = payment.get("order_id")

    # Fast path: a retried webhook whose original already committed.
    existing = db.query(BillingTransaction).filter(
        BillingTransaction.razorpay_payment_id == razorpay_payment_id
    ).first()
    if existing:
        return  # already handled — safe to ignore a retried webhook

    if not order_id:
        return  # not an order-backed payment we issued

    # --- Trust boundary -------------------------------------------------------
    # We NEVER read the amount or credit count from anything the browser can
    # influence:
    #   * `payment["notes"]` can be populated from the frontend Checkout call —
    #     do not trust it.
    #   * The order we created server-side is the source of truth. Its notes hold
    #     only the team id and the credit-pack id; the client has no API access to
    #     the order and cannot alter them.
    #   * The price and the number of credits are then looked up fresh from the
    #     `credit` catalog row — the database is the only authority for both.
    try:
        order = razorpay_client.order.fetch(order_id)
    except Exception:
        logger.exception("could not fetch Razorpay order %s", order_id)
        raise  # let Razorpay retry this webhook

    notes = order.get("notes") or {}
    team_id = notes.get("team_id")
    credit_pack_id = notes.get("credit_pack_id")
    if not team_id or not credit_pack_id:
        return  # not a credit-pack order

    pack = db.query(Credit).filter(Credit.id == credit_pack_id).first()
    if not pack:
        logger.error(
            "credit pack %s referenced by order %s no longer exists",
            credit_pack_id, order_id,
        )
        return

    # Defence in depth: the amount actually captured must match the catalog price
    # for that pack. Razorpay already pins the payment to the order amount, so a
    # mismatch here means the order was not created by our checkout endpoint.
    if amount_paid != pack.price or order.get("amount") != pack.price:
        logger.error(
            "amount mismatch for payment %s: paid=%s order=%s catalog=%s (pack %s)",
            razorpay_payment_id, amount_paid, order.get("amount"), pack.price, credit_pack_id,
        )
        db.add(BillingTransaction(
            team_id=team_id,
            type="credit_pack",
            razorpay_payment_id=razorpay_payment_id,
            amount=amount_paid,
            credits_added=0,
            status="amount_mismatch",
        ))
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
        return

    # A real customer paid real money here (Razorpay already captured the
    # payment) -- if the team it's for is soft-deleted, silently crediting a
    # team the owner can't currently see/use would leave them thinking their
    # purchase vanished, with no indication a restore would fix it. HOLD the
    # grant instead of skipping it outright: the BillingTransaction row (0
    # credits_added, a distinct status) is the audit trail support needs to
    # find this and either manually credit it after the team is restored or
    # process a refund -- this function never auto-refunds on its own
    # authority, that's a real money-movement decision for a human.
    team = db.query(Team).filter(Team.id == team_id).first()
    if team is not None and team.deleted_at is not None:
        logger.error(
            "MANUAL FOLLOW-UP NEEDED: payment %s captured for team %s, but that "
            "team is soft-deleted (deleted_at=%s) -- credits were NOT granted. "
            "Restore the team then manually credit it, or refund the payment.",
            razorpay_payment_id, team_id, team.deleted_at,
        )
        db.add(BillingTransaction(
            team_id=team_id,
            type="credit_pack",
            razorpay_payment_id=razorpay_payment_id,
            amount=amount_paid,
            credits_added=0,
            status="held_team_deleted",
        ))
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
        return

    credits = pack.credits  # server-side, straight from the database

    # Claim the payment id and grant the credits in ONE transaction. The unique
    # constraint on razorpay_payment_id is the idempotency guard: if a concurrent
    # (or racing retried) webhook already claimed it, the flush fails and we bail
    # out before touching the balance — no double-credit.
    db.add(BillingTransaction(
        team_id=team_id,
        type="credit_pack",
        razorpay_payment_id=razorpay_payment_id,
        amount=amount_paid,
        credits_added=credits,
        status="completed",
    ))
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        return  # another worker got here first

    team_after = add_topup_credits(db, team_id, credits, commit=False)
    balance_after = getattr(team_after, "topup_credits_balance", None)
    ledger_svc.record_credit_entry(
        db, team_id=team_id, pool="topup", entry_type=ledger_svc.TOPUP_PURCHASE,
        amount=credits, balance_after=balance_after if isinstance(balance_after, int) else None,
        idempotency_key=f"payment:{razorpay_payment_id}", source="webhook",
        razorpay_payment_id=razorpay_payment_id,
        metadata={"amount": amount_paid, "pack": getattr(pack, "slug", None)},
    )
    db.commit()


# Crude calendar math is fine for MVP — Razorpay's own state is authoritative
# for *when* the next charge happens; these values just drive the billing page.
_PERIOD = {"week": timedelta(weeks=1), "month": timedelta(days=30), "year": timedelta(days=365)}


def _credits_per_refill(plan: Subscription) -> int:
    """Yearly plans are delivered in 12 monthly slices; week/month get the full
    amount each period."""
    return plan.credits // 12 if plan.period_label == "year" else plan.credits


# Razorpay states in which cancelling the OLD subscription is already done, so a
# "cancel" error just means a previous (partly failed) delivery got that far.
_SUBSCRIPTION_ALREADY_ENDED = ("cancelled", "completed", "expired")


def _cancel_old_subscription_for_switch(old_razorpay_id: str, team_sub, switch_id, new_razorpay_id: str) -> None:
    """Cancel the plan being replaced. Idempotent across webhook retries: if the
    cancel fails but Razorpay says it has already ended (a prior delivery cancelled
    it and then failed to commit), that counts as success. Any other failure
    raises so Razorpay redelivers the webhook, after an ERROR that carries
    everything needed to find the pair by hand."""
    try:
        razorpay_client.subscription.cancel(old_razorpay_id)
        return
    except Exception as cancel_err:
        try:
            status = razorpay_client.subscription.fetch(old_razorpay_id).get("status")
        except Exception:
            status = None
        if status in _SUBSCRIPTION_ALREADY_ENDED:
            return
        logger.error(
            "switch activation: could not cancel the OLD subscription -- the user has paid "
            "for the new plan but the old one is still live. Raising so Razorpay retries: "
            "team_id=%s switch_id=%s old_razorpay_subscription_id=%s new_razorpay_subscription_id=%s "
            "old_status=%s",
            team_sub.team_id, switch_id, old_razorpay_id, new_razorpay_id, status,
            exc_info=cancel_err,
        )
        raise


def _promote_pending_switch(db: Session, team_sub, razorpay_subscription_id: str) -> None:
    """The replacement subscription of a pending switch was just paid for
    (`subscription.activated`). Cancel the old plan, move its leftover credits to
    top-up, install the new plan and clear the pending switch -- all in ONE commit.

    Idempotent: a redelivery no longer matches the (cleared) pending column and
    falls through to handle_subscription_activated's "already active" skip. If a
    delivery cancels the old plan and then fails to commit, the retry sees the old
    plan already ended and carries on.

    `team_sub` is already locked FOR UPDATE by the caller.
    """
    switch_id = team_sub.pending_switch_id
    plan = db.query(Subscription).filter(Subscription.id == team_sub.pending_subscription_id).first()
    if not plan:
        logger.error(
            "pending switch %s for team %s targets an unknown plan %s",
            switch_id, team_sub.team_id, team_sub.pending_subscription_id,
        )
        return

    # Same server-to-server verification as a normal activation: never trust the
    # webhook payload, and make sure Razorpay's subscription really is the plan
    # and switch we stored.
    try:
        razor_sub = razorpay_client.subscription.fetch(razorpay_subscription_id)
    except Exception:
        logger.exception("could not fetch Razorpay subscription %s", razorpay_subscription_id)
        raise  # let Razorpay retry
    notes = razor_sub.get("notes") or {}
    if razor_sub.get("plan_id") != plan.razorpay_plan_id or notes.get("switch_id") != str(switch_id):
        logger.error(
            "pending switch mismatch on subscription.activated: sub=%s switch_id=%s "
            "expected_plan=%s got_plan=%s notes_switch_id=%s",
            razorpay_subscription_id, switch_id, plan.razorpay_plan_id,
            razor_sub.get("plan_id"), notes.get("switch_id"),
        )
        return

    old_razorpay_id = team_sub.razorpay_subscription_id
    if old_razorpay_id:
        _cancel_old_subscription_for_switch(old_razorpay_id, team_sub, switch_id, razorpay_subscription_id)

    now = datetime.now(timezone.utc)
    period = _PERIOD[plan.period_label]
    credits_per_refill = _credits_per_refill(plan)

    team = db.query(Team).filter(Team.id == team_sub.team_id).with_for_update().first()
    if team is None:
        raise ValueError("Team not found")
    # Unused credits of the plan being replaced are kept (moved to top-up), then
    # the new plan's first slice replaces the pool -- the same as a fresh activation.
    leftover = team.subscription_credits_remaining
    team.topup_credits_balance += leftover
    team.subscription_credits_remaining = credits_per_refill
    if leftover:
        for pool, amount, balance_after, leg in (
            ("subscription", -leftover, 0, "out"),
            ("topup", leftover, team.topup_credits_balance, "in"),
        ):
            ledger_svc.record_credit_entry(
                db, team_id=team_sub.team_id, pool=pool, entry_type=ledger_svc.SWITCH_TRANSFER,
                amount=amount, balance_after=balance_after,
                idempotency_key=f"switch:{switch_id}:{leg}", source="webhook",
                razorpay_subscription_id=razorpay_subscription_id, switch_id=switch_id,
            )
    ledger_svc.record_credit_entry(
        db, team_id=team_sub.team_id, pool="subscription", entry_type=ledger_svc.SUBSCRIPTION_GRANT,
        amount=credits_per_refill, balance_after=credits_per_refill,
        idempotency_key=f"sub:{razorpay_subscription_id}:cycle:1", source="webhook",
        razorpay_subscription_id=razorpay_subscription_id, switch_id=switch_id,
        metadata=ledger_svc.plan_metadata(plan, cycle=1, charged=True),
    )

    team_sub.subscription_id = plan.id
    team_sub.razorpay_subscription_id = razorpay_subscription_id
    team_sub.status = "active"
    team_sub.credits_per_refill = credits_per_refill
    team_sub.last_paid_count = 0
    team_sub.next_refill_at = now + (timedelta(days=30) if plan.period_label == "year" else period)
    team_sub.current_period_end = now + period
    team_sub.renewal_notice_sent_at = None
    clear_pending_switch(team_sub)

    try:
        db.commit()
    except Exception:
        logger.error(
            "pending switch commit failed AFTER the old plan was cancelled: team_id=%s "
            "switch_id=%s old_razorpay_subscription_id=%s new_razorpay_subscription_id=%s "
            "-- raising so Razorpay retries (the retry will skip the already-ended old plan)",
            team_sub.team_id, switch_id, old_razorpay_id, razorpay_subscription_id, exc_info=True,
        )
        db.rollback()
        raise


def handle_subscription_activated(db: Session, event: dict) -> None:
    razorpay_subscription_id = event["payload"]["subscription"]["entity"]["id"]

    # An upgrade's replacement lives in the pending_* columns, not in
    # razorpay_subscription_id (the current plan must stay live until this
    # moment), so check there first.
    pending_row = db.query(TeamSubscription).filter(
        TeamSubscription.pending_razorpay_subscription_id == razorpay_subscription_id
    ).with_for_update().first()
    if pending_row is not None and pending_row.pending_razorpay_subscription_id == razorpay_subscription_id:
        _promote_pending_switch(db, pending_row, razorpay_subscription_id)
        return

    # Checkout stores the real razorpay_subscription_id on a `pending` row, so
    # this exact subscription is normally already on a row. Lock it and act on its
    # status:
    #   * active    -> already promoted (idempotent replay) -> skip
    #   * cancelled -> the owner cancelled during the pending window -> a
    #                  late-arriving webhook must NOT resurrect it -> skip
    #   * pending   -> this is the activation we've been waiting for -> promote
    #   * (no row)  -> fall through to the notes-based lookup below
    locked = db.query(TeamSubscription).filter(
        TeamSubscription.razorpay_subscription_id == razorpay_subscription_id
    ).with_for_update().first()
    if locked is not None and locked.status in ("active", "cancelled"):
        logger.info(
            "subscription.activated for %s ignored — row already %s",
            razorpay_subscription_id, locked.status,
        )
        return

    # Fetch server-to-server — never trust the webhook payload's own notes (same
    # lesson as the credit-pack fix). This pulls the notes OUR backend set at
    # checkout, which the browser cannot touch.
    try:
        razor_sub = razorpay_client.subscription.fetch(razorpay_subscription_id)
    except Exception:
        logger.exception("could not fetch Razorpay subscription %s", razorpay_subscription_id)
        raise  # let Razorpay retry

    notes = razor_sub.get("notes") or {}
    team_id = notes.get("team_id")
    subscription_id = notes.get("subscription_id")
    if not team_id or not subscription_id:
        return  # not one of ours, ignore safely

    plan = db.query(Subscription).filter(Subscription.id == subscription_id).first()
    if not plan:
        logger.error("subscription.activated for unknown plan %s", subscription_id)
        return

    # Defense in depth: the plan_id on the real Razorpay subscription must match
    # the plan we think this is — catches any tampering with the checkout call.
    if razor_sub.get("plan_id") != plan.razorpay_plan_id:
        logger.error(
            "plan mismatch on subscription.activated: sub=%s expected=%s got=%s",
            razorpay_subscription_id, plan.razorpay_plan_id, razor_sub.get("plan_id"),
        )
        return

    now = datetime.now(timezone.utc)
    period = _PERIOD[plan.period_label]
    credits_per_refill = _credits_per_refill(plan)
    # next_refill_at is the moment the NEXT slice is due (end of this period). The
    # first `subscription.charged` fires right after activation and must not
    # re-grant this period — it checks now vs next_refill_at.
    next_refill_at = now + (timedelta(days=30) if plan.period_label == "year" else period)
    # A fresh lifecycle (first activation OR a resubscribe reusing the row) always
    # starts this counter at 0, regardless of what a PREVIOUS subscription on this
    # same row left behind.
    last_paid_count = 0

    # Exactly one subscription row per team (UNIQUE team_id). Normally `locked`
    # (found by razorpay_subscription_id) IS the team's row — a `pending` row from
    # the in-progress checkout, promoted here in place. Only fall back to a
    # team_id lookup when this id isn't on any row yet.
    team_sub = locked
    if team_sub is None:
        team_sub = db.query(TeamSubscription).filter(
            TeamSubscription.team_id == team_id
        ).with_for_update().first()

    if (
        team_sub is not None
        and team_sub.status in ("active", "cancelled")
        and team_sub.razorpay_subscription_id not in (None, razorpay_subscription_id)
    ):
        # this team already has a terminal subscription under a different Razorpay
        # id — don't clobber it (checkout should never have allowed this).
        logger.error(
            "subscription.activated %s but team %s row is %s under %s",
            razorpay_subscription_id, team_id, team_sub.status,
            team_sub.razorpay_subscription_id,
        )
        return

    if team_sub is None:
        team_sub = TeamSubscription(team_id=team_id, subscription_id=subscription_id)
        db.add(team_sub)

    team_sub.subscription_id = subscription_id
    team_sub.razorpay_subscription_id = razorpay_subscription_id
    team_sub.status = "active"
    team_sub.credits_per_refill = credits_per_refill
    team_sub.last_paid_count = last_paid_count
    team_sub.next_refill_at = next_refill_at
    team_sub.current_period_end = now + period

    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        return  # a concurrent activation won the race

    # The ledger entry goes in BEFORE the refill: refill_subscription_credits commits,
    # and the entry must land in that same transaction.
    before = ledger_svc.subscription_pool_balance(db, team_id)
    ledger_svc.record_subscription_grant(
        db, team_id=team_id, balance_before=before, balance_after=credits_per_refill,
        idempotency_key=f"sub:{razorpay_subscription_id}:cycle:1", source="webhook",
        razorpay_subscription_id=razorpay_subscription_id,
        metadata=ledger_svc.plan_metadata(plan, cycle=1, charged=True),
    )
    refill_subscription_credits(db, team_id, credits_per_refill)
    db.commit()


def apply_subscription_charge(db: Session, team_sub, plan, razor_sub: dict, source: str = "webhook") -> bool:
    """Apply a charge Razorpay reports for `team_sub`: mark it active, and if the
    charge is new (paid_count beyond what was last recorded) advance the period,
    grant the renewal credits and record them in the ledger. Returns True if it
    was a new charge.

    Shared by the `subscription.charged` webhook and the reconciliation job, so a
    charge whose webhook was lost is applied by exactly the same code -- and the
    paid_count guard makes it a no-op if both ever see it. The caller holds the
    row lock and commits.
    """
    paid_count = razor_sub.get("paid_count", 0)
    now = datetime.now(timezone.utc)
    period = _PERIOD[plan.period_label]
    is_new_charge = paid_count > team_sub.last_paid_count

    team_sub.status = "active"

    # Idempotency: paid_count is Razorpay's own monotonic counter for this
    # subscription, not something we derive from a clock. A redelivered webhook
    # reports the SAME paid_count as the original, so only a value strictly
    # greater than what we've already recorded is genuinely new — this is what
    # stops a redelivery from re-refilling (refill REPLACES the pool, so acting
    # twice would wipe out anything spent between deliveries) and from
    # re-advancing next_refill_at a second time.
    if paid_count > team_sub.last_paid_count:
        team_sub.last_paid_count = paid_count
        # Advance the period only for a genuinely new charge (a redelivery must
        # not push the end date out again). Razorpay's own `current_end` is the
        # true end of the cycle just paid for -- exact for calendar months --
        # with now + period as the fallback if it is missing.
        current_end = razor_sub.get("current_end")
        if isinstance(current_end, int) and current_end > 0:
            team_sub.current_period_end = datetime.fromtimestamp(current_end, tz=timezone.utc)
        else:
            team_sub.current_period_end = now + period
        if plan.period_label != "year" and paid_count > 1:
            team_sub.next_refill_at = now + period
            before = ledger_svc.subscription_pool_balance(db, team_sub.team_id)
            ledger_svc.record_subscription_grant(
                db, team_id=team_sub.team_id, balance_before=before,
                balance_after=team_sub.credits_per_refill,
                idempotency_key=f"sub:{team_sub.razorpay_subscription_id}:cycle:{paid_count}",
                source=source, razorpay_subscription_id=team_sub.razorpay_subscription_id,
                metadata=ledger_svc.plan_metadata(plan, cycle=paid_count, charged=True),
            )
            refill_subscription_credits(db, team_sub.team_id, team_sub.credits_per_refill)

            # Renewal notice: warn the owner one cycle before this subscription
            # naturally completes (total_count reached). Best-effort only --
            # send_renewal_notice_email never raises, so a failed SMTP send
            # never surfaces as an unhandled 500 to Razorpay for a webhook
            # that actually succeeded, and since last_paid_count already
            # advanced above, a retry would never re-attempt the email anyway.
            if plan.total_count and paid_count == plan.total_count - 1:
                send_renewal_notice_email(db, team_sub)
    # Yearly plans: a renewal charge only extends current_period_end -- the 12
    # monthly credit slices between once-a-year charges are delivered by the
    # daily refill_due_subscriptions scheduler in worker.py, and the renewal
    # notice for a yearly plan is sent by that same worker's
    # send_yearly_renewal_notices cron (keyed off current_period_end
    # directly, since there's no second "charged" webhook to key off).

    return is_new_charge


def handle_subscription_charged(db: Session, event: dict) -> None:
    razorpay_subscription_id = event["payload"]["subscription"]["entity"]["id"]

    # Lock before reading — same as handle_subscription_activated. Razorpay can
    # redeliver this exact webhook, and without the lock a redelivery arriving
    # while the first is still mid-flight could read the same pre-update row.
    team_sub = db.query(TeamSubscription).filter(
        TeamSubscription.razorpay_subscription_id == razorpay_subscription_id
    ).with_for_update().first()
    if not team_sub:
        return  # activated never processed — nothing to update yet

    plan = db.query(Subscription).filter(Subscription.id == team_sub.subscription_id).first()
    if not plan:
        return

    # paid_count == 1 is the initial charge that fires alongside activation, which
    # already granted this period's credits. Only paid_count > 1 is a renewal.
    try:
        razor_sub = razorpay_client.subscription.fetch(razorpay_subscription_id)
    except Exception:
        logger.exception("could not fetch Razorpay subscription %s", razorpay_subscription_id)
        raise

    if team_sub.status == "cancelled":
        logger.info(
            "subscription.charged for %s ignored — row is cancelled",
            razorpay_subscription_id,
        )
        return

    if team_sub.status == "pending" and team_sub.credits_per_refill == 0:
        logger.info(
            "subscription.charged for %s arrived before activated — deferring",
            razorpay_subscription_id,
        )
        return

    apply_subscription_charge(db, team_sub, plan, razor_sub)
    db.commit()

def handle_subscription_completed(db: Session, event: dict):
    sub_entity = event["payload"]["subscription"]["entity"]
    team_sub = db.query(TeamSubscription).filter(
        TeamSubscription.razorpay_subscription_id == sub_entity["id"]
    ).with_for_update().first()

    if team_sub and team_sub.status == "active":
        team_sub.status = "cancelled"
        db.commit()

def handle_subscription_halted(db: Session, event: dict):
    sub_entity = event["payload"]["subscription"]["entity"]
    team_sub = db.query(TeamSubscription).filter(
        TeamSubscription.razorpay_subscription_id == sub_entity["id"]
    ).with_for_update().first()
    if team_sub:
        team_sub.status = "halted"
        db.commit()


def handle_subscription_cancelled(db: Session, event: dict):
    sub_entity = event["payload"]["subscription"]["entity"]
    team_sub = db.query(TeamSubscription).filter(
        TeamSubscription.razorpay_subscription_id == sub_entity["id"]
    ).with_for_update().first()
    if team_sub:
        team_sub.status = "cancelled"
        db.commit()


def handle_subscription_pending(db: Session, event: dict) -> None:
    # A renewal payment failed; Razorpay is in its 3-day retry window. Pause
    # refills (the worker skips non-active rows) but leave the existing balance
    # alone. Only active -> pending; never touch pending/cancelled/halted.
    sub_entity = event["payload"]["subscription"]["entity"]
    team_sub = db.query(TeamSubscription).filter(
        TeamSubscription.razorpay_subscription_id == sub_entity["id"]
    ).with_for_update().first()
    if team_sub and team_sub.status == "active":
        team_sub.status = "pending"
        db.commit()


