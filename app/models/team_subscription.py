from sqlalchemy import Column, Text, DateTime, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from app.core.database import Base


class TeamSubscription(Base):
    __tablename__ = "team_subscriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False, unique=True)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey("subscription.id"), nullable=False)
    razorpay_subscription_id = Column(Text, nullable=True, unique=True)
    status = Column(Text, nullable=False, default="active")
    credits_per_refill = Column(Integer, nullable=False)
    # Razorpay's own monotonic charge counter, last acted on. Lets
    # handle_subscription_charged tell a genuinely new renewal apart from a
    # redelivered webhook for a charge it already processed.
    last_paid_count = Column(Integer, nullable=False, default=0)
    next_refill_at = Column(DateTime(timezone=True), nullable=False)
    current_period_end = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # Yearly plans have no per-cycle "charged" webhook to key a renewal
    # notice off (total_count=1 -- there's only ever ONE charge, at the
    # start of the year) -- worker.py's send_yearly_renewal_notices cron
    # checks current_period_end directly instead, and this timestamp is
    # its own idempotency guard (set once sent, so a daily cron tick never
    # re-sends). Reset to NULL on a fresh subscribe (see
    # create_subscription_checkout) so a new cycle can send its own notice.
    renewal_notice_sent_at = Column(DateTime(timezone=True), nullable=True)
    # A pending plan switch: an upgrade whose replacement Razorpay subscription
    # has been created but not yet paid. The current plan (everything above) stays
    # live until that subscription's `activated` webhook promotes the switch --
    # see billing.switch_subscription / webhooks._promote_pending_switch. The
    # four columns are all set or all NULL (enforced by a CHECK in the migration).
    # Deliberately NOT in razorpay_subscription_id: the replacement's own
    # webhooks (including the `cancelled` we trigger when replacing or expiring
    # it) must not match this row's live subscription.
    pending_subscription_id = Column(UUID(as_uuid=True), ForeignKey("subscription.id"), nullable=True)
    pending_razorpay_subscription_id = Column(Text, nullable=True, unique=True)
    pending_switch_id = Column(UUID(as_uuid=True), nullable=True)
    pending_switch_expires_at = Column(DateTime(timezone=True), nullable=True)
    # When services/reconciliation.py last compared this row with Razorpay.
    last_reconciled_at = Column(DateTime(timezone=True), nullable=True)