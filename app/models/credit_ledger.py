from sqlalchemy import Column, Text, DateTime, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
import uuid
from app.core.database import Base


class CreditLedger(Base):
    """Append-only record of credits that came IN (subscription grants, plan-switch
    transfers, paid top-ups). Written in the same transaction as the balance change
    it describes.

    It is an audit trail and an idempotency guard, NOT the balance: the balances
    that generation spends against stay on `teams`. `idempotency_key` is UNIQUE so
    a webhook redelivery, the reconciliation job and the original webhook can all
    try to record the same event and only the first one lands.
    """

    __tablename__ = "credit_ledger"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    pool = Column(Text, nullable=False)          # "subscription" | "topup"
    entry_type = Column(Text, nullable=False)    # "subscription_grant" | "switch_transfer" | "topup_purchase"
    amount = Column(Integer, nullable=False)     # signed change to that pool
    # NULL only for rows backfilled from billing_transactions, which never
    # recorded the balance at the time.
    balance_after = Column(Integer, nullable=True)
    idempotency_key = Column(Text, nullable=False, unique=True)
    source = Column(Text, nullable=False)        # "webhook" | "reconcile" | "worker" | "backfill"
    razorpay_subscription_id = Column(Text, nullable=True)
    razorpay_payment_id = Column(Text, nullable=True)
    switch_id = Column(UUID(as_uuid=True), nullable=True)
    # "metadata" is reserved on declarative classes, hence the attribute name.
    entry_metadata = Column("metadata", JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
