from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from uuid import UUID


class PendingSwitchOut(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    plan: str | None = None  # slug of the plan being upgraded to
    expires_at: str | None = None


class TeamBillingOut(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    total_credits: int
    subscription_credits: int
    topup_credits: int
    plan: str | None = None  # subscription plan slug, e.g. "monthly-pro"; None when no subscription
    subscription_status: str | None = None
    current_period_end: str | None = None
    # An upgrade waiting on payment; the current plan above stays live meanwhile.
    pending_switch: PendingSwitchOut | None = None


class BillingHistoryItemOut(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: str
    type: str                       # subscription_grant | switch_transfer | topup_purchase
    pool: str                       # subscription | topup
    credits: int                    # signed change to that pool
    balance_after: int | None = None
    plan: str | None = None         # plan slug, for subscription entries
    period: str | None = None       # week | month | year
    cycle: int | None = None        # which billing cycle of the subscription
    amount_paid: int | None = None  # smallest currency unit; None for free slices and transfers
    reference: str | None = None    # payment id for a top-up
    created_at: str | None = None


class BillingHistoryOut(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    items: list[BillingHistoryItemOut]
    next_cursor: str | None = None
