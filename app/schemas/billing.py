from pydantic import BaseModel, ConfigDict, field_validator
from pydantic.alias_generators import to_camel
from uuid import UUID
from typing import Optional


def _normalize_info(value):
    # JSONB can come back as None or a list of str/objects. Keep it a list[str].
    if not value:
        return []
    return [item if isinstance(item, str) else str(item) for item in value]


class _CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


class SubscriptionOut(_CamelModel):
    id: UUID
    slug: str
    name: str
    price: int
    billing_period_days: int
    period_label: str
    credits: int
    info: list[str] = []
    tag: Optional[str] = None
    is_popular: bool = False
    sort_order: int = 0

    _fix_info = field_validator("info", mode="before")(_normalize_info)


class CreditOut(_CamelModel):
    id: UUID
    slug: str
    name: str
    price: int
    credits: int
    info: list[str] = []
    tag: Optional[str] = None
    sort_order: int = 0

    _fix_info = field_validator("info", mode="before")(_normalize_info)


class BillingResponse(_CamelModel):
    subscriptions: list[SubscriptionOut]
    credits: list[CreditOut]
