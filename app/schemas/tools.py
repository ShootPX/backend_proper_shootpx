from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from uuid import UUID


class _CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


def derive_tool_status(stage: int, is_active: bool) -> str:
    """"live" | "soon" -- the SAME condition GET /tools/{feature_type}/schema
    gates on (stage == 1 and is_active), deliberately NOT is_coming_soon.
    is_coming_soon is independent editorial copy for the card badge; if the
    grid called a tool "live" off is_coming_soon while /schema 404'd it off
    stage/is_active, the tool would render as usable and then break on click.
    """
    return "live" if (stage == 1 and is_active) else "soon"


class ToolOut(_CamelModel):
    feature_type: str
    display_name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    category: str
    is_coming_soon: bool
    card_sort_order: int = 0
    thumb_url: Optional[str] = None
    # A plain, stored field -- computed ONCE from the ORM row (see
    # ToolOut.from_row below), not derived lazily on every read. This response
    # is Redis-cached (routes/tools.py's CACHE_KEY): a computed_field/property
    # version of this was tried first and looked right against a live DB read,
    # but broke on every CACHE HIT -- the source columns it depended on
    # (stage/is_active) were excluded from serialization, so reconstructing a
    # ToolOut from the cached JSON silently recomputed status from field
    # DEFAULTS (stage=0, is_active=False) instead of the real values, always
    # yielding "soon". Storing the already-derived string sidesteps the whole
    # cache round-trip problem: there's nothing left to recompute.
    status: str

    @classmethod
    def from_row(cls, tool) -> "ToolOut":
        return cls(
            feature_type=tool.feature_type,
            display_name=tool.display_name,
            description=tool.description,
            icon=tool.icon,
            category=tool.category,
            is_coming_soon=tool.is_coming_soon,
            card_sort_order=tool.card_sort_order,
            thumb_url=tool.thumb_url,
            status=derive_tool_status(tool.stage, tool.is_active),
        )


class ToolsResponse(_CamelModel):
    tools: list[ToolOut]


class HomepageSlideOut(_CamelModel):
    id: UUID
    title: str
    subtitle: Optional[str] = None
    image_url: str
    cta_label: str
    deeplink: Any
    sort_order: int = 0


class HomepageSlidesResponse(_CamelModel):
    slides: list[HomepageSlideOut]
