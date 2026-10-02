import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.database import get_db
from app.deps import get_current_user
from app.main import app
from app.models.tool_definition import ToolDefinition

client = TestClient(app)


@pytest.mark.real_db
def test_tools_returns_list():
    res = client.get("/landing/tools")
    assert res.status_code == 200

    body = res.json()
    assert "tools" in body
    assert isinstance(body["tools"], list)


@pytest.mark.real_db
def test_tool_shape():
    res = client.get("/landing/tools")
    tools = res.json()["tools"]
    if tools:
        first = tools[0]
        for key in ("featureType", "displayName", "category", "isComingSoon", "cardSortOrder", "thumbUrl", "status"):
            assert key in first


@pytest.mark.real_db
def test_tool_shape_never_leaks_raw_stage_or_is_active():
    """stage/is_active are ORM-row-only inputs used to DERIVE `status` at
    construction time (ToolOut.from_row) -- they are never fields on ToolOut
    itself, so they can't leak into the response."""
    res = client.get("/landing/tools")
    for tool in res.json()["tools"]:
        assert "stage" not in tool
        assert "isActive" not in tool
        assert "is_active" not in tool


@pytest.mark.real_db
def test_status_is_live_for_real_active_tools():
    """The actual regression this guards: real, working tools (is_active=true,
    stage=1) were showing as isComingSoon=false with no `status` field at
    all, and the frontend's grid was reported to render everything as "SOON"
    regardless -- because it was reading a `status` field this endpoint never
    sent. `status` must be "live" for every tool with is_active=true,
    stage=1 -- checked against whatever is_coming_soon=false rows currently
    exist in the DB, rather than hardcoding a specific placeholder tool's
    feature_type (the "coming soon" placeholder rows are apparently volatile
    -- present earlier in this same session, absent by the time this test
    ran; the 6 real, launched tools are the stable fixture to assert against)."""
    res = client.get("/landing/tools")
    tools = res.json()["tools"]
    assert tools, "expected at least the real launched tools to be present"

    live_tools = [t for t in tools if not t["isComingSoon"]]
    assert live_tools, "expected at least one is_coming_soon=false tool"
    for t in live_tools:
        assert t["status"] == "live", f"{t['featureType']} has isComingSoon=false but status={t['status']!r}"

    # recolor is one of the known-real, always-expected tools
    by_ft = {t["featureType"]: t for t in tools}
    assert by_ft["recolor"]["status"] == "live"
    assert by_ft["recolor"]["isComingSoon"] is False


def test_status_derivation_uses_stage_and_is_active_not_is_coming_soon():
    """Unit-level proof the derivation is stage+is_active, not is_coming_soon
    -- constructs a row where the two would disagree if status were derived
    from the wrong field."""
    from unittest.mock import MagicMock
    from app.schemas.tools import ToolOut

    # is_coming_soon=False (so a naive "not is_coming_soon" derivation would
    # say "live"), but stage/is_active say this tool is NOT actually usable.
    tricky_row = MagicMock(
        feature_type="half_flipped", display_name="Half Flipped", description=None,
        icon=None, category="shoot", is_coming_soon=False, card_sort_order=0,
        stage=0, is_active=True,
    )
    assert ToolOut.from_row(tricky_row).status == "soon"

    live_row = MagicMock(
        feature_type="recolor", display_name="Recolor", description=None,
        icon=None, category="shoot", is_coming_soon=False, card_sort_order=0,
        stage=1, is_active=True,
    )
    assert ToolOut.from_row(live_row).status == "live"


def test_status_survives_a_cache_round_trip():
    """The actual bug found live: a computed_field version of `status` looked
    correct against a fresh DB read, but a cache HIT reconstructs ToolOut
    from the cached JSON dict -- if status were derived lazily from
    excluded/missing stage+is_active fields, that reconstruction would fall
    back to field defaults and silently flip every cached tool to "soon".
    Proves status is a plain stored value that survives dict -> model ->
    dict unchanged."""
    from app.schemas.tools import ToolOut

    original = ToolOut(
        feature_type="recolor", display_name="Recolor", category="shoot",
        is_coming_soon=False, card_sort_order=0, status="live",
    )
    as_json = original.model_dump(mode="json", by_alias=True)
    assert as_json["status"] == "live"

    # Simulates exactly what a cache HIT does: FastAPI reconstructs the
    # response model from the plain dict handed back by get_cached().
    reconstructed = ToolOut.model_validate(as_json)
    assert reconstructed.status == "live"


@pytest.mark.real_db
def test_tools_only_returns_rows_with_display_name():
    """enhance_prompt / model_shoot_generate_model are internal utility tools
    with no marketing copy (display_name is null) -- they must never show up
    on the public tool grid."""
    res = client.get("/landing/tools")
    feature_types = [t["featureType"] for t in res.json()["tools"]]
    assert "enhance_prompt" not in feature_types
    assert "model_shoot_generate_model" not in feature_types


# --------------------------------------------------------------------------- #
# GET /landing/homepage-slides
# --------------------------------------------------------------------------- #

@pytest.mark.real_db
def test_homepage_slides_returns_list():
    res = client.get("/landing/homepage-slides")
    assert res.status_code == 200
    body = res.json()
    assert "slides" in body
    assert isinstance(body["slides"], list)


@pytest.mark.real_db
def test_homepage_slide_shape():
    res = client.get("/landing/homepage-slides")
    slides = res.json()["slides"]
    if slides:
        first = slides[0]
        for key in ("id", "title", "imageUrl", "ctaLabel", "deeplink", "sortOrder"):
            assert key in first


# --------------------------------------------------------------------------- #
# GET /tools/{feature_type}/schema -- deliberately filtered subset of
# tool_definitions, used by clients to build the /generate form dynamically.
# --------------------------------------------------------------------------- #

@pytest.fixture
def schema_client():
    fake_user = MagicMock(id=uuid.uuid4())
    db = MagicMock()
    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_db] = lambda: db
    yield TestClient(app), db
    app.dependency_overrides.clear()


def test_tool_schema_returns_only_the_frontend_safe_fields(schema_client):
    """This is the real regression test for the bug found live during this
    audit: ToolDefinition.stage doesn't exist as an attribute error is raised
    -- a MagicMock stand-in for `db` does NOT hide this, because
    `ToolDefinition.stage == 1` is evaluated as a real expression against the
    actual SQLAlchemy model class before it ever reaches `db`."""
    client, db = schema_client
    tool = ToolDefinition(
        feature_type="recolor",
        fal_model_id="fal-ai/some-recolor-model",  # must NEVER reach the response
        max_input_images=3,
        max_output_resolution="2048x2048",
        default_output_count=1,
        credit_cost_per_output=5,
        param_schema=[{"name": "color", "type": "text", "required": True, "label": "Color"}],
        is_active=True,
        ai_steps={"detect_target": {"model": "fal-ai/moondream-next"}},  # must NEVER reach the response
        stage=1,
        generation_timeout_seconds=60,
    )
    db.query.return_value.filter.return_value.first.return_value = tool

    res = client.get("/tools/recolor/schema", headers={"Authorization": "Bearer x"})

    assert res.status_code == 200
    body = res.json()
    assert body == {
        "featureType": "recolor",
        "maxInputImages": 3,
        "paramSchema": [{"name": "color", "type": "text", "required": True, "label": "Color"}],
        "generationTimeoutSeconds": 60,   # the ONLY addition -- a pass-through of the row's budget
    }
    assert "falModelId" not in body
    assert "aiSteps" not in body
    assert "fal_model_id" not in body
    assert "ai_steps" not in body


def test_tool_schema_404s_when_tool_missing(schema_client):
    client, db = schema_client
    db.query.return_value.filter.return_value.first.return_value = None

    res = client.get("/tools/ghost_tool/schema", headers={"Authorization": "Bearer x"})

    assert res.status_code == 404


def test_tool_schema_404s_when_tool_inactive(schema_client):
    client, db = schema_client
    tool = ToolDefinition(
        feature_type="recolor", fal_model_id="m", max_input_images=1,
        max_output_resolution="1024x1024", default_output_count=1,
        credit_cost_per_output=5, param_schema=[], is_active=False, ai_steps={}, stage=1,
    )
    db.query.return_value.filter.return_value.first.return_value = tool

    res = client.get("/tools/recolor/schema", headers={"Authorization": "Bearer x"})

    assert res.status_code == 404
