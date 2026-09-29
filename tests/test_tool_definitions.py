"""app/services/tool_definitions.py::get_tool_definition -- caches
ToolDefinition rows in Redis (60s TTL) so /generate, worker.py, and
enhance_prompt.py's source-tool lookup don't all hit Postgres on every call.

These tests deliberately re-enable the REAL get_cached/set_cached (the
autouse _disable_tool_definition_cache fixture in conftest.py turns them into
no-ops for every other test in the suite, so a MagicMock ToolDefinition never
gets serialized into real dev Redis) -- against the real redis_client, same
pattern tests/test_cache_admin.py already uses for the /landing/tools cache.
"""

from unittest.mock import MagicMock

import pytest

from app.core.cache import get_cached, redis_client, set_cached
from app.services import tool_definitions as tool_definitions_svc


@pytest.fixture
def real_cache(monkeypatch):
    """Overrides the autouse no-op fixture for just this file -- exercises
    the actual Redis-backed get_cached/set_cached."""
    monkeypatch.setattr(tool_definitions_svc, "get_cached", get_cached)
    monkeypatch.setattr(tool_definitions_svc, "set_cached", set_cached)
    yield
    for key in redis_client.keys("tooldef:*"):
        redis_client.delete(key)


def _tool(feature_type="recolor"):
    return MagicMock(
        feature_type=feature_type, fal_model_id="openai/gpt-image-2/edit",
        max_input_images=5, max_output_resolution="4096x4096",
        default_output_count=1, credit_cost_per_output=2,
        param_schema=[{"name": "color", "type": "color", "label": "Color", "required": True}],
        is_active=True, stage=1,
        ai_steps={"detect_target": {"model": "fal-ai/moondream-next"}},
        generation_timeout_seconds=60,
    )


def _db(tool):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = tool
    return db


@pytest.mark.real_db
def test_cache_miss_queries_the_db_and_populates_the_cache(real_cache):
    tool = _tool()
    db = _db(tool)

    result = tool_definitions_svc.get_tool_definition(db, "recolor")

    assert result.feature_type == "recolor"
    assert result.ai_steps == {"detect_target": {"model": "fal-ai/moondream-next"}}
    db.query.assert_called_once()
    assert redis_client.exists("tooldef:recolor")


def test_cache_hit_never_touches_the_db(real_cache):
    tool = _tool()
    db = _db(tool)
    tool_definitions_svc.get_tool_definition(db, "recolor")  # warms the cache
    db.query.reset_mock()

    result = tool_definitions_svc.get_tool_definition(db, "recolor")

    assert result.feature_type == "recolor"
    assert result.param_schema == [{"name": "color", "type": "color", "label": "Color", "required": True}]
    db.query.assert_not_called()


def test_cached_result_exposes_every_attribute_real_callers_read(real_cache):
    """Every attribute worker.py / generation.py / routes actually access on
    a ToolDefinition must survive the cache round-trip."""
    tool = _tool()
    db = _db(tool)
    tool_definitions_svc.get_tool_definition(db, "recolor")

    cached = tool_definitions_svc.get_tool_definition(_db(None), "recolor")  # DB unreachable -- must not be needed

    assert cached.feature_type == "recolor"
    assert cached.fal_model_id == "openai/gpt-image-2/edit"
    assert cached.max_input_images == 5
    assert cached.default_output_count == 1
    assert cached.credit_cost_per_output == 2
    assert cached.is_active is True
    assert cached.stage == 1
    assert cached.generation_timeout_seconds == 60


def test_unknown_feature_type_returns_none_and_caches_nothing(real_cache):
    db = _db(None)

    result = tool_definitions_svc.get_tool_definition(db, "ghost_tool")

    assert result is None
    assert not redis_client.exists("tooldef:ghost_tool")


def test_none_feature_type_returns_none_without_crashing(real_cache):
    """Regression: `CACHE_KEY_PREFIX + feature_type` raises a raw TypeError
    for feature_type=None if this isn't guarded -- every real caller only
    reaches here with a validated non-empty string today (e.g. enhance_prompt
    requires source_feature_type in its param_schema), but this is the shared
    boundary function, so a None/empty feature_type must fail as a clean "no
    such tool" lookup rather than crash the caller with an unrelated TypeError."""
    db = _db(None)

    result = tool_definitions_svc.get_tool_definition(db, None)

    assert result is None
    db.query.assert_not_called()


def test_different_feature_types_are_cached_independently(real_cache):
    recolor_db = _db(_tool("recolor"))
    creative_db = _db(_tool("creative_photoshoot"))

    tool_definitions_svc.get_tool_definition(recolor_db, "recolor")
    tool_definitions_svc.get_tool_definition(creative_db, "creative_photoshoot")

    assert redis_client.exists("tooldef:recolor")
    assert redis_client.exists("tooldef:creative_photoshoot")


def test_admin_cache_clear_also_clears_tool_definition_cache(real_cache):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.core.config import settings

    client = TestClient(app)
    tool_definitions_svc.get_tool_definition(_db(_tool()), "recolor")
    assert redis_client.exists("tooldef:recolor")

    res = client.post("/admin/cache/clear", headers={"x-cache-secret": settings.cache_clear_secret})

    assert res.status_code == 200
    assert not redis_client.exists("tooldef:recolor")


# --------------------------------------------------------------------------- #
# The autouse conftest fixture itself -- confirms every OTHER test in this
# suite really does get a permanent cache miss with no real Redis writes.
# --------------------------------------------------------------------------- #

def test_default_fixture_makes_lookups_a_permanent_miss_with_no_writes():
    db = _db(_tool())

    tool_definitions_svc.get_tool_definition(db, "recolor")
    tool_definitions_svc.get_tool_definition(db, "recolor")

    assert db.query.call_count == 2  # every call hit the DB mock -- never cached
    assert not redis_client.exists("tooldef:recolor")
