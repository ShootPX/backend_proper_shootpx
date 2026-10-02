"""app/services/usage.py::get_team_usage -- against the real dev DB (same
style as test_tools.py's /landing/tools tests), since mocking the
multi-level group-by/with_entities SQLAlchemy chain would test the mock, not
the actual query. Assertions are structural (shape + internal consistency),
not hardcoded totals, since this runs against real, growing dev data.
"""
from datetime import datetime, timedelta, timezone

from app.core.database import SessionLocal
from app.services import usage as usage_svc
import pytest

pytestmark = pytest.mark.real_db  # reads the real catalog / teams tables

REAL_TEAM_ID = "a85531bd-d19f-4865-8f2d-b148192fcf46"


def test_get_team_usage_shape():
    db = SessionLocal()
    try:
        result = usage_svc.get_team_usage(db, REAL_TEAM_ID, period="month")
    finally:
        db.close()

    assert "creditsUsed" in result
    assert "byMember" in result
    assert "byTool" in result
    assert isinstance(result["byMember"], list)
    assert isinstance(result["byTool"], list)


def test_get_team_usage_totals_are_internally_consistent():
    """creditsUsed must equal the sum of byTool (and of byMember) -- both are
    just different groupings of the exact same underlying completed-job set."""
    db = SessionLocal()
    try:
        result = usage_svc.get_team_usage(db, REAL_TEAM_ID, period="month")
    finally:
        db.close()

    assert sum(t["credits"] for t in result["byTool"]) == result["creditsUsed"]
    assert sum(m["credits"] for m in result["byMember"]) == result["creditsUsed"]


def test_get_team_usage_unknown_team_returns_zero():
    db = SessionLocal()
    try:
        result = usage_svc.get_team_usage(db, "00000000-0000-0000-0000-000000000000", period="month")
    finally:
        db.close()

    assert result == {"creditsUsed": 0, "byMember": [], "byTool": []}


def test_resolve_range_month_starts_at_first_of_month():
    start, end = usage_svc._resolve_range("month", None, None)
    now = datetime.now(timezone.utc)
    assert start.day == 1
    assert start.month == now.month
    assert end is None


def test_resolve_range_week_starts_at_this_weeks_monday():
    start, end = usage_svc._resolve_range("week", None, None)
    assert start.weekday() == 0  # Monday
    assert end is None


def test_resolve_range_explicit_custom_range_wins_over_period():
    custom_from = datetime(2026, 1, 1, tzinfo=timezone.utc)
    custom_to = datetime(2026, 1, 31, tzinfo=timezone.utc)
    start, end = usage_svc._resolve_range("month", custom_from, custom_to)
    assert start == custom_from
    assert end == custom_to


def test_usage_requires_auth():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    res = client.get(f"/teams/{REAL_TEAM_ID}/usage")
    assert res.status_code == 401
