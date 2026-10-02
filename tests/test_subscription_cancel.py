"""Subscription cancellation + team-deletion-cancels-Razorpay."""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.database import get_db
from app.deps import get_current_user
from app.main import app
from app.services import billing as billing_svc
from app.services import teams as teams_svc
from app.services import webhooks as webhooks_svc

TEAM_ID = uuid.uuid4()


# --------------------------------------------------------------------------- #
# cancel_subscription (service)
# --------------------------------------------------------------------------- #

def _cancel_db(team_sub):
    db = MagicMock()
    (db.query.return_value.filter.return_value
       .with_for_update.return_value.first.return_value) = team_sub
    return db


def test_cancel_no_subscription_raises_and_never_touches_razorpay(monkeypatch):
    calls = []
    monkeypatch.setattr(
        billing_svc.razorpay_client.subscription, "cancel", lambda sid: calls.append(sid)
    )
    db = _cancel_db(None)

    with pytest.raises(ValueError, match="no active subscription to cancel"):
        billing_svc.cancel_subscription(db, TEAM_ID)

    assert calls == []
    db.commit.assert_not_called()


def test_cancel_calls_razorpay_before_local_status_change(monkeypatch):
    team_sub = MagicMock(pending_razorpay_subscription_id=None, razorpay_subscription_id="sub_live_1", status="active")
    seen = {}

    def fake_cancel(sid):
        seen["sid"] = sid
        seen["status_at_call_time"] = team_sub.status  # must still be 'active'

    monkeypatch.setattr(billing_svc.razorpay_client.subscription, "cancel", fake_cancel)
    db = _cancel_db(team_sub)

    result = billing_svc.cancel_subscription(db, TEAM_ID)

    assert seen == {"sid": "sub_live_1", "status_at_call_time": "active"}
    assert team_sub.status == "cancelled"        # flipped only after Razorpay returned
    db.commit.assert_called_once()
    assert result is team_sub


def test_cancel_razorpay_failure_leaves_status_unchanged(monkeypatch):
    team_sub = MagicMock(pending_razorpay_subscription_id=None, razorpay_subscription_id="sub_live_1", status="active")

    def boom(sid):
        raise RuntimeError("razorpay 500")

    monkeypatch.setattr(billing_svc.razorpay_client.subscription, "cancel", boom)
    db = _cancel_db(team_sub)

    with pytest.raises(billing_svc.RazorpayCancelError):
        billing_svc.cancel_subscription(db, TEAM_ID)

    assert team_sub.status == "active"           # NOT falsely 'cancelled'
    db.commit.assert_not_called()


def test_cancel_pending_row_with_stored_id_still_calls_razorpay(monkeypatch):
    """Checkout now stores the real id on the pending row, so cancelling during
    the pending window DOES reach Razorpay."""
    calls = []
    monkeypatch.setattr(
        billing_svc.razorpay_client.subscription, "cancel", lambda sid: calls.append(sid)
    )
    team_sub = MagicMock(pending_razorpay_subscription_id=None, razorpay_subscription_id="sub_live_7", status="pending")
    db = _cancel_db(team_sub)

    billing_svc.cancel_subscription(db, TEAM_ID)

    assert calls == ["sub_live_7"]
    assert team_sub.status == "cancelled"


def test_pending_cancel_then_late_activated_does_not_resurrect(monkeypatch):
    """The full sequence: pending row with a real id -> owner cancels (Razorpay
    reached) -> a late subscription.activated for that id is a clean no-op."""
    SUB = "sub_live_42"
    row = MagicMock(status="pending", razorpay_subscription_id=SUB, pending_razorpay_subscription_id=None,
                    subscription_id=uuid.uuid4(), team_id=str(TEAM_ID))

    # cancel during the pending window
    cancel_calls = []
    monkeypatch.setattr(billing_svc.razorpay_client.subscription, "cancel",
                        lambda sid: cancel_calls.append(sid))
    billing_svc.cancel_subscription(_cancel_db(row), TEAM_ID)
    assert cancel_calls == [SUB]          # id not null -> Razorpay actually told to stop
    assert row.status == "cancelled"

    # late-arriving subscription.activated for the same id
    refills = []
    fetch = MagicMock()
    monkeypatch.setattr(webhooks_svc, "refill_subscription_credits",
                        lambda *a, **k: refills.append(a))
    monkeypatch.setattr(webhooks_svc.razorpay_client.subscription, "fetch", fetch)

    wdb = MagicMock()
    (wdb.query.return_value.filter.return_value
       .with_for_update.return_value.first.return_value) = row
    webhooks_svc.handle_subscription_activated(
        wdb,
        {"event": "subscription.activated",
         "payload": {"subscription": {"entity": {"id": SUB}}}},
    )

    assert row.status == "cancelled"      # NOT resurrected
    assert refills == []                  # no credits granted
    fetch.assert_not_called()             # short-circuited before any work
    wdb.commit.assert_not_called()


# --------------------------------------------------------------------------- #
# POST /billing/teams/{team_id}/subscriptions/cancel (route)
# --------------------------------------------------------------------------- #

@pytest.fixture
def cancel_client(monkeypatch):
    fake_user = MagicMock(id=uuid.uuid4())
    state = {"team_sub": MagicMock(pending_razorpay_subscription_id=None, razorpay_subscription_id="sub_live_1", status="active"),
             "razorpay_ok": True}

    db = MagicMock()
    (db.query.return_value.filter.return_value
       .with_for_update.return_value.first.side_effect) = lambda: state["team_sub"]

    def cancel(sid):
        if not state["razorpay_ok"]:
            raise RuntimeError("razorpay down")

    monkeypatch.setattr(billing_svc.razorpay_client.subscription, "cancel", cancel)

    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr("app.routes.checkout.is_team_owner", lambda db, tid, uid: True)

    yield TestClient(app, raise_server_exceptions=False), state
    app.dependency_overrides.clear()


def _url():
    return f"/billing/teams/{TEAM_ID}/subscriptions/cancel"


def test_cancel_route_requires_owner(cancel_client, monkeypatch):
    client, state = cancel_client
    monkeypatch.setattr("app.routes.checkout.is_team_owner", lambda db, tid, uid: False)
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 403


def test_cancel_route_success(cancel_client):
    client, state = cancel_client
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 200
    assert res.json() == {"status": "cancelled", "teamId": str(TEAM_ID)}
    assert state["team_sub"].status == "cancelled"


def test_cancel_route_no_subscription_400(cancel_client):
    client, state = cancel_client
    state["team_sub"] = None
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 400


def test_cancel_route_razorpay_failure_400(cancel_client):
    client, state = cancel_client
    state["razorpay_ok"] = False
    res = client.post(_url(), headers={"Authorization": "Bearer x"})
    assert res.status_code == 400
    assert "razorpay" in res.json()["detail"].lower()
    assert state["team_sub"].status == "active"      # unchanged


# --------------------------------------------------------------------------- #
# soft_delete_team -> cancel_subscription (money stops immediately; see
# services/teams.py's two-clock deletion design). The actual row cleanup
# only happens later, in purge_team (called by the grace-period sweep).
# --------------------------------------------------------------------------- #

def _fake_team(deleted_at=None):
    from unittest.mock import MagicMock
    return MagicMock(id=TEAM_ID, deleted_at=deleted_at)


def test_soft_delete_team_cancels_subscription_first(monkeypatch):
    calls = []
    monkeypatch.setattr(teams_svc, "cancel_subscription", lambda db, tid: calls.append(tid))
    monkeypatch.setattr(teams_svc, "get_team", lambda db, tid: _fake_team())
    db = MagicMock()

    teams_svc.soft_delete_team(db, TEAM_ID)

    assert calls == [TEAM_ID]
    db.commit.assert_called_once()


def test_soft_delete_team_with_no_subscription_still_deletes(monkeypatch):
    def no_sub(db, tid):
        raise ValueError("This team has no active subscription to cancel")

    monkeypatch.setattr(teams_svc, "cancel_subscription", no_sub)
    monkeypatch.setattr(teams_svc, "get_team", lambda db, tid: _fake_team())
    db = MagicMock()

    teams_svc.soft_delete_team(db, TEAM_ID)  # must not raise

    db.commit.assert_called_once()


def test_soft_delete_team_rejects_an_already_deleted_team(monkeypatch):
    monkeypatch.setattr(teams_svc, "get_team", lambda db, tid: _fake_team(deleted_at=datetime.now(timezone.utc)))
    db = MagicMock()

    with pytest.raises(ValueError, match="already scheduled"):
        teams_svc.soft_delete_team(db, TEAM_ID)


def test_soft_delete_team_razorpay_failure_still_deletes_and_logs(monkeypatch, caplog):
    def boom(db, tid):
        raise billing_svc.RazorpayCancelError("provider down")

    monkeypatch.setattr(teams_svc, "cancel_subscription", boom)
    monkeypatch.setattr(teams_svc, "get_team", lambda db, tid: _fake_team())
    db = MagicMock()

    with caplog.at_level(logging.ERROR, logger="app.services.teams"):
        teams_svc.soft_delete_team(db, TEAM_ID)

    db.commit.assert_called_once()
    assert "MANUAL FOLLOW-UP NEEDED" in caplog.text
    # exc_info=True attaches the traceback
    assert any(r.exc_info for r in caplog.records)


# --------------------------------------------------------------------------- #
# purge_team -- the actual hard-delete, run only by the grace-period sweep
# --------------------------------------------------------------------------- #

def test_purge_team_clears_generation_jobs_before_deleting_the_team():
    """Real bug found live: generation_jobs.team_id is also a FK to teams, but
    wasn't in the original cleanup list -- any team that ever ran a single
    generation hit a real Postgres FK violation deleting the Team row. Must
    be cleared, and before Team itself (FK-child-first ordering)."""
    from app.models.generation_job import GenerationJob
    from app.models.team import Team

    db = MagicMock()

    teams_svc.purge_team(db, TEAM_ID)

    queried_models = [c.args[0] for c in db.query.call_args_list]
    assert GenerationJob in queried_models
    assert queried_models.index(GenerationJob) < queried_models.index(Team)


def test_restore_team_within_grace_window(monkeypatch):
    team = _fake_team(deleted_at=datetime.now(timezone.utc) - timedelta(days=5))
    monkeypatch.setattr(teams_svc, "get_team", lambda db, tid: team)
    db = MagicMock()

    result = teams_svc.restore_team(db, TEAM_ID)

    assert result.deleted_at is None
    db.commit.assert_called_once()


def test_restore_team_rejects_after_grace_window_expires(monkeypatch):
    team = _fake_team(deleted_at=datetime.now(timezone.utc) - timedelta(days=teams_svc.GRACE_PERIOD_DAYS + 1))
    monkeypatch.setattr(teams_svc, "get_team", lambda db, tid: team)
    db = MagicMock()

    with pytest.raises(ValueError, match="expired"):
        teams_svc.restore_team(db, TEAM_ID)


def test_restore_team_rejects_a_team_that_was_never_deleted(monkeypatch):
    monkeypatch.setattr(teams_svc, "get_team", lambda db, tid: _fake_team(deleted_at=None))
    db = MagicMock()

    with pytest.raises(ValueError, match="not scheduled"):
        teams_svc.restore_team(db, TEAM_ID)


# --------------------------------------------------------------------------- #
# Exact boundary-day tests -- restore_team's real enforcement is
# GRACE_PERIOD_DAYS (30, internal), NOT GRACE_PERIOD_PUBLIC_DAYS (15, the
# number the DELETE response's `recoverableUntil` shows the frontend). A
# team can be genuinely restorable via this API for up to 15 days AFTER the
# UI has told the user their window closed -- that gap is deliberate safety
# margin, not a bug, but it means the frontend must stop OFFERING restore at
# day 15 even though the backend would still honor a call past it.
#
# `_FixedClock` freezes `teams_svc.datetime.now()` to an exact instant so
# these assertions land on the exact day, not "roughly around" it (no
# freezegun dependency in this project -- this is the same fixed-clock
# monkeypatch trick, scoped to just this module).
# --------------------------------------------------------------------------- #

class _FixedClock(datetime):
    _fixed_now = None

    @classmethod
    def now(cls, tz=None):
        return cls._fixed_now


def _freeze_teams_clock(monkeypatch, fixed_now):
    frozen = type("_FixedClock", (_FixedClock,), {"_fixed_now": fixed_now})
    monkeypatch.setattr(teams_svc, "datetime", frozen)
    return frozen


FIXED_NOW = datetime(2026, 1, 31, 12, 0, 0, tzinfo=timezone.utc)


def test_restore_public_boundary_day_15_minus_seconds_still_succeeds(monkeypatch):
    """15 days is ONLY the public-facing number shown in `recoverableUntil` --
    restore_team itself does not enforce it at all. Confirms that explicitly:
    a team just under 15 days old restores fine (as expected either way)."""
    _freeze_teams_clock(monkeypatch, FIXED_NOW)
    team = _fake_team(deleted_at=FIXED_NOW - timedelta(days=15) + timedelta(seconds=5))
    monkeypatch.setattr(teams_svc, "get_team", lambda db, tid: team)
    db = MagicMock()

    result = teams_svc.restore_team(db, TEAM_ID)

    assert result.deleted_at is None


def test_restore_public_boundary_day_15_plus_seconds_still_succeeds_too(monkeypatch):
    """The real point of this test: restore does NOT reject at day 15 + a few
    seconds, because 15 is not the enforced boundary -- 30 is. If this ever
    starts failing, GRACE_PERIOD_PUBLIC_DAYS leaked into the enforcement
    check somewhere, which would silently break restores between day 15 and
    day 30 that the two-clock design explicitly promises to still honor."""
    _freeze_teams_clock(monkeypatch, FIXED_NOW)
    team = _fake_team(deleted_at=FIXED_NOW - timedelta(days=15) - timedelta(seconds=5))
    monkeypatch.setattr(teams_svc, "get_team", lambda db, tid: team)
    db = MagicMock()

    result = teams_svc.restore_team(db, TEAM_ID)  # must NOT raise

    assert result.deleted_at is None


def test_restore_internal_boundary_day_29_succeeds(monkeypatch):
    _freeze_teams_clock(monkeypatch, FIXED_NOW)
    team = _fake_team(deleted_at=FIXED_NOW - timedelta(days=29))
    monkeypatch.setattr(teams_svc, "get_team", lambda db, tid: team)
    db = MagicMock()

    result = teams_svc.restore_team(db, TEAM_ID)

    assert result.deleted_at is None


def test_restore_internal_boundary_exactly_day_30_still_succeeds(monkeypatch):
    """Exactly GRACE_PERIOD_DAYS elapsed (to the second) is still inside the
    window -- the check is a strict `>`, so equality does not reject. This is
    also the exact instant list_teams_past_grace_period's own boundary was
    aligned to (see that function's docstring) so the two never overlap."""
    _freeze_teams_clock(monkeypatch, FIXED_NOW)
    team = _fake_team(deleted_at=FIXED_NOW - timedelta(days=teams_svc.GRACE_PERIOD_DAYS))
    monkeypatch.setattr(teams_svc, "get_team", lambda db, tid: team)
    db = MagicMock()

    result = teams_svc.restore_team(db, TEAM_ID)  # must NOT raise

    assert result.deleted_at is None


def test_restore_internal_boundary_day_30_plus_one_second_fails(monkeypatch):
    _freeze_teams_clock(monkeypatch, FIXED_NOW)
    team = _fake_team(deleted_at=FIXED_NOW - timedelta(days=teams_svc.GRACE_PERIOD_DAYS) - timedelta(seconds=1))
    monkeypatch.setattr(teams_svc, "get_team", lambda db, tid: team)
    db = MagicMock()

    with pytest.raises(ValueError, match="expired"):
        teams_svc.restore_team(db, TEAM_ID)


def test_restore_internal_boundary_day_31_fails(monkeypatch):
    _freeze_teams_clock(monkeypatch, FIXED_NOW)
    team = _fake_team(deleted_at=FIXED_NOW - timedelta(days=31))
    monkeypatch.setattr(teams_svc, "get_team", lambda db, tid: team)
    db = MagicMock()

    with pytest.raises(ValueError, match="expired"):
        teams_svc.restore_team(db, TEAM_ID)


# --------------------------------------------------------------------------- #
# Purge sweep boundary -- against the REAL database, not a mocked query
# chain. list_teams_past_grace_period's boundary is a SQLAlchemy column
# expression (Team.deleted_at < cutoff) that a MagicMock db can't actually
# evaluate -- a mock-based test here would only prove the test's own
# Python-side re-implementation of the predicate, not the real SQL. Uses
# real, disposable teams (created and purged within the test) against the
# live dev DB, same pattern as this session's earlier soft-delete smoke test.
# --------------------------------------------------------------------------- #

@pytest.mark.real_db
def test_purge_sweep_exact_boundary_against_real_db():
    from app.core.database import SessionLocal
    from app.models.team import Team

    db = SessionLocal()
    now = datetime.now(timezone.utc)
    created_ids = []
    try:
        # +/-60s margins (not a mathematically exact zero-drift instant) --
        # this runs against the REAL, unfrozen wall clock (list_teams_past_
        # grace_period calls datetime.now() itself, after 4 sequential
        # INSERT+flush round trips to the real (Supabase-pooled, real
        # network latency) DB following `now` being captured here) -- a
        # true zero-margin boundary test would be flaky against that
        # latency, and a 2s margin measured flaky in practice (this test
        # failed once at 2s from real round-trip drift alone, confirmed by
        # re-running with more margin). 60s comfortably exceeds any
        # realistic round-trip here while staying negligible against a
        # 30-DAY window. The exact, zero-drift instant is already covered
        # precisely by the frozen-clock restore_team tests above, which
        # share the same GRACE_PERIOD_DAYS constant and the same intended
        # boundary.
        offsets_days = {
            "day_29": 29,                                                          # must NOT be purged
            "day_30_minus_60s": teams_svc.GRACE_PERIOD_DAYS - (60 / 86400),         # must NOT be purged (still inside)
            "day_30_plus_60s": teams_svc.GRACE_PERIOD_DAYS + (60 / 86400),          # MUST be purged (just past)
            "day_31": 31,                                                          # must be purged
        }
        teams_by_label = {}
        for label, days in offsets_days.items():
            team = Team(name=f"Boundary test {label}", deleted_at=now - timedelta(days=days))
            db.add(team)
            db.flush()
            teams_by_label[label] = team.id
            created_ids.append(team.id)
        db.commit()

        eligible_ids = {t.id for t in teams_svc.list_teams_past_grace_period(db)}

        assert teams_by_label["day_29"] not in eligible_ids
        assert teams_by_label["day_30_minus_60s"] not in eligible_ids
        assert teams_by_label["day_30_plus_60s"] in eligible_ids
        assert teams_by_label["day_31"] in eligible_ids
    finally:
        for team_id in created_ids:
            teams_svc.purge_team(db, team_id)
        db.close()


@pytest.mark.real_db
def test_day_30_restore_and_purge_cannot_both_succeed_on_the_same_team(monkeypatch):
    """The actual regression this fix closes: ONE real team, sitting at
    EXACTLY GRACE_PERIOD_DAYS, checked against BOTH operations. Before the
    `<=` -> `<` fix in list_teams_past_grace_period, this exact team would
    have been simultaneously purge-ELIGIBLE (old: elapsed >= 30 days) and
    restore-ACCEPTED (restore_team: elapsed > 30 days is the only rejection
    condition) -- a live race between the daily sweep and an owner's restore
    click, where whichever ran first would decide the team's fate out from
    under the other.

    Proves the race is closed by construction, not just "each boundary is
    independently correct": the purge sweep's own eligibility LIST for the
    whole table must exclude this team's id, which means restore is the ONLY
    code path that can act on it at this exact instant -- there is no window
    where a concurrent purge run could also grab it.

    Uses the frozen clock (not real wall-clock `now`) for both operations --
    list_teams_past_grace_period runs a REAL SQL comparison in Postgres, but
    the *cutoff value* it sends is computed from teams_svc.datetime.now() in
    Python first, so freezing that clock pins the cutoff to an exact instant
    regardless of real DB round-trip latency between the insert and the
    query (the same drift that made the plain wall-clock version of this
    kind of test measurably flaky earlier in this file at a 2s margin).
    """
    from app.core.database import SessionLocal
    from app.models.team import Team

    frozen = _freeze_teams_clock(monkeypatch, FIXED_NOW)
    db = SessionLocal()
    team = Team(name="Race boundary team", deleted_at=frozen._fixed_now - timedelta(days=teams_svc.GRACE_PERIOD_DAYS))
    db.add(team)
    db.commit()
    team_id = team.id

    try:
        # 1. The purge sweep's eligibility list must NOT contain this team --
        #    if it did, a sweep running right now could hard-delete it out
        #    from under a concurrent restore call.
        purge_eligible_ids = {t.id for t in teams_svc.list_teams_past_grace_period(db)}
        assert team_id not in purge_eligible_ids

        # 2. Restore on this EXACT SAME row, at this EXACT SAME frozen
        #    instant, succeeds -- confirming restore is genuinely still the
        #    only thing that can happen to this team right now, not that it
        #    merely happens to also be excluded from both operations.
        restored = teams_svc.restore_team(db, team_id)
        assert restored.deleted_at is None
    finally:
        teams_svc.purge_team(db, team_id)
        db.close()
