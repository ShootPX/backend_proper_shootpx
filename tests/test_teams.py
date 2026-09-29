import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

FAKE_TEAM = str(uuid.uuid4())
FAKE_TOKEN = "sometoken"


def test_list_teams_requires_auth():
    res = client.get("/teams")
    assert res.status_code == 401


def test_create_team_requires_auth():
    res = client.post("/teams", json={"name": "New Team"})
    assert res.status_code == 401


def test_create_team_rejects_empty_name():
    from unittest.mock import MagicMock

    from app.services.teams import create_team

    with pytest.raises(ValueError, match="empty"):
        create_team(MagicMock(), "u1", "   ")


def test_create_team_makes_caller_the_owner():
    from unittest.mock import MagicMock

    from app.models.team import Team
    from app.models.team_member import TeamMember
    from app.services.teams import create_team

    db = MagicMock()
    added = []
    db.add.side_effect = added.append

    def fake_flush():
        for obj in added:
            if isinstance(obj, Team) and obj.id is None:
                obj.id = FAKE_TEAM

    db.flush.side_effect = fake_flush

    result = create_team(db, "u1", "  New Team  ")

    assert result == {"id": str(FAKE_TEAM), "name": "New Team", "role": "owner"}
    member = next(obj for obj in added if isinstance(obj, TeamMember))
    assert member.role == "owner"
    assert member.user_id == "u1"


def test_create_team_starts_with_zero_credits():
    """The signup bonus is a one-time, anti-abuse-capped grant tied to
    account creation -- a manually created extra team must NOT get it too,
    or a single account could farm free credits by just creating teams."""
    from unittest.mock import MagicMock

    from app.models.team import Team
    from app.services.teams import create_team

    db = MagicMock()
    added = []
    db.add.side_effect = added.append

    create_team(db, "u1", "New Team")

    team = next(obj for obj in added if isinstance(obj, Team))
    assert team.topup_credits_balance in (0, None)  # ORM default, never set explicitly here


def test_invite_requires_auth():
    res = client.post(f"/teams/{FAKE_TEAM}/invite", json={"email": "a@b.com"})
    assert res.status_code == 401


def test_accept_requires_auth():
    res = client.post(f"/invites/{FAKE_TOKEN}/accept")
    assert res.status_code == 401


def test_members_requires_auth():
    res = client.get(f"/teams/{FAKE_TEAM}/members")
    assert res.status_code == 401


def test_rename_requires_auth():
    res = client.patch(f"/teams/{FAKE_TEAM}", json={"name": "X"})
    assert res.status_code == 401


def test_remove_member_requires_auth():
    res = client.delete(f"/teams/{FAKE_TEAM}/members/{uuid.uuid4()}")
    assert res.status_code == 401


def test_delete_team_requires_auth():
    res = client.delete(f"/teams/{FAKE_TEAM}")
    assert res.status_code == 401


def test_list_invites_requires_auth():
    res = client.get(f"/teams/{FAKE_TEAM}/invites")
    assert res.status_code == 401


def test_cancel_invite_requires_auth():
    res = client.delete(f"/teams/{FAKE_TEAM}/invites/{uuid.uuid4()}")
    assert res.status_code == 401


def test_remove_member_blocks_last_owner(monkeypatch):
    from unittest.mock import MagicMock

    from app.services import teams

    fake_member = MagicMock(role="owner")
    monkeypatch.setattr(teams, "_membership", lambda db, tid, uid: fake_member)
    monkeypatch.setattr(teams, "owner_count", lambda db, tid: 1)

    try:
        teams.remove_member(MagicMock(), FAKE_TEAM, "u1")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "last owner" in str(e).lower()


def test_invite_validates_body():
    res = client.post(
        f"/teams/{FAKE_TEAM}/invite",
        json={},  # missing email
        headers={"Authorization": "Bearer not-real"},
    )
    # rejected — either body validation (422) or auth (401), never a 500
    assert res.status_code in (401, 422)


def test_invite_rejects_bad_role():
    res = client.post(
        f"/teams/{FAKE_TEAM}/invite",
        json={"email": "a@b.com", "role": "superadmin"},
        headers={"Authorization": "Bearer not-real"},
    )
    # invalid role -> 422 body validation (or 401 if auth is checked first)
    assert res.status_code in (401, 422)


def _invite_client(monkeypatch):
    """Authenticated owner, with create_invite stubbed to record its role --
    so these tests get PAST auth and exercise the role validation itself
    (a bare unauthenticated call returns 401 for any body, proving nothing)."""
    import uuid as uuid_mod
    from unittest.mock import MagicMock

    from app.core.database import get_db
    from app.deps import get_current_user
    from app.main import app
    from app.routes import teams as teams_route

    app.dependency_overrides[get_current_user] = lambda: MagicMock(id=uuid_mod.uuid4())
    app.dependency_overrides[get_db] = lambda: MagicMock()
    monkeypatch.setattr(teams_route, "is_team_owner", lambda db, tid, uid: True)
    seen = []

    def fake_create_invite(db, team_id, email, invited_by, role):
        seen.append(role)
        return MagicMock(email=email, role=role)

    monkeypatch.setattr(teams_route, "create_invite", fake_create_invite)
    return TestClient(app), seen


def test_teams_have_exactly_two_roles_owner_and_editor():
    from app.services import team_invites

    assert team_invites.ALLOWED_ROLES == ("editor", "owner")


def test_invite_rejects_viewer_role(monkeypatch):
    """There is no viewer role -- only owner and editor exist."""
    from app.main import app

    client, seen = _invite_client(monkeypatch)
    try:
        res = client.post(
            f"/teams/{FAKE_TEAM}/invite",
            json={"email": "a@b.com", "role": "viewer"},
            headers={"Authorization": "Bearer x"},
        )
    finally:
        app.dependency_overrides.clear()

    assert res.status_code == 422
    assert seen == []  # never reached the service


@pytest.mark.parametrize("role", ["editor", "owner"])
def test_invite_accepts_both_real_roles(monkeypatch, role):
    from app.main import app

    client, seen = _invite_client(monkeypatch)
    try:
        res = client.post(
            f"/teams/{FAKE_TEAM}/invite",
            json={"email": "a@b.com", "role": role},
            headers={"Authorization": "Bearer x"},
        )
    finally:
        app.dependency_overrides.clear()

    assert res.status_code == 200
    assert res.json()["role"] == role
    assert seen == [role]


def test_invite_role_defaults_to_editor(monkeypatch):
    from app.main import app

    client, seen = _invite_client(monkeypatch)
    try:
        res = client.post(
            f"/teams/{FAKE_TEAM}/invite", json={"email": "a@b.com"},
            headers={"Authorization": "Bearer x"},
        )
    finally:
        app.dependency_overrides.clear()

    assert res.status_code == 200
    assert seen == ["editor"]


def test_generations_requires_auth():
    res = client.get(f"/teams/{FAKE_TEAM}/generations")
    assert res.status_code == 401


def test_usage_requires_auth():
    res = client.get(f"/teams/{FAKE_TEAM}/usage")
    assert res.status_code == 401


def test_restore_requires_auth():
    res = client.post(f"/teams/{FAKE_TEAM}/restore")
    assert res.status_code == 401


@pytest.mark.real_db
def test_soft_deleted_team_blocks_member_and_owner_access_real_db():
    """The access-blocking guarantee behind soft deletion, against the real
    database. is_team_member/is_team_owner now enforce it in a single joined
    query (`WHERE teams.deleted_at IS NULL`) rather than a membership lookup
    followed by a separate team lookup -- a SQL predicate a MagicMock cannot
    evaluate, so this has to hit real Postgres to mean anything. Uses a
    disposable team + a real user (team_members.user_id is a FK), purged at
    the end."""
    from datetime import datetime, timezone

    from app.core.database import SessionLocal
    from app.models.team import Team
    from app.models.team_member import TeamMember
    from app.models.user import User
    from app.services import teams as teams_svc

    db = SessionLocal()
    real_user = db.query(User).first()
    team = Team(name="Access gate test")
    db.add(team)
    db.flush()
    db.add(TeamMember(team_id=team.id, user_id=real_user.id, role="owner"))
    db.commit()

    try:
        # Alive: the owner has access.
        assert teams_svc.is_team_member(db, team.id, real_user.id) is True
        assert teams_svc.is_team_owner(db, team.id, real_user.id) is True

        team.deleted_at = datetime.now(timezone.utc)
        db.commit()

        # Soft-deleted: same membership row, same user -- access is gone.
        assert teams_svc.is_team_member(db, team.id, real_user.id) is False
        assert teams_svc.is_team_owner(db, team.id, real_user.id) is False
        # ...except through the one sanctioned bypass, so restore stays reachable.
        assert teams_svc.is_team_owner_including_deleted(db, team.id, real_user.id) is True
    finally:
        teams_svc.purge_team(db, team.id)
        db.close()


def test_is_team_owner_including_deleted_still_true_for_deleted_team(monkeypatch):
    from unittest.mock import MagicMock

    from app.services import teams as teams_svc

    fake_member = MagicMock(role="owner")
    monkeypatch.setattr(teams_svc, "_membership", lambda db, tid, uid: fake_member)

    # Deliberately does NOT check deleted_at at all -- this is the one
    # sanctioned bypass, used only by the delete/restore routes themselves.
    assert teams_svc.is_team_owner_including_deleted(MagicMock(), FAKE_TEAM, "u1") is True


def test_delete_team_route_returns_recoverable_until(monkeypatch):
    import uuid as uuid_mod
    from datetime import datetime, timedelta, timezone
    from unittest.mock import MagicMock

    from app.core.database import get_db
    from app.deps import get_current_user
    from app.main import app
    from app.routes import teams as teams_route

    fake_user = MagicMock(id=uuid_mod.uuid4())
    db = MagicMock()
    deleted_at = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)
    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr(teams_route, "is_team_owner_including_deleted", lambda db, tid, uid: True)
    monkeypatch.setattr(teams_route, "soft_delete_team", lambda db, tid: deleted_at)

    try:
        tc = TestClient(app)
        res = tc.delete(f"/teams/{FAKE_TEAM}", headers={"Authorization": "Bearer x"})
    finally:
        app.dependency_overrides.clear()

    assert res.status_code == 200
    body = res.json()
    assert body["deleted"] == str(FAKE_TEAM)
    assert body["recoverableUntil"] == (deleted_at + timedelta(days=teams_route.GRACE_PERIOD_PUBLIC_DAYS)).isoformat()


def test_delete_team_route_already_deleted_returns_400(monkeypatch):
    import uuid as uuid_mod
    from unittest.mock import MagicMock

    from app.core.database import get_db
    from app.deps import get_current_user
    from app.main import app
    from app.routes import teams as teams_route

    fake_user = MagicMock(id=uuid_mod.uuid4())
    db = MagicMock()
    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr(teams_route, "is_team_owner_including_deleted", lambda db, tid, uid: True)

    def already_deleted(db, tid):
        raise ValueError("This team is already scheduled for deletion")

    monkeypatch.setattr(teams_route, "soft_delete_team", already_deleted)

    try:
        tc = TestClient(app)
        res = tc.delete(f"/teams/{FAKE_TEAM}", headers={"Authorization": "Bearer x"})
    finally:
        app.dependency_overrides.clear()

    assert res.status_code == 400


def test_notify_team_restored_emails_every_member(monkeypatch):
    from unittest.mock import MagicMock, call

    from app.services import teams as teams_svc

    fake_team = MagicMock(name="Team", id=FAKE_TEAM)
    fake_team.name = "Whats Up Bro"
    monkeypatch.setattr(teams_svc, "get_team", lambda db, tid: fake_team)
    monkeypatch.setattr(teams_svc, "list_team_members", lambda db, tid: [
        {"email": "priya@brandco.com"}, {"email": "jordan@brandco.com"}, {"email": None},
    ])
    sent = []
    monkeypatch.setattr("app.core.email.send_email", lambda **kwargs: sent.append(kwargs["to"]))

    teams_svc.notify_team_restored(MagicMock(), FAKE_TEAM)

    assert sent == ["priya@brandco.com", "jordan@brandco.com"]  # None email skipped, not a crash


def test_notify_team_restored_one_bad_email_does_not_stop_the_rest(monkeypatch):
    from unittest.mock import MagicMock

    from app.services import teams as teams_svc

    fake_team = MagicMock(id=FAKE_TEAM)
    fake_team.name = "Whats Up Bro"
    monkeypatch.setattr(teams_svc, "get_team", lambda db, tid: fake_team)
    monkeypatch.setattr(teams_svc, "list_team_members", lambda db, tid: [
        {"email": "broken@brandco.com"}, {"email": "fine@brandco.com"},
    ])

    def flaky_send(**kwargs):
        if kwargs["to"] == "broken@brandco.com":
            raise RuntimeError("SMTP down")
        sent.append(kwargs["to"])

    sent = []
    monkeypatch.setattr("app.core.email.send_email", flaky_send)

    teams_svc.notify_team_restored(MagicMock(), FAKE_TEAM)  # must not raise

    assert sent == ["fine@brandco.com"]


def test_restore_route_still_succeeds_if_notification_email_fails(monkeypatch):
    """The restore itself must succeed even if notify_team_restored blows up
    unexpectedly -- it's a best-effort side effect, never a reason to turn a
    committed restore into a failed response."""
    import uuid as uuid_mod
    from unittest.mock import MagicMock

    from app.core.database import get_db
    from app.deps import get_current_user
    from app.main import app
    from app.routes import teams as teams_route

    fake_user = MagicMock(id=uuid_mod.uuid4())
    db = MagicMock()
    restored_team = MagicMock(id=FAKE_TEAM)
    restored_team.name = "Whats Up Bro"
    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr(teams_route, "is_team_owner_including_deleted", lambda db, tid, uid: True)
    monkeypatch.setattr(teams_route, "restore_team", lambda db, tid: restored_team)

    def boom(db, tid):
        raise RuntimeError("unexpected notify failure")

    monkeypatch.setattr(teams_route, "notify_team_restored", boom)

    try:
        tc = TestClient(app)
        res = tc.post(f"/teams/{FAKE_TEAM}/restore", headers={"Authorization": "Bearer x"})
    finally:
        app.dependency_overrides.clear()

    assert res.status_code == 200
    assert res.json()["restored"] is True


def test_restore_team_route_success(monkeypatch):
    import uuid as uuid_mod
    from unittest.mock import MagicMock

    from app.core.database import get_db
    from app.deps import get_current_user
    from app.main import app
    from app.routes import teams as teams_route

    fake_user = MagicMock(id=uuid_mod.uuid4())
    db = MagicMock()
    restored_team = MagicMock(id=FAKE_TEAM)
    restored_team.name = "Whats Up Bro"  # MagicMock(name=...) is reserved for the mock's own repr, not an attribute
    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr(teams_route, "is_team_owner_including_deleted", lambda db, tid, uid: True)
    monkeypatch.setattr(teams_route, "restore_team", lambda db, tid: restored_team)

    try:
        tc = TestClient(app)
        res = tc.post(f"/teams/{FAKE_TEAM}/restore", headers={"Authorization": "Bearer x"})
    finally:
        app.dependency_overrides.clear()

    assert res.status_code == 200
    assert res.json() == {"id": str(FAKE_TEAM), "name": "Whats Up Bro", "restored": True}


def test_list_team_generations_includes_batch_id_and_output_url():
    import uuid as uuid_mod
    from datetime import datetime, timezone
    from unittest.mock import MagicMock

    from app.services.generation import list_team_generations

    job_id = uuid_mod.uuid4()
    batch_id = uuid_mod.uuid4()
    created_at = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)
    fake_job = MagicMock(
        id=job_id, batch_id=batch_id, title="My shoot", feature_type="recolor",
        status="completed", created_at=created_at, output_url="https://x/1.png",
    )
    db = MagicMock()
    query = db.query.return_value
    query.filter.return_value = query
    query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [fake_job]

    result = list_team_generations(db, FAKE_TEAM, limit=10, offset=0)

    assert result == [{
        "jobId": str(job_id),
        "batchId": str(batch_id),
        "title": "My shoot",
        "featureType": "recolor",
        "status": "completed",
        "createdAt": created_at.isoformat(),
        "outputUrl": "https://x/1.png",
        "deeplink": {"type": "page", "route": "library", "params": {"job_id": str(job_id)}},
    }]


def test_list_team_generations_applies_feature_type_and_user_filters():
    from unittest.mock import MagicMock

    from app.services.generation import list_team_generations

    db = MagicMock()
    query = db.query.return_value
    query.filter.return_value = query
    query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []

    list_team_generations(db, FAKE_TEAM, feature_type="recolor", user_id="u1")

    # 2 filter() calls: the base conditions (team_id + feature_type, one call)
    # and the user_id filter. The real filtering behavior is proven against
    # Postgres in test_generations_status_and_internal_tool_filters_real_db.
    assert query.filter.call_count == 2


@pytest.mark.real_db
def test_generations_status_and_internal_tool_filters_real_db():
    """status + user_id + the default internal-tool exclusion, against real
    Postgres (the predicates are SQL a MagicMock can't evaluate). Disposable
    team with a controlled mix of jobs across two real users, purged after."""
    from app.core.database import SessionLocal
    from app.models.generation_job import GenerationJob
    from app.models.team import Team
    from app.models.user import User
    from app.services import teams as teams_svc
    from app.services.generation import list_team_generations

    db = SessionLocal()
    users = db.query(User).limit(2).all()
    assert len(users) == 2, "needs two real users to prove the user_id filter"
    a, b = users
    team = Team(name="Generations filter test")
    db.add(team)
    db.flush()

    def job(user, feature_type, status, output="https://x/o.png"):
        j = GenerationJob(
            team_id=team.id, user_id=user.id, feature_type=feature_type,
            status=status, input_params={}, credits_charged=1,
            output_url=output if status == "completed" else None,
        )
        db.add(j)
        db.flush()
        return str(j.id)

    a_done = job(a, "recolor", "completed")
    a_failed = job(a, "recolor", "failed")
    a_processing = job(a, "model_shoot", "processing")
    a_enhance = job(a, "enhance_prompt", "completed")
    a_gen_model = job(a, "model_shoot_generate_model", "completed")
    b_done = job(b, "recolor", "completed")
    db.commit()

    try:
        def ids(**kw):
            return {g["jobId"] for g in list_team_generations(db, team.id, limit=50, **kw)}

        # the exact Home "Recent Work" query
        assert ids(user_id=a.id, status="completed") == {a_done}

        # internal tools excluded by default, even with no other filter
        everything = ids()
        assert a_enhance not in everything and a_gen_model not in everything
        assert everything == {a_done, a_failed, a_processing, b_done}

        # status alone
        assert ids(status="failed") == {a_failed}
        assert ids(status="processing") == {a_processing}
        assert ids(status="completed") == {a_done, b_done}

        # user_id alone: only that user's, still no internal jobs
        assert ids(user_id=a.id) == {a_done, a_failed, a_processing}
        assert ids(user_id=b.id) == {b_done}

        # an explicitly requested feature_type wins over the default exclusion
        assert ids(feature_type="enhance_prompt") == {a_enhance}

        # --- Library view: paid history, differs from Home's "recent" view ---
        # enhance_prompt is hidden (it never has an outputUrl) and FAILED jobs
        # are hidden (refunded, no image). model_shoot_generate_model is a paid
        # deliverable and IS included; in-progress jobs stay visible.
        assert ids(view="library") == {a_done, a_processing, a_gen_model, b_done}
        assert a_enhance not in ids(view="library")
        assert a_failed not in ids(view="library")
        # an explicit status is still honoured, including failed
        assert ids(view="library", status="completed") == {a_done, a_gen_model, b_done}
        assert ids(view="library", status="failed") == {a_failed}
        assert ids(view="library", user_id=a.id) == {a_done, a_processing, a_gen_model}
        # ...and the default (no view) is still the Home behaviour, unchanged
        assert ids() == everything
    finally:
        teams_svc.purge_team(db, team.id)
        db.close()


def test_generations_rejects_an_invalid_status_value(monkeypatch):
    import uuid as uuid_mod
    from unittest.mock import MagicMock

    from app.core.database import get_db
    from app.deps import get_current_user
    from app.main import app
    from app.routes import teams as teams_route

    app.dependency_overrides[get_current_user] = lambda: MagicMock(id=uuid_mod.uuid4())
    app.dependency_overrides[get_db] = lambda: MagicMock()
    monkeypatch.setattr(teams_route, "is_team_member", lambda db, tid, uid: True)
    try:
        res = TestClient(app).get(
            f"/teams/{FAKE_TEAM}/generations?status=done", headers={"Authorization": "Bearer x"},
        )
    finally:
        app.dependency_overrides.clear()

    assert res.status_code == 422


def test_generations_endpoint_passes_status_through(monkeypatch):
    import uuid as uuid_mod
    from unittest.mock import MagicMock

    from app.core.database import get_db
    from app.deps import get_current_user
    from app.main import app
    from app.routes import teams as teams_route

    app.dependency_overrides[get_current_user] = lambda: MagicMock(id=uuid_mod.uuid4())
    app.dependency_overrides[get_db] = lambda: MagicMock()
    monkeypatch.setattr(teams_route, "is_team_member", lambda db, tid, uid: True)
    captured = {}
    monkeypatch.setattr(teams_route, "list_team_generations", lambda db, tid, **kw: captured.update(kw) or [])
    try:
        res = TestClient(app).get(
            f"/teams/{FAKE_TEAM}/generations?limit=8&status=completed",
            headers={"Authorization": "Bearer x"},
        )
    finally:
        app.dependency_overrides.clear()

    assert res.status_code == 200
    assert captured["status"] == "completed"
    assert captured["limit"] == 8


def test_generations_endpoint_caps_limit_at_page_size(monkeypatch):
    import uuid as uuid_mod
    from unittest.mock import MagicMock

    from app.core.database import get_db
    from app.deps import get_current_user
    from app.main import app
    from app.routes import teams as teams_route

    fake_user = MagicMock(id=uuid_mod.uuid4())
    db = MagicMock()
    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_db] = lambda: db
    monkeypatch.setattr(teams_route, "is_team_member", lambda db, tid, uid: True)
    captured = {}

    def fake_list(db, team_id, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(teams_route, "list_team_generations", fake_list)

    try:
        tc = TestClient(app)
        res = tc.get(f"/teams/{FAKE_TEAM}/generations?limit=500", headers={"Authorization": "Bearer x"})
    finally:
        app.dependency_overrides.clear()

    assert res.status_code == 200
    assert captured["limit"] == teams_route.MAX_GENERATIONS_PAGE_SIZE


def test_list_team_generations_falls_back_to_generated_title():
    import uuid as uuid_mod
    from datetime import datetime, timezone
    from unittest.mock import MagicMock

    from app.services.generation import list_team_generations

    job_id = uuid_mod.uuid4()
    created_at = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)
    fake_job = MagicMock(
        id=job_id, batch_id=None, output_url=None, title=None,
        feature_type="recolor", status="completed", created_at=created_at,
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [fake_job]

    result = list_team_generations(db, FAKE_TEAM, limit=10)

    assert result == [{
        "jobId": str(job_id),
        "batchId": None,
        "title": f"recolor — {created_at.isoformat()}",
        "featureType": "recolor",
        "status": "completed",
        "createdAt": created_at.isoformat(),
        "outputUrl": None,
        "deeplink": {"type": "page", "route": "library", "params": {"job_id": str(job_id)}},
    }]


def test_list_team_generations_uses_real_title_when_set():
    import uuid as uuid_mod
    from datetime import datetime, timezone
    from unittest.mock import MagicMock

    from app.services.generation import list_team_generations

    job_id = uuid_mod.uuid4()
    created_at = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)
    fake_job = MagicMock(
        id=job_id, batch_id=None, output_url=None, title="Blue jacket relaunch",
        feature_type="recolor", status="completed", created_at=created_at,
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [fake_job]

    result = list_team_generations(db, FAKE_TEAM, limit=10)

    assert result[0]["title"] == "Blue jacket relaunch"


def test_create_invite_blocks_when_team_full(monkeypatch):
    from unittest.mock import MagicMock

    from app.services import team_invites

    # no existing pending invite -> code reaches the seat check
    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = None
    monkeypatch.setattr(team_invites, "team_seat_count", lambda db, tid: team_invites.MAX_TEAM_MEMBERS)

    try:
        team_invites.create_invite(fake_db, FAKE_TEAM, "x@y.com", "some-user-id")
        assert False, "expected TeamFullError"
    except team_invites.TeamFullError as e:
        assert "full" in str(e).lower()


@pytest.mark.real_db
def test_only_an_owner_can_delete_or_restore_a_team_real_db(monkeypatch):
    """An editor calling DELETE / restore on their own team must get 403 and
    change nothing -- exercised through the real routes against real Postgres
    (the earlier delete tests stubbed the owner check, so they could not catch
    a broken role check). Disposable team with one real owner and one real
    editor, purged after."""
    from app.core.database import SessionLocal, get_db
    from app.deps import get_current_user
    from app.main import app
    from app.models.team import Team
    from app.models.team_member import TeamMember
    from app.models.user import User
    from app.routes import teams as teams_route
    from app.services import teams as teams_svc

    # A successful restore emails EVERY team member, and the owner/editor here
    # are REAL users from the dev database -- so without this stub each run of
    # the suite sent real "team restored" emails to real inboxes through real
    # SMTP (which is how this was noticed). Stubbed, and asserted below.
    notified = []
    monkeypatch.setattr(teams_route, "notify_team_restored", lambda db_, tid: notified.append(tid))

    db = SessionLocal()
    owner, editor = db.query(User).limit(2).all()
    team = Team(name="Owner-only delete test")
    db.add(team)
    db.flush()
    db.add_all([
        TeamMember(team_id=team.id, user_id=owner.id, role="owner"),
        TeamMember(team_id=team.id, user_id=editor.id, role="editor"),
    ])
    db.commit()
    team_id = team.id

    def call_as(user, method, path):
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_db] = lambda: db
        try:
            return getattr(TestClient(app), method)(path, headers={"Authorization": "Bearer x"})
        finally:
            app.dependency_overrides.clear()

    try:
        res = call_as(editor, "delete", f"/teams/{team_id}")
        assert res.status_code == 403
        assert res.json()["detail"] == "Only the team owner can do this"
        db.refresh(team)
        assert team.deleted_at is None  # untouched

        # the owner can
        res = call_as(owner, "delete", f"/teams/{team_id}")
        assert res.status_code == 200
        db.refresh(team)
        assert team.deleted_at is not None

        # a soft-deleted team's editor still cannot restore it
        res = call_as(editor, "post", f"/teams/{team_id}/restore")
        assert res.status_code == 403
        db.refresh(team)
        assert team.deleted_at is not None
        assert notified == []   # a refused restore must notify nobody

        # ...but the owner can
        res = call_as(owner, "post", f"/teams/{team_id}/restore")
        assert res.status_code == 200
        db.refresh(team)
        assert team.deleted_at is None
        assert notified == [team_id]   # members are notified exactly once, on the successful restore
    finally:
        teams_svc.purge_team(db, team_id)
        db.close()
