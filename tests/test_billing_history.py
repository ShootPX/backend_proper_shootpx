"""GET /teams/{team_id}/billing/history and services/ledger.list_billing_history.

The pagination is a real SQL predicate (keyset on created_at, id), so it runs
against an in-memory SQLite table rather than a mock.
"""

import base64
import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.database import get_db
from app.deps import get_current_user
from app.main import app
from app.models.credit_ledger import CreditLedger
from app.services import ledger as ledger_svc

TEAM_ID = uuid.uuid4()
OTHER_TEAM_ID = uuid.uuid4()
T0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def session():
    from sqlalchemy import create_engine
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.core.database import Base

    @compiles(JSONB, "sqlite")
    def _jsonb(type_, compiler, **kw):
        return "JSON"

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    # FKs to teams are not enforced by SQLite; only the ledger table is needed.
    Base.metadata.create_all(engine, tables=[CreditLedger.__table__])
    s = sessionmaker(bind=engine)()
    yield s
    s.close()
    engine.dispose()


def _entry(s, n, *, team_id=TEAM_ID, at=None, **kw):
    fields = dict(
        id=uuid.uuid4(), team_id=team_id, pool="topup", entry_type="topup_purchase", amount=100,
        balance_after=100, idempotency_key=f"k{n}-{uuid.uuid4()}", source="webhook",
        entry_metadata={}, created_at=at or T0 + timedelta(minutes=n),
    )
    fields.update(kw)
    row = CreditLedger(**fields)
    s.add(row)
    s.commit()
    return row


# --------------------------------------------------------------------------- #
# service
# --------------------------------------------------------------------------- #

def test_history_is_newest_first_and_scoped_to_the_team(session):
    old = _entry(session, 1)
    new = _entry(session, 5)
    _entry(session, 9, team_id=OTHER_TEAM_ID)              # someone else's

    page = ledger_svc.list_billing_history(session, TEAM_ID)

    assert [i["id"] for i in page["items"]] == [str(new.id), str(old.id)]
    assert page["next_cursor"] is None


def test_an_empty_history_is_an_empty_page(session):
    assert ledger_svc.list_billing_history(session, TEAM_ID) == {"items": [], "next_cursor": None}


def test_pages_chain_through_the_cursor_without_gaps_or_repeats(session):
    made = [_entry(session, n) for n in range(1, 8)]        # 7 entries
    expected = [str(e.id) for e in reversed(made)]

    seen, cursor, pages = [], None, 0
    while True:
        page = ledger_svc.list_billing_history(session, TEAM_ID, limit=3, cursor=cursor)
        seen += [i["id"] for i in page["items"]]
        pages += 1
        cursor = page["next_cursor"]
        if cursor is None:
            break

    assert seen == expected
    assert pages == 3                                       # 3 + 3 + 1


def test_the_last_page_has_no_cursor_even_when_it_is_exactly_full(session):
    for n in range(1, 4):
        _entry(session, n)
    page = ledger_svc.list_billing_history(session, TEAM_ID, limit=3)
    assert len(page["items"]) == 3 and page["next_cursor"] is None


def test_entries_sharing_a_timestamp_are_split_across_pages_by_id(session):
    """The three entries of one plan switch are written in one transaction and get the same timestamp."""
    same = [_entry(session, 1, at=T0) for _ in range(5)]

    first = ledger_svc.list_billing_history(session, TEAM_ID, limit=2)
    second = ledger_svc.list_billing_history(session, TEAM_ID, limit=2, cursor=first["next_cursor"])
    third = ledger_svc.list_billing_history(session, TEAM_ID, limit=2, cursor=second["next_cursor"])

    ids = [i["id"] for p in (first, second, third) for i in p["items"]]
    assert sorted(ids) == sorted(str(e.id) for e in same)   # every row exactly once
    assert len(set(ids)) == 5
    assert third["next_cursor"] is None


def test_an_entry_added_between_requests_does_not_shift_the_next_page(session):
    for n in range(1, 6):
        _entry(session, n)
    first = ledger_svc.list_billing_history(session, TEAM_ID, limit=2)

    newest = _entry(session, 99)                            # arrives after page 1 was served
    second = ledger_svc.list_billing_history(session, TEAM_ID, limit=2, cursor=first["next_cursor"])

    shown = {i["id"] for i in first["items"]} | {i["id"] for i in second["items"]}
    assert str(newest.id) not in shown
    assert len(shown) == 4                                  # no repeat, no skip


def test_limit_is_clamped(session):
    for n in range(1, 4):
        _entry(session, n)
    assert len(ledger_svc.list_billing_history(session, TEAM_ID, limit=0)["items"]) == 1
    assert len(ledger_svc.list_billing_history(session, TEAM_ID, limit=-5)["items"]) == 1
    assert len(ledger_svc.list_billing_history(session, TEAM_ID, limit=10_000)["items"]) == 3


@pytest.mark.parametrize("cursor", [
    "not-base64!!", base64.urlsafe_b64encode(b"not json").decode(),
    base64.urlsafe_b64encode(json.dumps({"t": "yesterday", "i": str(uuid.uuid4())}).encode()).decode(),
    base64.urlsafe_b64encode(json.dumps({"t": T0.isoformat(), "i": "nope"}).encode()).decode(),
    base64.urlsafe_b64encode(json.dumps({"t": T0.isoformat()}).encode()).decode(),
])
def test_a_malformed_cursor_is_a_value_error(session, cursor):
    with pytest.raises(ValueError, match="Invalid cursor"):
        ledger_svc.list_billing_history(session, TEAM_ID, cursor=cursor)


def test_items_describe_each_kind_of_entry(session):
    topup = _entry(session, 1, entry_type="topup_purchase", amount=500, balance_after=510,
                   razorpay_payment_id="pay_1", entry_metadata={"amount": 50000, "pack": "pack-500"})
    grant = _entry(session, 2, pool="subscription", entry_type="subscription_grant", amount=350,
                   balance_after=350, razorpay_subscription_id="sub_secret",
                   entry_metadata={"plan": "monthly", "period": "month", "cycle": 3, "price": 99900})
    slice_ = _entry(session, 3, pool="subscription", entry_type="subscription_grant", amount=100,
                    balance_after=100, entry_metadata={"plan": "yearly", "period": "year"})
    transfer = _entry(session, 4, pool="topup", entry_type="switch_transfer", amount=40,
                      balance_after=50, entry_metadata={})

    items = {i["id"]: i for i in ledger_svc.list_billing_history(session, TEAM_ID)["items"]}

    t_item = dict(items[str(topup.id)])
    assert t_item.pop("created_at")                          # present; its tz rendering is the DB's business
    assert t_item == {
        "id": str(topup.id), "type": "topup_purchase", "pool": "topup", "credits": 500,
        "balance_after": 510, "plan": None, "period": None, "cycle": None,
        "amount_paid": 50000, "reference": "pay_1",
    }
    g = items[str(grant.id)]
    assert (g["plan"], g["period"], g["cycle"], g["amount_paid"], g["reference"]) == ("monthly", "month", 3, 99900, None)
    assert items[str(slice_.id)]["amount_paid"] is None      # a monthly slice of a yearly plan costs nothing new
    t = items[str(transfer.id)]
    assert (t["type"], t["credits"], t["amount_paid"]) == ("switch_transfer", 40, None)
    assert "sub_secret" not in json.dumps(list(items.values()))     # subscription ids stay internal


def test_a_backfilled_entry_with_no_balance_is_still_listed(session):
    row = _entry(session, 1, balance_after=None, source="backfill",
                 entry_metadata={"amount": 50000, "backfilled": True}, razorpay_payment_id="pay_old")
    (item,) = ledger_svc.list_billing_history(session, TEAM_ID)["items"]
    assert item["balance_after"] is None and item["amount_paid"] == 50000 and item["id"] == str(row.id)


# --------------------------------------------------------------------------- #
# route
# --------------------------------------------------------------------------- #

@pytest.fixture
def client(session, monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: MagicMock(id=uuid.uuid4())
    app.dependency_overrides[get_db] = lambda: session
    monkeypatch.setattr("app.routes.teams.is_team_member", lambda db, tid, uid: True)
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def _get(client, query=""):
    return client.get(f"/teams/{TEAM_ID}/billing/history{query}", headers={"Authorization": "Bearer x"})


def test_the_route_requires_team_membership(client, monkeypatch, session):
    _entry(session, 1)
    monkeypatch.setattr("app.routes.teams.is_team_member", lambda db, tid, uid: False)

    res = _get(client)

    assert res.status_code == 403
    assert "items" not in res.json()


def test_the_route_returns_camel_case_items_and_a_cursor(client, session):
    for n in range(1, 4):
        _entry(session, n, entry_metadata={"amount": 50000})

    res = _get(client, "?limit=2")

    assert res.status_code == 200
    body = res.json()
    assert len(body["items"]) == 2 and body["nextCursor"]
    first = body["items"][0]
    assert set(first) == {"id", "type", "pool", "credits", "balanceAfter", "plan", "period",
                          "cycle", "amountPaid", "reference", "createdAt"}
    assert first["amountPaid"] == 50000

    rest = _get(client, f"?limit=2&cursor={body['nextCursor']}").json()
    assert len(rest["items"]) == 1 and rest["nextCursor"] is None


def test_the_route_rejects_a_bad_cursor_with_400(client):
    res = _get(client, "?cursor=garbage")
    assert res.status_code == 400
    assert res.json()["detail"] == "Invalid cursor"


def test_the_route_for_another_team_shows_none_of_this_teams_entries(client, session):
    _entry(session, 1)
    res = client.get(f"/teams/{OTHER_TEAM_ID}/billing/history", headers={"Authorization": "Bearer x"})
    assert res.status_code == 200 and res.json()["items"] == []
