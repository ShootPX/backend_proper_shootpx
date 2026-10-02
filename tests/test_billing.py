from fastapi.testclient import TestClient

from app.main import app
import pytest

pytestmark = pytest.mark.real_db  # reads the real catalog / teams tables

client = TestClient(app)


def test_billing_returns_subscriptions_and_credits():
    res = client.get("/landing/billing")
    assert res.status_code == 200

    body = res.json()
    assert "subscriptions" in body
    assert "credits" in body
    assert isinstance(body["subscriptions"], list)
    assert isinstance(body["credits"], list)


def test_billing_subscription_shape():
    res = client.get("/landing/billing")
    subs = res.json()["subscriptions"]
    if subs:
        first = subs[0]
        for key in ("id", "slug", "name", "price", "billingPeriodDays", "credits", "info"):
            assert key in first
