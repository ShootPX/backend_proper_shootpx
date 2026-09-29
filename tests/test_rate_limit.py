from fastapi.testclient import TestClient

from app.main import app
from app.core.limiter import limiter
import pytest

client = TestClient(app)


@pytest.mark.real_db
def test_rate_limit_triggers_429(monkeypatch):
    # re-enable the limiter just for this test (conftest disables it globally)
    limiter.enabled = True
    limiter.reset()
    try:
        limit = 30  # matches @limiter.limit("30/minute") on /landing/tools
        codes = [client.get("/landing/tools").status_code for _ in range(limit + 2)]
    finally:
        limiter.reset()
        limiter.enabled = False

    assert codes[0] == 200
    assert 429 in codes
    assert codes.count(200) <= limit
