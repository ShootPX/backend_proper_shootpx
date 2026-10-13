"""fal.ai webhook signature verification (ED25519 + JWKS), per fal's real docs:
https://fal.ai/docs/model-apis/model-endpoints/webhooks

Message = request_id + "\n" + user_id + "\n" + timestamp + "\n" + sha256_hex(raw_body),
signed with ED25519, signature sent hex-encoded, public keys served as a JWKS at
https://rest.fal.ai/.well-known/jwks.json (base64url-encoded raw Ed25519 key in
each key's "x" field). Timestamp must be within +/-300s of now.
"""

import base64
import hashlib
import json
import time
import uuid
from unittest.mock import MagicMock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi.testclient import TestClient

from app.core import fal_webhook as fal_webhook_mod
from app.core.database import get_db
from app.core.fal_webhook import FalWebhookVerificationError, verify_fal_webhook
from app.core.limiter import limiter
from app.main import app

limiter.enabled = False


def _b64url_no_pad(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _keypair_and_jwk():
    priv = Ed25519PrivateKey.generate()
    pub_raw = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    jwk = {"kty": "OKP", "crv": "Ed25519", "x": _b64url_no_pad(pub_raw), "use": "sig"}
    return priv, jwk


def _sign(priv, request_id, user_id, timestamp, raw_body: bytes) -> str:
    body_hash = hashlib.sha256(raw_body).hexdigest()
    message = f"{request_id}\n{user_id}\n{timestamp}\n{body_hash}".encode()
    return priv.sign(message).hex()


class _Headers(dict):
    """Case-insensitive-enough stand-in for Starlette's Headers, for unit tests
    that call verify_fal_webhook directly instead of going through the route."""
    def get(self, key, default=None):
        for k, v in self.items():
            if k.lower() == key.lower():
                return v
        return default


# --------------------------------------------------------------------------- #
# Unit tests: verify_fal_webhook itself, real ED25519 crypto, JWKS network
# call replaced with a fixed keypair so tests are deterministic and offline.
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_jwks(monkeypatch):
    priv, jwk = _keypair_and_jwk()
    monkeypatch.setattr(fal_webhook_mod, "_fetch_jwks", lambda: [jwk])
    return priv


def _valid_headers(priv, body: bytes, request_id="req-1", user_id="user-1", timestamp=None):
    ts = timestamp or str(int(time.time()))
    sig = _sign(priv, request_id, user_id, ts, body)
    return _Headers({
        "X-Fal-Webhook-Request-Id": request_id,
        "X-Fal-Webhook-User-Id": user_id,
        "X-Fal-Webhook-Timestamp": ts,
        "X-Fal-Webhook-Signature": sig,
    })


def test_valid_signature_passes(fake_jwks):
    body = json.dumps({"status": "OK"}).encode()
    headers = _valid_headers(fake_jwks, body)
    verify_fal_webhook(headers, body)  # must not raise


def test_missing_headers_rejected(fake_jwks):
    body = b'{"status": "OK"}'
    for missing in ["X-Fal-Webhook-Request-Id", "X-Fal-Webhook-User-Id",
                    "X-Fal-Webhook-Timestamp", "X-Fal-Webhook-Signature"]:
        headers = _valid_headers(fake_jwks, body)
        del headers[missing]
        with pytest.raises(FalWebhookVerificationError):
            verify_fal_webhook(headers, body)


def test_tampered_body_rejected(fake_jwks):
    original_body = json.dumps({"status": "OK"}).encode()
    headers = _valid_headers(fake_jwks, original_body)
    tampered_body = json.dumps({"status": "ERROR"}).encode()

    with pytest.raises(FalWebhookVerificationError):
        verify_fal_webhook(headers, tampered_body)


def test_wrong_key_signature_rejected(fake_jwks):
    """Signed with a DIFFERENT private key than the one in the JWKS -> reject."""
    other_priv = Ed25519PrivateKey.generate()
    body = b'{"status": "OK"}'
    headers = _valid_headers(other_priv, body)  # signed with the wrong key

    with pytest.raises(FalWebhookVerificationError):
        verify_fal_webhook(headers, body)


def test_garbage_signature_rejected(fake_jwks):
    body = b'{"status": "OK"}'
    headers = _valid_headers(fake_jwks, body)
    headers["X-Fal-Webhook-Signature"] = "not-valid-hex-zz"

    with pytest.raises(FalWebhookVerificationError):
        verify_fal_webhook(headers, body)


def test_stale_timestamp_rejected(fake_jwks):
    body = b'{"status": "OK"}'
    stale_ts = str(int(time.time()) - 3600)  # 1 hour old
    headers = _valid_headers(fake_jwks, body, timestamp=stale_ts)

    with pytest.raises(FalWebhookVerificationError):
        verify_fal_webhook(headers, body)


def test_no_keys_available_fails_closed(monkeypatch):
    """If fal's JWKS can't be fetched/is empty, we must reject (fail closed),
    never silently accept an unverifiable webhook."""
    monkeypatch.setattr(fal_webhook_mod, "_fetch_jwks", lambda: [])
    body = b'{"status": "OK"}'
    headers = _Headers({
        "X-Fal-Webhook-Request-Id": "r", "X-Fal-Webhook-User-Id": "u",
        "X-Fal-Webhook-Timestamp": str(int(time.time())),
        "X-Fal-Webhook-Signature": "aa",
    })

    with pytest.raises(FalWebhookVerificationError):
        verify_fal_webhook(headers, body)


# --------------------------------------------------------------------------- #
# Route-level: POST /webhooks/fal must reject bad signatures WITHOUT ever
# touching the job row (the exact requirement from the audit).
# --------------------------------------------------------------------------- #

@pytest.fixture
def webhook_client(monkeypatch):
    priv, jwk = _keypair_and_jwk()
    monkeypatch.setattr(fal_webhook_mod, "_fetch_jwks", lambda: [jwk])
    db = MagicMock()
    app.dependency_overrides[get_db] = lambda: db
    yield TestClient(app, raise_server_exceptions=False), priv, db
    app.dependency_overrides.clear()


def test_webhook_route_rejects_missing_signature_header_without_touching_db(webhook_client):
    client, priv, db = webhook_client
    job_id = uuid.uuid4()
    body = json.dumps({"status": "OK", "payload": {"images": [{"url": "https://x.test/a.png"}]}}).encode()

    res = client.post(
        f"/webhooks/fal?job_id={job_id}",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Fal-Webhook-Request-Id": "req-1",
            "X-Fal-Webhook-User-Id": "user-1",
            "X-Fal-Webhook-Timestamp": str(int(time.time())),
            # signature header deliberately omitted
        },
    )

    assert res.status_code == 401
    db.query.assert_not_called()
    db.commit.assert_not_called()


def test_webhook_route_rejects_invalid_signature_without_touching_db(webhook_client):
    client, priv, db = webhook_client
    job_id = uuid.uuid4()
    body = json.dumps({"status": "OK", "payload": {"images": [{"url": "https://x.test/a.png"}]}}).encode()
    request_id, user_id, ts = "req-1", "user-1", str(int(time.time()))
    real_sig = _sign(priv, request_id, user_id, ts, body)
    tampered_sig = ("0" if real_sig[0] != "0" else "1") + real_sig[1:]  # flip one hex char

    res = client.post(
        f"/webhooks/fal?job_id={job_id}",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Fal-Webhook-Request-Id": request_id,
            "X-Fal-Webhook-User-Id": user_id,
            "X-Fal-Webhook-Timestamp": ts,
            "X-Fal-Webhook-Signature": tampered_sig,
        },
    )

    assert res.status_code == 401
    db.query.assert_not_called()
    db.commit.assert_not_called()


def test_webhook_route_accepts_valid_signature(webhook_client, monkeypatch):
    client, priv, db = webhook_client
    job = MagicMock(status="queued", user_id=uuid.uuid4(), team_id=uuid.uuid4())
    (db.query.return_value.filter.return_value.populate_existing.return_value
       .with_for_update.return_value.first.return_value) = job
    db.query.return_value.filter.return_value.count.return_value = 0  # last (only) job in its batch
    monkeypatch.setattr("app.services.generation.release_generation_lock", lambda uid: None)
    monkeypatch.setattr("app.services.generation.download_from_url", lambda url: b"bytes")
    monkeypatch.setattr("app.services.generation.upload_to_storage",
                         lambda path, data, content_type="image/png": "https://our-storage.test/permanent/a.png")

    job_id = uuid.uuid4()
    body = json.dumps({"status": "OK", "payload": {"images": [{"url": "https://x.test/a.png"}]}}).encode()
    request_id, user_id, ts = "req-1", "user-1", str(int(time.time()))
    sig = _sign(priv, request_id, user_id, ts, body)

    res = client.post(
        f"/webhooks/fal?job_id={job_id}",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Fal-Webhook-Request-Id": request_id,
            "X-Fal-Webhook-User-Id": user_id,
            "X-Fal-Webhook-Timestamp": ts,
            "X-Fal-Webhook-Signature": sig,
        },
    )

    assert res.status_code == 200
    assert job.status == "completed"
