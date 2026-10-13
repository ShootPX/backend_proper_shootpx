"""Storage integration: fal.ai success -> download -> upload -> permanent
output_url, and every failure path along that chain refunds + fails cleanly.
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.core.storage import upload_to_storage
from app.services import generation as generation_svc


def _job(status="queued", credits_charged=5, from_sub=3, from_topup=2, seconds_old=12):
    return MagicMock(
        status=status,
        credits_charged=credits_charged,
        credits_from_subscription=from_sub,
        credits_from_topup=from_topup,
        team_id=uuid.uuid4(),
        feature_type="test_tool",
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        output_url=None,
        error_message=None,
        created_at=datetime.now(timezone.utc) - timedelta(seconds=seconds_old),
    )


def _webhook_db(job, remaining_in_batch=0):
    db = MagicMock()
    (db.query.return_value.filter.return_value.populate_existing.return_value
       .with_for_update.return_value.first.return_value) = job
    db.query.return_value.filter.return_value.count.return_value = remaining_in_batch
    return db


def _ok_payload(url="https://fal.ai/tmp/expires-soon.png"):
    return {"status": "OK", "payload": {"images": [{"url": url}]}}


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #

def test_success_downloads_uploads_and_stores_permanent_url(monkeypatch):
    job = _job()
    db = _webhook_db(job)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    download = MagicMock(return_value=b"fake-image-bytes")
    upload = MagicMock(return_value="https://our-storage.test/permanent/output.png")
    monkeypatch.setattr(generation_svc, "download_from_url", download)
    monkeypatch.setattr(generation_svc, "upload_to_storage", upload)
    refund = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)

    generation_svc.handle_fal_webhook(db, uuid.uuid4(), _ok_payload("https://fal.ai/tmp/x.png"))

    download.assert_called_once_with("https://fal.ai/tmp/x.png")
    expected_path = f"{job.team_id}/{job.feature_type}/{job.id}/output.png"
    upload.assert_called_once_with(expected_path, b"fake-image-bytes", "image/png")
    assert job.status == "completed"
    assert job.output_url == "https://our-storage.test/permanent/output.png"  # OUR url, not fal's
    refund.assert_not_called()


# --------------------------------------------------------------------------- #
# Failure paths -- all three must refund + fail + not touch each other
# --------------------------------------------------------------------------- #

def test_empty_images_array_fails_and_refunds(monkeypatch):
    job = _job(credits_charged=5, from_sub=3, from_topup=2)
    db = _webhook_db(job)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    download = MagicMock()
    upload = MagicMock()
    monkeypatch.setattr(generation_svc, "download_from_url", download)
    monkeypatch.setattr(generation_svc, "upload_to_storage", upload)
    refund = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)

    generation_svc.handle_fal_webhook(db, uuid.uuid4(), {"status": "OK", "payload": {"images": []}})

    assert job.status == "failed"
    assert "no image" in job.error_message.lower()
    refund.assert_called_once_with(db, job.team_id, 5, 3, 2)
    download.assert_not_called()
    upload.assert_not_called()


def test_download_failure_fails_refunds_with_generic_message_and_logs_real_error(monkeypatch, caplog):
    """Restores the download-vs-upload distinction, but learns the FAL_KEY
    lesson from earlier: job.error_message (user-facing via GET /jobs/{id})
    must never contain raw exception text, ever -- not even redacted. Instead
    it's one of two fixed, generic-but-distinct strings, and the real
    exception detail goes to the server log via logger.exception(), never to
    the client."""
    job = _job(credits_charged=5, from_sub=3, from_topup=2)
    db = _webhook_db(job)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "download_from_url",
                         MagicMock(side_effect=RuntimeError("fal temp URL expired (404)")))
    upload = MagicMock()
    monkeypatch.setattr(generation_svc, "upload_to_storage", upload)
    refund = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)

    with caplog.at_level("ERROR", logger="app.services.generation"):
        generation_svc.handle_fal_webhook(db, uuid.uuid4(), _ok_payload())

    assert job.status == "failed"
    assert job.error_message == "Could not retrieve the generated image. Please try again."
    assert "upload" not in job.error_message.lower()
    assert "fal temp URL expired (404)" not in job.error_message  # never leaks raw exception text
    refund.assert_called_once_with(db, job.team_id, 5, 3, 2)
    upload.assert_not_called()  # never reached
    assert "fal temp URL expired (404)" in caplog.text  # but IS logged server-side for debugging
    assert any(r.exc_info for r in caplog.records)  # logger.exception, not logger.error


def test_upload_failure_fails_refunds_with_generic_message_and_logs_real_error(monkeypatch, caplog):
    job = _job(credits_charged=5, from_sub=3, from_topup=2)
    db = _webhook_db(job)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "download_from_url", MagicMock(return_value=b"bytes"))
    monkeypatch.setattr(generation_svc, "upload_to_storage",
                         MagicMock(side_effect=RuntimeError("bucket permission denied")))
    refund = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)

    with caplog.at_level("ERROR", logger="app.services.generation"):
        generation_svc.handle_fal_webhook(db, uuid.uuid4(), _ok_payload())

    assert job.status == "failed"
    assert job.error_message == "Could not save the generated image. Please try again."
    assert "bucket permission denied" not in job.error_message
    refund.assert_called_once_with(db, job.team_id, 5, 3, 2)
    assert "bucket permission denied" in caplog.text
    assert any(r.exc_info for r in caplog.records)


def test_download_and_upload_failures_use_distinct_messages():
    """The two messages must actually differ -- that's the whole point."""
    from app.services.generation import DOWNLOAD_FAILED_MESSAGE, UPLOAD_FAILED_MESSAGE
    assert DOWNLOAD_FAILED_MESSAGE != UPLOAD_FAILED_MESSAGE


# --------------------------------------------------------------------------- #
# duration_seconds: populated for every terminal outcome, completed_at - created_at.
# --------------------------------------------------------------------------- #

def test_duration_seconds_populated_on_successful_completion(monkeypatch):
    job = _job(seconds_old=12)
    db = _webhook_db(job)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "download_from_url", MagicMock(return_value=b"bytes"))
    monkeypatch.setattr(generation_svc, "upload_to_storage", MagicMock(return_value="https://stored.test/x.png"))

    generation_svc.handle_fal_webhook(db, uuid.uuid4(), _ok_payload())

    assert job.status == "completed"
    assert job.duration_seconds == int((job.completed_at - job.created_at).total_seconds())
    assert 11 <= job.duration_seconds <= 13  # ~12s, small tolerance for real wall-clock time


def test_late_webhook_success_after_timeout_failure_delivers_output_without_recharging(monkeypatch):
    """The other half of the timeout money-loss fix (see app/worker.py's
    check_generation_timeouts): a job already failed+refunded by a timeout
    can still have the REAL fal.ai webhook show up afterward reporting
    success. It must be delivered -- completed, with the real output -- but
    credits must NOT be charged again; the timeout's own refund stands."""
    from app.services.generation import TIMEOUT_MESSAGE

    job = _job(status="failed", seconds_old=90)
    job.error_message = TIMEOUT_MESSAGE
    job.output_url = None
    db = _webhook_db(job)
    download = MagicMock(return_value=b"real-bytes")
    upload = MagicMock(return_value="https://our-storage.test/permanent/late.png")
    monkeypatch.setattr(generation_svc, "download_from_url", download)
    monkeypatch.setattr(generation_svc, "upload_to_storage", upload)
    refund = MagicMock()
    release = MagicMock()
    release_slot = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)
    monkeypatch.setattr(generation_svc, "release_generation_lock", release)
    monkeypatch.setattr(generation_svc, "release_fal_slot", release_slot)

    generation_svc.handle_fal_webhook(
        db, uuid.uuid4(),
        {"status": "OK", "payload": {"images": [{"url": "https://fal.test/late.png"}]}},
    )

    download.assert_called_once_with("https://fal.test/late.png")
    upload.assert_called_once_with(f"{job.team_id}/{job.feature_type}/{job.id}/output.png", b"real-bytes", "image/png")
    assert job.status == "completed"
    assert job.output_url == "https://our-storage.test/permanent/late.png"
    refund.assert_not_called()           # no double-refund
    release.assert_not_called()          # already released back when the timeout first failed it
    release_slot.assert_not_called()     # same -- must not double-decrement the fal concurrency counter
    db.commit.assert_called_once()


def test_late_webhook_failure_after_timeout_failure_is_a_safe_noop(monkeypatch):
    """A late webhook reporting FAILURE (not success) for an already
    timed-out-and-refunded job has nothing new to deliver -- must stay a
    silent no-op exactly like before, not attempt a second refund."""
    from app.services.generation import TIMEOUT_MESSAGE

    job = _job(status="failed", seconds_old=90)
    job.error_message = TIMEOUT_MESSAGE
    db = _webhook_db(job)
    refund = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)

    generation_svc.handle_fal_webhook(db, uuid.uuid4(), {"status": "ERROR", "error": "fal also failed"})

    assert job.status == "failed"
    assert job.error_message == TIMEOUT_MESSAGE  # untouched
    refund.assert_not_called()
    db.commit.assert_not_called()


def test_duration_seconds_populated_on_failure(monkeypatch):
    """A terminal FAILURE must get duration_seconds too, not just success --
    the download-failure path exercised here reaches the same completed_at
    line as every other outcome in handle_fal_webhook."""
    job = _job(seconds_old=30)
    db = _webhook_db(job)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "download_from_url", MagicMock(side_effect=RuntimeError("boom")))
    monkeypatch.setattr(generation_svc, "refund_credits", MagicMock())

    generation_svc.handle_fal_webhook(db, uuid.uuid4(), _ok_payload())

    assert job.status == "failed"
    assert job.duration_seconds == int((job.completed_at - job.created_at).total_seconds())
    assert 29 <= job.duration_seconds <= 31


# --------------------------------------------------------------------------- #
# Idempotency: replay after ANY of the above must not re-run storage logic
# or double-refund.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("final_status", ["completed", "failed"])
def test_replay_after_terminal_never_touches_storage_or_refunds_again(monkeypatch, final_status):
    job = _job(status=final_status)
    job.output_url = "https://our-storage.test/already-set.png" if final_status == "completed" else None
    job.error_message = None if final_status == "completed" else "already failed"
    db = _webhook_db(job)
    download = MagicMock()
    upload = MagicMock()
    refund = MagicMock()
    release = MagicMock()
    monkeypatch.setattr(generation_svc, "download_from_url", download)
    monkeypatch.setattr(generation_svc, "upload_to_storage", upload)
    monkeypatch.setattr(generation_svc, "refund_credits", refund)
    monkeypatch.setattr(generation_svc, "release_generation_lock", release)

    generation_svc.handle_fal_webhook(db, uuid.uuid4(), _ok_payload())

    download.assert_not_called()
    upload.assert_not_called()
    refund.assert_not_called()
    release.assert_not_called()
    db.commit.assert_not_called()


# --------------------------------------------------------------------------- #
# Path safety
# --------------------------------------------------------------------------- #

def test_permanent_path_uses_only_uuid_and_admin_controlled_feature_type(monkeypatch):
    """team_id and job.id are always real UUIDs (no traversal characters
    possible), and feature_type can only ever be a value that already exists
    as a primary key in tool_definitions (admin-controlled) -- create_generation_batch
    rejects any feature_type that isn't already a real row. Confirmed here by
    showing the exact composed path is exactly the 4 expected safe segments."""
    job = _job()
    db = _webhook_db(job)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "download_from_url", MagicMock(return_value=b"x"))
    upload = MagicMock(return_value="https://our-storage.test/x.png")
    monkeypatch.setattr(generation_svc, "upload_to_storage", upload)

    generation_svc.handle_fal_webhook(db, uuid.uuid4(), _ok_payload())

    path = upload.call_args[0][0]
    parts = path.split("/")
    assert len(parts) == 4
    assert parts[0] == str(job.team_id)
    assert parts[1] == job.feature_type
    assert parts[2] == str(job.id)
    assert parts[3] == "output.png"


@pytest.mark.parametrize("malicious_path", [
    "../../etc/passwd",
    "team/../../secret/x.png",
    "/etc/passwd",
    "team\\..\\..\\x.png",
    "team/feature/job/output.png\x00.jpg",
])
def test_upload_to_storage_rejects_path_traversal(malicious_path):
    """Defense in depth: feature_type is admin-gated today (create_generation_batch
    rejects any feature_type not already a tool_definitions row), so this isn't
    reachable through the current API -- but upload_to_storage is the actual
    security boundary that touches external storage, so it must not blindly
    trust a composed path either."""
    with pytest.raises(ValueError):
        upload_to_storage(malicious_path, b"data")


def test_upload_to_storage_accepts_a_normal_safe_path(monkeypatch):
    import httpx
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    monkeypatch.setattr("app.core.storage.httpx.post", MagicMock(return_value=fake_response))

    safe_path = f"{uuid.uuid4()}/test_tool/{uuid.uuid4()}/output.png"
    result = upload_to_storage(safe_path, b"data")  # must not raise
    assert safe_path in result


def test_upload_to_storage_sends_key_on_apikey_header_not_authorization(monkeypatch):
    """Regression test for a real bug found live against the actual dev
    Supabase project: this project's key is the newer sb_secret_... format,
    which is not a JWT. Sending it as `Authorization: Bearer` makes Supabase
    try to parse it as a JWT and reject it ("Invalid Compact JWS") -- it must
    go on the `apikey` header instead."""
    post = MagicMock()
    post.return_value.raise_for_status = MagicMock()
    monkeypatch.setattr("app.core.storage.httpx.post", post)

    upload_to_storage(f"{uuid.uuid4()}/test_tool/{uuid.uuid4()}/output.png", b"data")

    headers = post.call_args.kwargs["headers"]
    assert headers.get("apikey") == settings.supabase_service_role_key
    assert "Authorization" not in headers


# --------------------------------------------------------------------------- #
# Supabase service role key must never leak into a stored error message.
# --------------------------------------------------------------------------- #

def test_supabase_key_never_appears_in_webhook_error_message(monkeypatch):
    job = _job()
    db = _webhook_db(job)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "download_from_url", MagicMock(return_value=b"x"))
    monkeypatch.setattr(
        generation_svc, "upload_to_storage",
        MagicMock(side_effect=RuntimeError(f"request failed, Authorization: Bearer {settings.supabase_service_role_key}")),
    )
    monkeypatch.setattr(generation_svc, "refund_credits", MagicMock())

    generation_svc.handle_fal_webhook(db, uuid.uuid4(), _ok_payload())

    assert settings.supabase_service_role_key not in job.error_message
