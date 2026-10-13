"""Cloudflare R2 storage backend + backend dispatch. No network: boto3 is mocked."""
import uuid
from unittest.mock import MagicMock

import httpx
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from app.core import storage

CACHE = "public, max-age=31536000, immutable"
PATH = f"{uuid.uuid4()}/recolor/{uuid.uuid4()}/output.png"


@pytest.fixture
def r2(monkeypatch):
    monkeypatch.setattr(storage.settings, "storage_backend", "r2")
    monkeypatch.setattr(storage.settings, "r2_bucket", "bkt")
    monkeypatch.setattr(storage.settings, "r2_public_base_url", "https://cdn.example.com")
    monkeypatch.setattr(storage.settings, "r2_key_prefix", "")
    client = MagicMock()
    monkeypatch.setattr(storage, "_get_r2_client", lambda: client)
    return client


def test_put_object_args(r2):
    url = storage.upload_to_storage(PATH, b"IMG", "image/webp")
    r2.put_object.assert_called_once_with(
        Bucket="bkt", Key=PATH, Body=b"IMG", ContentType="image/webp", CacheControl=CACHE,
    )
    assert url == f"https://cdn.example.com/{PATH}"


def test_default_content_type_is_png(r2):
    storage.upload_to_storage(PATH, b"IMG")
    assert r2.put_object.call_args.kwargs["ContentType"] == "image/png"


def test_prefix_applied_exactly_once_and_url_has_no_double_slashes(r2, monkeypatch):
    monkeypatch.setattr(storage.settings, "r2_key_prefix", "dev/")
    url = storage.upload_to_storage(PATH, b"IMG")
    assert r2.put_object.call_args.kwargs["Key"] == f"dev/{PATH}"
    assert url == f"https://cdn.example.com/dev/{PATH}"
    assert "//" not in url.split("://", 1)[1]


@pytest.mark.parametrize("bad", ["../etc/passwd", "a/../b.png", "a\\b.png", "/abs.png", "", "a/./b.png"])
def test_path_validation_rejects_unsafe_paths(r2, bad):
    with pytest.raises(ValueError):
        storage.upload_to_storage(bad, b"x")
    r2.put_object.assert_not_called()


def test_validation_runs_on_the_final_key(r2, monkeypatch):
    monkeypatch.setattr(storage.settings, "r2_key_prefix", "../")
    with pytest.raises(ValueError):
        storage.upload_to_storage(PATH, b"x")
    r2.put_object.assert_not_called()


def test_client_error_surfaces_as_httpx_status_error(r2):
    r2.put_object.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "secret-ish"}, "ResponseMetadata": {"HTTPStatusCode": 403}},
        "PutObject",
    )
    with pytest.raises(httpx.HTTPStatusError) as exc:
        storage.upload_to_storage(PATH, b"x")
    assert exc.value.response.status_code == 403
    assert "secret-ish" not in str(exc.value)


def test_botocore_error_surfaces_as_httpx_transport_error(r2):
    r2.put_object.side_effect = EndpointConnectionError(endpoint_url="https://x")
    with pytest.raises(httpx.TransportError):
        storage.upload_to_storage(PATH, b"x")


def test_dispatch_picks_r2(r2, monkeypatch):
    sup = MagicMock()
    monkeypatch.setattr(storage, "_upload_supabase", sup)
    storage.upload_to_storage(PATH, b"x")
    sup.assert_not_called()
    r2.put_object.assert_called_once()


def test_dispatch_picks_supabase(monkeypatch):
    monkeypatch.setattr(storage.settings, "storage_backend", "supabase")
    sup = MagicMock(return_value="https://p.supabase.co/x")
    r2_up = MagicMock()
    monkeypatch.setattr(storage, "_upload_supabase", sup)
    monkeypatch.setattr(storage, "_upload_r2", r2_up)
    assert storage.upload_to_storage(PATH, b"x", "image/jpeg") == "https://p.supabase.co/x"
    sup.assert_called_once_with(PATH, b"x", "image/jpeg")
    r2_up.assert_not_called()


def test_lazy_client_is_built_once_with_r2_settings(monkeypatch):
    monkeypatch.setattr(storage.settings, "r2_account_id", "acct")
    monkeypatch.setattr(storage.settings, "r2_access_key_id", "ak")
    monkeypatch.setattr(storage.settings, "r2_secret_access_key", "sk")
    monkeypatch.setattr(storage, "_r2_client", None)
    factory = MagicMock()
    monkeypatch.setattr(storage.boto3, "client", factory)
    first, second = storage._get_r2_client(), storage._get_r2_client()
    assert first is second
    factory.assert_called_once()
    kw = factory.call_args.kwargs
    assert kw["endpoint_url"] == "https://acct.r2.cloudflarestorage.com"
    assert kw["region_name"] == "auto"
    cfg = kw["config"]
    assert cfg.signature_version == "s3v4"
    assert cfg.retries == {"max_attempts": 3, "mode": "standard"}
    assert (cfg.connect_timeout, cfg.read_timeout) == (10, 30)


@pytest.mark.parametrize("data,expected", [
    (b"\x89PNG\r\n\x1a\n" + b"0" * 8, ("image/png", "png")),
    (b"\xff\xd8\xff\xe0" + b"0" * 8, ("image/jpeg", "jpg")),
    (b"RIFF\x00\x00\x00\x00WEBPVP8 ", ("image/webp", "webp")),
    (b"GIF89a", ("image/png", "png")),
    (b"", ("image/png", "png")),
])
def test_detect_image_type(data, expected):
    assert storage.detect_image_type(data) == expected


LEGACY_URL = "https://abc.supabase.co/storage/v1/object/public/generation-storage/t/recolor/j/output.png"


def test_existing_supabase_output_url_passes_through_job_read_path_unchanged():
    """Rows written before the R2 switch keep their full Supabase URL: the
    history/library read path and the fal-input resolver return it verbatim,
    with no host check, rewrite or validation."""
    from datetime import datetime, timezone

    from app.routes.generation import _resolve_source_job_urls
    from app.services.generation import list_team_generations

    job_id = uuid.uuid4()
    query = MagicMock()
    for method in ("filter", "order_by", "offset", "limit"):
        getattr(query, method).return_value = query
    db = MagicMock()
    db.query.return_value = query

    query.all.return_value = [(job_id, LEGACY_URL)]
    assert _resolve_source_job_urls(db, uuid.uuid4(), [job_id]) == {job_id: LEGACY_URL}

    job = MagicMock(
        id=job_id, batch_id=None, feature_type="recolor", status="completed",
        output_url=LEGACY_URL, created_at=datetime.now(timezone.utc),
    )
    query.all.return_value = [job]
    items = list_team_generations(db, uuid.uuid4(), view="library")
    assert items[0]["outputUrl"] == LEGACY_URL
