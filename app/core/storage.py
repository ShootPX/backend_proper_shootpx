import logging
import re
import threading

import boto3
import httpx
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from app.core.config import settings
from app.core.http_retry import send_with_retry

logger = logging.getLogger(__name__)

BUCKET_NAME = "generation-storage"

# Every real caller composes this path from a team_id/job_id (always plain
# UUIDs) and a feature_type that must already exist as a tool_definitions
# primary key -- but this is the actual boundary that touches external
# storage, so it validates the composed path itself rather than trusting
# that invariant to hold forever upstream.
_SAFE_PATH = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]|/(?!\.\.?(?:/|$)))*$")


def _validate_storage_path(file_path: str) -> None:
    if not file_path or "\\" in file_path or not _SAFE_PATH.match(file_path):
        raise ValueError(f"Unsafe storage path: {file_path!r}")


def detect_image_type(data: bytes) -> tuple[str, str]:
    """(content_type, extension) from magic bytes; falls back to PNG."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", "webp"
    return "image/png", "png"


def upload_to_storage(file_path: str, file_bytes: bytes, content_type: str = "image/png") -> str:
    """
    Uploads a file to the configured storage backend (STORAGE_BACKEND).
    Returns the public URL of what was just written. URLs already stored in the
    database are never touched.
    """
    if settings.storage_backend == "r2":
        return _upload_r2(file_path, file_bytes, content_type)
    return _upload_supabase(file_path, file_bytes, content_type)


_r2_client = None
_r2_client_lock = threading.Lock()


def _get_r2_client():
    global _r2_client
    if _r2_client is None:
        with _r2_client_lock:
            if _r2_client is None:
                _r2_client = boto3.client(
                    "s3",
                    endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
                    aws_access_key_id=settings.r2_access_key_id,
                    aws_secret_access_key=settings.r2_secret_access_key,
                    region_name="auto",
                    config=Config(
                        signature_version="s3v4",
                        retries={"max_attempts": 3, "mode": "standard"},
                        connect_timeout=10,
                        read_timeout=30,
                    ),
                )
    return _r2_client


def _upload_r2(file_path: str, file_bytes: bytes, content_type: str) -> str:
    key = f"{settings.r2_key_prefix}{file_path}"
    _validate_storage_path(key)
    try:
        _get_r2_client().put_object(
            Bucket=settings.r2_bucket,
            Key=key,
            Body=file_bytes,
            ContentType=content_type,
            CacheControl="public, max-age=31536000, immutable",
        )
    except ClientError as e:
        # Same exception types the Supabase path raises, so callers' failure
        # handling (mark job failed + refund) is unchanged. Only the error
        # code is included: never credentials.
        code = e.response.get("Error", {}).get("Code", "unknown")
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") or 502
        logger.exception("R2 storage upload rejected [%s]: %s", status, code)
        request = httpx.Request("PUT", f"{settings.r2_public_base_url}/{key}")
        raise httpx.HTTPStatusError(
            f"R2 upload failed: {code}", request=request, response=httpx.Response(status, request=request)
        ) from e
    except BotoCoreError as e:
        logger.exception("R2 storage upload failed: %s", type(e).__name__)
        raise httpx.TransportError(f"R2 upload failed: {type(e).__name__}") from e
    return f"{settings.r2_public_base_url}/{key}"


def _upload_supabase(file_path: str, file_bytes: bytes, content_type: str) -> str:
    """Uploads a file to the Supabase Storage bucket at the given path."""
    _validate_storage_path(file_path)
    url = f"{settings.supabase_url}/storage/v1/object/{BUCKET_NAME}/{file_path}"

    # Retried on transient failures (app/core/http_retry.py). A retry after an
    # ambiguous read timeout may land on an upload that in fact succeeded the
    # first time; without x-upsert Supabase would answer that with a 400
    # "Duplicate" (a 4xx, never retried), failing a job whose image is already
    # safely stored. x-upsert makes the retry an idempotent overwrite. The path
    # is unique per job (team/feature/job_id), so upsert can only ever replace
    # this same job's own file.
    response = send_with_retry(
        lambda: httpx.post(
            url,
            # Supabase's newer sb_secret_... keys are not JWTs, so they must go on
            # the `apikey` header -- `Authorization: Bearer` tries to parse them
            # as a JWT and fails ("Invalid Compact JWS"). Confirmed live against
            # this project's real key during this audit.
            headers={
                "apikey": settings.supabase_service_role_key,
                "Content-Type": content_type,
                "x-upsert": "true",
            },
            content=file_bytes,
            timeout=30,
        ),
        "Supabase storage upload",
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        # The exception's own str() only has the status code -- Supabase's
        # response body is where the real reason (bad bucket, RLS rejection,
        # bad key) lives. Caller (handle_fal_webhook) already stores only a
        # generic message on the job; this is what makes the real cause
        # visible in server logs.
        logger.exception("Supabase storage upload rejected [%s]: %s", response.status_code, response.text)
        raise

    return f"{settings.supabase_url}/storage/v1/object/public/{BUCKET_NAME}/{file_path}"


def download_from_url(url: str) -> bytes:
    """
    Downloads a file from any URL (used to fetch fal.ai's output before
    it expires, so we can re-upload it to our own permanent storage).
    """
    # Retried on transient failures (app/core/http_retry.py). A plain GET, so a
    # retry is always safe. A 403/404 (fal's output link expired or was never
    # valid) is a client error and is NOT retried.
    response = send_with_retry(lambda: httpx.get(url, timeout=30), "fal.ai output download")
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.exception("Failed to download fal.ai output [%s]: %s", response.status_code, response.text)
        raise
    return response.content