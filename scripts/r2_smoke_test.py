"""Smoke test for the Cloudflare R2 storage backend. Not imported by the app.

Usage: set R2_* in the environment (or a local .env), then
    python scripts/r2_smoke_test.py

Uploads a small object under {R2_KEY_PREFIX}smoke-test/, fetches it through
R2_PUBLIC_BASE_URL, checks CORS for the production frontend origin, then deletes
its own object. Prints only PASS/FAIL lines -- never secret values.
"""
import os
import sys
import time
import uuid

import boto3
import httpx
from botocore.config import Config
from dotenv import load_dotenv

ORIGIN = "https://www.shootpx.com"
REQUIRED = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_PUBLIC_BASE_URL")


def report(ok: bool, label: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'}: {label}")
    return ok


def main() -> int:
    load_dotenv()
    missing = [n for n in REQUIRED if not os.environ.get(n, "").strip()]
    if missing:
        for n in missing:
            report(False, f"{n} is not set")
        return 1

    env = {n: os.environ[n].strip() for n in REQUIRED}
    prefix = os.environ.get("R2_KEY_PREFIX", "").strip().strip("/")
    prefix = f"{prefix}/" if prefix else ""
    base = env["R2_PUBLIC_BASE_URL"].rstrip("/")
    key = f"{prefix}smoke-test/{int(time.time())}-{uuid.uuid4().hex[:8]}.txt"
    body = b"r2 smoke test"

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=env["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=10,
            read_timeout=30,
        ),
    )

    failed = False
    uploaded = False
    try:
        try:
            client.put_object(
                Bucket=env["R2_BUCKET"], Key=key, Body=body,
                ContentType="text/plain", CacheControl="public, max-age=60",
            )
            uploaded = True
            report(True, "upload object")
        except Exception as e:  # report the type only, never the message
            report(False, f"upload object ({type(e).__name__})")
            return 1

        url = f"{base}/{key}"
        try:
            r = httpx.get(url, timeout=30)
            ok = r.status_code == 200 and r.content == body
            failed |= not report(ok, f"public GET returns 200 (got {r.status_code})")
        except httpx.HTTPError as e:
            failed |= not report(False, f"public GET ({type(e).__name__})")

        try:
            r = httpx.get(url, headers={"Origin": ORIGIN}, timeout=30)
            allow = r.headers.get("access-control-allow-origin")
            state = "present" if allow else "missing"
            failed |= not report(allow in ("*", ORIGIN), f"CORS access-control-allow-origin for {ORIGIN} ({state})")
        except httpx.HTTPError as e:
            failed |= not report(False, f"CORS request ({type(e).__name__})")
    finally:
        if uploaded:
            try:
                client.delete_object(Bucket=env["R2_BUCKET"], Key=key)
                report(True, "delete smoke-test object")
            except Exception as e:
                failed = True
                report(False, f"delete smoke-test object ({type(e).__name__})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
