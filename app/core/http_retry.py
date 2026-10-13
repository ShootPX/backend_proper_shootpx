"""
One retry policy for every outbound HTTP call that talks to fal.ai or Supabase
(including Supabase Storage uploads; Cloudflare R2 uploads go through boto3, which
has its own retry config in app/core/storage.py and does not use this module).

Why: an intermittent network stall on a single call used to fail a whole
generation outright ("fal submit failed: The read operation timed out", seen
repeatedly). Those stalls are transient, so a couple of quick retries make
them invisible to the user.

The rules (deliberately identical everywhere, so this can't drift call site to
call site):
  * retry TRANSPORT failures (timeouts, connection errors, dropped/reset
    connections) and HTTP 429 / 502 / 503 / 504;
  * NEVER retry any other 4xx -- 422/400/401/403/404 mean the request itself
    was rejected, so repeating it just fails again (and could cost money);
  * at most 3 attempts, sleeping 1s then 3s between them;
  * after the last attempt the real error propagates unchanged, so the caller's
    existing failure handling (mark failed + refund) is exactly as before.

Callers pass a zero-argument function that performs ONE attempt and returns
the httpx.Response, and still call raise_for_status() themselves.

IMPORTANT: this blocks the calling thread while it sleeps (up to ~4s) and while
each attempt waits (up to 30s each). Never call it directly from an event loop
-- run it in a thread (asyncio.to_thread / run_in_threadpool).
"""
import logging
import time
from typing import Callable

import httpx

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
RETRY_DELAYS_SECONDS = (1.0, 3.0)
RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})


def send_with_retry(send: Callable[[], httpx.Response], what: str) -> httpx.Response:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        is_last = attempt == MAX_ATTEMPTS
        try:
            response = send()
        except httpx.TransportError as e:
            if is_last:
                raise
            logger.warning(
                "%s attempt %d/%d failed (%s: %s) -- retrying",
                what, attempt, MAX_ATTEMPTS, type(e).__name__, e,
            )
            time.sleep(RETRY_DELAYS_SECONDS[attempt - 1])
            continue

        if response.status_code in RETRYABLE_STATUS_CODES and not is_last:
            logger.warning(
                "%s attempt %d/%d got HTTP %s -- retrying",
                what, attempt, MAX_ATTEMPTS, response.status_code,
            )
            time.sleep(RETRY_DELAYS_SECONDS[attempt - 1])
            continue

        return response
