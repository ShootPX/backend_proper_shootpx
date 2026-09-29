# ShootPX Backend

Simple FastAPI backend. Runs on `http://localhost:8000`.

## Setup (first time)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
copy .env.example .env         # Windows   (cp on macOS / Linux)
```

Then fill in `.env` — every variable is required, the app won't start without them.

## Run (dev)

```bash
uvicorn app.main:app --reload
```

- `--reload` restarts the server when you change any file (new routes included).
- Stop with `Ctrl+C`.
- Default address: `http://127.0.0.1:8000`.

Open:

- http://localhost:8000/docs           — Swagger UI (try the endpoints here)
- http://localhost:8000/health         — health check
- http://localhost:8000/landing/billing
- http://localhost:8000/landing/tools

## Background worker (arq)

[arq](https://arq-docs.helpmanual.io/) is a Redis-backed async task queue for
Python. This backend uses it to run everything that shouldn't block an HTTP
request: submitting a generation job to fal.ai, sweeping/timing-out stuck
jobs, refilling subscription credits, and reconciling fal concurrency
counters. `POST /generate` just enqueues a job and returns immediately — the
actual work happens in a **separate process** (the arq worker), which is why
it has to be started on its own, alongside `uvicorn`, not instead of it.

`app/worker.py` defines the jobs (`WorkerSettings.functions`) and the cron
schedule (`WorkerSettings.cron_jobs`):

| Job | Trigger |
|-----|---------|
| `submit_generation_to_fal` | enqueued by `POST /generate` (one per job) |
| `sweep_stale_generation_jobs` | cron, every 15 min |
| `check_generation_timeouts` | cron, every 15 sec |
| `reconcile_fal_slots` | cron, every 10 min |
| `refill_due_subscriptions` | cron, daily at 03:00 |
| `cleanup_stale_pending_subscriptions` | cron, daily at 04:00 |

**Run it** (same `.env` as the API — it reads settings through
`app.core.config`, so nothing extra to configure beyond what's already there):

```bash
arq app.worker.WorkerSettings
```

Run this in its own terminal, alongside `uvicorn app.main:app --reload` — both
need to be running for generation jobs to actually complete (the API enqueues,
the worker executes). Add `--watch app` to auto-restart it on code changes,
same idea as uvicorn's `--reload`:

```bash
arq app.worker.WorkerSettings --watch app
```

- Needs `REDIS_URL` reachable (same Redis as caching/rate-limiting — arq uses
  it as the job queue, not just a cache here).
- Needs `FAL_KEY` and `PUBLIC_BACKEND_URL` set — `submit_generation_to_fal`
  calls fal.ai and gives it a webhook URL (`PUBLIC_BACKEND_URL/webhooks/fal`)
  to call back on completion. Locally, fal.ai can't reach `localhost`, so
  webhook delivery only works with a publicly reachable `PUBLIC_BACKEND_URL`
  (e.g. an ngrok tunnel) — without one, jobs still submit but rely on the
  worker's own polling fallback (`check_generation_timeouts`) to eventually
  notice completion instead of getting the webhook immediately.
- Stop with `Ctrl+C` — `WorkerSettings.on_shutdown` logs a clean shutdown.

### Running the worker in production (read this before going live)

The worker is a **separate process/service** from the API (`arq app.worker.WorkerSettings`
as its own Render *Background Worker*, systemd unit, container, etc.). If it is
down or frozen, generations sit in `queued` and nothing else fails loudly — so
it needs three layers of protection, all of which exist in this repo:

1. **Automatic restart of a frozen worker (in-process watchdog).** The worker
   runs every job and cron on one event loop, and much of the work (psycopg2,
   sync httpx) blocks. One hung call — e.g. a DB connection that died silently
   mid-query — froze a worker for 40+ minutes while the process stayed *alive*,
   so nothing restarted it. `app/core/watchdog.py` runs a thread that watches
   the loop; if the loop is silent for `WORKER_WATCHDOG_SECONDS` (default 180)
   it logs `WORKER_FROZEN` at CRITICAL, posts to `ALERT_WEBHOOK_URL` (if set)
   and hard-exits with **code 70**. **Your process manager must be configured
   to restart the worker whenever it exits** — Render restarts a crashed
   service automatically; systemd needs `Restart=always`; Docker needs
   `restart: unless-stopped`; supervisor needs `autorestart=true`. Without a
   restart-on-exit policy the watchdog just turns "frozen" into "dead".
2. **External alerting (`GET /health/worker`).** Point an uptime monitor
   (UptimeRobot, Better Stack, Render health checks…) at
   `https://<api>/health/worker`. It returns **503** when the worker heartbeat
   is missing or due jobs are overdue by more than 2 minutes, and covers the
   case the in-process watchdog cannot: the worker process being completely
   dead. Alert after 2–3 consecutive failures (a single legitimate 30s
   blocking HTTP call can briefly delay the heartbeat). The same heartbeat is
   available to container probes as `arq --check app.worker.WorkerSettings`.
3. **Stuck-job cleanup.** `sweep_stale_generation_jobs` (every 15 min, inside
   the worker) fails any job stuck in `queued`/`processing` for over 10 minutes
   and refunds its credits. Note it runs *in the same worker*, so it only
   helps once the worker is running again — layers 1 and 2 are what get it
   running.

Database connections use TCP keepalives + a connect timeout
(`app/core/database.py`) so a dead connection raises an error within about a
minute instead of hanging forever.

## Cache

`GET /landing/billing` and `GET /landing/tools` are cached in Redis for 1 hour
(keys `landing:billing`, `landing:tools`). The cache is best-effort — if Redis is
down the endpoints still work, they just hit the database every time.

### Clear the cache

After changing rows in the `subscription`, `credit`, or `tools` tables, clear the
cache so the next request re-reads the database:

```bash
curl -X POST http://127.0.0.1:8000/admin/cache/clear \
  -H "x-cache-secret: <CACHE_CLEAR_SECRET>"
```

- Replace `<CACHE_CLEAR_SECRET>` with the value from your `.env`.
- Deletes all `landing:*` keys. Response: `{"cleared": <count>}`.
- Wrong/missing secret → `403` / `422`.

## Auth

Firebase ID tokens. The frontend signs in with Firebase and sends the token on
protected routes:

```
Authorization: Bearer <firebase-id-token>
```

`app/deps.py` → `get_current_user` dependency:

1. Extracts and validates the `Bearer` token (missing / wrong scheme / empty → `401`).
2. Verifies it with the Firebase Admin SDK (invalid / expired → `401`).
3. `app/services/users.py` looks up the user by `firebase_uid`; on first login it
   creates the `users` row plus a personal `teams` row and an `owner`
   `team_members` row.

Routes:

| Route | Auth | Purpose |
|-------|------|---------|
| `GET /auth/me` | required | current user `{ id, email, name, avatarUrl }` |
| `POST /auth/authmail` | none | body `{ email, continue_url }` — emails a Firebase sign-in link |

`/landing/*` and `/health` are public.

### Teams

| Route | Auth | Purpose |
|-------|------|---------|
| `GET /teams` | required | the current user's teams `[{ id, name, role }]` |
| `GET /teams/{team_id}/members` | any member | `{ id, name, members: [{ userId, email, name, avatarUrl, role, joinedAt }] }` |
| `POST /teams/{team_id}/invite` | owner only | body `{ email, role? }` (`role`: `editor` default / `owner`) — emails an accept link |
| `POST /invites/{token}/accept` | required | accepting user's email must match the invite; joins at the invite's role |
| `PATCH /teams/{team_id}` | owner only | body `{ name }` — rename the team |
| `DELETE /teams/{team_id}/members/{member_user_id}` | owner only | remove a member (can't remove the last owner) |
| `DELETE /teams/{team_id}` | owner only | delete team + its memberships + its invites |
| `GET /teams/{team_id}/billing` | any member | `{ totalCredits, subscriptionCredits, topupCredits, plan, subscriptionStatus, currentPeriodEnd }` |
| `POST /billing/teams/{team_id}/credit-packs/{pack_id}/checkout` | owner only | creates a Razorpay order → `{ order_id, amount, currency, key_id }` |
| `POST /billing/webhook` | Razorpay (signed) | credit-pack `payment.captured` → adds to the topup wallet, idempotent on `razorpay_payment_id` |

- **Size cap:** `MAX_TEAM_MEMBERS = 5` (owner + editors). Pending invites count toward it.
  Over the cap → `409` on invite and on accept.
- The invite email link is a **Firebase email sign-in link**: clicking it signs the
  invitee in as exactly that email (no account picker), lands on
  `FRONTEND_URL/testconsole.html?invite_token=…&invite_email=…`, and the page
  auto-calls `/invites/{token}/accept`.
- Reuses an existing pending invite for the same team + email; re-inviting with a
  different role updates it.
- Accept checks the token's email against the signed-in user's email. Already a
  member → role is updated to the invite's role, no duplicate row.
- `remove_member` refuses to delete the last `owner` → `400`.

Add auth to a route:

```python
from app.deps import get_current_user
from app.models.user import User

@router.get("/something")
def handler(user: User = Depends(get_current_user)):
    ...
```

## Rate limiting

Per-client-IP limits (via `slowapi`), counters in Redis so the limit is shared
across all workers / instances. Selected limits:

| Endpoint | Limit |
|----------|-------|
| `GET /landing/billing`, `GET /landing/tools` | 30 / minute |
| `GET /teams`, `GET /teams/{id}/members` | 20–30 / minute |
| `POST /teams/{id}/invite`, `POST /invites/{token}/accept` | 10 / minute |
| `PATCH`/`DELETE /teams/{id}`, remove member | 5–10 / minute |
| `POST /admin/cache/clear` | 5 / minute |

Over the limit → `429 {"error": "Rate limit exceeded: ..."}`.
`GET /` and `GET /health` are not limited (uptime checks).

**Client IP behind a proxy:** the key is resolved by `client_ip()` in
`app/core/limiter.py`. Locally it uses the socket peer. In production set
**`TRUST_PROXY=true`** — then it reads the real client IP from the end of
`X-Forwarded-For` (the address the trusted proxy saw). Without this, every user
behind the load balancer would share one bucket. Keep it `false` locally, or the
header is spoofable.

## Test

```bash
pytest
```

## Config = environment

All values come from `.env` (local) or real environment variables.
No silent defaults — if a variable is missing or invalid, the app refuses to start.

| Variable | Example | Meaning |
|----------|---------|---------|
| `APP_NAME` | `shootpx-backend` | App name (shown in `/docs` and `GET /`) |
| `ENV` | `development` | `development` or `production` |
| `HOST` | `127.0.0.1` | Address to bind |
| `PORT` | `8000` | Port to bind |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated frontend origins allowed to call the API |
| `DATABASE_URL` | `postgresql://...` | Postgres / Supabase connection string |
| `REDIS_URL` | `rediss://...` | Redis / Upstash connection string (caching + rate limits) |
| `CACHE_CLEAR_SECRET` | `long-random-string` | Secret required by `POST /admin/cache/clear` |
| `FIREBASE_CREDENTIALS_PATH` | `./firebase-service-account.json` | Firebase service account JSON (verifies ID tokens) |
| `TRUST_PROXY` | `false` local / `true` on Render | read client IP from `X-Forwarded-For` for rate limiting |
| `FRONTEND_URL` | `http://localhost:3000` | Base URL where the console/frontend runs — used to build links in emails (no trailing slash) |
| `RAZORPAY_KEY_ID` | `rzp_test_xxx` | Razorpay API key id (test or live) |
| `RAZORPAY_KEY_SECRET` | `xxx` | Razorpay API key secret |
| `RAZORPAY_WEBHOOK_SECRET` | `xxx` | Secret configured on the Razorpay webhook — verifies `POST /billing/webhook` signatures |
| `FAL_KEY` | `xxx` | fal.ai API key — the arq worker calls fal.ai's generation models with it |
| `PUBLIC_BACKEND_URL` | `https://api.shootpx.com` | Publicly reachable base URL fal.ai's webhook calls back to (`{PUBLIC_BACKEND_URL}/webhooks/fal`) — needs a tunnel (e.g. ngrok) to work locally |
| `SUPABASE_URL` | `https://xxx.supabase.co` | Supabase project URL — used for storing generated output images |
| `SUPABASE_SERVICE_ROLE_KEY` | `xxx` | Supabase service role key (storage uploads) |
| `FAL_PER_TEAM_CONCURRENCY_LIMIT` | `5` | Max fal.ai jobs in flight per team at once |
| `WORKER_WATCHDOG_SECONDS` | `180` (optional) | Seconds the worker's event loop may be silent before it alerts and exits for a restart. `0` disables |
| `ALERT_WEBHOOK_URL` | *(unset)* (optional) | Slack/Discord-style incoming webhook that receives the `WORKER_FROZEN` alert |

## Structure

Layered — each folder has one job, dependencies point downward only
(`routes → services → models`, everything may use `core`).

```
app/
  main.py            assembles the app: middleware + api_router. Nothing else.
  deps.py            shared FastAPI dependencies (get_current_user)

  core/              infrastructure — no business logic
    config.py        settings from env
    database.py      SQLAlchemy engine + get_db()
    cache.py         Redis client + get/set helpers (best-effort)
    limiter.py       slowapi rate limiter (Redis-backed, keyed by IP)
    firebase.py      Firebase Admin init + verify_token()
    email.py         low-level SMTP send_email()

  models/            SQLAlchemy tables (user, team, team_member, team_invite, subscription, credit, tool)
  schemas/           Pydantic request/response models (auth, billing, tools)

  services/          business logic — the "what actually happens"
    users.py         get_or_create_user (provision user + personal team)
    auth_links.py    generate Firebase email link + send it
    teams.py         membership checks, member/seat counts, rename/remove/delete
    team_invites.py  create / accept team invites

  routes/            thin HTTP layer — parse request, call a service, return
    __init__.py      api_router — includes every route module below
    health.py        GET /            GET /health
    auth.py          GET /auth/me     POST /auth/authmail
    teams.py         GET /teams  ·  GET|PATCH|DELETE /teams/{id}  ·  members  ·  invite  ·  accept
    billing.py       GET /landing/billing
    tools.py         GET /landing/tools
    cache.py         POST /admin/cache/clear

tests/               pytest
```

## Adding a new endpoint

1. `app/models/<thing>.py`   — table model (if it needs the DB)
2. `app/schemas/<thing>.py`  — request / response models
3. `app/services/<thing>.py` — the business logic (DB queries, external calls)
4. `app/routes/<thing>.py`   — `router = APIRouter(prefix=...)`, handlers call the service
5. `app/routes/__init__.py`  — `api_router.include_router(<thing>.router)`
6. `tests/test_<thing>.py`   — a quick check

To rate-limit a route, add `@limiter.limit("N/minute")` under the `@router.get(...)`
decorator and give the function a `request: Request` first argument (slowapi needs it).
