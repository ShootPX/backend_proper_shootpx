from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App configuration.

    The environment is the single source of truth. Every value must come from a
    real environment variable (cloud) or the local `.env` file. There are no
    silent fallbacks: if something is missing or invalid, the app refuses to
    start instead of running with wrong settings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # A secret pasted into Render's env var UI (or a .env file saved with
        # a trailing newline) can silently carry a trailing "\n" into the
        # value. That's invisible everywhere except when the value is later
        # sent as a raw HTTP header, where h11 rejects it outright
        # (httpx.LocalProtocolError: "Illegal header value ...\n") -- found
        # live via supabase_service_role_key breaking every storage upload.
        # Stripping whitespace on every string field closes this for all
        # current and future settings, not just this one key.
        str_strip_whitespace=True,
    )

    app_name: str
    env: Literal["development", "production"]
    host: str
    port: int
    # Comma-separated list of allowed frontend origins, e.g.
    # "http://localhost:3000,https://app.shootpx.com"
    cors_origins: str
    # Base URL of the frontend, used to build links in emails (no trailing slash)
    frontend_url: str
    database_url: str
    redis_url: str
    cache_clear_secret: str
    firebase_credentials_path: str
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    email_from_name: str
    email_from_address: str
    # True when the app runs behind a reverse proxy / load balancer (Render, nginx…).
    # Makes rate limiting use the real client IP from X-Forwarded-For instead of the
    # proxy's IP. Keep False for local dev (X-Forwarded-For would be spoofable).
    trust_proxy: bool
    razorpay_key_id: str
    razorpay_key_secret: str
    razorpay_webhook_secret: str
    fal_key: str
    public_backend_url: str
    supabase_url: str
    supabase_service_role_key: str
    fal_per_team_concurrency_limit: int
    # --- worker health monitoring (both OPTIONAL, unlike everything above) ----
    # Monitoring must not become a new way for a deploy to fail to boot, so
    # these have safe defaults instead of being required.
    # How long the arq worker's event loop may go without ticking before the
    # in-process watchdog declares it frozen, alerts, and hard-exits so the
    # process manager restarts it. Must comfortably exceed the longest
    # LEGITIMATE blocking stretch (a few sequential 30s HTTP timeouts in one
    # job). 0 disables the watchdog.
    worker_watchdog_seconds: int = 180
    # Optional Slack/Discord-style incoming webhook. When set, a frozen-worker
    # event POSTs a message there in addition to the CRITICAL log line.
    alert_webhook_url: str | None = None
    # How long an unpaid plan-switch (upgrade) checkout may stay pending before
    # the worker cancels the unpaid Razorpay subscription and clears it. OPTIONAL
    # with a default, like the worker settings above. 24h leaves room for slow
    # UPI-mandate / bank-OTP authorisations.
    pending_switch_ttl_hours: int = 24

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.env == "production"


def load_settings() -> Settings:
    try:
        return Settings()
    except Exception as exc:  # missing var, wrong type, bad value
        raise RuntimeError(
            "Environment configuration is invalid or incomplete. "
            "Set every variable listed in .env.example (as real env vars in the "
            f"cloud, or in a local .env file).\nDetails: {exc}"
        ) from exc


settings = load_settings()