import asyncio
import logging
import math
logging.basicConfig(level=logging.INFO)

from datetime import datetime, timedelta, timezone

from sqlalchemy import func
from arq.connections import RedisSettings
from arq.cron import cron

from app.core.config import settings
from app.core.cache import redis_client
from app.core.database import SessionLocal
from app.core.watchdog import WorkerWatchdog
from app.core.fal_client import submit_to_fal, check_fal_status, fetch_fal_result
from app.core.arq_pool import get_arq_pool
import app.models  # noqa: F401 -- registers every model on Base.metadata before any query runs (see app/models/__init__.py)
from app.models.team_subscription import TeamSubscription
from app.models.subscription import Subscription
from app.models.team import Team
from app.models.generation_job import GenerationJob
from app.services.tool_definitions import get_tool_definition
from app.services.credits import refill_subscription_credits, refund_credits
from app.services.webhooks import send_renewal_notice_email
from app.services import ledger as ledger_svc
from app.services.generation_lock import (
    release_generation_lock, try_reserve_fal_slot, release_fal_slot, INFLIGHT_KEY_GLOBAL,
)
from app.services import generation as generation_svc
from app.services.generation import TIMEOUT_MESSAGE, SWEEP_TIMEOUT_MESSAGE
from app.tools.registry import TOOL_HANDLERS
from app.tools import recolor, creative_photoshoot, listing_planner, model_shoot

logger = logging.getLogger(__name__)

# Recolor, creative_photoshoot and listing_photoshoot no longer use "size" at
# all -- each now builds its own explicit {width, height} from its own
# ai_steps.size_map (recolor.build_image_size / *.resolve_generation_params,
# see each tool's own branch below), keyed by aspect_ratio+resolution rather
# than a single "size" string. This table is kept for any tool that still
# sends a plain "size" field of its own (none currently do -- it's dead code
# until the next one does), written against gpt-image-2/edit's vocabulary
# confirmed at https://fal.ai/models/openai/gpt-image-2/edit/api; if a future
# tool reuses the "size" field name with different values against a different
# fal model, this translation would need to become per-tool rather than global.

# fal has no named preset for a plain 2:3 / 3:2 ratio, so those fall back to
# an explicit {width, height} object. Constraints (per the docs above): both
# dimensions multiples of 16, max edge 3840px, aspect ratio <= 3:1, total
# pixels between 655,360 and 8,294,400 -- 1024x1536 (and its 1536x1024 swap)
# satisfy all four comfortably.
FAL_SIZE_TRANSLATION = {
    "original": "auto",
    "1:1": "square_hd",
    "9:16": "portrait_16_9",
    "3:4": "portrait_4_3",
    "4:3": "landscape_4_3",
    "16:9": "landscape_16_9",
    "2:3": {"width": 1024, "height": 1536},
    "3:2": {"width": 1536, "height": 1024},
}

# "color" and "target_area" are Recolor's own UI/schema fields, already
# consumed by build_instruction() (above, in the caller) to produce the
# `prompt` string -- they are not part of fal-ai/flux-2/edit's real input
# schema and must not be forwarded raw. ("aspect_ratio"/"resolution" are also
# Recolor's own fields, not real flux-2/edit fields either, but are stripped
# in the feature_type-scoped "recolor" branch below instead of here, next to
# where image_size is already being built from them.)
# "source_feature_type" is enhance_prompt's own UI/schema field, already
# consumed by build_instruction() to look up the source tool's enhance_hint
# -- not part of openrouter/router's real input schema, must not be forwarded.
# "idea" is creative_photoshoot's own UI/schema field, already consumed by
# build_instruction() to build the scene prompt -- not part of
# gpt-image-2/edit's real input schema, must not be forwarded raw.
# "shot_type" is listing_photoshoot's own per-job planning metadata (written
# by plan_listing_shots() into each job's input_params via per_job_overrides,
# before the job ever exists -- there is no build_instruction step for this
# tool) -- not part of gpt-image-2/edit's real input schema either.
# "gender"/"age_bracket"/"ethnicity"/"skin_tone"/"body_type"/"notes" are
# model_shoot_generate_model's own structured-attribute UI fields, already
# consumed by build_instruction() into the AI-written prompt -- not part of
# openrouter/router's (or any other tool's) real input schema.
FAL_NON_SCHEMA_FIELDS = {
    "color", "target_area", "source_feature_type", "idea", "shot_type",
    "gender", "age_bracket", "ethnicity", "skin_tone", "body_type", "notes",
}

# model_shoot_generate_model's real fal model (openai/gpt-image-2 -- the
# plain text-to-image endpoint, NOT /edit) takes NO image input at all:
# confirmed against https://fal.ai/models/openai/gpt-image-2/api, whose full
# input schema is prompt/image_size/background/quality/num_images/
# output_format/sync_mode -- no image_urls or image_url field exists in it.
# /generate's generic path still always sets input_params["image_urls"] to a
# list (empty here, since this tool's max_input_images=0 blocks any upload
# attempt) -- strip it explicitly for this one tool rather than gambling on
# whether an empty array on an undocumented extra field is silently ignored.
MODEL_SHOOT_GENERATE_MODEL_NON_SCHEMA_FIELDS = {"image_urls"}

def _translate_fal_params(params: dict, feature_type: str | None = None) -> dict:
    """
    Applied only to the payload actually sent to fal.ai -- the caller's own
    dict (job.input_params) is never mutated, so the user-facing size value
    remains untouched for display and quality remains untouched for both
    display and _resolve_credit_cost's lookup (it's already fal's real value).
    """
    translated = dict(params)

    size_value = translated.pop("size", None)
    if size_value is not None:
        translated["image_size"] = FAL_SIZE_TRANSLATION.get(size_value, size_value)

    # recolor, creative_photoshoot, listing_photoshoot and model_shoot:
    # image_size (and quality, for the first three) are already computed and
    # set by the caller (from ai_steps.size_map / resolve_generation_params,
    # see submit_generation_to_fal) -- aspect_ratio and resolution are each
    # tool's own UI/schema fields, not real fal input fields, and must not be
    # forwarded raw.
    if feature_type in ("recolor", "creative_photoshoot", "listing_photoshoot", "model_shoot"):
        translated.pop("aspect_ratio", None)
        translated.pop("resolution", None)

    if feature_type == "model_shoot":
        # seedream's real image field is a single "image_urls" list (its
        # schema: prompt + image_urls, both required, max 10) -- the
        # model_image/garments/reference_images split only exists so
        # plan_model_shoot() can label each image differently in its vision
        # prompt (see app/tools/model_shoot.py); the real generation call
        # needs them recombined into one list, in the SAME order the planner
        # labeled them (#Image1 = model reference, then each garment group's
        # images in order, then style/pose references) so the edit model's
        # "Figure N" references line up with what the planner's prompts
        # actually meant. flatten_garment_image_urls is the same flattening
        # plan_model_shoot itself used to build its vision-call labels --
        # reused here so the two orderings can never drift apart.
        model_image_url = translated.pop("model_image", None)
        garments = translated.pop("garments", None) or []
        reference_image_urls = translated.pop("reference_images", None) or []
        translated["image_urls"] = (
            ([model_image_url] if model_image_url else [])
            + model_shoot.flatten_garment_image_urls(garments)
            + reference_image_urls
        )

    if feature_type == "model_shoot_generate_model":
        for field in MODEL_SHOOT_GENERATE_MODEL_NON_SCHEMA_FIELDS:
            translated.pop(field, None)

    for field in FAL_NON_SCHEMA_FIELDS:
        translated.pop(field, None)

    return translated


async def refill_due_subscriptions(ctx):
    db = SessionLocal()
    now = datetime.now(timezone.utc)

    due = db.query(TeamSubscription).filter(
        TeamSubscription.status == "active",
        TeamSubscription.next_refill_at <= now,
    ).all()

    for team_sub in due:
        try:
            locked_sub = db.query(TeamSubscription).filter(
                TeamSubscription.id == team_sub.id
            ).populate_existing().with_for_update().first()

            if not locked_sub or locked_sub.status != "active":
                continue
            if locked_sub.next_refill_at > now:
                continue

            plan = db.query(Subscription).filter(
                Subscription.id == locked_sub.subscription_id
            ).first()
            if not plan:
                continue

            step = {"week": timedelta(weeks=1), "month": timedelta(days=30), "year": timedelta(days=30)}
            if plan.period_label not in step:
                continue

            # Same transaction as the refill and the next_refill_at advance below.
            # Keyed by the slice's own due time, so a retried run can't record it twice.
            due_at = locked_sub.next_refill_at
            before = ledger_svc.subscription_pool_balance(db, locked_sub.team_id)
            ledger_svc.record_subscription_grant(
                db, team_id=locked_sub.team_id, balance_before=before,
                balance_after=locked_sub.credits_per_refill,
                idempotency_key=f"refill:{locked_sub.id}:{due_at.isoformat()}", source="worker",
                razorpay_subscription_id=locked_sub.razorpay_subscription_id,
                metadata=ledger_svc.plan_metadata(plan),
            )
            refill_subscription_credits(db, locked_sub.team_id, locked_sub.credits_per_refill, commit=False)
            locked_sub.next_refill_at = now + step[plan.period_label]
            db.commit()
            logger.info(
                "Refilled subscription for team %s (+%s credits)",
                locked_sub.team_id, locked_sub.credits_per_refill,
            )
        except Exception:
            db.rollback()
            logger.exception(
                "refill failed for team %s — skipping, will retry next run",
                team_sub.team_id,
            )

    lapsed = (
        db.query(TeamSubscription)
        .join(Team, Team.id == TeamSubscription.team_id)
        .filter(
            TeamSubscription.status == "cancelled",
            TeamSubscription.next_refill_at <= now,
            Team.subscription_credits_remaining > 0,
        )
        .all()
    )
    for team_sub in lapsed:
        try:
            team = db.query(Team).filter(Team.id == team_sub.team_id).with_for_update().first()
            if team:
                team.subscription_credits_remaining = 0
            db.commit()
            logger.info("Lapsed cancelled subscription credits for team %s", team_sub.team_id)
        except Exception:
            db.rollback()
            logger.exception("failed to lapse cancelled subscription for team %s", team_sub.team_id)

    db.close()


# How long before a yearly plan's current_period_end to warn its owner.
# Yearly plans have no second "charged" webhook to key a notice off
# (total_count=1 -- there's only ever ONE charge, at the start of the year),
# unlike week/month plans whose notice fires off handle_subscription_charged's
# own paid_count-based check (see app/services/webhooks.py) -- this cron
# checks current_period_end directly instead. A window (not an exact-day
# match) so a missed run (worker down for a day) still catches up on the next
# tick instead of silently skipping the notice for that team entirely.
YEARLY_RENEWAL_NOTICE_WINDOW_DAYS = 30


async def send_yearly_renewal_notices(ctx):
    db = SessionLocal()
    now = datetime.now(timezone.utc)
    window_end = now + timedelta(days=YEARLY_RENEWAL_NOTICE_WINDOW_DAYS)

    due = (
        db.query(TeamSubscription)
        .join(Subscription, Subscription.id == TeamSubscription.subscription_id)
        .filter(
            TeamSubscription.status == "active",
            Subscription.period_label == "year",
            TeamSubscription.current_period_end <= window_end,
            TeamSubscription.renewal_notice_sent_at.is_(None),
        )
        .all()
    )

    for team_sub in due:
        try:
            locked_sub = db.query(TeamSubscription).filter(
                TeamSubscription.id == team_sub.id
            ).populate_existing().with_for_update().first()

            if not locked_sub or locked_sub.status != "active":
                continue
            if locked_sub.renewal_notice_sent_at is not None:
                continue  # a concurrent run already sent it
            if locked_sub.current_period_end > window_end:
                continue  # stale read -- period_end moved out since the query above

            send_renewal_notice_email(db, locked_sub)
            locked_sub.renewal_notice_sent_at = now
            db.commit()
            logger.info(
                "Sent yearly renewal notice for team %s (current_period_end=%s)",
                locked_sub.team_id, locked_sub.current_period_end,
            )
        except Exception:
            db.rollback()
            logger.exception(
                "yearly renewal notice failed for team %s — skipping, will retry next run",
                team_sub.team_id,
            )

    db.close()


async def cleanup_stale_pending_subscriptions(ctx):
    """Housekeeping for checkouts that were started and never paid.

    ONLY never-activated rows (credits_per_refill == 0). "pending" is also the
    status of a real, paying subscription whose renewal payment failed while
    Razorpay retries for days (webhooks.handle_subscription_pending) -- and
    created_at is the row's ORIGINAL signup date, so without this condition
    the sweep deleted such a customer's subscription outright the next time it
    ran. Abandoned attempts no longer block re-checkout (see
    billing.create_subscription_checkout), so this is tidy-up, not a gate.
    """
    db = SessionLocal()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)

    stale = db.query(TeamSubscription).filter(
        TeamSubscription.status == "pending",
        TeamSubscription.credits_per_refill == 0,
        TeamSubscription.created_at <= cutoff,
    ).all()

    for row in stale:
        logger.info("Removing stale pending subscription for team %s", row.team_id)
        db.delete(row)

    db.commit()
    db.close()


async def expire_pending_switches(ctx):
    """Cancel upgrade checkouts nobody paid for within PENDING_SWITCH_TTL_HOURS.

    The team's current plan was never touched by the switch, so this only
    cancels the unpaid replacement at Razorpay and clears the pending columns.
    A checkout that was actually paid is left alone (see
    billing.expire_pending_switch). One row's failure never blocks the rest.
    """
    from app.services.billing import expire_pending_switch

    db = SessionLocal()
    try:
        due_ids = [
            row_id for (row_id,) in db.query(TeamSubscription.id).filter(
                TeamSubscription.pending_switch_expires_at.isnot(None),
                TeamSubscription.pending_switch_expires_at <= datetime.now(timezone.utc),
            ).all()
        ]
        db.rollback()  # end the read transaction before the per-row locking work
        for row_id in due_ids:
            try:
                expire_pending_switch(db, row_id)
            except Exception:
                db.rollback()
                logger.exception("failed to expire pending switch for subscription row %s", row_id)
    finally:
        db.close()


async def reconcile_subscriptions(ctx):
    """Hourly: compare live subscriptions with Razorpay and repair what webhooks
    missed (see services/reconciliation.py). The whole batch runs in a thread --
    it is a series of blocking Razorpay calls, and every job and cron shares this
    worker's one event loop (a stalled loop trips the watchdog)."""
    from app.services.reconciliation import reconcile_due_subscriptions

    def run() -> dict:
        db = SessionLocal()
        try:
            return reconcile_due_subscriptions(db)
        finally:
            db.close()

    try:
        await asyncio.to_thread(run)
    except Exception:
        logger.exception("reconcile_subscriptions run failed -- will retry next hour")


async def purge_expired_deleted_teams(ctx):
    """
    The second half of the two-clock team-deletion design (see
    services/teams.py soft_delete_team/restore_team): a team past its
    GRACE_PERIOD_DAYS window gets permanently hard-deleted here. Razorpay was
    already cancelled the moment deletion was requested, not now.
    """
    from app.services.teams import list_teams_past_grace_period, purge_team

    db = SessionLocal()
    try:
        expired = list_teams_past_grace_period(db)
        for team in expired:
            try:
                logger.info(
                    "Purging team %s -- deleted_at=%s past the grace window",
                    team.id, team.deleted_at,
                )
                purge_team(db, team.id)
            except Exception:
                db.rollback()
                logger.exception("failed to purge team %s -- will retry next run", team.id)
    finally:
        db.close()


GENERIC_START_FAILED_MESSAGE = "Failed to start generation. Please try again."

# At-capacity re-enqueues use pool.enqueue_job(), NOT arq's own Retry
# mechanism -- each call starts a brand-new arq job with its own try counter
# reset to 1, so arq's max_tries does not bound this loop at all. Without an
# explicit deadline here, a permanently-stuck concurrency counter (or any
# other reason try_reserve_fal_slot keeps returning False -- now only the
# per-team cap; see generation_lock.try_reserve_fal_slot) would retry
# forever.
#
# The deadline is a wall-clock budget, not a fixed retry COUNT (the old
# MAX_CAPACITY_RETRIES=12 * a flat 5s defer =~ 1 minute) -- that budget was
# far shorter than the jobs actually HOLDING the slots this one is waiting
# on are allowed to run for, so any batch deeper than ~2-3 items reliably
# failed its own later jobs the moment generation ran long. Found live: a
# 4-job listing_photoshoot batch against fal_per_team_concurrency_limit=3
# failed its 4th job outright, never even reaching fal.
#
# Derived, not a bare magic number, so a future change to either input can't
# silently reintroduce the bug: one team's own single batch, at its tool's
# own max output_count, can need up to ceil(max_output_count /
# fal_per_team_concurrency_limit) sequential "waves" for its LAST job to get
# a slot -- each wave bounded by the slowest tool's own
# generation_timeout_seconds. +1 wave of margin, since jobs don't finish in
# perfect lockstep and other in-flight work for the same team (a different
# batch, a different tool) can occupy slots too.
_MAX_TOOL_GENERATION_TIMEOUT_SECONDS = 300  # ceiling across all tools today: listing_photoshoot, model_shoot, model_shoot_generate_model -- confirmed against the live tool_definitions table
_MAX_OUTPUT_COUNT_FOR_SLOT_QUEUEING = 8  # listing_photoshoot's and model_shoot's own per-batch cap (see listing_planner.MAX_OUTPUT_COUNT and model_shoot.resolve_generation_params)
_SLOT_QUEUE_WAVES = math.ceil(_MAX_OUTPUT_COUNT_FOR_SLOT_QUEUEING / settings.fal_per_team_concurrency_limit)
SLOT_ACQUISITION_DEADLINE_SECONDS = (_SLOT_QUEUE_WAVES + 1) * _MAX_TOOL_GENERATION_TIMEOUT_SECONDS

# Polling backoff while waiting for a slot: a tight 5s poll for the first
# minute (most contention clears fast -- a couple of short jobs finishing),
# then a much lighter 20s poll for the rest of the (now much longer)
# SLOT_ACQUISITION_DEADLINE_SECONDS budget, so a job waiting several minutes
# doesn't hammer Redis with a capacity check every 5 seconds the whole time.
_SLOT_RETRY_FAST_POLL_SECONDS = 5
_SLOT_RETRY_FAST_POLL_WINDOW_SECONDS = 60
_SLOT_RETRY_SLOW_POLL_SECONDS = 20


def _next_slot_retry_defer_seconds(elapsed_seconds: float) -> int:
    if elapsed_seconds < _SLOT_RETRY_FAST_POLL_WINDOW_SECONDS:
        return _SLOT_RETRY_FAST_POLL_SECONDS
    return _SLOT_RETRY_SLOW_POLL_SECONDS


async def submit_generation_to_fal(ctx, job_id: str, attempt: int = 1):
    """
    Picks up a queued GenerationJob and submits it to fal.ai for real.
    On success: status -> processing, waits for the real webhook.
    On failure: status -> failed, credits refunded, lock (+ fal slot, if one
    was actually reserved) released.
    """
    db = SessionLocal()
    try:
        job = db.query(GenerationJob).filter(GenerationJob.id == job_id).with_for_update().first()
        if not job or job.status != "queued":
            return

        tool = get_tool_definition(db, job.feature_type)
        if not tool:
            return

        def fail_cleanly(message: str) -> None:
            job.status = "failed"
            job.error_message = message
            refund_credits(
                db, job.team_id, job.credits_charged,
                job.credits_from_subscription, job.credits_from_topup,
            )
            db.commit()
            release_generation_lock(job.user_id)

        # Everything from here through the fal submission must have a clean
        # failure path -- an uncaught exception anywhere in this function
        # leaves the job stuck "queued" forever with credits spent and the
        # lock held, since nothing else will ever mark it failed (the 10-min
        # sweep eventually would, but there's no reason to make a user wait
        # that long for what's already a definitive failure).
        try:
            slot_reserved = try_reserve_fal_slot(job.team_id)
        except Exception:
            logger.exception("try_reserve_fal_slot raised for job %s", job.id)
            fail_cleanly(GENERIC_START_FAILED_MESSAGE)
            return

        if not slot_reserved:
            # Measured from job.created_at (when credits were charged and the
            # user's wait actually started), not from this attempt or the
            # first retry -- an arq worker restart or delayed pickup before
            # attempt 1 even runs must count against the same budget, not
            # reset it.
            elapsed = (datetime.now(timezone.utc) - job.created_at).total_seconds()
            if elapsed >= SLOT_ACQUISITION_DEADLINE_SECONDS:
                logger.warning(
                    "job %s gave up waiting for a fal concurrency slot after %.0fs "
                    "(attempt %d, deadline %ds)",
                    job.id, elapsed, attempt, SLOT_ACQUISITION_DEADLINE_SECONDS,
                )
                fail_cleanly(GENERIC_START_FAILED_MESSAGE)
                return

            # At the per-team cap right now -- re-enqueue this same job to try
            # again shortly, so one team's batch can't starve the others. (The
            # account-wide fal limit is no longer pre-checked or retried on
            # here -- fal's own documented behavior already queues and
            # dispatches automatically once that's hit, so submission below
            # just proceeds and lets fal handle it.) Credits stay reserved and
            # the job stays "queued" for the whole wait -- fail_cleanly (and
            # its refund) only runs once, above, if the deadline is truly
            # exceeded, never on an ordinary retry.
            try:
                pool = await get_arq_pool()
                defer_seconds = _next_slot_retry_defer_seconds(elapsed)
                await pool.enqueue_job(
                    "submit_generation_to_fal", job_id, attempt + 1, _defer_by=defer_seconds,
                )
            except Exception:
                logger.exception("failed to re-enqueue job %s for a later capacity retry", job.id)
                fail_cleanly(GENERIC_START_FAILED_MESSAGE)
            return

        webhook_url = f"{settings.public_backend_url}/webhooks/fal?job_id={job.id}"

        try:
            build_instruction = TOOL_HANDLERS.get(job.feature_type)
            if build_instruction:
                # build_instruction can itself make a blocking, synchronous
                # HTTP call (recolor's vision step -> call_fal_sync). Run it
                # in a thread, not inline -- called directly, it would freeze
                # this WHOLE worker's event loop (every other job, arq's own
                # heartbeat, graceful shutdown) for however long that call
                # takes. Found live: this is very likely why a job caught
                # mid-vision-call by a worker restart was left stuck
                # "queued" instead of cleanly failed -- the loop wasn't free
                # to run the except block/cleanup below, let alone arq's own
                # shutdown handler.
                instruction = await asyncio.to_thread(build_instruction, job, tool, db)
                params = {**job.input_params, "prompt": instruction}
                # The tool's own ai_steps can name which underlying model the
                # fal endpoint itself should route to (e.g. enhance_prompt's
                # tool_definitions row: ai_steps={"model": "google/gemini-2.5-flash"}
                # for openrouter/router) -- distinct from ai_steps sub-keys
                # like recolor's detect_target, which build_instruction()
                # already fully consumes itself and never surfaces here.
                #
                # Deliberately scoped to enhance_prompt only -- unlike
                # openrouter/router, neither recolor's (fal-ai/flux-2/edit)
                # nor creative_photoshoot's (gpt-image-2/edit) real fal model
                # has a "model" field in its schema at all. tool.ai_steps is
                # a raw JSONB column hand-edited via SQL with no schema
                # enforcement; a top-level "model" key added to recolor's or
                # creative_photoshoot's row by mistake (easy to do since
                # nested steps like detect_target/scene_vision already use
                # "model" as a sub-key) would otherwise silently inject an
                # invalid field into every real image generation call for
                # that tool.
                if job.feature_type == "enhance_prompt" and tool.ai_steps.get("model"):
                    params["model"] = tool.ai_steps["model"]
                # recolor's real fal model (fal-ai/flux-2/edit) takes an
                # explicit {width, height} image_size, looked up from
                # ai_steps.size_map by the user's aspect_ratio + resolution
                # choice -- aspect_ratio/resolution are recolor's own UI
                # fields, not real flux-2/edit input fields, so they must
                # never be forwarded raw (stripped in _translate_fal_params
                # below). resolution's credit_cost was already resolved off
                # the untranslated job.input_params back at job-creation time
                # in create_generation_batch -- unrelated to this step.
                if job.feature_type == "recolor":
                    params["image_size"] = recolor.build_image_size(job, tool)
                # creative_photoshoot's real fal model has no aspect_ratio/
                # resolution fields of its own -- resolve_generation_params
                # turns the user's aspect_ratio+resolution+quality picks into
                # an explicit {width, height} image_size (from ai_steps.size_map)
                # and the real fal quality value. Credit cost was already
                # resolved off the same function back at job-creation time in
                # create_generation_batch -- unrelated to this step.
                if job.feature_type == "creative_photoshoot":
                    resolved = creative_photoshoot.resolve_generation_params(job, tool)
                    params["image_size"] = resolved["image_size"]
                    params["quality"] = resolved["quality"]
                # model_shoot_generate_model has no per-request quality/size
                # picks of its own at all (its param_schema is pure structured
                # attributes -- gender/age/ethnicity/etc) -- the real fal
                # request was going out as just {"prompt": ...}, silently
                # falling back to the endpoint's own defaults (quality=high,
                # a landscape preset) instead of what's actually configured
                # for this tool. ai_steps.generation_defaults is the single
                # fixed {quality, image_size} for every request; merged in
                # (not overwritten by) params so "prompt" from build_instruction
                # above always wins if a key ever collided.
                if job.feature_type == "model_shoot_generate_model":
                    params = {**tool.ai_steps.get("generation_defaults", {}), **params}
            else:
                # A copy, never the job's own live input_params dict -- the
                # listing_photoshoot branch just below mutates `params` with
                # image_size/quality, and job.input_params must stay exactly
                # what the user submitted (it's still read for display/audit
                # and by _resolve_credit_cost at job-creation time).
                params = dict(job.input_params)

            # listing_photoshoot has no build_instruction step at all (its
            # per-job prompt is already baked in at job-creation time via
            # plan_listing_shots' per_job_overrides) -- handled here, outside
            # the build_instruction branch above, for exactly that reason.
            # Same reasoning as creative_photoshoot just above: its real fal
            # model has no aspect_ratio/resolution fields of its own, so
            # resolve_generation_params turns the user's picks into an
            # explicit {width, height} image_size (from ai_steps.size_map)
            # and the real fal quality value. Credit cost was already
            # resolved off the same function back at job-creation time in
            # create_generation_batch -- unrelated to this step.
            if job.feature_type == "listing_photoshoot":
                resolved = listing_planner.resolve_generation_params(job, tool)
                params["image_size"] = resolved["image_size"]
                params["quality"] = resolved["quality"]

            # model_shoot also has no build_instruction step (its per-job
            # prompt is baked in at job-creation time via plan_model_shoot's
            # per_job_overrides). Its real fal model (seedream v4.5/edit) has
            # no quality field, only image_size -- resolve_generation_params
            # turns aspect_ratio+resolution into an explicit {width, height}
            # from ai_steps.size_map, replacing the stale aspect-ratio-only
            # preset table _translate_fal_params used to fall back to (which
            # silently ignored resolution entirely). Credit cost was already
            # resolved off the same function back at job-creation time in
            # create_generation_batch -- unrelated to this step.
            if job.feature_type == "model_shoot":
                resolved = model_shoot.resolve_generation_params(job, tool)
                params["image_size"] = resolved["image_size"]

            params = _translate_fal_params(params, job.feature_type)

            # Same reasoning -- submit_to_fal() is also a blocking httpx call.
            fal_request_id = await asyncio.to_thread(submit_to_fal, tool.fal_model_id, params, webhook_url)
            job.fal_request_id = fal_request_id
            job.status = "processing"
            db.commit()
            logger.info("Submitted job %s to fal.ai (request_id=%s)", job.id, fal_request_id)
            # NOTE: the fal slot stays reserved -- it is only released once
            # this job reaches a real terminal state (in handle_fal_webhook
            # on success/failure, or below in sweep_stale_generation_jobs
            # on timeout). It is NOT released here on a successful submit.
        except Exception as e:
            fail_cleanly(f"fal submit failed: {str(e).replace(settings.fal_key, '[REDACTED]')}")
            release_fal_slot(job.team_id)
            logger.info("fal submit failed for job %s, refunded and released lock+slot", job.id)
    finally:
        db.close()


def _fail_stale_job(db, job_id, message: str) -> bool:
    """
    Shared cleanup for a single stuck job, used by both the frequent
    per-tool timeout check and the 10-minute catch-all sweep below. Re-locks
    and re-reads the row under with_for_update() + populate_existing() (NOT
    reusing the caller's unlocked copy) so a job a concurrent webhook already
    resolved a moment ago is correctly left alone instead of being
    double-refunded -- see test_sweep_populate_existing_prevents_stale_identity_map_read
    for why populate_existing() specifically is required here.

    Returns True if the job was actually failed, False if it was skipped
    (already resolved, or gone).
    """
    locked_job = db.query(GenerationJob).filter(
        GenerationJob.id == job_id
    ).populate_existing().with_for_update().first()

    if not locked_job or locked_job.status not in ("queued", "processing"):
        return False

    # A fal slot is only ever reserved once a job reaches "processing"
    # (submit_generation_to_fal). A job still "queued" here (e.g. the worker
    # never picked it up, or bailed early on an unknown tool) never held one
    # -- releasing it anyway would decrement a counter nothing incremented,
    # corrupting the concurrency cap.
    had_fal_slot = locked_job.status == "processing"

    locked_job.status = "failed"
    locked_job.error_message = message
    refund_credits(
        db, locked_job.team_id, locked_job.credits_charged,
        locked_job.credits_from_subscription, locked_job.credits_from_topup,
    )
    locked_job.completed_at = datetime.now(timezone.utc)
    db.commit()
    # The generation lock is ONE key per USER, not per job. Releasing it
    # unconditionally here could free the lock of a NEWER generation the user
    # started after this stuck job's own lock TTL had already lapsed (the lock
    # expires on its own -- tool timeout + 30s -- long before the 10-minute
    # sweep gets here): the sweep would then silently let them start a second
    # concurrent generation, defeating the one-at-a-time rule. Only release
    # once this was the user's last active job. (The job just failed above is
    # already committed, so it no longer counts as active.)
    if not generation_svc._user_has_active_generation_job(db, locked_job.user_id):
        release_generation_lock(locked_job.user_id)
    if had_fal_slot:
        release_fal_slot(locked_job.team_id)
    return True


async def check_generation_timeouts(ctx):
    """
    Two jobs in one pass over every "processing" job, both driven by fal's own
    real queue status:

    1. DELIVERY: as soon as fal reports COMPLETED, the result is fetched and
       applied immediately -- regardless of how much of the tool's budget is
       left. This is what actually resolves jobs whenever the fal webhook
       never arrives, which is ALWAYS the case in local dev: fal's cloud
       cannot reach a PUBLIC_BACKEND_URL of 127.0.0.1, so the webhook is
       never delivered and this poll is the only path that finishes a job.
       This check used to sit behind the budget gate below, which meant a job
       fal had finished in 40s still sat "processing" until its full budget
       elapsed (60s for recolor -- looked fine; 180s for creative_photoshoot
       -- looked broken). In production it's also a genuine safety net for a
       webhook that's lost or arrives late.

    2. TIMEOUT: only once a job is past its own tool's generation_timeout_seconds
       AND fal has not reported it COMPLETED is it failed + refunded. Checking
       fal first matters: our budget elapsing doesn't mean fal's work did too,
       and blindly failing+refunding a job fal already delivered is a real
       money-loss bug.

    Only targets "processing" jobs -- a job still "queued" hasn't been
    submitted to fal yet, so there's nothing to poll and no tool-specific
    budget to measure it against; the 10-minute sweep's flat cutoff still
    catches a job stuck in that earlier state.
    """
    db = SessionLocal()
    now = datetime.now(timezone.utc)

    processing = db.query(GenerationJob).filter(
        GenerationJob.status == "processing",
    ).all()

    tools_by_feature_type = {}

    for job in processing:
        try:
            if job.feature_type not in tools_by_feature_type:
                tools_by_feature_type[job.feature_type] = get_tool_definition(db, job.feature_type)

            tool = tools_by_feature_type[job.feature_type]
            if not tool:
                continue  # unknown tool -- the 10-minute sweep will still catch it

            past_budget = job.created_at <= now - timedelta(seconds=tool.generation_timeout_seconds)

            # check_fal_status / fetch_fal_result / handle_fal_webhook (which
            # itself downloads + uploads the output) are all blocking,
            # synchronous HTTP calls -- run them in a thread so a slow fal
            # status check or storage round-trip doesn't freeze this whole
            # worker's event loop for every other job in flight.
            fal_status = None
            if job.fal_request_id:
                try:
                    status_payload = await asyncio.to_thread(
                        check_fal_status, tool.fal_model_id, job.fal_request_id,
                    )
                    fal_status = status_payload.get("status")
                except Exception:
                    logger.exception(
                        "fal status check failed for job %s (fal_request_id=%s) -- "
                        "falling back to the local timeout",
                        job.id, job.fal_request_id,
                    )

            if fal_status == "COMPLETED":
                # fal has finished (successfully or not) -- deliver the real
                # result now, through the exact same path a real webhook
                # delivery would use. Deliberately NOT gated on the budget:
                # this is what finishes jobs at all whenever the webhook never
                # arrives (always, in local dev).
                result_payload = await asyncio.to_thread(
                    fetch_fal_result, tool.fal_model_id, job.fal_request_id,
                )
                logger.info(
                    "job %s: fal reports COMPLETED -- delivering the real result "
                    "(past_budget=%s)",
                    job.id, past_budget,
                )
                await asyncio.to_thread(generation_svc.handle_fal_webhook, db, job.id, result_payload)
                continue

            if not past_budget:
                continue  # still within its own tool's budget -- let it keep running

            if fal_status in ("IN_QUEUE", "IN_PROGRESS"):
                # Measuring this specifically: how often our timeout fires
                # while fal itself confirms the work is still genuinely
                # ongoing (as opposed to us just never hearing back).
                logger.warning(
                    "job %s timed out while fal confirmed still in-progress "
                    "(fal_status=%s) -- failing per the timeout as designed",
                    job.id, fal_status,
                )

            if _fail_stale_job(db, job.id, TIMEOUT_MESSAGE):
                logger.info(
                    "Timed out job %s (feature_type=%s, budget=%ds)",
                    job.id, job.feature_type, tool.generation_timeout_seconds,
                )
        except Exception:
            db.rollback()
            logger.exception("failed to check timeout for job %s — will retry next run", job.id)

    db.close()


async def sweep_stale_generation_jobs(ctx):
    """
    Catches jobs stuck in queued/processing for too long — a crashed worker,
    a lost webhook, or fal.ai never responding. Marks them failed and refunds.
    A catch-all safety net behind check_generation_timeouts above: it uses one
    fixed 10-minute cutoff (vs. each tool's own, much shorter budget), so it
    also still catches jobs stuck "queued" (never even submitted) which the
    per-tool check above doesn't look at.
    """
    db = SessionLocal()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)

    stale = db.query(GenerationJob).filter(
        GenerationJob.status.in_(["queued", "processing"]),
        GenerationJob.created_at <= cutoff,
    ).all()

    for job in stale:
        try:
            if _fail_stale_job(db, job.id, SWEEP_TIMEOUT_MESSAGE):
                logger.info("Swept stale job %s (was stuck since %s)", job.id, job.created_at)
        except Exception:
            db.rollback()
            logger.exception("failed to sweep job %s — will retry next run", job.id)

    db.close()


async def reconcile_fal_slots(ctx):
    """
    Self-healing for the fal in-flight Redis counters (try_reserve_fal_slot /
    release_fal_slot in generation_lock.py): they're plain INCR/DECR
    counters with no TTL, unlike the generation lock. Any time the worker is
    killed between reserving and releasing a slot -- a Ctrl+C, a crash, a
    redeploy -- the leaked count sticks around forever with nothing to ever
    correct it. Confirmed live more than once this session (a per-team
    counter sitting at 1 with zero jobs actually processing).

    Recomputes the TRUE per-team in-flight count directly from Postgres (the
    real source of truth: jobs genuinely still "processing" right now) and
    resets Redis to match -- both the per-team keys and the global one.
    """
    db = SessionLocal()
    try:
        true_counts = {
            str(team_id): count
            for team_id, count in (
                db.query(GenerationJob.team_id, func.count(GenerationJob.id))
                .filter(GenerationJob.status == "processing")
                .group_by(GenerationJob.team_id)
                .all()
            )
        }

        # KEYS is O(N) over the keyspace -- fine for this app's scale run
        # every 10 minutes, but would need SCAN instead at real production
        # volume.
        remaining_team_keys = {
            key.split(":", 2)[2] for key in redis_client.keys("fal:inflight_count:*")
        }

        for team_id_str, true_count in true_counts.items():
            key = f"fal:inflight_count:{team_id_str}"
            current = int(redis_client.get(key) or 0)
            if current != true_count:
                logger.warning(
                    "reconcile_fal_slots: team %s Redis count was %d, Postgres truth "
                    "is %d -- correcting",
                    team_id_str, current, true_count,
                )
                redis_client.set(key, true_count)
            remaining_team_keys.discard(team_id_str)

        # Any team key left here has a nonzero-or-stale Redis count but zero
        # real "processing" jobs for it right now -- a pure leak, zero it.
        for team_id_str in remaining_team_keys:
            key = f"fal:inflight_count:{team_id_str}"
            current = int(redis_client.get(key) or 0)
            if current != 0:
                logger.warning(
                    "reconcile_fal_slots: team %s Redis count was %d with zero jobs "
                    "actually processing -- leaked, resetting to 0",
                    team_id_str, current,
                )
                redis_client.set(key, 0)

        corrected_global = sum(true_counts.values())
        current_global = int(redis_client.get(INFLIGHT_KEY_GLOBAL) or 0)
        if current_global != corrected_global:
            logger.warning(
                "reconcile_fal_slots: global count was %d, Postgres truth is %d -- correcting",
                current_global, corrected_global,
            )
            redis_client.set(INFLIGHT_KEY_GLOBAL, corrected_global)
    finally:
        db.close()


async def startup(ctx):
    logger.info("arq worker started")

    # Frozen-worker watchdog (see app/core/watchdog.py for the full story: a
    # single hung blocking call froze this worker for 40+ minutes, alive but
    # doing nothing, with no alert and no restart).
    if settings.worker_watchdog_seconds > 0:
        watchdog = WorkerWatchdog(settings.worker_watchdog_seconds)
        watchdog.attach_to_loop(asyncio.get_running_loop())
        watchdog.start()
        ctx["watchdog"] = watchdog
        logger.info("worker watchdog armed (threshold %ss)", settings.worker_watchdog_seconds)
    else:
        logger.warning("worker watchdog DISABLED (WORKER_WATCHDOG_SECONDS=0)")


async def shutdown(ctx):
    watchdog = ctx.get("watchdog")
    if watchdog:
        watchdog.stop()
    logger.info("arq worker shutting down")


class WorkerSettings:
    functions = [
        refill_due_subscriptions,
        send_yearly_renewal_notices,
        cleanup_stale_pending_subscriptions,
        expire_pending_switches,
        reconcile_subscriptions,
        submit_generation_to_fal,
        sweep_stale_generation_jobs,
        check_generation_timeouts,
        reconcile_fal_slots,
        purge_expired_deleted_teams,
    ]
    cron_jobs = [
        cron(refill_due_subscriptions, hour=3, minute=0),
        cron(send_yearly_renewal_notices, hour=3, minute=30),
        cron(cleanup_stale_pending_subscriptions, minute={7, 22, 37, 52}),
        cron(expire_pending_switches, minute={3, 18, 33, 48}),
        cron(reconcile_subscriptions, minute={11}),
        cron(sweep_stale_generation_jobs, minute={0, 15, 30, 45}),
        cron(check_generation_timeouts, second={0, 15, 30, 45}),
        cron(reconcile_fal_slots, minute={0, 10, 20, 30, 40, 50}),
        cron(purge_expired_deleted_teams, hour=4, minute=30),
    ]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    # arq refreshes its health sentinel (Redis key "arq:queue:health-check")
    # from the worker's own main loop, so a frozen loop stops refreshing it and
    # the key expires after interval+1 seconds. The DEFAULT interval is 3600s,
    # which made the sentinel outlive a freeze by up to an hour and useless as
    # a signal. 60s means a loop that has been blocked for ~1 minute is
    # visible to `arq --check app.worker.WorkerSettings` (a Docker/K8s health
    # probe) and to GET /health/worker -- long enough not to flap on a single
    # legitimate 30s blocking HTTP call.
    health_check_interval = 60


WorkerSettings.on_startup = startup
WorkerSettings.on_shutdown = shutdown