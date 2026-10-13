import logging
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from sqlalchemy.orm import Session

from app.models.generation_job import GenerationJob
from app.services.tool_definitions import get_tool_definition
from app.services.generation_lock import (
    acquire_generation_lock, release_generation_lock, release_fal_slot, generation_lock_age_seconds,
    extend_generation_lock_ttl,
)
from app.services.credits import spend_credits_up_to, refund_credits
from app.core.storage import upload_to_storage, download_from_url, detect_image_type
from app.tools.creative_photoshoot import resolve_generation_params
from app.tools import listing_planner, model_shoot

logger = logging.getLogger(__name__)

DOWNLOAD_FAILED_MESSAGE = "Could not retrieve the generated image. Please try again."
UPLOAD_FAILED_MESSAGE = "Could not save the generated image. Please try again."

# The exact error_message a job gets when worker.py fails it for running past
# its own timeout budget (worker.check_generation_timeouts, the per-tool
# check) or the flat 10-minute catch-all (worker.sweep_stale_generation_jobs).
# Defined here (not in worker.py) so handle_fal_webhook below can recognize
# "this job was failed+refunded specifically because of a timeout, not a real
# fal.ai failure" without worker.py and generation.py importing each other.
TIMEOUT_MESSAGE = "Generation took too long. Please try again."
SWEEP_TIMEOUT_MESSAGE = "Generation timed out — no response received"
TIMEOUT_FAILURE_MESSAGES = {TIMEOUT_MESSAGE, SWEEP_TIMEOUT_MESSAGE}


def _job_history_title(job: GenerationJob) -> str:
    # title is nullable and nothing sets it yet (see list_team_generations) --
    # this fallback keeps the history list usable in the meantime. Whether
    # title ends up auto-generated at job-creation time or user-editable
    # after the fact is a separate, not-yet-decided piece of work.
    if job.title:
        return job.title
    return f"{job.feature_type} — {job.created_at.isoformat()}"


PERIOD_PRESETS = {
    "last_7_days": timedelta(days=7),
    "last_30_days": timedelta(days=30),
}


def _resolve_period(period: str | None, from_dt, to_dt):
    """Shared by list_team_generations and usage.get_team_usage.
    Explicit from_/to_ wins if given; otherwise a named preset resolves to a
    rolling window ending now; "all_time" (or nothing at all) means no filter."""
    if from_dt or to_dt:
        return from_dt, to_dt
    if period and period in PERIOD_PRESETS:
        return datetime.now(timezone.utc) - PERIOD_PRESETS[period], None
    return None, None


# What each view of the history hides by default. The two views differ on
# purpose:
#   * "recent" (Home's Recent Work strip) shows finished, displayable results
#     only, so it also hides model_shoot_generate_model -- an internal step of
#     the Model Shoot flow, not a standalone result to feature on Home.
#   * "library" is the full paid-history view. model_shoot_generate_model is
#     something the user paid credits for and a legitimate deliverable, so it
#     belongs there. Only enhance_prompt is hidden, because it produces text
#     and never an outputUrl -- there is nothing to ever show.
# Library also hides `failed` jobs by default (see list_team_generations) --
# they were refunded and have no image -- while queued/processing stay visible.
EXCLUDED_BY_VIEW = {
    "recent": ("enhance_prompt", "model_shoot_generate_model"),
    "library": ("enhance_prompt",),
}


def list_team_generations(
    db: Session, team_id, limit: int = 10, offset: int = 0,
    feature_type: str | None = None, user_id=None,
    period: str | None = None, from_dt=None, to_dt=None,
    status: str | None = None, view: str = "recent",
) -> list[dict]:
    # One filter() call for the always-on conditions. An explicitly requested
    # feature_type wins over the view's default exclusion (the exclusion is a
    # default, not a hard block), so the two are mutually exclusive rather
    # than stacked.
    conditions = [GenerationJob.team_id == team_id]
    if feature_type:
        conditions.append(GenerationJob.feature_type == feature_type)
    else:
        conditions.append(GenerationJob.feature_type.notin_(EXCLUDED_BY_VIEW[view]))
    query = db.query(GenerationJob).filter(*conditions)

    if status:
        query = query.filter(GenerationJob.status == status)
    elif view == "library":
        # A failed generation is refunded (see fail_and_release/refund_credits),
        # so there is nothing the user paid for to keep in their paid-history
        # view -- and a failed card has no image to show. Hidden by default;
        # an explicit status=failed still returns them. Filtered here, not in
        # the client, so pages stay full and offsets stay correct.
        query = query.filter(GenerationJob.status != "failed")
    if user_id:
        query = query.filter(GenerationJob.user_id == user_id)

    resolved_from, resolved_to = _resolve_period(period, from_dt, to_dt)
    if resolved_from:
        query = query.filter(GenerationJob.created_at >= resolved_from)
    if resolved_to:
        query = query.filter(GenerationJob.created_at <= resolved_to)

    jobs = (
        query.order_by(GenerationJob.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [
        {
            "jobId": str(job.id),
            "batchId": str(job.batch_id) if job.batch_id else None,
            "title": _job_history_title(job),
            "featureType": job.feature_type,
            "status": job.status,
            "createdAt": job.created_at.isoformat() if job.created_at else None,
            "outputUrl": job.output_url,
            "deeplink": {"type": "page", "route": "library", "params": {"job_id": str(job.id)}},
        }
        for job in jobs
    ]


def validate_input_params(param_schema: list, input_params: dict) -> None:
    for field in param_schema:
        name = field["name"]

        # output_count is a batch-size control, not a per-job input -- the
        # route resolves it from its own dedicated Form field (falling back
        # to tool.default_output_count) and it's never part of the
        # input_params dict passed here. A param_schema row for it (e.g.
        # listing_photoshoot's, with its own min/max/default) exists only so
        # the frontend can render a bounded number input -- it must not be
        # checked as a "required" per-job field, or every request would fail
        # with "Missing required field" regardless of what the caller sends.
        if name == "output_count":
            continue

        value = input_params.get(name)

        if field.get("required") and not value:
            raise ValueError(f"Missing required field: {field['label']}")

        if field["type"] == "select" and value:
            options = field.get("options", [])
            valid_values = [
                opt["value"] if isinstance(opt, dict) else opt
                for opt in options
            ]
            if value not in valid_values:
                raise ValueError(f"Invalid value for {field['label']}: {value}")


def _resolve_credit_cost(tool, feature_type: str, input_params: dict, output_count: int) -> int:
    # creative_photoshoot's cost isn't a flat per-option price on param_schema
    # (its select options are plain strings, not {value, credit_cost} dicts)
    # -- it's quality_multiplier * resolution_multiplier off ai_steps, and the
    # exact width/height combo must be in ai_steps.size_map or this raises.
    # No GenerationJob row exists yet at this point, so resolve_generation_params
    # (which only reads job.input_params/job.id) is given a minimal stand-in.
    if feature_type == "creative_photoshoot":
        stub_job = SimpleNamespace(id="pending", input_params=input_params)
        return resolve_generation_params(stub_job, tool)["credit_cost"]

    if feature_type == "listing_photoshoot":
        # listing_planner.resolve_generation_params prices the WHOLE batch
        # (quality_multiplier * resolution_multiplier * output_count, one
        # priced unit per shot) -- output_count isn't part of input_params
        # (it's create_generation_batch's own already-resolved argument), so
        # it's added onto the stand-in job's input_params here. Divide the
        # total back down to a per-job cost: this function's only contract
        # with its caller is "the price of ONE output", so create_generation_
        # batch's own `per_output_cost * output_count` reconstructs the exact
        # same total (exact division -- output_count is a multiplied-in
        # factor here, never a remainder).
        stub_job = SimpleNamespace(
            id="pending", input_params={**input_params, "output_count": output_count},
        )
        total_cost = listing_planner.resolve_generation_params(stub_job, tool)["credit_cost"]
        return total_cost // output_count

    if feature_type == "model_shoot":
        # Same shape as listing_photoshoot just above: model_shoot.
        # resolve_generation_params prices the WHOLE batch (resolution_credit
        # * output_count, one priced unit per pose) -- output_count isn't
        # part of input_params, so it's added onto the stand-in job here, and
        # the total is divided back down to a per-job cost.
        stub_job = SimpleNamespace(
            id="pending", input_params={**input_params, "output_count": output_count},
        )
        total_cost = model_shoot.resolve_generation_params(stub_job, tool)["credit_cost"]
        return total_cost // output_count

    for field in tool.param_schema:
        if field["type"] != "select":
            continue
        options = field.get("options", [])
        if not options or not isinstance(options[0], dict):
            continue
        if "credit_cost" not in options[0]:
            continue

        selected_value = input_params.get(field["name"])
        for opt in options:
            if opt["value"] == selected_value:
                return opt["credit_cost"]

    return tool.credit_cost_per_output


# How old an unclaimed lock has to be before we'll consider it orphaned
# rather than a genuinely in-flight request. Real job creation (tool lookup,
# validation, credit spend, insert, commit) finishes in well under a second
# even under DB latency -- 30s gives a very wide safety margin before ever
# treating a real in-flight request's lock as stale.
STALE_LOCK_GRACE_SECONDS = 30

ALREADY_IN_PROGRESS_MESSAGE = "You already have a generation in progress. Please wait for it to finish."

# Shared with app/routes/generation.py (imported from here, not redefined) --
# a request-shape limit independent of credit cost, see that module for the
# full rationale. Also enforced here, not just at the route, because
# create_generation_batch's own output_count can come from
# tool.default_output_count (a DB value the route never validates against
# this cap) rather than only from the caller -- and because output_count
# is later used as a divisor below, a misconfigured default_output_count of
# 0 (or negative) must be rejected here as a clean ValueError instead of
# crashing the credit split with a ZeroDivisionError.
MAX_OUTPUT_COUNT = 20


def _user_has_active_generation_job(db: Session, user_id) -> bool:
    return db.query(GenerationJob).filter(
        GenerationJob.user_id == user_id,
        GenerationJob.status.in_(("queued", "processing")),
    ).first() is not None


def acquire_or_heal_generation_lock(db: Session, user_id) -> bool:
    """
    True if the lock is now held for this user -- either freshly acquired, or
    healed from a stale orphan and re-acquired. False if a real generation is
    genuinely already in progress and the caller should reject the request.

    Callers that need to reject BEFORE doing any expensive/billable work
    (e.g. listing_photoshoot's shot-planning vision call in the /generate
    route) should call this directly instead of going through
    create_generation_batch's own internal check, which only runs after such
    work has already happened -- see lock_already_held below.
    """
    if acquire_generation_lock(user_id):
        return True

    # Self-heal a lock orphaned by a crash/restart between
    # acquire_generation_lock() and the job actually being created (or the
    # except-block release running) -- a hard process kill mid-request skips
    # that cleanup entirely, otherwise stranding the lock for its full
    # 5-minute TTL. Only clears it when BOTH the lock is old enough to rule
    # out a genuinely in-flight request AND there's no real job in the DB
    # backing it -- a lock with an actual queued/processing job behind it is
    # never touched here, it stays blocked exactly as intended until that
    # job resolves.
    age = generation_lock_age_seconds(user_id)
    if age is not None and age > STALE_LOCK_GRACE_SECONDS and not _user_has_active_generation_job(db, user_id):
        logger.warning(
            "user %s: releasing an orphaned generation lock (age=%ds, no active job found)",
            user_id, age,
        )
        release_generation_lock(user_id)
        return acquire_generation_lock(user_id)

    return False


def create_generation_batch(
    db: Session, team_id, user_id, feature_type: str, input_params: dict,
    output_count: int = 1, per_job_overrides: list[dict] | None = None,
    lock_already_held: bool = False,
) -> list[GenerationJob]:
    if not lock_already_held and not acquire_or_heal_generation_lock(db, user_id):
        raise ValueError(ALREADY_IN_PROGRESS_MESSAGE)

    try:
        tool = get_tool_definition(db, feature_type)
        if not tool:
            raise ValueError("Unknown tool")
        # Same condition as GET /tools/{feature_type}/schema's own gate
        # (stage != 1 or not is_active) -- these two must never disagree, or
        # a tool mid-launch (is_active flipped true while still at stage 0
        # for internal testing) would be invisible to the schema endpoint
        # but generatable anyway.
        if tool.stage != 1 or not tool.is_active:
            raise ValueError("This tool is not currently available")

        # The lock (freshly acquired above, or already held by a caller like
        # listing_photoshoot/model_shoot's own pre-planning acquire) starts
        # with the generic LOCK_TTL_SECONDS safety net -- now that the real
        # tool is known, re-derive its TTL from that tool's own
        # generation_timeout_seconds so a slow-but-legitimate job never
        # outlives its own lock (see generation_lock.py).
        extend_generation_lock_ttl(user_id, tool.generation_timeout_seconds)

        validate_input_params(tool.param_schema, input_params)

        if per_job_overrides:
            output_count = len(per_job_overrides)
        else:
            output_count = output_count if output_count and output_count > 0 else tool.default_output_count

        if not output_count or output_count <= 0 or output_count > MAX_OUTPUT_COUNT:
            raise ValueError(
                f"output_count must be between 1 and {MAX_OUTPUT_COUNT} "
                f"(resolved to {output_count!r} for feature_type={feature_type!r})"
            )

        per_output_cost = _resolve_credit_cost(tool, feature_type, input_params, output_count)

        # spend_credits_up_to (not the plain all-or-nothing spend_credits)
        # because this is a shared TEAM balance: another request against the
        # same team can spend concurrently between whenever the caller last
        # saw the balance and this call, so a multi-output batch is sized
        # down to whatever the team can actually afford right now rather
        # than rejecting the whole request outright. A single-output request
        # (output_count=1) behaves identically to spend_credits either way --
        # either that one unit is affordable or this raises the same
        # "Insufficient credits" error.
        _, granted_count, from_subscription, from_topup = spend_credits_up_to(
            db, team_id, per_output_cost, output_count, commit=False,
        )

        if granted_count < output_count:
            logger.info(
                "team %s: requested %d x %d-credit output(s) for %s but could only "
                "afford %d right now -- granting a partial batch instead of "
                "rejecting the whole request",
                team_id, output_count, per_output_cost, feature_type, granted_count,
            )
            output_count = granted_count
            if per_job_overrides:
                per_job_overrides = per_job_overrides[:output_count]

        per_job_sub = from_subscription // output_count
        per_job_top = from_topup // output_count
        remainder_sub = from_subscription - (per_job_sub * output_count)
        remainder_top = from_topup - (per_job_top * output_count)

        batch_id = uuid.uuid4()
        jobs = []

        for i in range(output_count):
            job = GenerationJob(
                team_id=team_id,
                user_id=user_id,
                feature_type=feature_type,
                input_params={**input_params, **per_job_overrides[i]} if per_job_overrides else input_params,
                batch_id=batch_id,
                credits_charged=per_output_cost,
                credits_from_subscription=per_job_sub + (remainder_sub if i == 0 else 0),
                credits_from_topup=per_job_top + (remainder_top if i == 0 else 0),
                status="queued",
            )
            db.add(job)
            jobs.append(job)

        db.commit()
        return jobs

    except Exception:
        db.rollback()
        release_generation_lock(user_id)
        raise


def fail_and_release(db: Session, job: GenerationJob, error_message: str) -> None:
    """
    Only used for the enqueue-failure path (/generate -> arq), before the job
    has ever reached the worker -- so no fal slot was ever reserved for it
    (that only happens inside submit_generation_to_fal). Must NOT call
    release_fal_slot here, or it decrements a counter nothing incremented.
    """
    job.status = "failed"
    job.error_message = error_message
    refund_credits(db, job.team_id, job.credits_charged, job.credits_from_subscription, job.credits_from_topup)
    job.completed_at = datetime.now(timezone.utc)

    remaining = db.query(GenerationJob).filter(
        GenerationJob.batch_id == job.batch_id,
        GenerationJob.status.in_(("queued", "processing")),
    ).count()

    if remaining <= 1:
        release_generation_lock(job.user_id)

    db.commit()


def _deliver_late_success_after_timeout(db: Session, job: GenerationJob, payload: dict) -> None:
    """
    A job we already failed+refunded for running past its timeout budget
    (see TIMEOUT_FAILURE_MESSAGES) has now had the REAL fal.ai webhook arrive
    reporting success -- fal actually finished the work. Deliver the real
    output and mark it completed, but deliberately do NOT charge credits
    again: the refund from the timeout stands, so this output is free. This
    is the money-loss case the timeout mechanism can otherwise create (fail
    the user's job, refund them, then fal quietly succeeds anyway and the
    real result would previously just be discarded here).
    """
    images = payload.get("payload", {}).get("images", [])
    fal_url = images[0]["url"] if images else None
    if not fal_url:
        return

    try:
        file_bytes = download_from_url(fal_url)
        content_type, ext = detect_image_type(file_bytes)
        permanent_path = f"{job.team_id}/{job.feature_type}/{job.id}/output.{ext}"
        job.output_url = upload_to_storage(permanent_path, file_bytes, content_type)
    except Exception:
        logger.exception(
            "late success webhook arrived for already-timed-out-and-refunded job %s, "
            "but storing the real output failed -- job stays failed/refunded as-is",
            job.id,
        )
        return

    job.status = "completed"
    db.commit()
    logger.warning(
        "job %s: late success delivered free of charge -- fal.ai's real webhook "
        "reported success after this job was already failed+refunded by a timeout; "
        "output stored, credits NOT re-charged (the timeout refund stands)",
        job.id,
    )


def handle_fal_webhook(db: Session, job_id, payload: dict):
    job = db.query(GenerationJob).filter(
        GenerationJob.id == job_id
    ).populate_existing().with_for_update().first()
    if not job:
        return

    # The URL's job_id (?job_id=...) is our OWN identifier, not part of what
    # fal.ai signs -- verify_fal_webhook only proves this payload genuinely
    # came from fal for SOME request, not that it's for THIS job. Without
    # this check, a real (fal-signed) webhook for one job could be replayed
    # onto a completely different job_id within the signature's replay
    # window, letting an authenticated user overwrite/complete/fail another
    # team's job with their own request's content.
    incoming_request_id = payload.get("request_id")
    if job.fal_request_id and incoming_request_id and incoming_request_id != job.fal_request_id:
        logger.warning(
            "job %s: webhook request_id %r does not match this job's own "
            "fal_request_id %r -- ignoring (misdirected or replayed delivery)",
            job.id, incoming_request_id, job.fal_request_id,
        )
        return

    if job.status in ("completed", "failed"):
        if (
            job.status == "failed"
            and job.error_message in TIMEOUT_FAILURE_MESSAGES
            and payload.get("status") == "OK"
        ):
            _deliver_late_success_after_timeout(db, job, payload)
        return

    if payload.get("status") == "OK":
        if job.feature_type == "enhance_prompt":
            output_text = payload.get("payload", {}).get("output")
            if output_text:
                job.output_text = output_text
                job.status = "completed"
            else:
                job.status = "failed"
                job.error_message = "fal reported success but returned no text"
                refund_credits(db, job.team_id, job.credits_charged, job.credits_from_subscription, job.credits_from_topup)
            job.completed_at = datetime.now(timezone.utc)
            job.duration_seconds = int((job.completed_at - job.created_at).total_seconds())

            remaining = db.query(GenerationJob).filter(
                GenerationJob.batch_id == job.batch_id,
                GenerationJob.status.in_(("queued", "processing")),
            ).count()

            if remaining <= 1:
                release_generation_lock(job.user_id)
                release_fal_slot(job.team_id)

            db.commit()
            return

        images = payload.get("payload", {}).get("images", [])
        fal_url = images[0]["url"] if images else None
        
        if fal_url:
            try:
                file_bytes = download_from_url(fal_url)
            except Exception:
                logger.exception("Failed to download fal.ai output for job %s", job.id)
                job.status = "failed"
                job.error_message = DOWNLOAD_FAILED_MESSAGE
                refund_credits(db, job.team_id, job.credits_charged, job.credits_from_subscription, job.credits_from_topup)
            else:
                try:
                    content_type, ext = detect_image_type(file_bytes)
                    permanent_path = f"{job.team_id}/{job.feature_type}/{job.id}/output.{ext}"
                    job.output_url = upload_to_storage(permanent_path, file_bytes, content_type)
                    job.status = "completed"
                except Exception:
                    logger.exception("Failed to upload output to storage for job %s", job.id)
                    job.status = "failed"
                    job.error_message = UPLOAD_FAILED_MESSAGE
                    refund_credits(db, job.team_id, job.credits_charged, job.credits_from_subscription, job.credits_from_topup)
        else:
            job.status = "failed"
            job.error_message = "fal reported success but returned no image"
            refund_credits(db, job.team_id, job.credits_charged, job.credits_from_subscription, job.credits_from_topup)
    else:
        job.status = "failed"
        job.error_message = payload.get("error", "Generation failed")
        refund_credits(db, job.team_id, job.credits_charged, job.credits_from_subscription, job.credits_from_topup)

    job.completed_at = datetime.now(timezone.utc)
    job.duration_seconds = int((job.completed_at - job.created_at).total_seconds())

    remaining = db.query(GenerationJob).filter(
        GenerationJob.batch_id == job.batch_id,
        GenerationJob.status.in_(("queued", "processing")),
    ).count()

    if remaining <= 1:
        release_generation_lock(job.user_id)
        release_fal_slot(job.team_id)

    db.commit()