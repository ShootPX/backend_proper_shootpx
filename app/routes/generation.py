import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, File, Form, UploadFile
from sqlalchemy.orm import Session
from pydantic import BaseModel
from uuid import UUID
from typing import List

from app.core.database import get_db
from app.core.fal_webhook import FalWebhookVerificationError, verify_fal_webhook
from app.core.limiter import limiter
from app.deps import get_current_user
from app.models.user import User
from app.services.teams import is_team_member
from app.services.generation import (
    create_generation_batch, fail_and_release, handle_fal_webhook, validate_input_params,
    acquire_or_heal_generation_lock, ALREADY_IN_PROGRESS_MESSAGE, MAX_OUTPUT_COUNT,
)
from app.services.generation_lock import release_generation_lock
from app.models.generation_job import GenerationJob
from app.models.model_preset import ModelPreset
from app.core.arq_pool import get_arq_pool
from app.core.fal_client import upload_image_to_fal
from app.tools.listing_planner import plan_listing_shots, MAX_OUTPUT_COUNT as LISTING_MAX_OUTPUT_COUNT
from app.tools.model_shoot import plan_model_shoot
from app.services.tool_definitions import get_tool_definition
from app.services.job_errors import public_job_error

logger = logging.getLogger(__name__)

# 10 MB/image is generous for a product photo while still bounding memory use
# (await image.read() loads the whole file) and fal upload cost for obviously
# bad input.
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp"}

# The shoot-analyst's own "reason" text (plan_model_shoot's LLM-authored
# classification of why it blocked a shoot) must never reach the frontend
# directly -- it's free-text model output describing exactly what it judged
# unsafe/inappropriate about the user's own uploaded images, which is not
# something to echo back verbatim in an API response. Logged server-side
# (see the blocked branch below) for internal visibility instead.
MODEL_SHOOT_BLOCKED_MESSAGE = "This combination of model and garment cannot be generated."

router = APIRouter(tags=["generation"])


async def _read_and_upload_images(images: List[UploadFile]) -> List[str]:
    """
    Shared by every image-bearing field on /generate (the generic `images`
    list, and model_shoot's model_image/top_images/bottom_images/extra_images_*/
    reference_images) -- validates content-type and size, reads every file, THEN uploads, so a
    request that's about to be rejected (bad type, oversized file) never
    costs any fal upload quota, no matter which field it came through.
    """
    read_images = []
    for image in images:
        if image.content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image type: {image.content_type}. "
                       f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_CONTENT_TYPES))}",
            )
        file_bytes = await image.read()
        if len(file_bytes) > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"{image.filename} is over the {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)}MB limit per image",
            )
        read_images.append((file_bytes, image.filename, image.content_type))

    return [
        upload_image_to_fal(file_bytes, filename, content_type)
        for file_bytes, filename, content_type in read_images
    ]


LIBRARY_SOURCE_UNAVAILABLE_MESSAGE = "One or more selected images are not available"


def _resolve_source_job_urls(db: Session, team_id, job_ids: list[UUID]) -> dict[UUID, str]:
    """
    job id -> that job's output_url, for reusing Library outputs as inputs.

    ONE query for every id across every slot (each extra query is a full
    network round trip). A job only counts if it belongs to THIS team, is
    completed, and actually has an output -- anything else (someone else's
    job, a failed/processing one, an unknown id) fails the whole request with
    the same generic message, so a caller can't probe which ids exist in other
    teams. Runs BEFORE any fal upload so a bad selection never costs quota.
    """
    unique = set(job_ids)
    if not unique:
        return {}

    rows = (
        db.query(GenerationJob.id, GenerationJob.output_url)
        .filter(
            GenerationJob.id.in_(unique),
            GenerationJob.team_id == team_id,
            GenerationJob.status == "completed",
            GenerationJob.output_url.isnot(None),
        )
        .all()
    )
    found = {job_id: url for job_id, url in rows}
    if len(found) != len(unique):
        raise HTTPException(status_code=400, detail=LIBRARY_SOURCE_UNAVAILABLE_MESSAGE)
    return found


class GenerateRequest(BaseModel):
    team_id: UUID
    feature_type: str
    input_params: dict = {}
    output_count: int = 1


@router.post("/generate")
@limiter.limit("20/minute")
async def generate(
    request: Request,
    team_id: UUID = Form(...),
    feature_type: str = Form(...),
    color: str = Form(None),
    quality: str = Form(None),
    size: str = Form(None),
    target_area: str = Form(None),
    prompt: str = Form(None),
    source_feature_type: str = Form(None),
    idea: str = Form(None),
    # model_shoot_generate_model's own fields -- a text-to-image prompt built
    # entirely from structured attributes, no image upload at all.
    gender: str = Form(None),
    age_bracket: str = Form(None),
    ethnicity: str = Form(None),
    skin_tone: str = Form(None),
    body_type: str = Form(None),
    notes: str = Form(None),
    # model_shoot's own field -- distinct from "size" (an aspect-ratio string
    # like recolor's/creative_photoshoot's) because model_shoot's real fal
    # model (seedream v4.5/edit) only accepts a fixed aspect ratio when
    # quality="1k"; 2k/4k use an auto-resolution mode that ignores it
    # entirely (see worker.py::_translate_fal_params).
    aspect_ratio: str = Form(None),
    # recolor's own field -- standard/high, also what _resolve_credit_cost
    # reads its credit_cost off of (see app/services/generation.py).
    resolution: str = Form(None),
    output_count: int = Form(None),
    images: List[UploadFile] = File(default=[]),
    # model_shoot's own grouped image upload -- kept entirely separate from
    # the generic `images` field above. Its planner (plan_model_shoot) labels
    # each image differently in the vision prompt it sends ("#Image1 = model
    # reference" vs "garment product (Top), preserve fidelity (1/2)" vs
    # "style/pose reference only") -- merging these into one flat list the
    # way every other tool's images work would lose the model/garment-group/
    # reference distinction (and which images belong to the SAME garment
    # group) entirely. Every other tool keeps using the single `images`
    # field above, completely unchanged.
    model_image: UploadFile | None = File(None),
    # model_shoot's garments are GROUPED, not a flat list -- multiple images
    # of the SAME physical item (angles) must be told apart from multiple
    # DIFFERENT items, or the planner's vision call has no way to know
    # whether 3 uploaded images are 3 angles of one top, or a top + bottom +
    # watch (see app/tools/model_shoot.py::plan_model_shoot). Fixed, named
    # slots (not a dynamic list) since FastAPI's typed Form/File params can't
    # declare an unbounded number of labeled image groups: Top and Bottom are
    # first-class slots, then up to 3 additional named slots for anything
    # else (watch, hat, shoes, bag, ...) -- comfortably enough headroom under
    # the tool's own 10-image total cap (1 model + up to 5 garment groups +
    # references) without needing a JSON manifest.
    top_images: List[UploadFile] = File(default=[]),
    bottom_images: List[UploadFile] = File(default=[]),
    extra_label_1: str = Form(None),
    extra_images_1: List[UploadFile] = File(default=[]),
    extra_label_2: str = Form(None),
    extra_images_2: List[UploadFile] = File(default=[]),
    extra_label_3: str = Form(None),
    extra_images_3: List[UploadFile] = File(default=[]),
    reference_images: List[UploadFile] = File(default=[]),
    # Alternative to uploading model_image -- an id from GET /model-presets.
    # Exactly one of model_image / model_preset_id must be given for
    # model_shoot; a preset's image_url is already a real hosted URL, so it
    # skips _read_and_upload_images entirely (see below).
    model_preset_id: str = Form(None),
    # Reuse EXISTING outputs from the team's Library as inputs, instead of
    # uploading files. These take generation job ids (not raw URLs) on
    # purpose: each id is resolved server-side to that job's own output_url
    # and must belong to THIS team, be completed, and have an output -- so a
    # caller can only ever reference images their own team generated, and
    # can't make fal fetch an arbitrary URL. Combine freely with uploads.
    # Generic tools (recolor, creative_photoshoot, listing_photoshoot):
    source_job_ids: List[UUID] = Form(default=[]),
    # model_shoot's grouped slots, mirroring the upload fields above one to
    # one -- the model may come from a past model_shoot_generate_model result:
    model_source_job_id: UUID | None = Form(None),
    top_source_job_ids: List[UUID] = Form(default=[]),
    bottom_source_job_ids: List[UUID] = Form(default=[]),
    extra_source_job_ids_1: List[UUID] = Form(default=[]),
    extra_source_job_ids_2: List[UUID] = Form(default=[]),
    extra_source_job_ids_3: List[UUID] = Form(default=[]),
    reference_source_job_ids: List[UUID] = Form(default=[]),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not is_team_member(db, team_id, user.id):
        raise HTTPException(status_code=403, detail="You are not a member of this team")

    tool_check = get_tool_definition(db, feature_type)
    if not tool_check:
        raise HTTPException(status_code=400, detail="Unknown tool")
    # Checked here, first, deliberately BEFORE any image upload happens
    # below (_read_and_upload_images spends real fal quota per image) --
    # same reasoning already documented for validate_input_params further
    # down: a request that's going to be rejected must not cost anything
    # first. Same condition as GET /tools/{feature_type}/schema's gate and
    # create_generation_batch's own defense-in-depth check -- all three must
    # agree, or a tool visible to one and hidden from another is a real gap.
    if tool_check.stage != 1 or not tool_check.is_active:
        raise HTTPException(status_code=400, detail="This tool is not currently available")

    if feature_type == "model_shoot":
        if sum(bool(m) for m in (model_image, model_preset_id, model_source_job_id)) != 1:
            raise HTTPException(
                status_code=400,
                detail="Provide exactly one of model_image, model_preset_id or model_source_job_id",
            )
        # The model reference is always exactly 1 image toward fal's own
        # image_urls cap, whether it came from an upload, a preset or a past
        # Library result -- all end up in the same final list sent to
        # fal (see worker.py's model_shoot image recombination), so it
        # counts the same either way. Library-sourced images count too.
        total_input_images = (
            1
            + len(top_images) + len(top_source_job_ids)
            + len(bottom_images) + len(bottom_source_job_ids)
            + len(extra_images_1) + len(extra_source_job_ids_1)
            + len(extra_images_2) + len(extra_source_job_ids_2)
            + len(extra_images_3) + len(extra_source_job_ids_3)
            + len(reference_images) + len(reference_source_job_ids)
        )
        if total_input_images > tool_check.max_input_images:
            raise HTTPException(
                status_code=400,
                detail=f"This tool accepts at most {tool_check.max_input_images} image(s) total "
                       f"(model_image + top/bottom/extra garment images + reference_images)",
            )
        # The whole point of this tool is putting a garment on the model --
        # a request with zero garment images (e.g. a stale client still
        # posting the old flat "garment_images" field, which this route no
        # longer declares and FastAPI silently drops rather than rejects)
        # must never be allowed to quietly plan+generate a garment-less
        # "model only" shot and still charge credits for it.
        total_garment_images = (
            len(top_images) + len(top_source_job_ids)
            + len(bottom_images) + len(bottom_source_job_ids)
            + len(extra_images_1) + len(extra_source_job_ids_1)
            + len(extra_images_2) + len(extra_source_job_ids_2)
            + len(extra_images_3) + len(extra_source_job_ids_3)
        )
        if total_garment_images == 0:
            raise HTTPException(
                status_code=400,
                detail="At least one garment image is required (top_images, bottom_images, "
                       "or an extra_images_N slot) -- if you're using the test console, hard-refresh "
                       "the page, an older cached version may still be sending the removed "
                       "garment_images field.",
            )
    elif len(images) + len(source_job_ids) > tool_check.max_input_images:
        raise HTTPException(
            status_code=400,
            detail=f"This tool accepts at most {tool_check.max_input_images} image(s)",
        )

    # output_count left unset (None) means "use this tool's own default" --
    # was previously a hardcoded Form(1), which made tool.default_output_count
    # unreachable for every tool (listing_photoshoot's default of 4 could
    # never actually take effect without the caller explicitly passing 4).
    if output_count is not None and output_count > MAX_OUTPUT_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"output_count cannot exceed {MAX_OUTPUT_COUNT} per request",
        )

    # listing_photoshoot's own, tighter cap (see app/tools/listing_planner.py)
    # -- rejected here, before the billable shot-planning vision call ever
    # runs, rather than only discovered later when create_generation_batch's
    # own resolve_generation_params call rejects it after that vision call
    # already ran for nothing.
    if (
        feature_type == "listing_photoshoot"
        and output_count is not None
        and output_count > LISTING_MAX_OUTPUT_COUNT
    ):
        raise HTTPException(
            status_code=400,
            detail=f"listing_photoshoot accepts at most {LISTING_MAX_OUTPUT_COUNT} shot(s) per request",
        )

    # Validate the tool's own required/select fields (e.g. a missing color)
    # BEFORE paying to upload anything to fal -- a request that's going to be
    # rejected here shouldn't cost real fal upload quota every time.
    try:
        validate_input_params(
            tool_check.param_schema,
            {
                "color": color, "quality": quality, "size": size, "target_area": target_area,
                "prompt": prompt, "source_feature_type": source_feature_type, "idea": idea,
                "gender": gender, "age_bracket": age_bracket, "ethnicity": ethnicity,
                "skin_tone": skin_tone, "body_type": body_type, "notes": notes,
                "aspect_ratio": aspect_ratio, "resolution": resolution,
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # creative_photoshoot's param_schema marks idea and prompt each optional,
    # but at least one is required (see creative_photoshoot.build_instruction,
    # which raises without one). That check used to live ONLY in the worker,
    # so a request with neither was accepted, created a job, reserved credits
    # and then failed minutes later -- rejected here instead, before any
    # upload or credit reservation.
    if feature_type == "creative_photoshoot" and not (idea or "").strip() and not (prompt or "").strip():
        raise HTTPException(status_code=400, detail="Describe your scene or pick an idea")

    # Resolve every Library-sourced image id in ONE query, before any upload:
    # a bad selection (another team's job, a failed job, an unknown id) must
    # be rejected without having spent fal quota on the uploads beside it.
    if feature_type == "model_shoot":
        wanted_source_ids = [
            *([model_source_job_id] if model_source_job_id else []),
            *top_source_job_ids, *bottom_source_job_ids,
            *extra_source_job_ids_1, *extra_source_job_ids_2, *extra_source_job_ids_3,
            *reference_source_job_ids,
        ]
    else:
        wanted_source_ids = list(source_job_ids)
    source_urls = _resolve_source_job_urls(db, team_id, wanted_source_ids)

    if feature_type == "model_shoot":
        if model_preset_id:
            preset = db.query(ModelPreset).filter(ModelPreset.id == model_preset_id).first()
            if not preset or not preset.is_active:
                raise HTTPException(status_code=404, detail="Unknown model preset")
            # Already a real hosted CloudFront URL -- skip
            # _read_and_upload_images (and its content-type/size validation)
            # entirely. There's no per-request check on an uploaded
            # model_image beyond that generic upload validation today, and a
            # preset was never uploaded as a file in this request, so there's
            # nothing to validate here -- by design, not an oversight.
            model_image_url = preset.image_url
        elif model_source_job_id:
            # A past Library output (typically a model_shoot_generate_model
            # result) -- already a real hosted URL, nothing to upload.
            model_image_url = source_urls[model_source_job_id]
        else:
            model_image_url = (await _read_and_upload_images([model_image]))[0]
        # Build the grouped garments list -- one entry per non-empty slot, in
        # a fixed Top -> Bottom -> extra-1 -> extra-2 -> extra-3 order (this
        # order is what plan_model_shoot labels images in, and worker.py's
        # real generation call must reproduce it exactly, see
        # flatten_garment_image_urls). Within a slot, uploads come first, then
        # Library-sourced images, in the order they were sent.
        garments = []
        top_image_urls = (
            await _read_and_upload_images(top_images)
            + [source_urls[j] for j in top_source_job_ids]
        )
        if top_image_urls:
            garments.append({"label": "Top", "image_urls": top_image_urls})
        bottom_image_urls = (
            await _read_and_upload_images(bottom_images)
            + [source_urls[j] for j in bottom_source_job_ids]
        )
        if bottom_image_urls:
            garments.append({"label": "Bottom", "image_urls": bottom_image_urls})
        for extra_label, extra_images, extra_source_ids in (
            (extra_label_1, extra_images_1, extra_source_job_ids_1),
            (extra_label_2, extra_images_2, extra_source_job_ids_2),
            (extra_label_3, extra_images_3, extra_source_job_ids_3),
        ):
            extra_image_urls = (
                await _read_and_upload_images(extra_images)
                + [source_urls[j] for j in extra_source_ids]
            )
            if extra_image_urls:
                garments.append({"label": extra_label or "Garment", "image_urls": extra_image_urls})
        reference_image_urls = (
            await _read_and_upload_images(reference_images)
            + [source_urls[j] for j in reference_source_job_ids]
        )
    else:
        image_urls = (
            await _read_and_upload_images(images)
            + [source_urls[j] for j in source_job_ids]
        )

    input_params = {
        "color": color,
        "quality": quality,
        "size": size,
        "target_area": target_area,
        "prompt": prompt,
        "source_feature_type": source_feature_type,
        "idea": idea,
        "gender": gender,
        "age_bracket": age_bracket,
        "ethnicity": ethnicity,
        "skin_tone": skin_tone,
        "body_type": body_type,
        "notes": notes,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
    }
    if feature_type == "model_shoot":
        input_params["model_image"] = model_image_url
        input_params["garments"] = garments  # always a list, even empty -- see the grouping comment above
        input_params["reference_images"] = reference_image_urls  # always a list, even empty
    else:
        input_params["image_urls"] = image_urls  # always a list now, even for single-image tools
    input_params = {k: v for k, v in input_params.items() if v is not None}

    try:
        if feature_type == "listing_photoshoot":
            # The billable shot-planning vision call must not run at all if a
            # generation is already in progress -- it used to sit BEFORE
            # create_generation_batch's own lock check, so a request that was
            # going to be rejected anyway still burned a real fal vision call
            # for nothing. Acquire (or heal) the lock here first; on any
            # failure before create_generation_batch takes over ownership of
            # it, this code path must release it itself.
            if not acquire_or_heal_generation_lock(db, user.id):
                raise HTTPException(status_code=400, detail=ALREADY_IN_PROGRESS_MESSAGE)

            effective_output_count = output_count if output_count and output_count > 0 else tool_check.default_output_count
            try:
                # Off the event loop: this is a blocking HTTP call that now retries
                # (up to ~94s worst case). Run inline it would freeze EVERY other
                # request on this server for that long, not just this one.
                shots = await asyncio.to_thread(
                    plan_listing_shots, tool_check, image_urls, prompt, effective_output_count,
                )
            except Exception:
                release_generation_lock(user.id)
                raise HTTPException(status_code=503, detail="Failed to plan listing shots. Please try again.")

            per_job_overrides = [{"prompt": shot["prompt"], "shot_type": shot["shot_type"]} for shot in shots]
            jobs = create_generation_batch(
                db, team_id, user.id, feature_type, input_params,
                effective_output_count, per_job_overrides=per_job_overrides,
                lock_already_held=True,
            )
        elif feature_type == "model_shoot":
            # Same reasoning as listing_photoshoot above -- plan_model_shoot
            # is a billable vision call that must not run if a generation is
            # already in progress for this user.
            if not acquire_or_heal_generation_lock(db, user.id):
                raise HTTPException(status_code=400, detail=ALREADY_IN_PROGRESS_MESSAGE)

            effective_output_count = output_count if output_count and output_count > 0 else tool_check.default_output_count
            try:
                # Off the event loop, same reason as plan_listing_shots above.
                plan = await asyncio.to_thread(
                    plan_model_shoot,
                    tool_check, model_image_url, garments, reference_image_urls,
                    prompt, effective_output_count,
                )
            except Exception:
                release_generation_lock(user.id)
                raise HTTPException(status_code=503, detail="Failed to plan the shoot. Please try again.")

            if plan["blocked"]:
                # No GenerationJob row, no credits spent -- reject up front,
                # same as any other request-shape validation failure. The
                # LLM-authored reason is logged for internal visibility only
                # -- never returned in the response body (see
                # MODEL_SHOOT_BLOCKED_MESSAGE above for why).
                logger.info(
                    "model_shoot blocked for user %s (team %s): %s",
                    user.id, team_id, plan["reason"],
                )
                release_generation_lock(user.id)
                raise HTTPException(status_code=400, detail=MODEL_SHOOT_BLOCKED_MESSAGE)

            per_job_overrides = [{"prompt": p} for p in plan["prompts"]]
            jobs = create_generation_batch(
                db, team_id, user.id, feature_type, input_params,
                effective_output_count, per_job_overrides=per_job_overrides,
                lock_already_held=True,
            )
        elif feature_type == "recolor" and image_urls:
            # One input image -> one output, each recolored on its own. Without
            # this every job got the whole image_urls list (flux-2/edit then
            # treated all N as references for ONE blended result, and the
            # vision step only ever looked at image 1). per_job_overrides makes
            # create_generation_batch size the batch to len(image_urls) and
            # bill per_output_cost * N, so the client's output_count (which
            # recolor's schema doesn't even have) is deliberately ignored.
            per_job_overrides = [{"image_urls": [url]} for url in image_urls]
            jobs = create_generation_batch(
                db, team_id, user.id, feature_type, input_params,
                len(per_job_overrides), per_job_overrides=per_job_overrides,
            )
        else:
            jobs = create_generation_batch(db, team_id, user.id, feature_type, input_params, output_count)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        pool = await get_arq_pool()
        for job in jobs:
            await pool.enqueue_job("submit_generation_to_fal", str(job.id))
    except Exception:
        for job in jobs:
            fail_and_release(db, job, "Failed to enqueue generation for processing")
        raise HTTPException(status_code=503, detail="Failed to start generation. Please try again.")

    # len(jobs) can be less than what was asked for -- create_generation_batch
    # sizes a multi-output batch down to whatever the team can actually afford
    # right now (see spend_credits_up_to) rather than rejecting the whole
    # request outright when the shared team balance can't cover the full ask.
    # Only compares against the caller's own EXPLICIT output_count (a real,
    # already-validated int straight from the request -- never a tool default
    # resolved internally) so this can never be "less than" some default the
    # caller never actually asked for.
    if feature_type == "recolor" and image_urls:
        # Recolor's batch size is the number of input images, not output_count.
        requested_output_count = len(image_urls)
    else:
        requested_output_count = output_count if output_count and output_count > 0 else len(jobs)
    return {
        "batchId": str(jobs[0].batch_id),
        "jobs": [{"jobId": str(j.id), "status": j.status} for j in jobs],
        "requestedCount": requested_output_count,
        "grantedCount": len(jobs),
        "partial": len(jobs) < requested_output_count,
    }


@router.get("/model-presets")
@limiter.limit("60/minute")
def get_model_presets(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    presets = db.query(ModelPreset).filter(ModelPreset.is_active == True).all()  # noqa: E712
    return {
        "presets": [
            {"id": p.id, "name": p.name, "thumbnailUrl": p.thumbnail_url}
            for p in presets
        ]
    }


@router.get("/jobs/{job_id}")
@limiter.limit("60/minute")
def get_job(
    job_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    job = db.query(GenerationJob).filter(GenerationJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not is_team_member(db, job.team_id, user.id):
        raise HTTPException(status_code=403, detail="You are not a member of this team")

    return {
        "jobId": str(job.id),
        "status": job.status,
        "outputUrl": job.output_url,
        "outputText": job.output_text,
        "errorMessage": public_job_error(job.status, job.error_message),
        "creditsCharged": job.credits_charged,
        # Only ever set for listing_photoshoot today (its shot planner writes
        # shot_type into input_params as a per_job_override -- see
        # create_generation_batch). model_shoot has no per-prompt label of
        # its own (plan_model_shoot returns bare prompt strings, nothing
        # structured to pull a label from), so this is always null there --
        # not a bug, there's genuinely nothing to show yet.
        "shotType": job.input_params.get("shot_type") if job.input_params else None,
    }


@router.get("/batches/{batch_id}")
@limiter.limit("60/minute")
def get_batch(
    batch_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    jobs = db.query(GenerationJob).filter(GenerationJob.batch_id == batch_id).all()
    if not jobs:
        raise HTTPException(status_code=404, detail="Batch not found")

    if not is_team_member(db, jobs[0].team_id, user.id):
        raise HTTPException(status_code=403, detail="You are not a member of this team")

    return {
        "batchId": str(batch_id),
        "jobs": [
            {
                "jobId": str(j.id), "status": j.status, "outputUrl": j.output_url,
                "outputText": j.output_text, "errorMessage": public_job_error(j.status, j.error_message),
                "creditsCharged": j.credits_charged,
                # See the matching comment on GET /jobs/{job_id} -- null for
                # every tool except listing_photoshoot today.
                "shotType": j.input_params.get("shot_type") if j.input_params else None,
            }
            for j in jobs
        ],
    }


@router.post("/webhooks/fal")
@limiter.limit("120/minute")
async def fal_webhook(request: Request, job_id: UUID, db: Session = Depends(get_db)):
    raw_body = await request.body()

    try:
        verify_fal_webhook(request.headers, raw_body)
    except FalWebhookVerificationError as e:
        raise HTTPException(status_code=401, detail=str(e))

    payload = json.loads(raw_body)
    # Off the event loop: handling a completed job downloads fal's output and
    # uploads it to our storage -- two blocking HTTP calls that now retry (a
    # stalled one could hold ~94s each). Inline in this async handler that
    # would freeze every other request on the server.
    await asyncio.to_thread(handle_fal_webhook, db, job_id, payload)
    return {"status": "ok"}