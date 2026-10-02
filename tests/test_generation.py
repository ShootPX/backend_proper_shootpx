"""Generation pipeline: per-user lock, credit reserve/refund, webhook idempotency,
and ownership checks on /generate and /jobs/{id}."""

import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.cache import redis_client
from app.core.database import get_db
from app.deps import get_current_user
from app.main import app
from app.models.generation_job import GenerationJob
from app.services import generation as generation_svc
from app.services.generation_lock import (
    LOCK_TTL_SECONDS,
    LOCK_TTL_BUFFER_SECONDS,
    acquire_generation_lock,
    extend_generation_lock_ttl,
    generation_lock_age_seconds,
    release_generation_lock,
)
from tests.prompt_fixtures import ai_steps_for

TEAM_ID = uuid.uuid4()

# release_fal_slot / try_reserve_fal_slot are auto-mocked for every test by
# the autouse fixture in conftest.py (they talk to real Redis, and almost
# nothing here is testing the fal-concurrency-limit feature itself).


# --------------------------------------------------------------------------- #
# Lock behavior — against the real Redis instance, same pattern as
# test_cache_admin.py uses for the real cache module. This is the actual
# mechanism (SET NX EX), not a mock of it.
# --------------------------------------------------------------------------- #

@pytest.fixture
def lock_user():
    user_id = uuid.uuid4()
    yield user_id
    release_generation_lock(user_id)  # always clean up, pass or fail


def test_lock_blocks_second_acquire_for_same_user(lock_user):
    assert acquire_generation_lock(lock_user) is True
    assert acquire_generation_lock(lock_user) is False  # already held


def test_lock_is_independent_per_user(lock_user):
    other_user = uuid.uuid4()
    try:
        assert acquire_generation_lock(lock_user) is True
        assert acquire_generation_lock(other_user) is True  # not blocked by lock_user's lock
    finally:
        release_generation_lock(other_user)


def test_release_then_acquire_succeeds_again(lock_user):
    assert acquire_generation_lock(lock_user) is True
    release_generation_lock(lock_user)
    assert acquire_generation_lock(lock_user) is True


def test_lock_has_a_ttl_safety_net(lock_user):
    """Must have a TTL so a crash mid-flow can never lock a user out forever."""
    acquire_generation_lock(lock_user)
    ttl = redis_client.ttl(f"genlock:user:{lock_user}")
    assert 0 < ttl <= LOCK_TTL_SECONDS


def test_release_is_a_safe_noop_when_nothing_held(lock_user):
    release_generation_lock(lock_user)  # never acquired -- must not raise


# --------------------------------------------------------------------------- #
# extend_generation_lock_ttl -- re-deriving the lock's TTL from a specific
# tool's own generation_timeout_seconds, against real Redis. Regression
# coverage for the real collision found live: listing_photoshoot's
# generation_timeout_seconds is exactly 300s, identical to the old fixed
# LOCK_TTL_SECONDS -- a job running right up to its own budget could have its
# lock expire out from under it, letting the same user start a second
# generation before the first one was done.
# --------------------------------------------------------------------------- #

def test_extend_generation_lock_ttl_derives_ttl_from_the_tools_own_timeout(lock_user):
    acquire_generation_lock(lock_user)  # starts at the generic LOCK_TTL_SECONDS

    extend_generation_lock_ttl(lock_user, generation_timeout_seconds=300)

    ttl = redis_client.ttl(f"genlock:user:{lock_user}")
    assert 0 < ttl <= 300 + LOCK_TTL_BUFFER_SECONDS
    # Specifically confirms the real collision: a tool timeout equal to the
    # OLD fixed LOCK_TTL_SECONDS now still gets its own margin on top, not
    # left sitting at exactly LOCK_TTL_SECONDS with zero headroom.
    assert ttl > LOCK_TTL_SECONDS


def test_extend_generation_lock_ttl_is_a_noop_when_no_lock_is_held(lock_user):
    """Must never resurrect a lock that isn't (or is no longer) held."""
    extend_generation_lock_ttl(lock_user, generation_timeout_seconds=300)

    assert redis_client.ttl(f"genlock:user:{lock_user}") == -2  # key doesn't exist


def test_generation_lock_age_seconds_is_correct_after_extending_past_the_default_ttl():
    """The lock's value stores its OWN real TTL (not a placeholder), so age
    is computed correctly even when a tool's timeout pushed the TTL above the
    generic LOCK_TTL_SECONDS default -- age must never come out negative."""
    user_id = uuid.uuid4()
    try:
        acquire_generation_lock(user_id)
        extend_generation_lock_ttl(user_id, generation_timeout_seconds=300)

        age = generation_lock_age_seconds(user_id)

        assert age is not None
        assert 0 <= age <= 2  # just acquired+extended, essentially zero seconds old
    finally:
        release_generation_lock(user_id)


# --------------------------------------------------------------------------- #
# create_generation_batch — lock + credit reserve, service level (mocked DB)
# --------------------------------------------------------------------------- #

def _tool(feature_type="test_tool", credit_cost=5, active=True, default_output_count=1,
          generation_timeout_seconds=60, stage=1):
    return MagicMock(feature_type=feature_type, credit_cost_per_output=credit_cost,
                      is_active=active, default_output_count=default_output_count,
                      generation_timeout_seconds=generation_timeout_seconds, stage=stage)


def _job_db(tool):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = tool
    return db


def test_create_batch_rejected_when_lock_already_held(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: False)
    spend = MagicMock()
    monkeypatch.setattr(generation_svc, "spend_credits_up_to", spend)
    db = _job_db(_tool())

    with pytest.raises(ValueError, match="already have a generation in progress"):
        generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "test_tool", {})

    spend.assert_not_called()  # lock is checked before anything else touches credits


# --------------------------------------------------------------------------- #
# Orphaned-lock self-healing: a crash/restart between acquire_generation_lock()
# and the job being created (or the except-block release running) leaves the
# lock stranded with nothing backing it. create_generation_batch must clear
# it and proceed ONLY when the lock is old enough to rule out a genuinely
# in-flight request AND there's no real job in the DB behind it.
# --------------------------------------------------------------------------- #

def _multi_model_db(tool, active_job):
    """Distinguishes the ToolDefinition lookup from the GenerationJob
    active-check -- a single db.query(...).filter(...).first() mock can't
    tell these apart since both create_generation_batch's own tool lookup
    and _user_has_active_generation_job's check share that exact shape."""
    db = MagicMock()

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "ToolDefinition":
            q.filter.return_value.first.return_value = tool
        elif name == "GenerationJob":
            q.filter.return_value.first.return_value = active_job
        return q

    db.query.side_effect = query
    return db


def test_stale_lock_is_cleared_and_retried_when_old_enough_and_no_active_job(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", MagicMock(side_effect=[False, True]))
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    monkeypatch.setattr(generation_svc, "generation_lock_age_seconds", lambda uid: 45)  # > STALE_LOCK_GRACE_SECONDS
    monkeypatch.setattr(generation_svc, "spend_credits_up_to", MagicMock(return_value=(MagicMock(), 1, 5, 0)))
    user_id = uuid.uuid4()
    db = _multi_model_db(_tool(credit_cost=5), active_job=None)  # no active job backing the lock

    jobs = generation_svc.create_generation_batch(db, TEAM_ID, user_id, "test_tool", {})

    assert len(jobs) == 1  # succeeded -- the stale lock did not block it
    assert released == [user_id]


def test_lock_stays_blocked_when_a_real_active_job_backs_it(monkeypatch):
    """The lock is old enough to otherwise qualify as stale, but a real
    queued/processing job is actually backing it -- must NOT be cleared."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: False)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    monkeypatch.setattr(generation_svc, "generation_lock_age_seconds", lambda uid: 45)
    db = _multi_model_db(_tool(), active_job=MagicMock())  # a real active job exists

    with pytest.raises(ValueError, match="already have a generation in progress"):
        generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "test_tool", {})

    assert released == []  # never touched -- a real job is backing this lock


def test_lock_stays_blocked_when_too_recent_even_with_no_active_job(monkeypatch):
    """Protects a genuinely in-flight request: job creation just hasn't
    committed yet, so no DB row exists -- but the lock is too young to
    assume it's orphaned rather than mid-flight."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: False)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    monkeypatch.setattr(generation_svc, "generation_lock_age_seconds", lambda uid: 5)  # well under the grace period
    db = _multi_model_db(_tool(), active_job=None)

    with pytest.raises(ValueError, match="already have a generation in progress"):
        generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "test_tool", {})

    assert released == []


def test_lock_stays_blocked_when_age_is_unknown(monkeypatch):
    """generation_lock_age_seconds returns None when the lock key doesn't
    actually exist in Redis (e.g. a race where it expired between the failed
    acquire and this check) -- must fail closed, never self-heal on unknown age."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: False)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    monkeypatch.setattr(generation_svc, "generation_lock_age_seconds", lambda uid: None)
    db = _multi_model_db(_tool(), active_job=None)

    with pytest.raises(ValueError, match="already have a generation in progress"):
        generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "test_tool", {})

    assert released == []


def test_create_batch_releases_lock_on_unknown_tool(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    user_id = uuid.uuid4()
    db = _job_db(None)  # tool lookup returns nothing

    with pytest.raises(ValueError, match="Unknown tool"):
        generation_svc.create_generation_batch(db, TEAM_ID, user_id, "nope", {})

    assert released == [user_id]


def test_create_batch_releases_lock_on_inactive_tool(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    user_id = uuid.uuid4()
    db = _job_db(_tool(active=False))

    with pytest.raises(ValueError, match="not currently available"):
        generation_svc.create_generation_batch(db, TEAM_ID, user_id, "test_tool", {})

    assert released == [user_id]


def test_create_batch_releases_lock_on_a_tool_at_the_wrong_stage(monkeypatch):
    """is_active alone isn't the gate -- this last-line-of-defense check must
    match GET /tools/{feature_type}/schema's condition exactly (stage != 1 or
    not is_active), so a tool live-but-still-staged can't slip through here
    while being hidden from the schema endpoint."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    user_id = uuid.uuid4()
    db = _job_db(_tool(active=True, stage=0))

    with pytest.raises(ValueError, match="not currently available"):
        generation_svc.create_generation_batch(db, TEAM_ID, user_id, "test_tool", {})

    assert released == [user_id]


def test_create_batch_releases_lock_on_insufficient_credits(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    monkeypatch.setattr(
        generation_svc, "spend_credits_up_to",
        MagicMock(side_effect=ValueError("Insufficient credits. Available: 0, needed: 5")),
    )
    user_id = uuid.uuid4()
    db = _job_db(_tool())

    with pytest.raises(ValueError, match="Insufficient credits"):
        generation_svc.create_generation_batch(db, TEAM_ID, user_id, "test_tool", {})

    assert released == [user_id]
    db.rollback.assert_called_once()


def test_create_batch_success_charges_and_keeps_lock_held(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    monkeypatch.setattr(
        generation_svc, "spend_credits_up_to",
        MagicMock(return_value=(MagicMock(), 1, 3, 2)),  # granted 1/1, split across both pools, real ints
    )
    user_id = uuid.uuid4()
    db = _job_db(_tool(credit_cost=5))

    jobs = generation_svc.create_generation_batch(db, TEAM_ID, user_id, "test_tool", {"prompt": "x"}, 1)

    assert len(jobs) == 1
    job = jobs[0]
    assert job.status == "queued"
    assert job.credits_charged == 5
    assert job.credits_from_subscription == 3
    assert job.credits_from_topup == 2
    db.add.assert_called_once_with(job)
    db.commit.assert_called_once()  # spend + job insert land in ONE transaction
    assert released == []  # lock stays held until the webhook resolves the job


def test_create_batch_extends_the_lock_ttl_to_the_resolved_tools_own_timeout(monkeypatch):
    """The real wiring for the lock-TTL-vs-tool-timeout fix: create_generation_
    batch must re-derive the lock's TTL from the ACTUAL tool it resolved
    (generation_timeout_seconds=300, e.g. listing_photoshoot), not leave it at
    the generic default -- this is what closes the exact collision found live
    (a tool's own timeout equal to the old fixed lock TTL)."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "spend_credits_up_to", MagicMock(return_value=(MagicMock(), 1, 5, 0)))
    extend = MagicMock()
    monkeypatch.setattr(generation_svc, "extend_generation_lock_ttl", extend)
    user_id = uuid.uuid4()
    db = _job_db(_tool(credit_cost=5, generation_timeout_seconds=300))

    generation_svc.create_generation_batch(db, TEAM_ID, user_id, "test_tool", {}, 1)

    extend.assert_called_once_with(user_id, 300)


# --------------------------------------------------------------------------- #
# per_job_overrides -- the generic mechanism listing_photoshoot/model_shoot's
# own planning paths both use: N jobs, each with its own planned prompt merged
# over the shared input_params, instead of every job in the batch sharing one
# identical input_params dict. Exercised here through a generic "batch_tool"
# feature_type (flat credit_cost_per_output, like most tools) rather than the
# real "listing_photoshoot" string, since that string now carries its own
# quality_multiplier x resolution_multiplier pricing (see the dedicated
# listing_photoshoot pricing section below) that's irrelevant to what this
# section is actually testing -- the batching mechanic itself.
# --------------------------------------------------------------------------- #

def test_per_job_overrides_gives_each_job_its_own_distinct_input_params(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "spend_credits_up_to", MagicMock(return_value=(MagicMock(), 2, 6, 0)))
    db = _job_db(_tool(credit_cost=3))

    overrides = [
        {"prompt": "Hero shot, studio lighting", "shot_type": "hero"},
        {"prompt": "Side angle, studio lighting", "shot_type": "side"},
    ]
    shared_input = {"size": "1:1", "image_urls": ["https://fal.test/product.png"]}

    jobs = generation_svc.create_generation_batch(
        db, TEAM_ID, uuid.uuid4(), "batch_tool", shared_input,
        per_job_overrides=overrides,
    )

    assert len(jobs) == 2
    assert jobs[0].input_params == {**shared_input, **overrides[0]}
    assert jobs[1].input_params == {**shared_input, **overrides[1]}
    assert jobs[0].input_params["prompt"] == "Hero shot, studio lighting"
    assert jobs[1].input_params["prompt"] == "Side angle, studio lighting"
    # the shared fields survive on every job, not just the first
    assert jobs[0].input_params["size"] == "1:1"
    assert jobs[1].input_params["size"] == "1:1"


def test_per_job_overrides_length_wins_even_when_output_count_argument_disagrees(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "spend_credits_up_to", MagicMock(return_value=(MagicMock(), 3, 9, 0)))
    db = _job_db(_tool(credit_cost=3, default_output_count=4))

    overrides = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]

    jobs = generation_svc.create_generation_batch(
        db, TEAM_ID, uuid.uuid4(), "batch_tool", {},
        output_count=1,  # deliberately disagrees with len(overrides)
        per_job_overrides=overrides,
    )

    assert len(jobs) == 3  # overrides length wins, not the output_count argument


def test_per_job_overrides_splits_credits_correctly_across_all_jobs(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "spend_credits_up_to", MagicMock(return_value=(MagicMock(), 3, 7, 2)))
    db = _job_db(_tool(credit_cost=3))

    overrides = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]

    jobs = generation_svc.create_generation_batch(
        db, TEAM_ID, uuid.uuid4(), "batch_tool", {}, per_job_overrides=overrides,
    )

    assert len(jobs) == 3
    assert all(j.credits_charged == 3 for j in jobs)
    # 7 // 3 = 2 each, remainder 1 on the first job; 2 // 3 = 0 each, remainder 2 on the first job
    assert [j.credits_from_subscription for j in jobs] == [3, 2, 2]
    assert [j.credits_from_topup for j in jobs] == [2, 0, 0]
    assert sum(j.credits_from_subscription for j in jobs) == 7
    assert sum(j.credits_from_topup for j in jobs) == 2


def test_without_per_job_overrides_every_job_shares_the_identical_input_params(monkeypatch):
    """Regression guard: every OTHER existing tool's behavior must stay
    exactly as before -- one shared input_params dict, not accidentally
    forked per job."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "spend_credits_up_to", MagicMock(return_value=(MagicMock(), 2, 10, 0)))
    db = _job_db(_tool(credit_cost=5))

    shared_input = {"color": "red", "image_urls": ["https://fal.test/1.png"]}
    jobs = generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "recolor", shared_input, output_count=2)

    assert len(jobs) == 2
    assert jobs[0].input_params == shared_input
    assert jobs[1].input_params == shared_input
    assert jobs[0].input_params is jobs[1].input_params is shared_input  # the exact same object, not a copy


def test_create_batch_spends_credits_without_committing_separately(monkeypatch):
    """Regression test: spend_credits must be called with commit=False so a
    failure creating the job rows can never leave credits spent with no job to
    account for them (see the credit-loss bug this closes)."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    spend = MagicMock(return_value=(MagicMock(), 1, 5, 0))  # real ints, needed for the // split math
    monkeypatch.setattr(generation_svc, "spend_credits_up_to", spend)
    db = _job_db(_tool(credit_cost=5))

    generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "test_tool", {})

    assert spend.call_args.kwargs.get("commit") is False


def test_create_batch_rolls_back_and_releases_lock_if_job_insert_fails(monkeypatch):
    """The actual credit-loss bug found during an earlier audit: spend_credits
    used to commit on its own, so a failure building/inserting a job row
    afterward left credits permanently spent with no job row and no way to
    refund them. Reproduced live against the real dev DB, then fixed by
    making the spend and the job insert(s) one transaction (commit=False +
    a single db.commit())."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    monkeypatch.setattr(generation_svc, "spend_credits_up_to", MagicMock(return_value=(MagicMock(), 1, 5, 0)))
    user_id = uuid.uuid4()
    db = _job_db(_tool(credit_cost=5))
    db.commit.side_effect = RuntimeError("db blip during job insert")

    with pytest.raises(RuntimeError):
        generation_svc.create_generation_batch(db, TEAM_ID, user_id, "test_tool", {})

    db.rollback.assert_called_once()  # the spend is rolled back along with the failed insert(s)
    assert released == [user_id]      # user is never left stuck


# --------------------------------------------------------------------------- #
# create_generation_batch — the batching logic itself (multi-output requests)
# --------------------------------------------------------------------------- #

def test_create_batch_of_five_creates_five_jobs_sharing_one_batch_id(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "spend_credits_up_to", MagicMock(return_value=(MagicMock(), 5, 25, 0)))
    db = _job_db(_tool(credit_cost=5))

    jobs = generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "test_tool", {}, 5)

    assert len(jobs) == 5
    batch_ids = {j.batch_id for j in jobs}
    assert len(batch_ids) == 1  # all 5 share exactly one batch_id
    assert db.add.call_count == 5
    db.commit.assert_called_once()  # one atomic transaction for the whole batch


def test_create_batch_splits_total_credit_charge_across_all_jobs_with_remainder_on_first(monkeypatch):
    """total_cost = 5 credits/output * 5 outputs = 25. Split unevenly on
    purpose (23 from subscription, 7 from topup -- neither divides evenly by
    5) so the remainder-handling is actually exercised, not just the clean
    case."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "spend_credits_up_to", MagicMock(return_value=(MagicMock(), 5, 23, 7)))
    db = _job_db(_tool(credit_cost=5))

    jobs = generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "test_tool", {}, 5)

    assert len(jobs) == 5
    # the sum across all 5 jobs must equal exactly what was charged, split or not
    assert sum(j.credits_from_subscription for j in jobs) == 23
    assert sum(j.credits_from_topup for j in jobs) == 7
    assert sum(j.credits_charged for j in jobs) == 25
    # even split is 23//5=4 remainder 3, and 7//5=1 remainder 2 -- both
    # remainders land on the first job only
    assert jobs[0].credits_from_subscription == 4 + 3
    assert jobs[0].credits_from_topup == 1 + 2
    for j in jobs[1:]:
        assert j.credits_from_subscription == 4
        assert j.credits_from_topup == 1
    # each job's own credits_charged is still the flat per-output cost, not a split
    assert all(j.credits_charged == 5 for j in jobs)


def test_create_batch_uses_tool_default_output_count_when_not_specified(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    spend = MagicMock(return_value=(MagicMock(), 3, 15, 0))
    monkeypatch.setattr(generation_svc, "spend_credits_up_to", spend)
    db = _job_db(_tool(credit_cost=5, default_output_count=3))

    jobs = generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "test_tool", {}, output_count=0)

    assert len(jobs) == 3
    # per-unit cost (5) and requested_units (3, the resolved default -- not 0)
    spend.assert_called_once_with(db, TEAM_ID, 5, 3, commit=False)


def test_misconfigured_zero_default_output_count_raises_a_clean_value_error(monkeypatch):
    """Regression: output_count is later used as a divisor for the
    credit-pool split. A tool_definitions row hand-edited via SQL to
    default_output_count=0 (or negative) used to reach that division
    directly and crash with an unhandled ZeroDivisionError -- which isn't a
    ValueError, so /generate's `except ValueError` never caught it and the
    request 500'd instead of failing cleanly. Must raise ValueError instead,
    same as every other "this request can't be fulfilled" case here."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    spend = MagicMock(return_value=(MagicMock(), 0, 0))
    monkeypatch.setattr(generation_svc, "spend_credits_up_to", spend)
    db = _job_db(_tool(default_output_count=0))

    with pytest.raises(ValueError, match="output_count must be between 1"):
        generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "test_tool", {}, output_count=0)

    spend.assert_not_called()  # rejected before any credits are touched


def test_output_count_over_the_max_raises_a_clean_value_error_even_from_a_tool_default(monkeypatch):
    """The MAX_OUTPUT_COUNT cap (see app/routes/generation.py) previously only
    guarded a caller-supplied output_count -- a tool_definitions row whose
    default_output_count exceeds it (DB-only misconfiguration, but nothing
    enforced it) could still create an oversized batch. Now enforced here too,
    on the resolved value regardless of where it came from."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    spend = MagicMock()
    monkeypatch.setattr(generation_svc, "spend_credits_up_to", spend)
    db = _job_db(_tool(default_output_count=generation_svc.MAX_OUTPUT_COUNT + 1))

    with pytest.raises(ValueError, match="output_count must be between 1"):
        generation_svc.create_generation_batch(db, TEAM_ID, uuid.uuid4(), "test_tool", {}, output_count=0)

    spend.assert_not_called()


# --------------------------------------------------------------------------- #
# _resolve_credit_cost — per-tier pricing (Standard/Advanced/Premium selects)
# --------------------------------------------------------------------------- #

QUALITY_TIER_SCHEMA = [
    {
        "name": "quality",
        "type": "select",
        "label": "Quality",
        "options": [
            {"value": "standard", "credit_cost": 5},
            {"value": "advanced", "credit_cost": 10},
            {"value": "premium", "credit_cost": 20},
        ],
    },
]


def _schema_tool(param_schema, credit_cost=1):
    return MagicMock(param_schema=param_schema, credit_cost_per_output=credit_cost)


@pytest.mark.parametrize("quality,expected_cost", [
    ("standard", 5),
    ("advanced", 10),
    ("premium", 20),
])
def test_resolve_credit_cost_picks_the_selected_tier(quality, expected_cost):
    tool = _schema_tool(QUALITY_TIER_SCHEMA, credit_cost=1)
    cost = generation_svc._resolve_credit_cost(tool, "some_tool", {"quality": quality}, output_count=1)
    assert cost == expected_cost


def test_resolve_credit_cost_falls_back_when_selected_value_matches_no_tier():
    tool = _schema_tool(QUALITY_TIER_SCHEMA, credit_cost=7)
    cost = generation_svc._resolve_credit_cost(tool, "some_tool", {"quality": "unknown-tier"}, output_count=1)
    assert cost == 7


def test_resolve_credit_cost_ignores_plain_string_options_without_credit_cost():
    """A select field whose options are plain strings (e.g. size/aspect ratio)
    must never be mistaken for a priced tier -- only options shaped as dicts
    with a credit_cost key opt into per-tier pricing."""
    schema = [{"name": "size", "type": "select", "options": ["1:1", "16:9"]}]
    tool = _schema_tool(schema, credit_cost=3)
    cost = generation_svc._resolve_credit_cost(tool, "some_tool", {"size": "16:9"}, output_count=1)
    assert cost == 3


# --------------------------------------------------------------------------- #
# _resolve_credit_cost — creative_photoshoot's own quality_multiplier *
# resolution_multiplier pricing (not a param_schema-embedded credit_cost)
# --------------------------------------------------------------------------- #

CREATIVE_PHOTOSHOOT_AI_STEPS = ai_steps_for("creative_photoshoot")

# Mirrors the real tool_definitions row's quality_multiplier x
# resolution_multiplier table exactly.
CREATIVE_PHOTOSHOOT_CREDIT_TABLE = {
    ("low", "1k"): 2, ("low", "2k"): 4, ("low", "4k"): 6,
    ("medium", "1k"): 4, ("medium", "2k"): 8, ("medium", "4k"): 12,
    ("high", "1k"): 10, ("high", "2k"): 20, ("high", "4k"): 30,
    ("xhigh", "1k"): 18, ("xhigh", "2k"): 36, ("xhigh", "4k"): 54,
    ("max", "1k"): 32, ("max", "2k"): 64, ("max", "4k"): 96,
}


@pytest.mark.parametrize("quality,resolution,expected_cost", [
    (q, r, cost) for (q, r), cost in CREATIVE_PHOTOSHOOT_CREDIT_TABLE.items()
])
def test_resolve_credit_cost_for_creative_photoshoot_matches_the_priced_table(quality, resolution, expected_cost):
    tool = MagicMock(ai_steps=CREATIVE_PHOTOSHOOT_AI_STEPS, credit_cost_per_output=3)
    cost = generation_svc._resolve_credit_cost(
        tool, "creative_photoshoot",
        {"aspect_ratio": "1:1", "resolution": resolution, "quality": quality},
        output_count=1,
    )
    assert cost == expected_cost


def test_resolve_credit_cost_for_creative_photoshoot_raises_on_an_unpriced_combo():
    """aspect_ratio/resolution combos not in ai_steps.size_map must be
    rejected up front, not silently charged some fallback amount."""
    tool = MagicMock(ai_steps=CREATIVE_PHOTOSHOOT_AI_STEPS, credit_cost_per_output=3)
    with pytest.raises(ValueError, match="does not support"):
        generation_svc._resolve_credit_cost(
            tool, "creative_photoshoot",
            {"aspect_ratio": "1:1", "resolution": "8k", "quality": "medium"},
            output_count=1,
        )


# --------------------------------------------------------------------------- #
# _resolve_credit_cost — listing_photoshoot's own quality_multiplier *
# resolution_multiplier * output_count pricing. Unlike creative_photoshoot,
# listing_planner.resolve_generation_params prices the WHOLE batch at once
# (one priced unit per shot) -- _resolve_credit_cost's job is to divide that
# back down to a per-job cost so create_generation_batch's own
# `per_output_cost * output_count` reconstructs the identical total.
# --------------------------------------------------------------------------- #

LISTING_PHOTOSHOOT_AI_STEPS = ai_steps_for("listing_photoshoot")


@pytest.mark.parametrize("quality,resolution,output_count,expected_total", [
    ("low", "1k", 1, 2), ("low", "1k", 4, 8),
    ("medium", "2k", 3, 24),
    ("high", "4k", 2, 60),
    ("max", "4k", 8, 768),
])
def test_resolve_credit_cost_for_listing_photoshoot_prices_per_job_so_the_total_matches(
    quality, resolution, output_count, expected_total,
):
    tool = MagicMock(ai_steps=LISTING_PHOTOSHOOT_AI_STEPS, credit_cost_per_output=0, default_output_count=4)
    per_job_cost = generation_svc._resolve_credit_cost(
        tool, "listing_photoshoot",
        {"aspect_ratio": "1:1", "resolution": resolution, "quality": quality},
        output_count=output_count,
    )
    assert per_job_cost * output_count == expected_total


def test_resolve_credit_cost_for_listing_photoshoot_raises_when_output_count_exceeds_its_own_cap():
    """listing_planner.MAX_OUTPUT_COUNT (8) -- rejected here (inside credit
    resolution) as a last line of defense even if the /generate route's own
    earlier cap check were ever bypassed."""
    tool = MagicMock(ai_steps=LISTING_PHOTOSHOOT_AI_STEPS, credit_cost_per_output=0, default_output_count=4)
    with pytest.raises(ValueError, match="out of the allowed range"):
        generation_svc._resolve_credit_cost(
            tool, "listing_photoshoot",
            {"aspect_ratio": "1:1", "resolution": "1k", "quality": "medium"},
            output_count=9,
        )


# --------------------------------------------------------------------------- #
# create_generation_batch x listing_photoshoot's real pricing, end to end --
# confirms _resolve_credit_cost's per-job price is what actually lands on
# each job's credits_charged (not just what the helper function returns in
# isolation).
# --------------------------------------------------------------------------- #

def test_create_batch_charges_listing_photoshoots_real_per_shot_price(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    # medium/2k = 2 * 4 = 8 credits/shot; 3 shots = 24 total, granted in full.
    monkeypatch.setattr(generation_svc, "spend_credits_up_to", MagicMock(return_value=(MagicMock(), 3, 24, 0)))
    tool = MagicMock(
        ai_steps=LISTING_PHOTOSHOOT_AI_STEPS, credit_cost_per_output=0, default_output_count=4,
        generation_timeout_seconds=300,
        param_schema=[], is_active=True, stage=1,
    )
    db = _job_db(tool)
    overrides = [{"prompt": "a", "shot_type": "hero"}, {"prompt": "b", "shot_type": "side"}, {"prompt": "c", "shot_type": "detail"}]

    jobs = generation_svc.create_generation_batch(
        db, TEAM_ID, uuid.uuid4(), "listing_photoshoot",
        {"aspect_ratio": "1:1", "resolution": "2k", "quality": "medium"},
        per_job_overrides=overrides,
    )

    assert len(jobs) == 3
    assert all(j.credits_charged == 8 for j in jobs)
    assert sum(j.credits_from_subscription for j in jobs) == 24


# --------------------------------------------------------------------------- #
# Partial-fulfillment on a shared, concurrently-spent team balance: the exact
# scenario reported live -- a batch of 6 shots at 2 credits each (12 total)
# against a team that shows 14 credits, but another concurrent request has
# already taken 4 of those, leaving 10 truly available right now. Rather than
# rejecting the whole request, create_generation_batch grants as many WHOLE
# shots as the team can actually afford (10 // 2 = 5) instead of all 6.
# --------------------------------------------------------------------------- #

def test_create_batch_grants_a_partial_listing_photoshoot_batch_when_team_credits_fall_short(monkeypatch):
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    tool = MagicMock(
        ai_steps=LISTING_PHOTOSHOOT_AI_STEPS, credit_cost_per_output=0, default_output_count=4,
        generation_timeout_seconds=300,
        param_schema=[], is_active=True, stage=1,
    )
    db = _job_db(tool)

    # low/1k = 1 * 2 = 2 credits/shot -- matches the "6 shots costing 2 each"
    # scenario reported live. Team shows 14 credits, but spend_credits_up_to
    # is where the REAL, currently-locked balance is read: another request
    # against the same team already spent 4 of those 14 concurrently, so only
    # 10 are truly available right now -- 10 // 2 = 5 whole shots, not 6.
    def fake_spend_credits_up_to(db_, team_id, per_unit_cost, requested_units, commit=True):
        team = MagicMock(subscription_credits_remaining=10, topup_credits_balance=0)
        granted_units = min(requested_units, 10 // per_unit_cost)
        return team, granted_units, min(10, per_unit_cost * granted_units), 0

    monkeypatch.setattr(generation_svc, "spend_credits_up_to", fake_spend_credits_up_to)

    overrides = [{"prompt": f"shot {i}", "shot_type": "hero"} for i in range(6)]

    jobs = generation_svc.create_generation_batch(
        db, TEAM_ID, uuid.uuid4(), "listing_photoshoot",
        {"aspect_ratio": "1:1", "resolution": "1k", "quality": "low"},
        per_job_overrides=overrides,
    )

    assert len(jobs) == 5  # requested 6, only 5 granted -- not rejected outright
    assert all(j.credits_charged == 2 for j in jobs)
    assert sum(j.credits_from_subscription for j in jobs) == 10


# --------------------------------------------------------------------------- #
# _resolve_credit_cost -- model_shoot's own resolution_credit * output_count
# pricing. Real bug fixed: model_shoot was never wired into the credit
# system at all -- credit_cost_per_output=0 in the DB row (a flat fallback)
# meant every model_shoot job was charged 0 credits, confirmed live. Same
# shape as listing_photoshoot: model_shoot.resolve_generation_params prices
# the WHOLE batch at once (one priced unit per pose), so _resolve_credit_cost
# divides that back down to a per-job cost.
# --------------------------------------------------------------------------- #

MODEL_SHOOT_AI_STEPS = ai_steps_for("model_shoot")

# Mirrors the real tool_definitions row's resolution_credit table exactly.
MODEL_SHOOT_CREDIT_TABLE = {"1k": 4, "2k": 6, "4k": 8}


@pytest.mark.parametrize("resolution,output_count,expected_total", [
    ("1k", 1, 4), ("1k", 4, 16),
    ("2k", 3, 18), ("2k", 6, 36),
    ("4k", 1, 8), ("4k", 8, 64),
])
def test_resolve_credit_cost_for_model_shoot_prices_per_job_so_the_total_matches(
    resolution, output_count, expected_total,
):
    tool = MagicMock(ai_steps=MODEL_SHOOT_AI_STEPS, credit_cost_per_output=0, default_output_count=4)
    per_job_cost = generation_svc._resolve_credit_cost(
        tool, "model_shoot",
        {"aspect_ratio": "1:1", "resolution": resolution},
        output_count=output_count,
    )
    assert per_job_cost * output_count == expected_total


def test_resolve_credit_cost_for_model_shoot_raises_when_output_count_exceeds_its_own_cap():
    tool = MagicMock(ai_steps=MODEL_SHOOT_AI_STEPS, credit_cost_per_output=0, default_output_count=4)
    with pytest.raises(ValueError, match="out of the allowed range"):
        generation_svc._resolve_credit_cost(
            tool, "model_shoot",
            {"aspect_ratio": "1:1", "resolution": "1k"},
            output_count=9,
        )


def test_resolve_credit_cost_for_model_shoot_raises_on_an_unpriced_combo():
    tool = MagicMock(ai_steps=MODEL_SHOOT_AI_STEPS, credit_cost_per_output=0, default_output_count=4)
    with pytest.raises(ValueError, match="does not support"):
        generation_svc._resolve_credit_cost(
            tool, "model_shoot",
            {"aspect_ratio": "1:1", "resolution": "8k"},
            output_count=4,
        )


def test_create_batch_charges_model_shoots_real_per_pose_price(monkeypatch):
    """End to end: confirms _resolve_credit_cost's per-job price is what
    actually lands on each job's credits_charged (not just what the helper
    function returns in isolation) -- before this fix, credits_charged would
    have been the flat DB fallback of 0 for every job regardless of
    resolution."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    # 2k = 6 credits/pose; 3 poses = 18 total, granted in full.
    monkeypatch.setattr(generation_svc, "spend_credits_up_to", MagicMock(return_value=(MagicMock(), 3, 18, 0)))
    tool = MagicMock(
        ai_steps=MODEL_SHOOT_AI_STEPS, credit_cost_per_output=0, default_output_count=4,
        generation_timeout_seconds=300,
        param_schema=[], is_active=True, stage=1,
    )
    db = _job_db(tool)
    overrides = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]

    jobs = generation_svc.create_generation_batch(
        db, TEAM_ID, uuid.uuid4(), "model_shoot",
        {"aspect_ratio": "3:4", "resolution": "2k"},
        per_job_overrides=overrides,
    )

    assert len(jobs) == 3
    assert all(j.credits_charged == 6 for j in jobs)
    assert sum(j.credits_from_subscription for j in jobs) == 18


def test_create_batch_charges_the_selected_tiers_credit_cost_not_the_flat_default(monkeypatch):
    """End-to-end through create_generation_batch: the tool's flat
    credit_cost_per_output must be overridden by the selected quality tier's
    own credit_cost, and that per-tier cost is what's split across every job
    in a multi-output batch."""
    monkeypatch.setattr(generation_svc, "acquire_generation_lock", lambda uid: True)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    spend = MagicMock(return_value=(MagicMock(), 2, 40, 0))
    monkeypatch.setattr(generation_svc, "spend_credits_up_to", spend)
    tool = _tool(credit_cost=5)
    tool.param_schema = QUALITY_TIER_SCHEMA
    db = _job_db(tool)

    jobs = generation_svc.create_generation_batch(
        db, TEAM_ID, uuid.uuid4(), "test_tool", {"quality": "premium"}, output_count=2,
    )

    assert len(jobs) == 2
    spend.assert_called_once_with(db, TEAM_ID, 20, 2, commit=False)  # 20/unit (premium), 2 units, not 5 (flat)
    assert all(j.credits_charged == 20 for j in jobs)


# --------------------------------------------------------------------------- #
# handle_fal_webhook — success/failure/refund-split/idempotent replay
# --------------------------------------------------------------------------- #

def _job(status="queued", credits_charged=5, from_sub=3, from_topup=2, batch_id=None, fal_request_id=None):
    return MagicMock(
        status=status,
        credits_charged=credits_charged,
        credits_from_subscription=from_sub,
        credits_from_topup=from_topup,
        team_id=TEAM_ID,
        user_id=uuid.uuid4(),
        batch_id=batch_id or uuid.uuid4(),
        output_url=None,
        error_message=None,
        fal_request_id=fal_request_id,
    )


def _webhook_db(job, remaining_in_batch=0):
    """remaining_in_batch: how many OTHER jobs in the same batch are still
    queued/processing -- 0 or 1 means "this was the last one" (the count
    includes this job's own not-yet-flushed row, see fail_and_release /
    handle_fal_webhook's `remaining <= 1` check), anything higher means
    siblings are still in flight and the lock must stay held."""
    db = MagicMock()
    (db.query.return_value.filter.return_value.populate_existing.return_value
       .with_for_update.return_value.first.return_value) = job
    db.query.return_value.filter.return_value.count.return_value = remaining_in_batch
    return db


# --------------------------------------------------------------------------- #
# request_id correlation: a genuinely fal-signed webhook for a DIFFERENT
# request must never be applied to this job (replay / cross-job substitution).
# --------------------------------------------------------------------------- #

def test_webhook_with_mismatched_request_id_is_ignored(monkeypatch):
    """Real gap found live: ?job_id=... isn't part of what fal signs, so
    nothing previously stopped a genuinely fal-signed webhook for one request
    being replayed onto a different job_id. Must be rejected before touching
    status/credits/lock at all."""
    job = _job(status="processing", fal_request_id="fal-req-REAL")
    db = _webhook_db(job)
    refund = MagicMock()
    release = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)
    monkeypatch.setattr(generation_svc, "release_generation_lock", release)

    generation_svc.handle_fal_webhook(
        db, uuid.uuid4(),
        {"status": "OK", "request_id": "fal-req-ATTACKERS-OWN",
         "payload": {"images": [{"url": "https://attacker.test/x.png"}]}},
    )

    assert job.status == "processing"  # untouched
    refund.assert_not_called()
    release.assert_not_called()
    db.commit.assert_not_called()


def test_webhook_with_matching_request_id_still_works(monkeypatch):
    """Regression check: the correlation check must not break the normal path."""
    job = _job(status="processing", fal_request_id="fal-req-REAL")
    db = _webhook_db(job)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "download_from_url", MagicMock(return_value=b"bytes"))
    monkeypatch.setattr(generation_svc, "upload_to_storage",
                         MagicMock(return_value="https://our-storage.test/ok.png"))

    generation_svc.handle_fal_webhook(
        db, uuid.uuid4(),
        {"status": "OK", "request_id": "fal-req-REAL",
         "payload": {"images": [{"url": "https://fal.test/ok.png"}]}},
    )

    assert job.status == "completed"


def test_webhook_without_a_request_id_field_is_not_rejected(monkeypatch):
    """Our own internal delivery (worker.check_generation_timeouts resolving
    a job via fal's real status API, not an actual HTTP webhook) always sets
    request_id correctly -- but stay permissive if it's ever absent rather
    than fail closed on a missing field that isn't itself suspicious."""
    job = _job(status="processing", fal_request_id="fal-req-REAL")
    db = _webhook_db(job)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "download_from_url", MagicMock(return_value=b"bytes"))
    monkeypatch.setattr(generation_svc, "upload_to_storage",
                         MagicMock(return_value="https://our-storage.test/ok.png"))

    generation_svc.handle_fal_webhook(
        db, uuid.uuid4(),
        {"status": "OK", "payload": {"images": [{"url": "https://fal.test/ok.png"}]}},
    )

    assert job.status == "completed"


def test_webhook_success_completes_job_and_releases_lock(monkeypatch):
    """Storage integration (see tests/test_storage_integration.py for the full
    download/upload matrix): output_url ends up as OUR permanent storage URL,
    not fal's raw temporary one."""
    job = _job()
    db = _webhook_db(job)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    monkeypatch.setattr(generation_svc, "download_from_url", MagicMock(return_value=b"bytes"))
    monkeypatch.setattr(generation_svc, "upload_to_storage",
                         MagicMock(return_value="https://our-storage.test/permanent/out.png"))
    refund = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)

    generation_svc.handle_fal_webhook(
        db, uuid.uuid4(),
        {"status": "OK", "payload": {"images": [{"url": "https://x.test/out.png"}]}},
    )

    assert job.status == "completed"
    assert job.output_url == "https://our-storage.test/permanent/out.png"
    assert released == [job.user_id]


# --------------------------------------------------------------------------- #
# enhance_prompt webhook branch -- text output, not an image download/upload
# --------------------------------------------------------------------------- #

def test_webhook_success_sets_output_text_for_enhance_prompt_without_touching_output_url(monkeypatch):
    job = _job()
    job.feature_type = "enhance_prompt"
    job.output_text = None
    db = _webhook_db(job)
    released = []
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: released.append(uid))
    download = MagicMock()
    upload = MagicMock()
    monkeypatch.setattr(generation_svc, "download_from_url", download)
    monkeypatch.setattr(generation_svc, "upload_to_storage", upload)
    refund = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)

    generation_svc.handle_fal_webhook(
        db, uuid.uuid4(),
        {"status": "OK", "payload": {"output": "A vivid, richly detailed enhanced prompt."}},
    )

    assert job.status == "completed"
    assert job.output_text == "A vivid, richly detailed enhanced prompt."
    assert job.output_url is None
    download.assert_not_called()
    upload.assert_not_called()
    refund.assert_not_called()
    assert released == [job.user_id]
    db.commit.assert_called_once()


def test_webhook_success_with_no_text_fails_and_refunds_enhance_prompt(monkeypatch):
    job = _job()
    job.feature_type = "enhance_prompt"
    job.output_text = None
    db = _webhook_db(job)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    refund = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)

    generation_svc.handle_fal_webhook(
        db, uuid.uuid4(),
        {"status": "OK", "payload": {}},
    )

    assert job.status == "failed"
    assert job.output_text is None
    assert job.error_message == "fal reported success but returned no text"
    refund.assert_called_once_with(db, job.team_id, job.credits_charged, job.credits_from_subscription, job.credits_from_topup)


def test_webhook_failure_sets_failed_and_refunds_matching_split(monkeypatch):
    job = _job(from_sub=3, from_topup=2, credits_charged=5)
    db = _webhook_db(job)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    refund = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)

    generation_svc.handle_fal_webhook(db, uuid.uuid4(), {"status": "ERROR", "error": "boom"})

    assert job.status == "failed"
    assert job.error_message == "boom"
    refund.assert_called_once_with(db, TEAM_ID, 5, 3, 2)  # exact pools, exact split
    db.commit.assert_called_once()


def test_webhook_failure_defaults_error_message_when_absent(monkeypatch):
    job = _job()
    db = _webhook_db(job)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    monkeypatch.setattr(generation_svc, "refund_credits", MagicMock())

    generation_svc.handle_fal_webhook(db, uuid.uuid4(), {"status": "ERROR"})

    assert job.status == "failed"
    assert job.error_message == "Generation failed"


def test_webhook_unknown_job_id_is_a_safe_noop(monkeypatch):
    db = MagicMock()
    (db.query.return_value.filter.return_value.populate_existing.return_value
       .with_for_update.return_value.first.return_value) = None
    refund = MagicMock()
    release = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)
    monkeypatch.setattr(generation_svc, "release_generation_lock", release)

    generation_svc.handle_fal_webhook(db, uuid.uuid4(), {"status": "OK", "payload": {}})

    refund.assert_not_called()
    release.assert_not_called()
    db.commit.assert_not_called()


@pytest.mark.parametrize("terminal_status", ["completed", "failed"])
def test_webhook_replay_on_terminal_job_is_a_safe_noop(monkeypatch, terminal_status):
    """Idempotent replay: a webhook delivered twice (or replayed by fal) for a
    job that's already terminal must not re-set fields, re-refund, or
    re-release an already-released lock in a way that breaks anything."""
    job = _job(status=terminal_status, from_sub=3, from_topup=2)
    job.output_url = "https://original.test/out.png"
    job.error_message = "original failure" if terminal_status == "failed" else None
    db = _webhook_db(job)
    refund = MagicMock()
    release = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)
    monkeypatch.setattr(generation_svc, "release_generation_lock", release)

    generation_svc.handle_fal_webhook(
        db, uuid.uuid4(),
        {"status": "OK", "payload": {"images": [{"url": "https://REPLAY-SHOULD-NOT-APPLY.test/x.png"}]}},
    )

    # nothing changed, nothing re-triggered
    assert job.output_url == "https://original.test/out.png"
    assert job.error_message == ("original failure" if terminal_status == "failed" else None)
    refund.assert_not_called()
    release.assert_not_called()
    db.commit.assert_not_called()


def test_webhook_no_images_in_success_payload_fails_and_refunds(monkeypatch):
    """Updated for the storage-integration behavior change: fal reporting
    success with no image is not a usable result, so it now fails + refunds
    (see tests/test_storage_integration.py for the full matrix) instead of
    completing with a null output_url."""
    job = _job()
    db = _webhook_db(job)
    monkeypatch.setattr(generation_svc, "release_generation_lock", lambda uid: None)
    refund = MagicMock()
    monkeypatch.setattr(generation_svc, "refund_credits", refund)

    generation_svc.handle_fal_webhook(db, uuid.uuid4(), {"status": "OK", "payload": {"images": []}})

    assert job.status == "failed"
    assert job.output_url is None
    refund.assert_called_once()


# --------------------------------------------------------------------------- #
# Batch-aware lock release: held until every job in the batch is terminal
# --------------------------------------------------------------------------- #

def test_lock_stays_held_after_only_first_of_five_jobs_completes(monkeypatch):
    job = _job(status="queued")
    db = _webhook_db(job, remaining_in_batch=4)  # 4 siblings still queued/processing
    release = MagicMock()
    monkeypatch.setattr(generation_svc, "release_generation_lock", release)
    monkeypatch.setattr(generation_svc, "download_from_url", MagicMock(return_value=b"bytes"))
    monkeypatch.setattr(generation_svc, "upload_to_storage",
                         MagicMock(return_value="https://our-storage.test/1.png"))

    generation_svc.handle_fal_webhook(
        db, uuid.uuid4(),
        {"status": "OK", "payload": {"images": [{"url": "https://fal.test/1.png"}]}},
    )

    assert job.status == "completed"
    release.assert_not_called()  # siblings still in flight -- lock must stay held


def test_lock_releases_once_the_last_job_in_the_batch_completes(monkeypatch):
    job = _job(status="queued")
    # 1, not 0: the count includes this job's own not-yet-flushed row (autoflush
    # is off), so "1 remaining" IS "this was the last one" -- see handle_fal_webhook.
    db = _webhook_db(job, remaining_in_batch=1)
    release = MagicMock()
    monkeypatch.setattr(generation_svc, "release_generation_lock", release)
    monkeypatch.setattr(generation_svc, "download_from_url", MagicMock(return_value=b"bytes"))
    monkeypatch.setattr(generation_svc, "upload_to_storage",
                         MagicMock(return_value="https://our-storage.test/5.png"))

    generation_svc.handle_fal_webhook(
        db, uuid.uuid4(),
        {"status": "OK", "payload": {"images": [{"url": "https://fal.test/5.png"}]}},
    )

    assert job.status == "completed"
    release.assert_called_once_with(job.user_id)


def test_lock_release_check_counts_only_queued_and_processing_siblings(monkeypatch):
    """A failure mid-batch must be able to release the lock too, once it was
    the last non-terminal job -- not just successes."""
    job = _job(status="processing")
    db = _webhook_db(job, remaining_in_batch=1)
    release = MagicMock()
    monkeypatch.setattr(generation_svc, "release_generation_lock", release)
    monkeypatch.setattr(generation_svc, "refund_credits", MagicMock())

    generation_svc.handle_fal_webhook(db, uuid.uuid4(), {"status": "ERROR", "error": "boom"})

    assert job.status == "failed"
    release.assert_called_once_with(job.user_id)


# --------------------------------------------------------------------------- #
# Route-level ownership checks: /generate and /jobs/{id}
# --------------------------------------------------------------------------- #

@pytest.fixture
def gen_client(monkeypatch):
    fake_user = MagicMock(id=uuid.uuid4())
    db = MagicMock()

    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_db] = lambda: db

    yield TestClient(app, raise_server_exceptions=False), fake_user, db
    app.dependency_overrides.clear()


def test_generate_rejects_malformed_team_id_with_422_not_500(gen_client):
    """Regression test: team_id used to be typed `str` on the request model, so
    a malformed value sailed past validation and hit the DB as a raw UUID
    comparison, raising an unhandled 500. Every other route in this codebase
    types team_id as UUID; this one now matches."""
    client, fake_user, db = gen_client

    res = client.post(
        "/generate",
        json={"team_id": "not-a-uuid", "feature_type": "test_tool", "input_params": {}},
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 422


def test_generate_rejects_non_team_member(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: False)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "test_tool"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 403


def test_generate_allows_team_member_and_surfaces_value_error_as_400(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(max_input_images=5, stage=1,)
    monkeypatch.setattr(
        "app.routes.generation.upload_image_to_fal",
        lambda *a, **k: "https://fal.test/uploaded.png",
    )
    monkeypatch.setattr(
        "app.routes.generation.create_generation_batch",
        MagicMock(side_effect=ValueError("You already have a generation in progress. Please wait for it to finish.")),
    )

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "test_tool"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "already have a generation in progress" in res.json()["detail"]


# --------------------------------------------------------------------------- #
# /generate: multi-image upload + max_input_images enforcement
# --------------------------------------------------------------------------- #

def test_generate_rejects_unknown_tool_before_any_upload_or_charge(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = None  # tool lookup finds nothing
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "ghost_tool"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert res.json()["detail"] == "Unknown tool"
    upload.assert_not_called()  # rejected before ever touching fal's CDN


def test_generate_rejects_inactive_tool_before_any_upload(gen_client, monkeypatch):
    """A tool_definitions row that exists but is_active=False (e.g. a
    coming-soon placeholder like magic_erase) must be rejected -- and
    rejected BEFORE _read_and_upload_images spends real fal upload quota,
    same reasoning as the unknown-tool and missing-field checks above."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        max_input_images=1, is_active=False, stage=0, param_schema=[],
    )
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "magic_erase"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert res.json()["detail"] == "This tool is not currently available"
    upload.assert_not_called()  # rejected before ever touching fal's CDN


def test_generate_rejects_active_tool_at_the_wrong_stage(gen_client, monkeypatch):
    """Same condition GET /tools/{feature_type}/schema checks (stage != 1 or
    not is_active) -- a tool flipped to is_active=True while still at
    stage=0 for internal testing must still be rejected here, not just
    hidden from the schema endpoint."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        max_input_images=1, is_active=True, stage=0, param_schema=[],
    )
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "some_tool_in_testing"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert res.json()["detail"] == "This tool is not currently available"
    upload.assert_not_called()


def test_generate_rejects_more_images_than_the_tools_max_input_images(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(max_input_images=1, stage=1,)
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "test_tool"},
        files=[
            ("images", ("one.png", b"one", "image/png")),
            ("images", ("two.png", b"two", "image/png")),
        ],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "at most 1" in res.json()["detail"]
    upload.assert_not_called()  # rejected before uploading anything to fal's CDN


# --------------------------------------------------------------------------- #
# /generate: schema validation + image checks must happen BEFORE any fal
# upload cost is paid -- a request that's going to be rejected anyway
# shouldn't burn real fal.ai upload quota every time.
# --------------------------------------------------------------------------- #

def test_generate_rejects_output_count_over_the_cap_before_any_upload(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(max_input_images=5, param_schema=[], stage=1,)
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "test_tool", "output_count": "9999"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "output_count" in res.json()["detail"]
    upload.assert_not_called()


def test_generate_rejects_listing_photoshoots_output_count_over_its_own_tighter_cap(gen_client, monkeypatch):
    """listing_photoshoot's own cap (8, see app/tools/listing_planner.py) is
    tighter than the generic MAX_OUTPUT_COUNT (20) -- must be rejected before
    the billable shot-planning vision call ever runs, not just eventually
    inside create_generation_batch."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(max_input_images=5, param_schema=[], stage=1,)
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)
    plan = MagicMock()
    monkeypatch.setattr("app.routes.generation.plan_listing_shots", plan)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "listing_photoshoot", "output_count": "9"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "8" in res.json()["detail"]
    upload.assert_not_called()
    plan.assert_not_called()  # rejected before the billable vision call


def test_generate_omitted_output_count_does_not_reject_with_500(gen_client, monkeypatch):
    """output_count left off the request entirely must not crash the
    MAX_OUTPUT_COUNT comparison (None > int) -- regression guard for the
    Form(1) -> Form(None) change that made tool.default_output_count
    actually reachable."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(max_input_images=5, param_schema=[], stage=1,)
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", MagicMock(return_value="https://fal.test/x.png"))
    monkeypatch.setattr(
        "app.routes.generation.create_generation_batch",
        MagicMock(return_value=[MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued")]),
    )

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "test_tool"},  # no output_count at all
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200


def test_generate_does_not_reject_listing_photoshoot_over_its_own_required_output_count_field(gen_client, monkeypatch):
    """Regression: listing_photoshoot's real param_schema declares its own
    "output_count" field as required (so the frontend renders a bounded
    number input) -- but output_count is a batch-size control resolved from
    its own dedicated Form field, never part of the input_params dict
    validate_input_params checks. Before the fix, EVERY listing_photoshoot
    request 400'd with "Missing required field: Number of Shots" regardless
    of what was sent, because that dict never carried an "output_count" key
    at all for validate_input_params to find."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        max_input_images=5, default_output_count=4, is_active=True, stage=1,
        param_schema=[
            {"name": "prompt", "type": "text", "label": "Prompt", "required": False},
            {"name": "aspect_ratio", "type": "select", "label": "Aspect Ratio", "default": "1:1",
             "options": ["1:1", "4:3", "3:4", "16:9", "9:16"], "required": True},
            {"name": "resolution", "type": "select", "label": "Resolution", "default": "1k",
             "options": ["1k", "2k", "4k"], "required": True},
            {"name": "quality", "type": "select", "label": "Quality", "default": "medium",
             "options": ["low", "medium", "high", "xhigh", "max"], "required": True},
            {"max": 8, "min": 1, "name": "output_count", "type": "number", "label": "Number of Shots",
             "default": 4, "required": True},
        ],
    )
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", MagicMock(return_value="https://fal.test/x.png"))
    plan = MagicMock(return_value=[{"shot_type": "hero", "prompt": "x"}] * 3)
    monkeypatch.setattr("app.routes.generation.plan_listing_shots", plan)
    monkeypatch.setattr(
        "app.routes.generation.create_generation_batch",
        MagicMock(return_value=[MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued") for _ in range(3)]),
    )

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)

    res = client.post(
        "/generate",
        data={
            "team_id": str(TEAM_ID), "feature_type": "listing_photoshoot",
            "aspect_ratio": "1:1", "resolution": "1k", "quality": "medium", "output_count": "3",
        },
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200


def test_generate_omitting_output_count_lets_listing_photoshoot_reach_its_own_default(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        max_input_images=5, default_output_count=4, param_schema=[], is_active=True, stage=1,
    )
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", MagicMock(return_value="https://fal.test/x.png"))
    plan = MagicMock(return_value=[{"shot_type": "hero", "prompt": "x"}] * 4)
    monkeypatch.setattr("app.routes.generation.plan_listing_shots", plan)
    monkeypatch.setattr(
        "app.routes.generation.create_generation_batch",
        MagicMock(return_value=[MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued") for _ in range(4)]),
    )

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "listing_photoshoot"},  # no output_count at all
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200
    plan.assert_called_once_with(db.query.return_value.filter.return_value.first.return_value, ["https://fal.test/x.png"], None, 4)


def test_generate_rejects_missing_required_field_before_any_upload(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        max_input_images=5, is_active=True, stage=1,
        param_schema=[{"name": "color", "type": "color", "label": "Color", "required": True}],
    )
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "recolor"},  # no "color" field sent
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "Missing required field: Color" in res.json()["detail"]
    upload.assert_not_called()  # rejected before ever touching fal's CDN


def test_generate_rejects_disallowed_image_content_type_before_any_upload(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(max_input_images=5, param_schema=[], stage=1,)
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "test_tool"},
        files=[("images", ("payload.exe", b"not-an-image", "application/x-msdownload"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "Unsupported image type" in res.json()["detail"]
    upload.assert_not_called()


def test_generate_rejects_oversized_image_before_any_upload(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(max_input_images=5, param_schema=[], stage=1,)
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)
    monkeypatch.setattr("app.routes.generation.MAX_IMAGE_SIZE_BYTES", 10)  # tiny, for the test

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "test_tool"},
        files=[("images", ("big.png", b"way more than ten bytes of data", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "MB limit" in res.json()["detail"]
    upload.assert_not_called()


def test_generate_uploads_every_image_and_passes_all_urls_through(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(max_input_images=3, stage=1,)

    upload = MagicMock(side_effect=["https://fal.test/1.png", "https://fal.test/2.png"])
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    captured = {}

    def fake_create_batch(db_, team_id, user_id, feature_type, input_params, output_count):
        captured["input_params"] = input_params
        return [MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued")]
    monkeypatch.setattr("app.routes.generation.create_generation_batch", fake_create_batch)

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "test_tool"},
        files=[
            ("images", ("one.png", b"one", "image/png")),
            ("images", ("two.png", b"two", "image/png")),
        ],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200
    assert upload.call_count == 2
    # image_urls is ALWAYS a list, in upload order, one URL per uploaded file
    assert captured["input_params"]["image_urls"] == ["https://fal.test/1.png", "https://fal.test/2.png"]


# --------------------------------------------------------------------------- #
# /generate response -- requestedCount/grantedCount/partial. create_generation_
# batch can grant fewer outputs than an explicit output_count when the team's
# shared credit balance falls short (see spend_credits_up_to); this is what
# lets the frontend/test console tell the user "you got 5, not 6" instead of
# silently under-delivering.
# --------------------------------------------------------------------------- #

def test_generate_response_reports_full_grant_when_every_output_is_created(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(max_input_images=3, param_schema=[], stage=1,)
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", MagicMock(return_value="https://fal.test/1.png"))
    monkeypatch.setattr(
        "app.routes.generation.create_generation_batch",
        MagicMock(return_value=[
            MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued") for _ in range(3)
        ]),
    )

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "test_tool", "output_count": "3"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200
    body = res.json()
    assert body["requestedCount"] == 3
    assert body["grantedCount"] == 3
    assert body["partial"] is False


def test_generate_response_reports_a_partial_grant_when_fewer_outputs_are_created(gen_client, monkeypatch):
    """create_generation_batch (mocked here) sized the batch down to 2 jobs
    even though the caller explicitly asked for 5 -- the response must say so
    rather than silently returning only 2 jobs with no explanation."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(max_input_images=3, param_schema=[], stage=1,)
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", MagicMock(return_value="https://fal.test/1.png"))
    monkeypatch.setattr(
        "app.routes.generation.create_generation_batch",
        MagicMock(return_value=[
            MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued") for _ in range(2)
        ]),
    )

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "test_tool", "output_count": "5"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200
    body = res.json()
    assert body["requestedCount"] == 5
    assert body["grantedCount"] == 2
    assert body["partial"] is True
    assert len(body["jobs"]) == 2


def test_generate_forwards_recolors_aspect_ratio_and_resolution_fields(gen_client, monkeypatch):
    """Real bug this guards against: recolor's param_schema grew "resolution"
    (standard/high, also what _resolve_credit_cost reads its credit_cost off
    of) but the /generate route's Form(...) signature was never updated to
    accept it -- FastAPI silently drops any multipart field with no matching
    Form parameter, so every recolor submission 400'd with "Missing required
    field: Resolution" even when the frontend sent it correctly."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        max_input_images=5, is_active=True, stage=1,
        param_schema=[
            {"name": "color", "type": "color", "label": "Color", "required": True},
            {"name": "aspect_ratio", "type": "select", "label": "Aspect Ratio", "required": True,
             "options": ["1:1", "16:9"]},
            {"name": "resolution", "type": "select", "label": "Resolution", "required": True,
             "options": [{"value": "standard", "credit_cost": 1}, {"value": "high", "credit_cost": 2}]},
        ],
    )
    upload = MagicMock(return_value="https://fal.test/uploaded.png")
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    captured = {}

    def fake_create_batch(db_, team_id, user_id, feature_type, input_params, output_count):
        captured["input_params"] = input_params
        return [MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued")]
    monkeypatch.setattr("app.routes.generation.create_generation_batch", fake_create_batch)

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)

    res = client.post(
        "/generate",
        data={
            "team_id": str(TEAM_ID), "feature_type": "recolor",
            "color": "#1874ec", "aspect_ratio": "16:9", "resolution": "high",
        },
        files=[("images", ("shirt.png", b"shirt", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200
    assert captured["input_params"]["aspect_ratio"] == "16:9"
    assert captured["input_params"]["resolution"] == "high"


def test_generate_forwards_idea_field_for_creative_photoshoot(gen_client, monkeypatch):
    """creative_photoshoot needs "idea" -- not one of Recolor's fields --
    forwarded through to create_generation_batch, same class of gap that
    previously blocked enhance_prompt's "prompt"/"source_feature_type"."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        max_input_images=5, is_active=True, stage=1,
        param_schema=[
            {"name": "idea", "type": "select", "label": "Idea", "options": ["Sci-Fi", "Luxury"], "required": False},
            {"name": "prompt", "type": "text", "label": "Prompt", "required": False},
        ],
    )
    upload = MagicMock(return_value="https://fal.test/uploaded.png")
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    captured = {}

    def fake_create_batch(db_, team_id, user_id, feature_type, input_params, output_count):
        captured["input_params"] = input_params
        return [MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued")]
    monkeypatch.setattr("app.routes.generation.create_generation_batch", fake_create_batch)

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "creative_photoshoot", "idea": "Sci-Fi"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200
    assert captured["input_params"]["idea"] == "Sci-Fi"


def test_generate_plans_shots_and_uses_per_job_overrides_for_listing_photoshoot(gen_client, monkeypatch):
    """listing_photoshoot is the one tool where /generate must NOT call
    create_generation_batch with a single shared input_params -- it plans N
    distinct shots first and passes per_job_overrides instead."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    tool = MagicMock(
        max_input_images=5, default_output_count=4, is_active=True, stage=1,
        param_schema=[{"name": "prompt", "type": "text", "label": "Prompt", "required": False}],
    )
    db.query.return_value.filter.return_value.first.return_value = tool
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", MagicMock(return_value="https://fal.test/uploaded.png"))

    planned_shots = [
        {"shot_type": "hero", "prompt": "Hero shot, studio lighting"},
        {"shot_type": "side", "prompt": "Side angle, studio lighting"},
    ]
    plan = MagicMock(return_value=planned_shots)
    monkeypatch.setattr("app.routes.generation.plan_listing_shots", plan)

    captured = {}

    def fake_create_batch(db_, team_id, user_id, feature_type, input_params, output_count=1, per_job_overrides=None, lock_already_held=False):
        captured["output_count"] = output_count
        captured["per_job_overrides"] = per_job_overrides
        return [MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued") for _ in (per_job_overrides or [None])]
    monkeypatch.setattr("app.routes.generation.create_generation_batch", fake_create_batch)

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "listing_photoshoot"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200
    plan.assert_called_once_with(tool, ["https://fal.test/uploaded.png"], None, 4)  # falls back to tool.default_output_count
    assert captured["per_job_overrides"] == planned_shots
    assert captured["output_count"] == 4


def test_generate_never_plans_shots_when_a_generation_is_already_in_progress(gen_client, monkeypatch):
    """Real bug found live: the lock used to be checked INSIDE
    create_generation_batch, which only runs AFTER plan_listing_shots -- a
    request that was going to be rejected anyway still burned a real,
    billable fal vision call for nothing. The lock must be checked first."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        max_input_images=5, default_output_count=4, param_schema=[], is_active=True, stage=1,
    )
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", MagicMock(return_value="https://fal.test/x.png"))
    monkeypatch.setattr("app.routes.generation.acquire_or_heal_generation_lock", lambda db, uid: False)
    plan = MagicMock()
    monkeypatch.setattr("app.routes.generation.plan_listing_shots", plan)
    create_batch = MagicMock()
    monkeypatch.setattr("app.routes.generation.create_generation_batch", create_batch)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "listing_photoshoot"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "already have a generation in progress" in res.json()["detail"]
    plan.assert_not_called()  # the whole point -- never burn the billable call
    create_batch.assert_not_called()


def test_generate_releases_the_lock_itself_when_shot_planning_fails_after_acquiring_it(gen_client, monkeypatch):
    """The route acquires the lock itself before planning (lock_already_held
    passes ownership to create_generation_batch only on success) -- if
    planning fails, the route must release it, or every subsequent request
    from this user would be wrongly blocked until the lock's TTL expires."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        max_input_images=5, default_output_count=4, param_schema=[], is_active=True, stage=1,
    )
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", MagicMock(return_value="https://fal.test/x.png"))
    monkeypatch.setattr("app.routes.generation.acquire_or_heal_generation_lock", lambda db, uid: True)
    monkeypatch.setattr("app.routes.generation.plan_listing_shots", MagicMock(side_effect=RuntimeError("fal is down")))
    release = MagicMock()
    monkeypatch.setattr("app.routes.generation.release_generation_lock", release)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "listing_photoshoot"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 503
    release.assert_called_once_with(fake_user.id)


def test_generate_surfaces_a_clean_503_when_shot_planning_fails(gen_client, monkeypatch):
    """The vision call is a real, billable external request -- a raw
    exception must not leak as an unhandled 500."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        max_input_images=5, default_output_count=4, param_schema=[], is_active=True, stage=1,
    )
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", MagicMock(return_value="https://fal.test/uploaded.png"))
    monkeypatch.setattr("app.routes.generation.plan_listing_shots", MagicMock(side_effect=RuntimeError("fal is down")))
    create_batch = MagicMock()
    monkeypatch.setattr("app.routes.generation.create_generation_batch", create_batch)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "listing_photoshoot"},
        files=[("images", ("test.png", b"fake-image-bytes", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 503
    create_batch.assert_not_called()  # never charged/created anything after the planning failure


# --------------------------------------------------------------------------- #
# model_shoot: 3-category image upload (model_image/garment_images/
# reference_images kept separate, never merged into the generic "images"
# field), and plan_model_shoot's blocked path creating zero jobs/zero credits.
# --------------------------------------------------------------------------- #

def _model_shoot_tool(max_input_images=10, default_output_count=2):
    return MagicMock(
        max_input_images=max_input_images, default_output_count=default_output_count,
        param_schema=[{"name": "prompt", "type": "text", "label": "Prompt", "required": False}],
        stage=1, is_active=True,
    )


# --------------------------------------------------------------------------- #
# GET /model-presets
# --------------------------------------------------------------------------- #

def test_get_model_presets_returns_id_name_and_thumbnail_url(gen_client):
    client, fake_user, db = gen_client
    # NOTE: MagicMock's own `name` kwarg sets its repr, not an attribute --
    # must be assigned separately.
    male = MagicMock(id="preset_male_1", thumbnail_url="https://cdn.test/male.png")
    male.name = "Male"
    female = MagicMock(id="preset_female_1", thumbnail_url="https://cdn.test/female.png")
    female.name = "Female"
    db.query.return_value.filter.return_value.all.return_value = [male, female]

    res = client.get("/model-presets", headers={"Authorization": "Bearer x"})

    assert res.status_code == 200
    assert res.json() == {
        "presets": [
            {"id": "preset_male_1", "name": "Male", "thumbnailUrl": "https://cdn.test/male.png"},
            {"id": "preset_female_1", "name": "Female", "thumbnailUrl": "https://cdn.test/female.png"},
        ]
    }


def test_get_model_presets_only_queries_active_rows(gen_client):
    """Confirms the query actually filters on is_active -- a mocked .all()
    can't prove the WHERE clause is right by itself, so this inspects the
    real SQLAlchemy expression the route builds."""
    client, fake_user, db = gen_client
    db.query.return_value.filter.return_value.all.return_value = []

    client.get("/model-presets", headers={"Authorization": "Bearer x"})

    filter_expression = db.query.return_value.filter.call_args.args[0]
    assert "is_active" in str(filter_expression)


def test_get_model_presets_requires_auth(gen_client):
    from app.deps import get_current_user
    from app.main import app

    client, fake_user, db = gen_client
    app.dependency_overrides.pop(get_current_user, None)  # simulate no token

    res = client.get("/model-presets")

    assert res.status_code == 401


def _model_shoot_db(tool, preset=None):
    """Distinguishes db.query(ToolDefinition) from db.query(ModelPreset) --
    needed once a single /generate call can look up both."""
    db = MagicMock()

    def query(model):
        q = MagicMock()
        name = getattr(model, "__name__", "")
        if name == "ToolDefinition":
            q.filter.return_value.first.return_value = tool
        elif name == "ModelPreset":
            q.filter.return_value.first.return_value = preset
        return q
    db.query.side_effect = query
    return db


def test_generate_rejects_model_shoot_with_neither_model_image_nor_preset(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = _model_shoot_tool()
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)
    plan = MagicMock()
    monkeypatch.setattr("app.routes.generation.plan_model_shoot", plan)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "model_shoot"},
        # no model_image, no model_preset_id -- only a garment image
        files=[("garment_images", ("shirt.png", b"shirt", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "exactly one of model_image, model_preset_id or model_source_job_id" in res.json()["detail"]
    upload.assert_not_called()  # rejected before any upload cost
    plan.assert_not_called()


def test_generate_rejects_model_shoot_with_both_model_image_and_preset(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = _model_shoot_tool()
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)
    plan = MagicMock()
    monkeypatch.setattr("app.routes.generation.plan_model_shoot", plan)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "model_shoot", "model_preset_id": "preset_male_1"},
        files=[("model_image", ("model.png", b"model", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "exactly one of model_image, model_preset_id or model_source_job_id" in res.json()["detail"]
    upload.assert_not_called()  # rejected before any upload cost -- neither is uploaded/looked up
    plan.assert_not_called()


def test_generate_404s_on_an_unknown_model_preset_id(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db_with_no_preset = _model_shoot_db(_model_shoot_tool(), preset=None)
    from app.core.database import get_db as real_get_db
    from app.main import app
    app.dependency_overrides[real_get_db] = lambda: db_with_no_preset
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)
    plan = MagicMock()
    monkeypatch.setattr("app.routes.generation.plan_model_shoot", plan)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "model_shoot", "model_preset_id": "preset_does_not_exist"},
        files=[("top_images", ("shirt.png", b"shirt", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 404
    assert "Unknown model preset" in res.json()["detail"]
    upload.assert_not_called()
    plan.assert_not_called()


def test_generate_404s_on_a_deactivated_model_preset(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    inactive_preset = MagicMock(id="preset_old", image_url="https://cdn.test/old.png", is_active=False)
    db_with_inactive_preset = _model_shoot_db(_model_shoot_tool(), preset=inactive_preset)
    from app.core.database import get_db as real_get_db
    from app.main import app
    app.dependency_overrides[real_get_db] = lambda: db_with_inactive_preset
    plan = MagicMock()
    monkeypatch.setattr("app.routes.generation.plan_model_shoot", plan)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "model_shoot", "model_preset_id": "preset_old"},
        files=[("top_images", ("shirt.png", b"shirt", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 404
    assert "Unknown model preset" in res.json()["detail"]
    plan.assert_not_called()


def test_generate_resolves_model_preset_id_to_its_image_url_and_skips_upload(gen_client, monkeypatch):
    """The real point of this feature: a preset's image_url is already a
    real hosted URL -- it must reach plan_model_shoot directly, with zero
    calls to upload_image_to_fal for the model reference slot."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    tool = _model_shoot_tool()
    preset = MagicMock(
        id="preset_male_1", name="Male",
        image_url="https://d2v5dzhdg4zhx3.cloudfront.net/graphics/9d5bac40-679b-4032-941c-7a778c3f8020.png",
        is_active=True,
    )
    db_with_preset = _model_shoot_db(tool, preset=preset)
    from app.core.database import get_db as real_get_db
    from app.main import app
    app.dependency_overrides[real_get_db] = lambda: db_with_preset

    upload = MagicMock(return_value="https://fal.test/garment.png")
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)
    plan = MagicMock(return_value={"blocked": False, "reason": "", "prompts": ["shot 1", "shot 2"]})
    monkeypatch.setattr("app.routes.generation.plan_model_shoot", plan)

    captured = {}

    def fake_create_batch(db_, team_id, user_id, feature_type, input_params, output_count=1, per_job_overrides=None, lock_already_held=False):
        captured["input_params"] = input_params
        return [MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued") for _ in (per_job_overrides or [None])]
    monkeypatch.setattr("app.routes.generation.create_generation_batch", fake_create_batch)

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "model_shoot", "model_preset_id": "preset_male_1"},
        files=[("top_images", ("shirt.png", b"shirt", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200
    plan.assert_called_once_with(
        tool, "https://d2v5dzhdg4zhx3.cloudfront.net/graphics/9d5bac40-679b-4032-941c-7a778c3f8020.png",
        [{"label": "Top", "image_urls": ["https://fal.test/garment.png"]}], [], None, 2,
    )
    assert captured["input_params"]["model_image"] == (
        "https://d2v5dzhdg4zhx3.cloudfront.net/graphics/9d5bac40-679b-4032-941c-7a778c3f8020.png"
    )
    # upload_image_to_fal was called exactly once -- for the garment image
    # only, never for the preset's model reference.
    upload.assert_called_once_with(b"shirt", "shirt.png", "image/png")


# --------------------------------------------------------------------------- #
# Reusing existing Library outputs as inputs (source_job_ids and friends)
# --------------------------------------------------------------------------- #

def _library_generate_setup(gen_client, monkeypatch, tool, resolved):
    """Shared wiring: team member, the given tool, a stubbed source-job
    resolver returning `resolved`, a capturing create_generation_batch."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = tool
    resolver = MagicMock(return_value=resolved)
    monkeypatch.setattr("app.routes.generation._resolve_source_job_urls", resolver)
    upload = MagicMock(return_value="https://fal.test/uploaded.png")
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)
    captured = {}

    def fake_create_batch(db_, team_id, user_id, feature_type, input_params, output_count=1, per_job_overrides=None, lock_already_held=False):
        captured["input_params"] = input_params
        return [MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued") for _ in (per_job_overrides or [None])]
    monkeypatch.setattr("app.routes.generation.create_generation_batch", fake_create_batch)
    monkeypatch.setattr("app.routes.generation.acquire_or_heal_generation_lock", lambda db, uid: True)

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)
    return client, resolver, upload, captured


def test_generate_accepts_source_job_ids_instead_of_uploads(gen_client, monkeypatch):
    """The Library "start a new shoot from these images" flow: no files at
    all, just ids of past outputs -- resolved to their URLs, zero uploads."""
    job_id = uuid.uuid4()
    tool = MagicMock(max_input_images=5, is_active=True, stage=1, param_schema=[])
    client, resolver, upload, captured = _library_generate_setup(
        gen_client, monkeypatch, tool, {job_id: "https://storage.test/out1.png"},
    )

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "recolor", "source_job_ids": [str(job_id)]},
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200
    assert captured["input_params"]["image_urls"] == ["https://storage.test/out1.png"]
    upload.assert_not_called()  # already-hosted output: no fal upload cost


def test_generate_combines_uploads_and_source_job_ids_uploads_first(gen_client, monkeypatch):
    job_id = uuid.uuid4()
    tool = MagicMock(max_input_images=5, is_active=True, stage=1, param_schema=[])
    client, resolver, upload, captured = _library_generate_setup(
        gen_client, monkeypatch, tool, {job_id: "https://storage.test/out1.png"},
    )

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "recolor", "source_job_ids": [str(job_id)]},
        files=[("images", ("new.png", b"new", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200
    assert captured["input_params"]["image_urls"] == [
        "https://fal.test/uploaded.png", "https://storage.test/out1.png",
    ]


def test_generate_counts_source_job_ids_toward_the_tools_image_cap(gen_client, monkeypatch):
    ids = [uuid.uuid4(), uuid.uuid4()]
    tool = MagicMock(max_input_images=2, is_active=True, stage=1, param_schema=[])
    client, resolver, upload, captured = _library_generate_setup(
        gen_client, monkeypatch, tool, {i: f"https://storage.test/{n}.png" for n, i in enumerate(ids)},
    )

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "recolor", "source_job_ids": [str(i) for i in ids]},
        files=[("images", ("new.png", b"new", "image/png"))],  # 1 upload + 2 library = 3 > cap of 2
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "at most 2 image" in res.json()["detail"]
    resolver.assert_not_called()  # cap rejected before even touching the DB
    upload.assert_not_called()


def test_generate_rejects_an_unavailable_source_job_before_any_upload(gen_client, monkeypatch):
    """A selection that includes another team's job, a failed job or an
    unknown id must fail the whole request BEFORE the sibling upload costs
    real fal quota."""
    from fastapi import HTTPException

    tool = MagicMock(max_input_images=5, is_active=True, stage=1, param_schema=[])
    client, resolver, upload, captured = _library_generate_setup(gen_client, monkeypatch, tool, {})
    resolver.side_effect = HTTPException(status_code=400, detail="One or more selected images are not available")

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "recolor", "source_job_ids": [str(uuid.uuid4())]},
        files=[("images", ("new.png", b"new", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert res.json()["detail"] == "One or more selected images are not available"
    upload.assert_not_called()
    assert "input_params" not in captured  # never reached job creation


def test_generate_rejects_a_malformed_source_job_id_with_422(gen_client, monkeypatch):
    tool = MagicMock(max_input_images=5, is_active=True, stage=1, param_schema=[])
    client, resolver, upload, captured = _library_generate_setup(gen_client, monkeypatch, tool, {})

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "recolor", "source_job_ids": ["not-a-uuid"]},
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 422


def test_generate_model_shoot_from_library_images_skips_upload(gen_client, monkeypatch):
    """The model can be a past model_shoot_generate_model result, and garments
    can be past outputs too -- all Library-sourced, zero fal uploads, each in
    its own slot with the same grouping as uploaded images."""
    model_job, top_job = uuid.uuid4(), uuid.uuid4()
    tool = _model_shoot_tool()
    client, resolver, upload, captured = _library_generate_setup(
        gen_client, monkeypatch, tool,
        {model_job: "https://storage.test/generated-model.png", top_job: "https://storage.test/top.png"},
    )
    plan = MagicMock(return_value={"blocked": False, "reason": "", "prompts": ["shot 1"]})
    monkeypatch.setattr("app.routes.generation.plan_model_shoot", plan)

    res = client.post(
        "/generate",
        data={
            "team_id": str(TEAM_ID), "feature_type": "model_shoot",
            "model_source_job_id": str(model_job), "top_source_job_ids": [str(top_job)],
        },
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200
    upload.assert_not_called()
    plan.assert_called_once_with(
        tool, "https://storage.test/generated-model.png",
        [{"label": "Top", "image_urls": ["https://storage.test/top.png"]}], [], None, 2,
    )
    assert captured["input_params"]["model_image"] == "https://storage.test/generated-model.png"


def test_generate_model_shoot_requires_exactly_one_model_source_including_library(gen_client, monkeypatch):
    """Sending BOTH a preset and a Library model is ambiguous -- rejected,
    same as sending both an upload and a preset always was."""
    top_job = uuid.uuid4()
    tool = _model_shoot_tool()
    client, resolver, upload, captured = _library_generate_setup(
        gen_client, monkeypatch, tool, {top_job: "https://storage.test/top.png"},
    )

    res = client.post(
        "/generate",
        data={
            "team_id": str(TEAM_ID), "feature_type": "model_shoot",
            "model_preset_id": "preset_x", "model_source_job_id": str(uuid.uuid4()),
            "top_source_job_ids": [str(top_job)],
        },
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "exactly one of model_image, model_preset_id or model_source_job_id" in res.json()["detail"]


def test_generate_model_shoot_library_garments_count_toward_the_garment_requirement(gen_client, monkeypatch):
    """A Library-only garment must satisfy the 'at least one garment' rule,
    and its absence must still fail it."""
    model_job = uuid.uuid4()
    tool = _model_shoot_tool()
    client, resolver, upload, captured = _library_generate_setup(
        gen_client, monkeypatch, tool, {model_job: "https://storage.test/m.png"},
    )

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "model_shoot", "model_source_job_id": str(model_job)},
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "At least one garment image is required" in res.json()["detail"]


@pytest.mark.real_db
def test_resolve_source_job_urls_real_db_enforces_team_status_and_output():
    """The security-relevant part, against real Postgres: only THIS team's
    completed jobs that actually have an output can be reused. Anything else
    -- another team's job, a failed job, an output-less job, an unknown id --
    fails the whole request with one generic message (so ids in other teams
    can't be probed)."""
    from fastapi import HTTPException

    from app.core.database import SessionLocal
    from app.models.generation_job import GenerationJob
    from app.models.team import Team
    from app.models.user import User
    from app.routes.generation import _resolve_source_job_urls
    from app.services import teams as teams_svc

    db = SessionLocal()
    user = db.query(User).first()
    mine, other = Team(name="Library source mine"), Team(name="Library source other")
    db.add_all([mine, other])
    db.flush()

    def job(team, status, output):
        j = GenerationJob(
            team_id=team.id, user_id=user.id, feature_type="recolor", status=status,
            input_params={}, credits_charged=1, output_url=output,
        )
        db.add(j)
        db.flush()
        return j.id

    ok1 = job(mine, "completed", "https://storage.test/ok1.png")
    ok2 = job(mine, "completed", "https://storage.test/ok2.png")
    failed = job(mine, "failed", None)
    no_output = job(mine, "completed", None)
    foreign = job(other, "completed", "https://storage.test/foreign.png")
    db.commit()

    try:
        # happy path, including a duplicate id and an empty selection
        assert _resolve_source_job_urls(db, mine.id, [ok1, ok2, ok1]) == {
            ok1: "https://storage.test/ok1.png", ok2: "https://storage.test/ok2.png",
        }
        assert _resolve_source_job_urls(db, mine.id, []) == {}

        for bad in ([failed], [no_output], [foreign], [uuid.uuid4()], [ok1, foreign]):
            with pytest.raises(HTTPException) as exc:
                _resolve_source_job_urls(db, mine.id, bad)
            assert exc.value.status_code == 400
            assert exc.value.detail == "One or more selected images are not available"
    finally:
        teams_svc.purge_team(db, mine.id)
        teams_svc.purge_team(db, other.id)
        db.close()


def test_generate_enforces_total_image_cap_across_all_three_model_shoot_categories(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = _model_shoot_tool(max_input_images=2)
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "model_shoot"},
        files=[
            ("model_image", ("model.png", b"model", "image/png")),
            ("top_images", ("shirt.png", b"shirt", "image/png")),
            ("reference_images", ("pose.png", b"pose", "image/png")),  # 1 + 1 + 1 = 3 > cap of 2
        ],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "image(s) total" in res.json()["detail"]
    upload.assert_not_called()


def test_generate_uploads_model_shoots_garment_groups_separately(gen_client, monkeypatch):
    """The real gap this closes: a flat garment image list gives the vision
    call no way to tell "2 angles of one top" apart from "2 separate
    garments" -- model_image/top_images/bottom_images/extra_images_*/
    reference_images must reach plan_model_shoot (and job.input_params) as
    distinct, LABELED garment groups, never flattened into one list."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    tool = _model_shoot_tool()
    db.query.return_value.filter.return_value.first.return_value = tool

    upload = MagicMock(side_effect=[
        "https://fal.test/model.png", "https://fal.test/shirt.png",
        "https://fal.test/pants.png", "https://fal.test/watch.png",
        "https://fal.test/pose.png",
    ])
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    plan = MagicMock(return_value={"blocked": False, "reason": "", "prompts": ["shot 1", "shot 2"]})
    monkeypatch.setattr("app.routes.generation.plan_model_shoot", plan)

    captured = {}

    def fake_create_batch(db_, team_id, user_id, feature_type, input_params, output_count=1, per_job_overrides=None, lock_already_held=False):
        captured["input_params"] = input_params
        captured["per_job_overrides"] = per_job_overrides
        return [MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued") for _ in (per_job_overrides or [None])]
    monkeypatch.setattr("app.routes.generation.create_generation_batch", fake_create_batch)

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)

    res = client.post(
        "/generate",
        data={
            "team_id": str(TEAM_ID), "feature_type": "model_shoot", "prompt": "studio look",
            "extra_label_1": "Watch",
        },
        files=[
            ("model_image", ("model.png", b"model", "image/png")),
            ("top_images", ("shirt.png", b"shirt", "image/png")),
            ("bottom_images", ("pants.png", b"pants", "image/png")),
            ("extra_images_1", ("watch.png", b"watch", "image/png")),
            ("reference_images", ("pose.png", b"pose", "image/png")),
        ],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200
    expected_garments = [
        {"label": "Top", "image_urls": ["https://fal.test/shirt.png"]},
        {"label": "Bottom", "image_urls": ["https://fal.test/pants.png"]},
        {"label": "Watch", "image_urls": ["https://fal.test/watch.png"]},
    ]
    plan.assert_called_once_with(
        tool, "https://fal.test/model.png",
        expected_garments,
        ["https://fal.test/pose.png"],
        "studio look", 2,  # falls back to tool.default_output_count
    )
    assert captured["input_params"]["model_image"] == "https://fal.test/model.png"
    assert captured["input_params"]["garments"] == expected_garments
    assert captured["input_params"]["reference_images"] == ["https://fal.test/pose.png"]
    assert "image_urls" not in captured["input_params"]  # never the generic flat field for this tool
    assert "garment_images" not in captured["input_params"]  # the old flat field is gone
    assert captured["per_job_overrides"] == [{"prompt": "shot 1"}, {"prompt": "shot 2"}]


def test_generate_omits_an_empty_extra_slot_from_the_garments_list(gen_client, monkeypatch):
    """An extra slot with no images uploaded must not become a garment group
    at all, regardless of whether a label was typed into it."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    tool = _model_shoot_tool()
    db.query.return_value.filter.return_value.first.return_value = tool

    upload = MagicMock(side_effect=["https://fal.test/model.png", "https://fal.test/shirt.png"])
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)
    plan = MagicMock(return_value={"blocked": False, "reason": "", "prompts": ["shot 1", "shot 2"]})
    monkeypatch.setattr("app.routes.generation.plan_model_shoot", plan)
    monkeypatch.setattr(
        "app.routes.generation.create_generation_batch",
        lambda *a, **k: [MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued") for _ in range(2)],
    )

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)

    res = client.post(
        "/generate",
        # "Hat" typed in but no extra_images_2 uploaded -- must be ignored.
        data={"team_id": str(TEAM_ID), "feature_type": "model_shoot", "extra_label_2": "Hat"},
        files=[
            ("model_image", ("model.png", b"model", "image/png")),
            ("top_images", ("shirt.png", b"shirt", "image/png")),
        ],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200
    plan.assert_called_once_with(
        tool, "https://fal.test/model.png",
        [{"label": "Top", "image_urls": ["https://fal.test/shirt.png"]}],
        [], None, 2,
    )


def test_generate_blocked_model_shoot_plan_creates_no_job_and_spends_no_credits(gen_client, monkeypatch):
    """Item 3's real requirement: if plan_model_shoot reports blocked=True,
    /generate must reject with a 400 immediately -- no GenerationJob row, no
    credits spent, and the lock it acquired for planning must be released so
    the user isn't wrongly locked out afterward."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = _model_shoot_tool()
    monkeypatch.setattr(
        "app.routes.generation.upload_image_to_fal",
        MagicMock(return_value="https://fal.test/x.png"),
    )
    plan = MagicMock(return_value={
        "blocked": True,
        "reason": "Blocked: intimate apparel combined with an ambiguous-age model.",
        "prompts": [],
    })
    monkeypatch.setattr("app.routes.generation.plan_model_shoot", plan)
    create_batch = MagicMock()
    monkeypatch.setattr("app.routes.generation.create_generation_batch", create_batch)
    release = MagicMock()
    monkeypatch.setattr("app.routes.generation.release_generation_lock", release)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "model_shoot"},
        files=[
            ("model_image", ("model.png", b"model", "image/png")),
            ("top_images", ("lingerie.png", b"lingerie", "image/png")),
        ],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    # The LLM-authored reason must NEVER reach the response body -- only the
    # fixed, generic constant. See test_generate_blocked_model_shoots_real_reason_is_logged_not_returned
    # below for where the real reason is expected to go instead (server logs).
    from app.routes.generation import MODEL_SHOOT_BLOCKED_MESSAGE
    assert res.json()["detail"] == MODEL_SHOOT_BLOCKED_MESSAGE
    assert "intimate apparel" not in res.text
    create_batch.assert_not_called()  # no job, no credits -- rejected before either
    release.assert_called_once_with(fake_user.id)


def test_generate_blocked_model_shoots_real_reason_is_logged_not_returned(gen_client, monkeypatch, caplog):
    """The shoot-analyst's actual free-text reason must still reach server
    logs for internal visibility, even though it's withheld from the
    response body (see the test above)."""
    import logging as _logging

    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = _model_shoot_tool()
    monkeypatch.setattr(
        "app.routes.generation.upload_image_to_fal",
        MagicMock(return_value="https://fal.test/x.png"),
    )
    monkeypatch.setattr("app.routes.generation.plan_model_shoot", MagicMock(return_value={
        "blocked": True,
        "reason": "Blocked: intimate apparel combined with an ambiguous-age model.",
        "prompts": [],
    }))
    monkeypatch.setattr("app.routes.generation.create_generation_batch", MagicMock())
    monkeypatch.setattr("app.routes.generation.release_generation_lock", MagicMock())

    with caplog.at_level(_logging.INFO, logger="app.routes.generation"):
        res = client.post(
            "/generate",
            data={"team_id": str(TEAM_ID), "feature_type": "model_shoot"},
            files=[
                ("model_image", ("model.png", b"model", "image/png")),
                ("top_images", ("lingerie.png", b"lingerie", "image/png")),
            ],
            headers={"Authorization": "Bearer x"},
        )

    assert res.status_code == 400
    assert "intimate apparel combined with an ambiguous-age model" in caplog.text
    assert str(fake_user.id) in caplog.text  # tagged with the user/request context


def test_generate_never_plans_model_shoot_when_a_generation_is_already_in_progress(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = _model_shoot_tool()
    monkeypatch.setattr(
        "app.routes.generation.upload_image_to_fal",
        MagicMock(return_value="https://fal.test/x.png"),
    )
    monkeypatch.setattr("app.routes.generation.acquire_or_heal_generation_lock", lambda db, uid: False)
    plan = MagicMock()
    monkeypatch.setattr("app.routes.generation.plan_model_shoot", plan)
    create_batch = MagicMock()
    monkeypatch.setattr("app.routes.generation.create_generation_batch", create_batch)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "model_shoot"},
        files=[
            ("model_image", ("model.png", b"model", "image/png")),
            ("top_images", ("shirt.png", b"shirt", "image/png")),
        ],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert "already have a generation in progress" in res.json()["detail"]
    plan.assert_not_called()  # never burn the billable vision call
    create_batch.assert_not_called()


def test_generate_accepts_enhance_prompt_with_no_images(gen_client, monkeypatch):
    """enhance_prompt has max_input_images=0 and needs "prompt" +
    "source_feature_type" instead of any of Recolor's fields -- confirms the
    route actually accepts both, requires no file upload, and forwards them
    through to create_generation_batch untouched."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        max_input_images=0, is_active=True, stage=1,
        param_schema=[
            {"name": "prompt", "type": "text", "label": "Prompt", "required": True},
            {"name": "source_feature_type", "type": "text", "label": "Source tool", "required": True},
        ],
    )
    upload = MagicMock()
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)

    captured = {}

    def fake_create_batch(db_, team_id, user_id, feature_type, input_params, output_count):
        captured["input_params"] = input_params
        return [MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued")]
    monkeypatch.setattr("app.routes.generation.create_generation_batch", fake_create_batch)

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)

    res = client.post(
        "/generate",
        data={
            "team_id": str(TEAM_ID), "feature_type": "enhance_prompt",
            "prompt": "a cool sneaker", "source_feature_type": "recolor",
        },
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200
    upload.assert_not_called()  # no images -- must never try to upload anything
    assert captured["input_params"]["prompt"] == "a cool sneaker"
    assert captured["input_params"]["source_feature_type"] == "recolor"
    assert captured["input_params"]["image_urls"] == []


def test_get_job_returns_output_text_for_enhance_prompt(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    job = MagicMock(
        id=uuid.uuid4(), team_id=TEAM_ID, status="completed",
        output_url=None, output_text="an enhanced, more vivid prompt",
        error_message=None, credits_charged=0,
    )
    db.query.return_value.filter.return_value.first.return_value = job
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)

    res = client.get(f"/jobs/{job.id}", headers={"Authorization": "Bearer x"})

    assert res.status_code == 200
    body = res.json()
    assert body["outputText"] == "an enhanced, more vivid prompt"
    assert body["outputUrl"] is None


def test_get_job_rejects_non_team_member(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    job = MagicMock(team_id=TEAM_ID)
    db.query.return_value.filter.return_value.first.return_value = job
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: False)

    res = client.get(f"/jobs/{uuid.uuid4()}", headers={"Authorization": "Bearer x"})

    assert res.status_code == 403


def test_get_job_404_when_missing(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    db.query.return_value.filter.return_value.first.return_value = None

    res = client.get(f"/jobs/{uuid.uuid4()}", headers={"Authorization": "Bearer x"})

    assert res.status_code == 404


def test_get_job_returns_status_for_team_member(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    job = MagicMock(
        id=uuid.uuid4(), team_id=TEAM_ID, status="queued",
        output_url=None, error_message=None, credits_charged=5,
    )
    db.query.return_value.filter.return_value.first.return_value = job
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)

    res = client.get(f"/jobs/{job.id}", headers={"Authorization": "Bearer x"})

    assert res.status_code == 200
    assert res.json()["status"] == "queued"
    assert res.json()["creditsCharged"] == 5


def test_get_job_returns_shot_type_for_listing_photoshoot(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    job = MagicMock(
        id=uuid.uuid4(), team_id=TEAM_ID, status="completed",
        output_url="https://x/1.png", error_message=None, credits_charged=2,
        input_params={"prompt": "front angle", "shot_type": "front"},
    )
    db.query.return_value.filter.return_value.first.return_value = job
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)

    res = client.get(f"/jobs/{job.id}", headers={"Authorization": "Bearer x"})

    assert res.json()["shotType"] == "front"


def test_get_job_shot_type_is_null_for_model_shoot(gen_client, monkeypatch):
    """model_shoot has no per-prompt label of its own -- shot_type must be
    null there, never a guessed/fabricated value."""
    client, fake_user, db = gen_client
    job = MagicMock(
        id=uuid.uuid4(), team_id=TEAM_ID, status="completed",
        output_url="https://x/1.png", error_message=None, credits_charged=5,
        input_params={"prompt": "a pose description"},
    )
    db.query.return_value.filter.return_value.first.return_value = job
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)

    res = client.get(f"/jobs/{job.id}", headers={"Authorization": "Bearer x"})

    assert res.json()["shotType"] is None


# --------------------------------------------------------------------------- #
# GET /batches/{batch_id}
# --------------------------------------------------------------------------- #

def test_get_batch_returns_status_for_every_job_in_the_batch(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    batch_id = uuid.uuid4()
    j1 = MagicMock(id=uuid.uuid4(), team_id=TEAM_ID, batch_id=batch_id,
                    status="completed", output_url="https://our-storage.test/1.png", error_message=None,
                    credits_charged=2, input_params={"shot_type": "detail"})
    j2 = MagicMock(id=uuid.uuid4(), team_id=TEAM_ID, batch_id=batch_id,
                    status="queued", output_url=None, error_message=None, credits_charged=2)
    j3 = MagicMock(id=uuid.uuid4(), team_id=TEAM_ID, batch_id=batch_id,
                    status="failed", output_url=None, error_message="fal reported success but returned no image",
                    credits_charged=1)
    db.query.return_value.filter.return_value.all.return_value = [j1, j2, j3]
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)

    res = client.get(f"/batches/{batch_id}", headers={"Authorization": "Bearer x"})

    assert res.status_code == 200
    body = res.json()
    assert body["batchId"] == str(batch_id)
    by_id = {j["jobId"]: j for j in body["jobs"]}
    assert by_id[str(j1.id)]["status"] == "completed"
    assert by_id[str(j1.id)]["outputUrl"] == "https://our-storage.test/1.png"
    assert by_id[str(j1.id)]["creditsCharged"] == 2
    assert by_id[str(j1.id)]["shotType"] == "detail"
    assert by_id[str(j2.id)]["status"] == "queued"
    assert by_id[str(j3.id)]["status"] == "failed"
    # the raw internal text is NEVER exposed -- users get the generic message
    assert by_id[str(j3.id)]["errorMessage"] == "We couldn't generate this image. Please try again."
    assert by_id[str(j3.id)]["creditsCharged"] == 1


def test_get_batch_404_when_no_jobs_found(gen_client):
    client, fake_user, db = gen_client
    db.query.return_value.filter.return_value.all.return_value = []

    res = client.get(f"/batches/{uuid.uuid4()}", headers={"Authorization": "Bearer x"})

    assert res.status_code == 404


def test_get_batch_rejects_non_team_member(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    job = MagicMock(id=uuid.uuid4(), team_id=TEAM_ID, status="queued", output_url=None, error_message=None)
    db.query.return_value.filter.return_value.all.return_value = [job]
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: False)

    res = client.get(f"/batches/{uuid.uuid4()}", headers={"Authorization": "Bearer x"})

    assert res.status_code == 403


# --------------------------------------------------------------------------- #
# creative_photoshoot: at least one of idea / prompt is required
# --------------------------------------------------------------------------- #

def _creative_client(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        max_input_images=3, is_active=True, stage=1, default_output_count=1,
        param_schema=[
            {"name": "idea", "type": "select", "label": "Idea", "options": ["Sci-Fi", "Luxury"], "required": False},
            {"name": "prompt", "type": "text", "label": "Prompt", "required": False},
        ],
    )
    upload = MagicMock(return_value="https://fal.test/up.png")
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", upload)
    created = []

    def fake_create_batch(db_, team_id, user_id, feature_type, input_params, output_count=1, **kw):
        created.append(input_params)
        return [MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued")]
    monkeypatch.setattr("app.routes.generation.create_generation_batch", fake_create_batch)

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)
    return client, upload, created


@pytest.mark.parametrize("extra", [{}, {"idea": ""}, {"prompt": "   "}, {"idea": "", "prompt": "   "}])
def test_creative_rejects_neither_idea_nor_prompt_before_upload_or_charge(gen_client, monkeypatch, extra):
    client, upload, created = _creative_client(gen_client, monkeypatch)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "creative_photoshoot", **extra},
        files=[("images", ("p.png", b"img", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 400
    assert res.json()["detail"] == "Describe your scene or pick an idea"
    upload.assert_not_called()   # rejected before spending fal upload quota
    assert created == []          # and before any job/credit reservation


@pytest.mark.parametrize("extra", [{"idea": "Luxury"}, {"prompt": "on a marble table"}, {"idea": "Sci-Fi", "prompt": "neon"}])
def test_creative_accepts_idea_or_prompt_or_both(gen_client, monkeypatch, extra):
    client, upload, created = _creative_client(gen_client, monkeypatch)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "creative_photoshoot", **extra},
        files=[("images", ("p.png", b"img", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )

    assert res.status_code == 200
    assert len(created) == 1


def test_idea_or_prompt_rule_is_creative_only(gen_client, monkeypatch):
    """Other tools keep their own rules -- a tool with neither field must not
    be blocked by creative_photoshoot's requirement."""
    client, fake_user, db = gen_client
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        max_input_images=3, is_active=True, stage=1, default_output_count=1, param_schema=[],
    )
    monkeypatch.setattr("app.routes.generation.upload_image_to_fal", MagicMock(return_value="https://fal.test/x.png"))
    monkeypatch.setattr(
        "app.routes.generation.create_generation_batch",
        lambda *a, **k: [MagicMock(id=uuid.uuid4(), batch_id=uuid.uuid4(), status="queued")],
    )

    async def fake_get_arq_pool():
        pool = MagicMock()

        async def enqueue_job(*a, **k):
            return None
        pool.enqueue_job = enqueue_job
        return pool
    monkeypatch.setattr("app.routes.generation.get_arq_pool", fake_get_arq_pool)

    res = client.post(
        "/generate",
        data={"team_id": str(TEAM_ID), "feature_type": "recolor"},
        files=[("images", ("p.png", b"img", "image/png"))],
        headers={"Authorization": "Bearer x"},
    )
    assert res.status_code == 200



# --------------------------------------------------------------------------- #
# errorMessage on /jobs and /batches never carries raw provider/internal text
# --------------------------------------------------------------------------- #

RAW_INTERNAL_ERRORS = [
    "fal submit failed: The read operation timed out",
    "fal.ai generation failed [422]",
    "Unexpected status code: 422",
    "fal reported success but returned no image",
]


@pytest.mark.parametrize("raw", RAW_INTERNAL_ERRORS)
def test_get_job_never_exposes_raw_internal_errors(gen_client, monkeypatch, raw):
    client, fake_user, db = gen_client
    job = MagicMock(
        id=uuid.uuid4(), team_id=TEAM_ID, status="failed", output_url=None,
        output_text=None, error_message=raw, credits_charged=1, input_params={},
    )
    db.query.return_value.filter.return_value.first.return_value = job
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)

    res = client.get(f"/jobs/{job.id}", headers={"Authorization": "Bearer x"})

    assert res.status_code == 200
    assert res.json()["errorMessage"] == "We couldn't generate this image. Please try again."
    assert raw not in res.text          # not anywhere in the body, not just that field
    assert "fal" not in res.json()["errorMessage"].lower()


@pytest.mark.parametrize("raw", RAW_INTERNAL_ERRORS)
def test_get_batch_never_exposes_raw_internal_errors(gen_client, monkeypatch, raw):
    client, fake_user, db = gen_client
    batch_id = uuid.uuid4()
    bad = MagicMock(id=uuid.uuid4(), team_id=TEAM_ID, batch_id=batch_id, status="failed",
                    output_url=None, output_text=None, error_message=raw, credits_charged=1, input_params={})
    ok = MagicMock(id=uuid.uuid4(), team_id=TEAM_ID, batch_id=batch_id, status="completed",
                   output_url="https://storage.test/ok.png", output_text=None, error_message=None,
                   credits_charged=1, input_params={})
    db.query.return_value.filter.return_value.all.return_value = [bad, ok]
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)

    res = client.get(f"/batches/{batch_id}", headers={"Authorization": "Bearer x"})

    by_id = {j["jobId"]: j for j in res.json()["jobs"]}
    assert by_id[str(bad.id)]["errorMessage"] == "We couldn't generate this image. Please try again."
    assert by_id[str(ok.id)]["errorMessage"] is None      # healthy sibling carries no error
    assert raw not in res.text


def test_a_timeout_is_shown_as_a_friendly_timeout_not_the_internal_wording(gen_client, monkeypatch):
    client, fake_user, db = gen_client
    job = MagicMock(
        id=uuid.uuid4(), team_id=TEAM_ID, status="failed", output_url=None, output_text=None,
        error_message="Generation timed out — no response received", credits_charged=1, input_params={},
    )
    db.query.return_value.filter.return_value.first.return_value = job
    monkeypatch.setattr("app.routes.generation.is_team_member", lambda db, tid, uid: True)

    res = client.get(f"/jobs/{job.id}", headers={"Authorization": "Bearer x"})

    assert res.json()["errorMessage"] == "This took longer than expected. Please try again."
