import logging
from datetime import datetime, timedelta
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.limiter import limiter
from app.deps import get_current_user
from app.models.user import User
from app.services.teams import (
    create_team,
    soft_delete_team,
    restore_team,
    notify_team_restored,
    get_team,
    is_team_member,
    is_team_owner,
    is_team_owner_including_deleted,
    get_team_members_page,
    list_user_teams,
    remove_member,
    rename_team,
    get_team_billing,
    GRACE_PERIOD_PUBLIC_DAYS,
)
from app.services.generation import list_team_generations
from app.services.usage import get_team_usage
from app.schemas.teams import BillingHistoryOut, TeamBillingOut
from app.services.ledger import DEFAULT_HISTORY_PAGE_SIZE, list_billing_history
from app.services.team_invites import (
    accept_invite, cancel_invite, create_invite, list_pending_invites,
    InviteEmailRateLimitedError, TeamFullError,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["teams"])


# --- request bodies --------------------------------------------------------

class InviteRequest(BaseModel):
    email: str
    role: Literal["editor", "owner"] = "editor"


class RenameTeamRequest(BaseModel):
    name: str


class CreateTeamRequest(BaseModel):
    name: str


def _require_owner(db: Session, team_id: UUID, user: User) -> None:
    if not is_team_owner(db, team_id, user.id):
        raise HTTPException(status_code=403, detail="Only the team owner can do this")


# --- reads ---------------------------------------------------------------

@router.get("/teams")
@limiter.limit("20/minute")
def my_teams(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return {"teams": list_user_teams(db, user.id)}


@router.post("/teams")
@limiter.limit("10/minute")
def create_team_route(
    request: Request,
    payload: CreateTeamRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        team = create_team(db, user.id, payload.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return team


@router.get("/teams/{team_id}/members")
@limiter.limit("30/minute")
def team_members(
    request: Request,
    team_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not is_team_member(db, team_id, user.id):
        raise HTTPException(status_code=403, detail="You are not a member of this team")

    page = get_team_members_page(db, team_id)
    if page is None:
        raise HTTPException(status_code=404, detail="Team not found")

    return page


# --- invites -----------------------------------------------------------

@router.post("/teams/{team_id}/invite")
@limiter.limit("10/minute")
def invite_to_team(
    request: Request,
    team_id: UUID,
    payload: InviteRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_owner(db, team_id, user)

    try:
        invite = create_invite(db, team_id, payload.email, user.id, payload.role)
    except TeamFullError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except InviteEmailRateLimitedError as e:
        raise HTTPException(status_code=429, detail=str(e))

    return {"sent": True, "email": invite.email, "role": invite.role}


@router.get("/teams/{team_id}/invites")
@limiter.limit("30/minute")
def pending_invites(
    request: Request,
    team_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_owner(db, team_id, user)
    return {"invites": list_pending_invites(db, team_id)}


@router.delete("/teams/{team_id}/invites/{invite_id}")
@limiter.limit("10/minute")
def cancel_team_invite(
    request: Request,
    team_id: UUID,
    invite_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_owner(db, team_id, user)

    try:
        cancel_invite(db, team_id, invite_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"cancelled": str(invite_id)}


@router.post("/invites/{token}/accept")
@limiter.limit("10/minute")
def accept_team_invite(
    request: Request,
    token: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        invite = accept_invite(db, token, user.id, user.email)
    except TeamFullError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"joinedTeamId": str(invite.team_id), "role": invite.role}


# --- owner management -------------------------------------------------

@router.patch("/teams/{team_id}")
@limiter.limit("10/minute")
def patch_team(
    request: Request,
    team_id: UUID,
    payload: RenameTeamRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_owner(db, team_id, user)

    team = get_team(db, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    try:
        rename_team(db, team, payload.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"id": str(team.id), "name": team.name}


@router.delete("/teams/{team_id}/members/{member_user_id}")
@limiter.limit("10/minute")
def remove_team_member(
    request: Request,
    team_id: UUID,
    member_user_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_owner(db, team_id, user)

    try:
        remove_member(db, team_id, member_user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"removed": str(member_user_id)}


@router.delete("/teams/{team_id}")
@limiter.limit("5/minute")
def delete_team_route(
    request: Request,
    team_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    # Deliberately NOT _require_owner/is_team_owner here -- those already
    # treat a soft-deleted team as inaccessible, which would misreport
    # "you're not the owner" for a second delete attempt instead of the
    # real "already scheduled for deletion" error soft_delete_team raises.
    if not is_team_owner_including_deleted(db, team_id, user.id):
        raise HTTPException(status_code=403, detail="Only the team owner can do this")

    try:
        deleted_at = soft_delete_team(db, team_id)
    except ValueError as e:
        status = 404 if "not found" in str(e).lower() else 400
        raise HTTPException(status_code=status, detail=str(e))

    recoverable_until = deleted_at + timedelta(days=GRACE_PERIOD_PUBLIC_DAYS)
    return {
        "deleted": str(team_id),
        "recoverableUntil": recoverable_until.isoformat(),
    }


@router.post("/teams/{team_id}/restore")
@limiter.limit("10/minute")
def restore_team_route(
    request: Request,
    team_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not is_team_owner_including_deleted(db, team_id, user.id):
        raise HTTPException(status_code=403, detail="Only the team owner can do this")

    try:
        team = restore_team(db, team_id)
    except ValueError as e:
        status = 404 if "not found" in str(e).lower() else 400
        raise HTTPException(status_code=status, detail=str(e))

    # Best-effort only -- notify_team_restored never raises, but the restore
    # itself is already committed by this point regardless, so even an
    # unexpected exception here must not turn a successful restore into a
    # failed response.
    try:
        notify_team_restored(db, team_id)
    except Exception:
        logger.exception("notify_team_restored raised unexpectedly for team %s", team_id)

    return {"id": str(team.id), "name": team.name, "restored": True}

@router.get("/teams/{team_id}/billing", response_model=TeamBillingOut)
@limiter.limit("30/minute")
def team_billing(
    request: Request,
    team_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not is_team_member(db, team_id, user.id):
        raise HTTPException(status_code=403, detail="You are not a member of this team")

    try:
        billing = get_team_billing(db, team_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return billing


@router.get("/teams/{team_id}/billing/history", response_model=BillingHistoryOut)
@limiter.limit("30/minute")
def team_billing_history(
    request: Request,
    team_id: UUID,
    limit: int = DEFAULT_HISTORY_PAGE_SIZE,
    cursor: str | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Credits that came in for this team (plan grants, plan-switch transfers, paid
    top-ups), newest first. Same visibility as GET /teams/{id}/billing: any member."""
    if not is_team_member(db, team_id, user.id):
        raise HTTPException(status_code=403, detail="You are not a member of this team")

    try:
        return list_billing_history(db, team_id, limit=limit, cursor=cursor)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


MAX_GENERATIONS_PAGE_SIZE = 50


@router.get("/teams/{team_id}/generations")
@limiter.limit("30/minute")
def team_generations(
    request: Request,
    team_id: UUID,
    limit: int = 10,
    offset: int = 0,
    feature_type: str | None = None,
    user_id: UUID | None = None,
    status: Literal["queued", "processing", "completed", "failed"] | None = None,
    view: Literal["recent", "library"] = "recent",
    period: Literal["last_7_days", "last_30_days", "all_time"] | None = None,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not is_team_member(db, team_id, user.id):
        raise HTTPException(status_code=403, detail="You are not a member of this team")

    limit = min(limit, MAX_GENERATIONS_PAGE_SIZE)
    generations = list_team_generations(
        db, team_id, limit=limit, offset=offset,
        feature_type=feature_type, user_id=user_id, status=status, view=view,
        period=None if period == "all_time" else period,
        from_dt=from_date, to_dt=to_date,
    )
    return {"generations": generations, "limit": limit, "offset": offset}


@router.get("/teams/{team_id}/usage")
@limiter.limit("30/minute")
def team_usage(
    request: Request,
    team_id: UUID,
    period: Literal["week", "month"] | None = None,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not is_team_member(db, team_id, user.id):
        raise HTTPException(status_code=403, detail="You are not a member of this team")

    return get_team_usage(db, team_id, period=period, from_dt=from_date, to_dt=to_date)