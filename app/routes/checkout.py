from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.limiter import limiter
from app.deps import get_current_user
from app.models.user import User
from app.services.teams import is_team_owner
from app.services.billing import (
    create_credit_pack_checkout,
    create_subscription_checkout,
    cancel_subscription,
    RazorpayCancelError,
    PlanSwitchBlocked,
    switch_subscription
)
router = APIRouter(prefix="/billing", tags=["checkout"])


@router.post("/teams/{team_id}/credit-packs/{pack_id}/checkout")
@limiter.limit("10/minute")
def checkout_credit_pack(
    request: Request,
    team_id: UUID,
    pack_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not is_team_owner(db, team_id, user.id):
        raise HTTPException(status_code=403, detail="Only the team owner can buy credits")

    try:
        checkout = create_credit_pack_checkout(db, team_id, pack_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return checkout


@router.post("/teams/{team_id}/subscriptions/{subscription_id}/checkout")
@limiter.limit("10/minute")
def checkout_subscription(
    request: Request,
    team_id: UUID,
    subscription_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not is_team_owner(db, team_id, user.id):
        raise HTTPException(status_code=403, detail="Only the team owner can subscribe")

    try:
        checkout = create_subscription_checkout(db, team_id, subscription_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return checkout

@router.post("/teams/{team_id}/subscriptions/cancel")
@limiter.limit("10/minute")
def cancel_team_subscription(
    request: Request,
    team_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not is_team_owner(db, team_id, user.id):
        raise HTTPException(status_code=403, detail="Only the team owner can cancel the subscription")

    try:
        cancel_subscription(db, team_id)
    except (ValueError, RazorpayCancelError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "cancelled", "teamId": str(team_id)}


@router.post("/teams/{team_id}/subscriptions/{new_subscription_id}/switch")
@limiter.limit("10/minute")
def switch_team_subscription(
    request: Request,
    team_id: UUID,
    new_subscription_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not is_team_owner(db, team_id, user.id):
        raise HTTPException(status_code=403, detail="Only the team owner can switch plans")

    try:
        checkout = switch_subscription(db, team_id, new_subscription_id)
    except PlanSwitchBlocked as e:
        raise HTTPException(
            status_code=409,
            detail={
                "code": e.code,
                "message": str(e),
                "current_period_end": e.current_period_end.isoformat(),
            },
        )
    except (ValueError, RazorpayCancelError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return checkout