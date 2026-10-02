import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.billing_transaction import BillingTransaction
from app.models.credit_ledger import CreditLedger
from app.models.generation_job import GenerationJob
from app.models.team import Team
from app.models.team_invite import TeamInvite
from app.models.team_member import TeamMember
from app.models.team_subscription import TeamSubscription
from app.models.user import User
from app.services.billing import cancel_subscription, is_unpaid_checkout

logger = logging.getLogger(__name__)

# Hard cap on team size: owner + editors. Pending invites also count toward it
# (each pending invite reserves a seat), so a team can never end up over the cap.
MAX_TEAM_MEMBERS = 5

# Team deletion is a two-clock design (see soft_delete_team/restore_team):
# money stops immediately (Razorpay is cancelled at delete time, not at
# purge time), but the team + all its data stay recoverable for a grace
# window. GRACE_PERIOD_DAYS is the REAL, enforced cutoff for both restore
# eligibility and the purge sweep -- intentionally longer than whatever
# public-facing copy the frontend shows ("15 days"), as safety margin for
# edge cases and support requests. Never surface this exact number in an
# API response; the frontend's own copy is the public commitment.
GRACE_PERIOD_DAYS = 30
# What the API tells the FRONTEND to show the user -- deliberately shorter
# than the real GRACE_PERIOD_DAYS enforcement above. Never derive this from
# GRACE_PERIOD_DAYS; it's an independent, conservative public commitment.
GRACE_PERIOD_PUBLIC_DAYS = 15


# --- membership checks ---------------------------------------------------------

def _membership(db: Session, team_id, user_id) -> TeamMember | None:
    return (
        db.query(TeamMember)
        .filter(TeamMember.team_id == team_id, TeamMember.user_id == user_id)
        .first()
    )


def _live_membership_role(db: Session, team_id, user_id) -> str | None:
    """
    The role this user holds on this team, or None if they're not a member
    OR the team is soft-deleted.

    ONE query, joined -- not a membership lookup followed by a separate team
    lookup. Every team-scoped route in the app funnels through this, and at
    the measured ~160ms round trip to the (Tokyo-hosted) database, the
    second query was costing a full extra round trip on literally every
    authenticated team request.
    """
    row = (
        db.query(TeamMember.role)
        .join(Team, Team.id == TeamMember.team_id)
        .filter(
            TeamMember.team_id == team_id,
            TeamMember.user_id == user_id,
            Team.deleted_at.is_(None),
        )
        .first()
    )
    return row[0] if row else None


def is_team_member(db: Session, team_id, user_id) -> bool:
    """False for a soft-deleted team -- deletion must actually block access,
    not just hide the team from list_user_teams. Restoring a team (which
    itself requires bypassing this, see is_team_owner_including_deleted) is
    the only way back in."""
    return _live_membership_role(db, team_id, user_id) is not None


def is_team_owner(db: Session, team_id, user_id) -> bool:
    return _live_membership_role(db, team_id, user_id) == "owner"


def is_team_owner_including_deleted(db: Session, team_id, user_id) -> bool:
    """The one deliberate bypass of the deleted_at gate above -- restoring a
    deleted team, and initiating its deletion in the first place, both need
    to authorize against a team that either already is, or is about to
    become, soft-deleted."""
    m = _membership(db, team_id, user_id)
    return m is not None and m.role == "owner"


# --- reads --------------------------------------------------------------------

def get_team(db: Session, team_id) -> Team | None:
    return db.query(Team).filter(Team.id == team_id).first()


def list_user_teams(db: Session, user_id) -> list[dict]:
    rows = (
        db.query(Team, TeamMember.role)
        .join(TeamMember, TeamMember.team_id == Team.id)
        .filter(TeamMember.user_id == user_id, Team.deleted_at.is_(None))
        .order_by(TeamMember.joined_at)
        .all()
    )
    # Credit balances live on the Team row this query already loaded, so
    # including them here is free -- and it saves the client a whole second
    # request (GET /teams -> GET /teams/{id}/billing) just to render a
    # credits figure, which it previously could not even START until this
    # response came back. GET /teams/{id}/billing remains the authority for
    # the full breakdown (plan, status, period end, pool split).
    return [
        {
            "id": str(t.id),
            "name": t.name,
            "role": role,
            "totalCredits": t.subscription_credits_remaining + t.topup_credits_balance,
        }
        for t, role in rows
    ]


def list_team_members(db: Session, team_id) -> list[dict]:
    rows = (
        db.query(TeamMember, User)
        .join(User, User.id == TeamMember.user_id)
        .filter(TeamMember.team_id == team_id)
        .order_by(TeamMember.joined_at)
        .all()
    )
    return [
        {
            "userId": str(u.id),
            "email": u.email,
            "name": u.name,
            "avatarUrl": u.avatar_url,
            "role": m.role,
            "joinedAt": m.joined_at.isoformat() if m.joined_at else None,
        }
        for m, u in rows
    ]


def get_team_members_page(db: Session, team_id) -> dict | None:
    """The whole GET /teams/{id}/members payload in ONE query.

    The route previously did `get_team()` (for the name) and then
    `list_team_members()` -- two round trips for data one join returns. The
    team name repeats on every row, which is free compared to a second
    ~160ms trip to the database.
    """
    rows = (
        db.query(TeamMember, User, Team.name)
        .join(User, User.id == TeamMember.user_id)
        .join(Team, Team.id == TeamMember.team_id)
        .filter(TeamMember.team_id == team_id)
        .order_by(TeamMember.joined_at)
        .all()
    )
    if not rows:
        return None

    return {
        "id": str(team_id),
        "name": rows[0][2],
        "members": [
            {
                "userId": str(u.id),
                "email": u.email,
                "name": u.name,
                "avatarUrl": u.avatar_url,
                "role": m.role,
                "joinedAt": m.joined_at.isoformat() if m.joined_at else None,
            }
            for m, u, _ in rows
        ],
    }


def team_member_count(db: Session, team_id) -> int:
    return db.query(TeamMember).filter(TeamMember.team_id == team_id).count()


def owner_count(db: Session, team_id) -> int:
    return (
        db.query(TeamMember)
        .filter(TeamMember.team_id == team_id, TeamMember.role == "owner")
        .count()
    )


def team_seat_count(db: Session, team_id) -> int:
    """Current members + still-pending, not-yet-expired invites."""
    pending = (
        db.query(TeamInvite)
        .filter(
            TeamInvite.team_id == team_id,
            TeamInvite.status == "pending",
            TeamInvite.expires_at > datetime.now(timezone.utc),
        )
        .count()
    )
    return team_member_count(db, team_id) + pending


# --- writes (all callers must have already checked the acting user is owner) ---

def _clean_team_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("Team name cannot be empty")
    return name


def create_team(db: Session, user_id, name: str) -> dict:
    """
    A user-initiated additional team, distinct from the one auto-created on
    signup (see services/users.py::_create_user_with_team) -- this one starts
    with ZERO credits, deliberately. The signup bonus is a one-time,
    anti-abuse-capped grant tied to account creation (see
    signup_abuse.py); handing out the same bonus here would let a single
    account farm it indefinitely by just creating more teams.
    """
    name = _clean_team_name(name)

    team = Team(name=name)
    db.add(team)
    db.flush()

    db.add(TeamMember(team_id=team.id, user_id=user_id, role="owner"))
    db.commit()

    return {"id": str(team.id), "name": team.name, "role": "owner"}


def rename_team(db: Session, team: Team, name: str) -> Team:
    team.name = _clean_team_name(name)
    db.commit()
    return team


def remove_member(db: Session, team_id, target_user_id) -> None:
    member = _membership(db, team_id, target_user_id)
    if member is None:
        raise ValueError("That user is not a member of this team")
    if member.role == "owner" and owner_count(db, team_id) <= 1:
        raise ValueError("Cannot remove the last owner — transfer ownership or delete the team")

    db.delete(member)
    db.commit()


def soft_delete_team(db: Session, team_id) -> datetime:
    """
    Starts the two-clock deletion: money stops NOW (Razorpay is cancelled
    immediately, same as the old hard-delete used to do), but nothing else
    is touched -- no rows deleted, no credit balance changed. The team just
    becomes inaccessible (is_team_member/is_team_owner both start returning
    False for it) until either restore_team() or the purge sweep runs.
    Returns the deleted_at timestamp actually stored.
    """
    team = get_team(db, team_id)
    if not team:
        raise ValueError("Team not found")
    if team.deleted_at is not None:
        raise ValueError("This team is already scheduled for deletion")

    try:
        cancel_subscription(db, team_id)
    except ValueError:
        pass  # no active subscription — expected, not an error
    except Exception:
        logger.error(
            "MANUAL FOLLOW-UP NEEDED: failed to cancel Razorpay subscription while "
            "deleting team %s — check the Razorpay dashboard directly. Deletion proceeded.",
            team_id, exc_info=True,
        )

    team.deleted_at = datetime.now(timezone.utc)
    db.commit()
    return team.deleted_at


def restore_team(db: Session, team_id) -> Team:
    """
    Un-deletes a team within its grace window. The Razorpay subscription
    does NOT resume -- it was genuinely cancelled at delete time, not
    paused -- so a restored team keeps whatever credit balance it had (both
    pools are untouched by deletion) but needs to actively re-subscribe once
    that balance runs out.
    """
    team = get_team(db, team_id)
    if not team:
        raise ValueError("Team not found")
    if team.deleted_at is None:
        raise ValueError("This team is not scheduled for deletion")
    if datetime.now(timezone.utc) - team.deleted_at > timedelta(days=GRACE_PERIOD_DAYS):
        raise ValueError("The recovery window for this team has expired")

    team.deleted_at = None
    db.commit()
    return team


def purge_team(db: Session, team_id) -> None:
    """
    The actual, permanent hard-delete -- only ever called by the daily purge
    sweep (worker.py) once a soft-deleted team's grace window has passed.
    Razorpay was already cancelled at soft_delete_team() time, so there's no
    provider call here.
    """
    # FKs to teams are ON DELETE NO ACTION, so every child row must go first --
    # every table with a team_id FK (checked: generation_jobs, billing_transactions,
    # credit_ledger, team_subscriptions, team_invites, team_members) must be listed here, or the
    # final Team delete below hits a real FK violation for any team that used it.
    db.query(GenerationJob).filter(GenerationJob.team_id == team_id).delete(synchronize_session=False)
    db.query(BillingTransaction).filter(BillingTransaction.team_id == team_id).delete(synchronize_session=False)
    db.query(CreditLedger).filter(CreditLedger.team_id == team_id).delete(synchronize_session=False)
    db.query(TeamSubscription).filter(TeamSubscription.team_id == team_id).delete(synchronize_session=False)
    db.query(TeamInvite).filter(TeamInvite.team_id == team_id).delete(synchronize_session=False)
    db.query(TeamMember).filter(TeamMember.team_id == team_id).delete(synchronize_session=False)
    db.query(Team).filter(Team.id == team_id).delete(synchronize_session=False)
    db.commit()


def notify_team_restored(db: Session, team_id) -> None:
    """
    Best-effort courtesy email to every CURRENT member once a team comes back
    from soft-deletion -- same "never raises" contract as webhooks.py's
    send_renewal_notice_email: an SMTP failure here must not fail the
    restore itself (the team is already un-deleted and committed by the
    time this runs), just logged and swallowed per-recipient so one bad
    address doesn't stop the rest of the team from being notified.
    """
    from app.core.email import send_email

    team = get_team(db, team_id)
    if not team:
        return

    for member in list_team_members(db, team_id):
        email = member.get("email")
        if not email:
            continue
        try:
            send_email(
                to=email,
                subject=f'"{team.name}" has been restored',
                html=(
                    f'<p>Good news — the <strong>{team.name}</strong> team on ShootPX '
                    f"has been restored by its owner. You have access again, and "
                    f"everything is exactly as it was before deletion.</p>"
                ),
                text=(
                    f'The "{team.name}" team on ShootPX has been restored by its owner. '
                    f"You have access again, and everything is exactly as it was before deletion."
                ),
            )
        except Exception:
            logger.error(
                "Failed to send restore-notification email to %s for team %s",
                email, team_id, exc_info=True,
            )


def list_teams_past_grace_period(db: Session) -> list[Team]:
    """
    Must be the exact complement of restore_team's own boundary check
    (`elapsed > GRACE_PERIOD_DAYS`) -- using `<=` here (elapsed >=
    GRACE_PERIOD_DAYS) would make a team sitting at EXACTLY the boundary
    both still-restorable AND already purge-eligible, a real race between
    this sweep and a same-moment restore call. `<` keeps the two windows
    adjacent with no overlap: restore_team accepts up to and including
    exactly GRACE_PERIOD_DAYS elapsed, this sweep only takes teams strictly
    past it.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=GRACE_PERIOD_DAYS)
    return db.query(Team).filter(Team.deleted_at.isnot(None), Team.deleted_at < cutoff).all()

def get_team_billing(db: Session, team_id: UUID) -> dict:
    from app.models.subscription import Subscription
    from app.models.team_subscription import TeamSubscription
    from app.services.credits import get_total_credits

    # One outer-joined query, not "fetch team" then "fetch its subscription" --
    # the second round trip was pure latency (see _live_membership_role's note
    # on the measured per-query cost). LEFT joins because a team with no
    # subscription at all is the normal free-tier case, not an error.
    row = (
        db.query(Team, TeamSubscription, Subscription)
        .outerjoin(TeamSubscription, TeamSubscription.team_id == Team.id)
        .outerjoin(Subscription, Subscription.id == TeamSubscription.subscription_id)
        .filter(Team.id == team_id)
        .first()
    )
    if row is None:
        raise ValueError("Team not found")

    team, sub, plan = row
    unpaid = is_unpaid_checkout(sub)

    pending_switch = None
    pending_plan_id = getattr(sub, "pending_subscription_id", None) if sub else None
    if pending_plan_id:
        pending_plan = db.query(Subscription).filter(Subscription.id == pending_plan_id).first()
        expires_at = sub.pending_switch_expires_at
        pending_switch = {
            "plan": pending_plan.slug if pending_plan else None,
            "expires_at": expires_at.isoformat() if expires_at else None,
        }

    return {
        "total_credits": get_total_credits(team),
        "subscription_credits": team.subscription_credits_remaining,
        "topup_credits": team.topup_credits_balance,
        "plan": plan.slug if plan else None,
        # "created" = a checkout was started but never paid (internally a
        # "pending" row with no activation yet) -- distinct from a real
        # "pending", which means an ACTIVE subscription whose renewal payment
        # failed and is being retried. The frontend must be able to tell them
        # apart: one is "payment not completed", the other is a live plan.
        "subscription_status": ("created" if unpaid else sub.status) if sub else None,
        # An unpaid attempt has no billing period -- the placeholder date the
        # checkout claim stores must never be shown as a renewal date.
        "current_period_end": sub.current_period_end.isoformat() if sub and not unpaid else None,
        "pending_switch": pending_switch,
    }