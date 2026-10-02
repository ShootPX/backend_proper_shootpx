# Every model must be imported here so it registers on Base.metadata no
# matter which entry point runs first (FastAPI app, arq worker, or a one-off
# script) -- a model imported only by whichever router happened to need it
# left "users" unregistered for app/worker.py, which touches generation_jobs
# (FK'd to users.id) but never imported app.models.user itself. Any process
# that does `from app.models import *` (or even just imports this package)
# now gets every table registered up front instead of depending on import order.
from app.models.user import User
from app.models.team import Team
from app.models.team_member import TeamMember
from app.models.team_invite import TeamInvite
from app.models.team_subscription import TeamSubscription
from app.models.subscription import Subscription
from app.models.credit import Credit
from app.models.billing_transaction import BillingTransaction
from app.models.credit_ledger import CreditLedger
from app.models.tool_definition import ToolDefinition
from app.models.generation_job import GenerationJob
from app.models.model_preset import ModelPreset
from app.models.homepage_slide import HomepageSlide
from app.models.signup_bonus_log import SignupBonusLog

__all__ = [
    "User", "Team", "TeamMember", "TeamInvite", "TeamSubscription",
    "Subscription", "Credit", "BillingTransaction", "CreditLedger", "ToolDefinition",
    "GenerationJob", "ModelPreset", "HomepageSlide",
    "SignupBonusLog",
]
