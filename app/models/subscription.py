from sqlalchemy import Column, Text, Integer, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from app.core.database import Base


class Subscription(Base):
    __tablename__ = "subscription"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    slug = Column(Text, nullable=False, unique=True)
    name = Column(Text, nullable=False)
    price = Column(Integer, nullable=False)
    billing_period_days = Column(Integer, nullable=False)
    period_label = Column(Text, nullable=False)
    credits = Column(Integer, nullable=False)
    info = Column(JSONB, nullable=False, default=list)
    tag = Column(Text, nullable=True)
    is_popular = Column(Boolean, nullable=False, default=False, server_default="false")
    sort_order = Column(Integer, nullable=False, default=0)
    razorpay_plan_id = Column(Text, nullable=True)  
    is_active = Column(Boolean, nullable=False, default=True)
    total_count = Column(Integer, nullable=True)