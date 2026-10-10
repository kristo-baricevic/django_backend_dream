"""Public service contract types; imports no Django or user credentials."""
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True)
class OwnerContext:
    domain_user_id: UUID
    auth_user_id: int
    is_staff: bool
    login_deadline: datetime
    session_key: str | None = None
