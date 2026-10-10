"""Strict test-only approved OwnerContext contract, pending task-12's actual helpers."""
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from django.core.exceptions import PermissionDenied
from django.db import transaction
from django.utils import timezone
from rest_framework.exceptions import NotAuthenticated

MAPPINGS = {}


def require_owner(request):
    if not request.user.is_authenticated or not request.user.is_active:
        raise NotAuthenticated()
    domain_id = next((id for id, auth_id in MAPPINGS.items() if auth_id == request.user.pk), None)
    if domain_id is None:
        raise NotAuthenticated()
    return OwnerContext(domain_id, request.user.pk, request.user.is_staff, timezone.now() + __import__("datetime").timedelta(hours=1))


@dataclass(frozen=True)
class OwnerContext:
    domain_user_id: UUID
    auth_user_id: int
    is_staff: bool
    login_deadline: datetime


def validate(owner):
    if MAPPINGS.get(owner.domain_user_id) != owner.auth_user_id or owner.login_deadline <= timezone.now():
        raise PermissionDenied("owner_contract_invalid")


def owned_queryset(model, owner):
    return model.objects.filter(user_id=owner.domain_user_id)


def require_owned_object(model, owner, id, for_update=False):
    query = owned_queryset(model, owner)
    if for_update:
        query = query.select_for_update()
    return query.get(pk=id)


def owned_atomic_write(*, owner_id, expected_auth_user_id, workflow_id, write, lock_state=None):
    from django.contrib.auth import get_user_model
    from myapp.models import User
    if MAPPINGS.get(owner_id) != expected_auth_user_id:
        raise PermissionDenied("owner_contract_invalid")
    with transaction.atomic():
        auth = get_user_model().objects.select_for_update().get(pk=expected_auth_user_id)
        if not auth.is_active:
            raise PermissionDenied("owner_contract_invalid")
        profile = User.objects.select_for_update().get(pk=owner_id)
        if lock_state is not None:
            lock_state(profile)
        return write(profile)
