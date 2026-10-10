"""Adapter to task-12's approved helpers; absence is a hard configuration error.

Configure the actual module after auth integration. This module implements no
authentication, default owner, account mapping or alternative locking policy.
"""
from importlib import import_module
from django.conf import settings


class ConfigurationRequired(RuntimeError):
    pass


def security():
    name = getattr(settings, "LONGITUDINAL_SECURITY_MODULE", "myapp.security.owner")
    if not name:
        raise ConfigurationRequired("approved_auth_contract_required")
    try:
        module = import_module(name)
    except ImportError as exc:
        raise ConfigurationRequired("approved_auth_contract_required") from exc
    for helper in ("owned_queryset", "require_owned_object", "owned_atomic_write"):
        if not callable(getattr(module, helper, None)):
            raise ConfigurationRequired("approved_auth_contract_incomplete")
    return module


def owner_write(owner, write):
    # Auth helper acquires auth-user then domain-owner locks. State locks come
    # only inside this callback; no feature path inverts that ordering.
    from myapp.longitudinal_models import LongitudinalState
    def lock_state(profile):
        state, _ = LongitudinalState.objects.get_or_create(user=profile)
        LongitudinalState.objects.select_for_update().get(pk=state.pk, user=profile)
    def checked_write(profile):
        if profile.pk != owner.domain_user_id:
            raise ConfigurationRequired("approved_owner_contract_mismatch")
        if getattr(owner, 'session_key', None):
            security().validate_request_session(owner)
        return write()
    return security().owned_atomic_write(owner_id=owner.domain_user_id,
        expected_auth_user_id=owner.auth_user_id, workflow_id=None,
        write=checked_write, lock_state=lock_state)
