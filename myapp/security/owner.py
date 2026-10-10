"""Domain UUID is the only private-data owner. No global or email fallback."""
from datetime import datetime, timezone
from math import isfinite
from uuid import UUID
from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth import logout
from django.db import transaction
from django.http import Http404
from rest_framework.exceptions import NotAuthenticated, PermissionDenied, ValidationError
from myapp.models import User, Settings, JournalEntry, WorkflowExecution
from .contracts import OwnerContext
from django.contrib.auth import load_backend
from django.utils import timezone as django_timezone
from django.utils.crypto import constant_time_compare

DEADLINE_KEY = 'dream_login_deadline'


def validate_request_session(owner):
    """Revalidate and lock the initiating Django session through response/commit.

    Logout deletion waits for this row lock. Password changes serialize on the
    existing auth lock and invalidate the session auth hash immediately.
    Durable accepted jobs deliberately have no session key and use account and
    source revocation instead; public request callers must carry a key.
    """
    from django.contrib.sessions.models import Session
    session = Session.objects.select_for_update().filter(
        session_key=owner.session_key, expire_date__gt=django_timezone.now()).first()
    now = django_timezone.now()
    if session is None or session.expire_date <= now or owner.login_deadline <= now:
        raise PermissionDenied('Request session unavailable.')
    data = session.get_decoded()
    backend = data.get('_auth_user_backend')
    if backend not in settings.AUTHENTICATION_BACKENDS or str(data.get('_auth_user_id')) != str(owner.auth_user_id):
        raise PermissionDenied('Request session unavailable.')
    auth = load_backend(backend).get_user(owner.auth_user_id)
    if auth is None or not auth.is_active:
        raise PermissionDenied('Request session unavailable.')
    hashes = [auth.get_session_auth_hash(), *auth.get_session_auth_fallback_hash()]
    stored = data.get('_auth_user_hash', '')
    deadline = data.get(DEADLINE_KEY)
    if (not isinstance(stored, str) or not any(constant_time_compare(stored, value) for value in hashes)
            or type(deadline) not in {int, float} or not isfinite(deadline)
            or deadline != owner.login_deadline.timestamp()):
        raise PermissionDenied('Request session unavailable.')


def _valid_profile(profile, auth_user):
    return (auth_user.is_active and profile.auth_user_id == auth_user.pk
            and profile.import_ready
            and (not profile.is_local_fixture or (
                getattr(settings, 'DREAM_ENV', '') == 'local'
                and getattr(settings, 'DREAM_ENABLE_LOCAL_FIXTURE', False))))


def require_owner(request):
    def deny():
        if request.user.is_authenticated:
            logout(request)
        raise NotAuthenticated('Authentication required.')
    auth = request.user
    if not auth.is_authenticated or not auth.is_active:
        deny()
    deadline = request.session.get(DEADLINE_KEY)
    if not isinstance(deadline, (int, float)) or not isfinite(deadline) or deadline <= datetime.now(timezone.utc).timestamp():
        deny()
    profile = User.objects.filter(auth_user_id=auth.pk).first()
    if profile is None or not _valid_profile(profile, auth):
        deny()
    # Reject-only intent hints keep a stale browser tab from sending one
    # account's private input with a different account's switched cookie.
    # Cookies and the linked active account remain the sole authority.
    if not request.path.rstrip('/').endswith('/auth/login'):
        principal_hint = request.headers.get('X-Dream-Principal')
        session_hint = request.headers.get('X-Dream-Session')
        if ((principal_hint is not None and principal_hint != str(profile.pk))
                or (session_hint is not None and session_hint != datetime.fromtimestamp(deadline, timezone.utc).isoformat())):
            raise PermissionDenied('Session changed. Reload and try again.')
    return OwnerContext(profile.pk, auth.pk, auth.is_staff,
                        datetime.fromtimestamp(deadline, timezone.utc), request.session.session_key)


def owned_queryset(model, owner):
    return model.objects.filter(user_id=owner.domain_user_id)


def require_owned_object(model, owner, object_id, *, for_update=False):
    try:
        object_id = UUID(str(object_id))
    except (ValueError, TypeError, AttributeError):
        raise Http404
    query = owned_queryset(model, owner)
    if for_update:
        query = query.select_for_update()
    try:
        return query.get(pk=object_id)
    except model.DoesNotExist:
        raise Http404


def load_owned_entries(owner, entry_ids):
    try:
        ids = [UUID(str(value)) for value in entry_ids]
    except (ValueError, TypeError, AttributeError):
        raise ValidationError('Invalid entry selection.')
    if len(ids) != len(set(ids)) or len(ids) > 1000:
        raise ValidationError('Invalid entry selection.')
    records = {row.pk: row for row in owned_queryset(JournalEntry, owner).filter(pk__in=ids)}
    if len(records) != len(ids):
        raise Http404
    return tuple(records[pk] for pk in ids)


def effective_settings(owner, *, for_update=False):
    profile_query = User.objects
    if for_update:
        profile_query = profile_query.select_for_update()
    profile = profile_query.get(pk=owner.domain_user_id, auth_user_id=owner.auth_user_id)
    if profile.active_settings_id is None:
        raise ValidationError('Settings selection is required.')
    query = Settings.objects.filter(pk=profile.active_settings_id, user_id=profile.pk)
    if for_update:
        query = query.select_for_update()
    row = query.first()
    if row is None:
        raise ValidationError('Settings selection is required.')
    return row


def initialize_settings(owner, *, new_login=False):
    def create(profile):
        if owner.session_key and not new_login:
            validate_request_session(owner)
        if profile.active_settings_id:
            return effective_settings(owner, for_update=True)
        if Settings.objects.filter(user_id=profile.pk).exists():
            raise ValidationError('Legacy settings require an explicit selection.')
        row = Settings.objects.create(user=profile)
        profile.active_settings = row
        profile.save(update_fields=['active_settings'])
        return row
    return owned_atomic_write(owner_id=owner.domain_user_id,
                              expected_auth_user_id=owner.auth_user_id,
                              workflow_id=None, write=create)


def owner_from_workflow(workflow_id, expected_owner_id, expected_auth_user_id):
    profile = User.objects.select_related('auth_user').filter(pk=expected_owner_id).first()
    if profile is None or profile.auth_user is None or not _valid_profile(profile, profile.auth_user):
        raise PermissionDenied('Job owner unavailable.')
    if profile.auth_user_id != expected_auth_user_id:
        raise PermissionDenied('Job owner unavailable.')
    if not WorkflowExecution.objects.filter(pk=workflow_id, user=profile,
            accepted_auth_user_id=expected_auth_user_id, is_historical=False).exists():
        raise PermissionDenied('Job owner unavailable.')
    return profile


def owned_atomic_write(*, owner_id, expected_auth_user_id, workflow_id, write,
                       lock_state=None):
    """Lock auth → domain → optional private state → workflow until commit.

    lock_state(profile) is the longitudinal integration hook: acquire privacy/source
    locks there before the workflow lock. The write callback resolves all private
    relations inside this transaction and explicitly assigns user=profile.
    """
    with transaction.atomic():
        auth = get_user_model().objects.select_for_update().filter(pk=expected_auth_user_id).first()
        profile = User.objects.select_for_update().filter(pk=owner_id).first()
        if auth is None or profile is None or not _valid_profile(profile, auth):
            raise PermissionDenied('Job owner unavailable.')
        if lock_state is not None:
            lock_state(profile)
        if workflow_id is not None:
            workflow = WorkflowExecution.objects.select_for_update().filter(
                pk=workflow_id, user=profile, accepted_auth_user_id=expected_auth_user_id,
                is_historical=False).first()
            if workflow is None:
                raise PermissionDenied('Job owner unavailable.')
        return write(profile)
