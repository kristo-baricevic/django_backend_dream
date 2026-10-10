"""Server-owned input snapshots and private result revocation.

Call while auth/domain/state locks are held. Hashes describe exact UTF8 input;
they are never a client-supplied grant or a substitute for owner authorization.
"""
from functools import wraps
from hashlib import sha256
import json
import re
from uuid import UUID
from django.core.exceptions import ValidationError as DjangoValidationError
from django.core.serializers.json import DjangoJSONEncoder
from django.http import Http404
from django.http import JsonResponse
from rest_framework.request import Request
from django.db.models import F
from rest_framework.exceptions import APIException, PermissionDenied
from myapp.models import (Analysis, CumulativeAnalysis, CustomQuestion, JournalEntry,
                          WorkflowExecution, DoctorProfile, LongitudinalState)
from .owner import owned_atomic_write, owned_queryset, require_owner, effective_settings, validate_request_session


class StaleSource(APIException):
    status_code = 409
    default_detail = 'Saved sources or settings changed. Reload and try again.'


def lock_state(profile):
    LongitudinalState.objects.get_or_create(user=profile)
    return LongitudinalState.objects.select_for_update().get(user=profile)


def content_hash(text):
    return sha256(text.encode('utf-8')).hexdigest()


def revision(profile):
    state = lock_state(profile)
    return {'source': state.source_revision, 'privacy': state.privacy_revision,
            'review': state.review_revision}


def settings_digest(owner):
    row = effective_settings(owner, for_update=True)
    values = {f.attname: getattr(row, f.attname) for f in row._meta.concrete_fields}
    # The database doctor profile is also an input, even when selected by name.
    name = row.doctor_personality or 'Academic'
    doctor = DoctorProfile.objects.select_for_update().filter(pk=row.doctor_id).first() if row.doctor_id else None
    doctor = doctor or DoctorProfile.objects.select_for_update().filter(name__iexact=name).first()
    values['doctor_profile'] = ({f.attname: getattr(doctor, f.attname)
                                for f in doctor._meta.concrete_fields} if doctor else None)
    return sha256(json.dumps(values, sort_keys=True, separators=(',', ':'),
                             cls=DjangoJSONEncoder).encode()).hexdigest()


def capture_snapshot(owner, entries):
    from myapp.models import User
    profile = User.objects.get(pk=owner.domain_user_id)
    return {'version': 1, 'owner_id': str(owner.domain_user_id),
            'auth_user_id': owner.auth_user_id, 'revision': revision(profile),
            'settings_digest': settings_digest(owner),
            'sources': [{'entry_id': str(row.pk), 'content_hash': content_hash(row.content)}
                        for row in entries]}


def snapshot_current(snapshot, owner):
    if not isinstance(snapshot, dict) or set(snapshot) != {
            'version', 'owner_id', 'auth_user_id', 'revision', 'settings_digest', 'sources'}:
        return False
    if (type(snapshot['version']) is not int or snapshot['version'] != 1
            or snapshot['owner_id'] != str(owner.domain_user_id)
            or type(snapshot['auth_user_id']) is not int or snapshot['auth_user_id'] != owner.auth_user_id
            or not isinstance(snapshot['revision'], dict) or set(snapshot['revision']) != {'source', 'privacy', 'review'}
            or any(type(v) is not int or v < 0 for v in snapshot['revision'].values())
            or not isinstance(snapshot['settings_digest'], str)
            or re.fullmatch(r'[0-9a-f]{64}', snapshot['settings_digest']) is None):
        return False
    from myapp.models import User
    profile = User.objects.get(pk=owner.domain_user_id)
    if snapshot['revision'] != revision(profile) or snapshot['settings_digest'] != settings_digest(owner):
        return False
    sources = snapshot['sources']
    if not isinstance(sources, list) or len(sources) > 500:
        return False
    ids = []
    for source in sources:
        if not isinstance(source, dict) or set(source) != {'entry_id', 'content_hash'}:
            return False
        if (not isinstance(source['entry_id'], str) or not isinstance(source['content_hash'], str)
                or re.fullmatch(r'[0-9a-f]{64}', source['content_hash']) is None):
            return False
        try:
            if str(UUID(source['entry_id'])) != source['entry_id']:
                return False
        except ValueError:
            return False
        ids.append(source['entry_id'])
    if len(set(ids)) != len(ids):
        return False
    try:
        rows = list(owned_queryset(JournalEntry, owner).filter(pk__in=ids).order_by('pk').select_for_update())
    except (ValueError, TypeError, OverflowError, DjangoValidationError):
        return False
    hashes = {str(row.pk): content_hash(row.content) for row in rows}
    return len(hashes) == len(ids) and all(hashes.get(s['entry_id']) == s['content_hash'] for s in sources)


def require_snapshot(snapshot, owner):
    if not snapshot_current(snapshot, owner):
        raise StaleSource()


def result_current(row, owner):
    if row.user_id != owner.domain_user_id:
        return False
    if isinstance(row, Analysis):
        if row.entry.user_id != owner.domain_user_id or row.entry.analysis_id != row.pk:
            return False
        metadata = row.grounding_metadata
        # Empty pre-migration records are explicitly historical, never grounded.
        if metadata == {} and row.is_historical:
            return True
        return isinstance(metadata, dict) and snapshot_current(metadata.get('source_snapshot'), owner)
    if isinstance(row, WorkflowExecution):
        if row.is_historical:
            return True
        if row.accepted_auth_user_id != owner.auth_user_id or not snapshot_current(row.source_metadata, owner):
            return False
        if row.status == 'completed':
            steps = list(row.steps.filter(step_type='answer_generation', status='completed'))
            if len(steps) != 1 or not isinstance(steps[0].output_data, dict):
                return False
            coverage = steps[0].output_data.get('coverage')
            if not isinstance(coverage, dict):
                return False
            dependencies = coverage.get('source_dependencies')
            if dependencies != row.source_metadata['sources']:
                return False
            models = (CumulativeAnalysis, CustomQuestion)
            try:
                matching = [result for model in models for result in model.objects.filter(
                    workflow_execution=row, user_id=owner.domain_user_id, pk=row.analysis_id)] if row.analysis_id else []
            except (ValueError, TypeError, DjangoValidationError):
                return False
            if len(matching) != 1:
                return False
        return True
    if isinstance(row, (CumulativeAnalysis, CustomQuestion)):
        if row.is_historical:
            return True
        workflow = row.workflow_execution
        return (workflow is not None and not workflow.is_historical and
                str(row.pk) == workflow.analysis_id and result_current(workflow, owner))
    return True


def eligible_queryset(model, owner, *, include_historical=True):
    query = owned_queryset(model, owner)
    if model is Analysis:
        query = query.filter(entry__user_id=owner.domain_user_id, entry__analysis_id=F('pk'))
        if not include_historical:
            query = query.filter(is_historical=False)
    elif model is WorkflowExecution and not include_historical:
        query = query.filter(is_historical=False)
    elif model in {CumulativeAnalysis, CustomQuestion} and not include_historical:
        query = query.filter(is_historical=False, workflow_execution__isnull=False, workflow_execution__is_historical=False)
    if model not in {Analysis, WorkflowExecution, CumulativeAnalysis, CustomQuestion}:
        return query
    ids = [row.pk for row in query if result_current(row, owner)]
    return query.filter(pk__in=ids)


def private_read(function):
    @wraps(function)
    def wrapped(request, *args, **kwargs):
        try:
            owner = require_owner(request)
            def fence(profile):
                lock_state(profile)
                validate_request_session(owner)
            return owned_atomic_write(owner_id=owner.domain_user_id,
                expected_auth_user_id=owner.auth_user_id, workflow_id=None,
                lock_state=fence, write=lambda profile: function(request, *args, **kwargs))
        except APIException as error:
            if isinstance(request, Request):
                raise
            return JsonResponse({'detail': 'Private data unavailable.'}, status=error.status_code)
    return wrapped


def private_read_method(function):
    @wraps(function)
    def wrapped(self, request, *args, **kwargs):
        return private_read(lambda req: function(self, req, *args, **kwargs))(request)
    return wrapped


class PrivateReadMixin:
    def list(self, request, *args, **kwargs):
        return private_read(super().list)(request, *args, **kwargs)


def eligible_feedback(owner):
    from myapp.models import AnalysisFeedback
    ids = []
    for row in owned_queryset(AnalysisFeedback, owner).select_related(
            'analysis__entry', 'cumulative_analysis__workflow_execution', 'custom_question__workflow_execution'):
        target = row.analysis or row.cumulative_analysis or row.custom_question
        if target is not None and result_current(target, owner):
            if isinstance(target, Analysis) and target.is_historical:
                continue
            if isinstance(target, (CumulativeAnalysis, CustomQuestion)) and (
                    target.is_historical or target.workflow_execution is None or target.workflow_execution.is_historical):
                continue
            ids.append(row.pk)
    return owned_queryset(AnalysisFeedback, owner).filter(pk__in=ids)


def source_preimage(entry):
    return {'entry_id': str(entry.pk), 'content_hash': content_hash(entry.content),
            'updated_at': entry.updated_at.isoformat()}


def require_source_ticket(request, owner, entry, operation):
    validate_request_session(owner)
    if request.headers.get('X-Dream-Principal') != str(owner.domain_user_id):
        raise PermissionDenied('Session changed. Reload and try again.')
    if request.data.get('expected_session') != owner.login_deadline.isoformat():
        raise StaleSource()
    expected = request.data.get('expected_revision')
    if not isinstance(expected, dict) or set(expected) != {'source', 'privacy', 'review'} or any(
            type(expected[key]) is not int or expected[key] < 0 for key in expected):
        raise StaleSource()
    if expected != revision(entry.user) or request.data.get('expected_operation') != operation:
        raise StaleSource()
    if request.data.get('expected_source') != source_preimage(entry):
        raise StaleSource()
