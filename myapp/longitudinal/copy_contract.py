"""Read-only copy adapter metadata. Auth workstream owns applying/rollback."""
from hashlib import sha256
import json
from datetime import datetime
from django.core.serializers.json import DjangoJSONEncoder
from uuid import UUID, uuid5
from .core import raw_identity, InvalidEvidence

COPY_POLICY = {
    'myapp.LongitudinalState': 'fresh_disabled',
    'myapp.EntryAuditPreference': 'fresh_excluded_unknown',
    'myapp.DreamReflection': 'explicit_authored_selection_only',
    'myapp.EntryAudit': 'omit_recompute_after_new_consent',
    'myapp.DreamObservation': 'omit_recompute_or_reauthor',
    'myapp.ObservationSuppression': 'omit_reconfirm_source_correction',
    'myapp.ThemeRecord': 'omit_recompute',
    'myapp.ThemeReview': 'omit_reconfirm',
    'myapp.ThemeMerge': 'omit_reauthor',
    'myapp.ThemeMergeMember': 'omit_reauthor',
    'myapp.ProfileSnapshot': 'omit_recompute',
    'myapp.SnapshotSource': 'omit_recompute',
    'myapp.AuditRun': 'omit_no_pending_job_copy',
    'myapp.AuditRunEntry': 'omit_no_pending_job_copy',
}
INDEX_NAMESPACES = ()  # Relational evidence only. No private FAISS/vector namespace.
ROLLBACK_DEPENDENCIES = {
    'myapp.DreamReflection': ('myapp.ThemeReview', 'myapp.SnapshotSource'),
    'myapp.JournalEntry': ('myapp.EntryAuditPreference', 'myapp.DreamReflection', 'myapp.EntryAudit', 'myapp.ObservationSuppression', 'myapp.AuditRunEntry', 'myapp.ThemeReview'),
    'myapp.EntryAudit': ('myapp.DreamObservation', 'myapp.AuditRunEntry', 'myapp.SnapshotSource'),
    'myapp.ThemeRecord': ('myapp.ThemeReview', 'myapp.ThemeMergeMember'),
    'myapp.ThemeMerge': ('myapp.ThemeMergeMember',),
    'myapp.ProfileSnapshot': ('myapp.SnapshotSource',),
    'myapp.AuditRun': ('myapp.AuditRunEntry',),
}


def inspect_private_inventory(model_labels, index_namespaces):
    if set(model_labels) - set(COPY_POLICY) or set(index_namespaces) - set(INDEX_NAMESPACES):
        raise InvalidEvidence('reviewed_private_copy_adapter_required')
    return {'policies': {label: COPY_POLICY[label] for label in model_labels}, 'owner_field': 'user',
            'migration_dependency': 'myapp.0019_auth_owner_infrastructure',
            'feature_migration': 'myapp.0020_longitudinal_evidence_profile',
            'indexes': [], 'rollback_dependencies': ROLLBACK_DEPENDENCIES}


def selected_reflection_plan(rows, *, selected_ids, source_owner_id, target_owner_id, copy_run_id, entry_map):
    """Explicitly selected authored text only; no consent/reviews/jobs transfer.

    Input/output remain private. Caller validates destination through actual
    owned_atomic_write and records local digests/rows in the copy ledger. New
    references created later must block rollback, never cascade silently.
    """
    source, target, run = UUID(str(source_owner_id)), UUID(str(target_owner_id)), UUID(str(copy_run_id))
    selected = {UUID(str(id)) for id in selected_ids}
    if len(selected) != len(selected_ids):
        raise InvalidEvidence('duplicate_reflection_selection')
    result, found = [], set()
    for row in rows:
        if set(row) - {'id', 'user_id', 'entry_id', 'text', 'created_at', 'updated_at', 'revision', 'use_for_profile', 'use_for_chat'}:
            raise InvalidEvidence('unknown_authored_reflection_field')
        id = UUID(str(row['id']))
        if id not in selected:
            continue
        if id in found or UUID(str(row['user_id'])) != source:
            raise InvalidEvidence('reflection_selection_owner_mismatch')
        text = row['text']
        raw_identity(text)
        if not text.strip() or len(text) > 4000:
            raise InvalidEvidence('invalid_reflection')
        entry_id = row.get('entry_id')
        if entry_id is not None and str(entry_id) not in entry_map:
            raise InvalidEvidence('reflection_source_expansion_required')
        timestamps = {}
        for field in ('created_at', 'updated_at'):
            value = row.get(field)
            if isinstance(value, str):
                try:
                    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                except ValueError:
                    raise InvalidEvidence('invalid_reflection_timestamp')
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise InvalidEvidence('aware_reflection_timestamp_required')
            timestamps[field] = value
        destination = {'id': str(uuid5(run, 'dreamreflection:' + str(id))), 'user_id': str(target),
            'entry_id': str(entry_map[str(entry_id)]) if entry_id is not None else None, 'text': text,
            'revision': 1, 'use_for_profile': False, 'use_for_chat': False,
            **timestamps}
        # Exact canonical serializer used by auth's ledger (including UUID/date).
        digest = sha256(json.dumps(destination, sort_keys=True, separators=(',', ':'), cls=DjangoJSONEncoder).encode()).hexdigest()
        result.append({'model_label': 'myapp.DreamReflection', 'source_id': str(id), 'destination': destination, 'digest': digest})
        found.add(id)
    if found != selected:
        raise InvalidEvidence('missing_selected_reflection')
    return result
