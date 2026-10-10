"""Copy-only local importer. Input is data, never Django deserialization or pickle."""
import hashlib
import json
from uuid import UUID, uuid5
from django.conf import settings
from django.core.serializers.json import DjangoJSONEncoder
from django.db import connection, transaction
from django.utils.dateparse import parse_datetime
from myapp.models import (User, JournalEntry, Analysis, CumulativeAnalysis, CustomQuestion,
    Settings, WorkflowExecution, WorkflowStep, StepCitation, AnalysisFeedback, DataCopyRun, DataCopyRecord)

MODELS = {'entries': JournalEntry, 'analyses': Analysis, 'cumulative': CumulativeAnalysis,
          'questions': CustomQuestion, 'settings': Settings, 'workflows': WorkflowExecution,
          'steps': WorkflowStep, 'citations': StepCitation, 'feedback': AnalysisFeedback}
ALIASES = {'doctorPersonality': 'doctor_personality', 'doctorImage': 'doctor_image',
           'personality': 'personality_type', 'medicalHistory': 'medical_history', 'influence': 'weights'}
RELATIONS = {'analysis': 'analyses', 'entry': 'entries', 'workflow_execution': 'workflows',
             'execution': 'workflows', 'step': 'steps', 'custom_question': 'questions',
             'cumulative_analysis': 'cumulative'}
PARTIAL_MISSING = ['workflow links and records', 'feedback records', 'doctor profiles and settings doctor link',
                   'original settings timestamps/doctor influence/owner', 'private files and vector namespaces inventory']


class CopyError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    cls=DjangoJSONEncoder).encode()).hexdigest()


def snapshot(row):
    return {field.attname: getattr(row, field.attname) for field in row._meta.concrete_fields}


def guard_destination():
    name = connection.settings_dict['NAME']
    if getattr(settings, 'DREAM_ENV', '') not in {'local', 'test'} or not str(name).startswith(('dream_auth_', 'test_dream_auth_')):
        raise CopyError('Copy commands require an explicitly isolated local/test database.')


def inspect_bundle(bundle, *, partial=False):
    if not isinstance(bundle, dict) or set(bundle) - set(MODELS):
        raise CopyError('Unknown resource: a reviewed adapter is required.')
    if not {'entries', 'analyses', 'cumulative', 'questions', 'settings'} <= set(bundle):
        raise CopyError('Incomplete exposed-resource export.')
    if not partial:
        raise CopyError('API exports omit source relations; explicitly acknowledge partial coverage.')
    ids, counts = {}, {}
    for label, rows in bundle.items():
        if not isinstance(rows, list): raise CopyError('Invalid resource list.')
        try:
            ids[label] = {UUID(str(row['id'])) for row in rows}
        except (KeyError, TypeError, ValueError):
            raise CopyError('Invalid source identity.')
        if len(ids[label]) != len(rows): raise CopyError('Duplicate source identity.')
        counts[label] = len(rows)
        for row in rows:
            known = {field.name for field in MODELS[label]._meta.concrete_fields} | set(ALIASES)
            if set(row) - known: raise CopyError('Unknown source field: reviewed adapter required.')
            # No new fixture, login, session, provider secret or authentication identity is copied.
            if row.get('session_id'): raise CopyError('Credential-bearing session field requires sanitized export.')
            relations = {field.name for field in MODELS[label]._meta.concrete_fields if field.is_relation}
            for field, target in RELATIONS.items():
                if field not in relations: continue
                value = row.get(field)
                if label == 'entries' and field == 'analysis' and isinstance(value, dict):
                    value = value.get('id')
                if value is not None:
                    try: UUID(str(value))
                    except (ValueError, TypeError): raise CopyError('Invalid relation identity.')
    for label, rows in bundle.items():
        for row in rows:
            for field, target in RELATIONS.items():
                if field not in {f.name for f in MODELS[label]._meta.concrete_fields if f.is_relation}: continue
                value = row.get(field)
                if isinstance(value, dict): value = value.get('id')
                if value is not None and UUID(str(value)) not in ids.get(target, set()):
                    raise CopyError('Unresolved exported relation; source export required.')
    entries = {str(row['id']): row for row in bundle['entries']}
    analyses = {str(row['id']): row for row in bundle['analyses']}
    for row in analyses.values():
        entry = entries.get(str(row.get('entry')))
        linked = entry.get('analysis') if entry else None
        if isinstance(linked, dict): linked = linked.get('id')
        if entry is None or (linked is not None and str(linked) != str(row['id'])):
            raise CopyError('Entry/analysis relationship mismatch.')
        if entry.get('user') != row.get('user'): raise CopyError('Mixed source ownership in linked records.')
    for entry in entries.values():
        linked = entry.get('analysis')
        if isinstance(linked, dict): linked = linked.get('id')
        if linked is not None and str(analyses[str(linked)]['entry']) != str(entry['id']):
            raise CopyError('Entry/analysis relationship mismatch.')
    owners = {str(row.get('user')) for label, rows in bundle.items() if label != 'settings' for row in rows if 'user' in row}
    if len(owners) > 1: raise CopyError('Mixed source owners require explicit partitioned export.')
    return {'counts': counts, 'manifest_digest': digest(bundle), 'partial': True, 'unresolved_coverage': PARTIAL_MISSING}


def copy_bundle(bundle, *, owner_id, key, source_identity, selected_settings_id=None, partial=False, dry_run=True):
    guard_destination()
    report = inspect_bundle(bundle, partial=partial)
    if len(bundle['settings']) > 1 and not selected_settings_id:
        raise CopyError('Explicit settings selection required; every version will be copied.')
    if selected_settings_id is None and bundle['settings']:
        selected_settings_id = str(bundle['settings'][0]['id'])
    if selected_settings_id and str(selected_settings_id) not in {str(row['id']) for row in bundle['settings']}:
        raise CopyError('Selected settings not in export.')
    if dry_run:
        if not User.objects.filter(pk=owner_id).exists(): raise CopyError('Destination owner required.')
        return report
    with transaction.atomic():
        profile = User.objects.select_for_update().get(pk=owner_id)
        existing = DataCopyRun.objects.select_for_update().filter(key=key).first()
        if existing:
            if existing.owner_id != profile.pk or existing.manifest_digest != report['manifest_digest'] or existing.source_identity != source_identity:
                raise CopyError('Import key is bound to a different source/destination.')
            expected_settings = existing.records.filter(model_label='settings', source_id=selected_settings_id).first() if selected_settings_id else None
            if existing.active_settings_copy != (expected_settings.destination_id if expected_settings else None):
                raise CopyError('Import key is bound to a different settings selection.')
            if existing.state != 'copied': raise CopyError('Import is not replayable in this state.')
            reconcile(existing)
            return report | {'run_id': str(existing.pk), 'replayed': True}
        run = DataCopyRun.objects.create(key=key, owner=profile, source_identity=source_identity,
            manifest_digest=report['manifest_digest'], settings_preimage=profile.active_settings_id, counts=report['counts'])
        mapped = {(label, str(row['id'])): uuid5(run.pk, label + ':' + str(row['id'])) for label, rows in bundle.items() for row in rows}
        deferred = []
        for label in ('entries', 'workflows', 'analyses', 'cumulative', 'questions', 'settings', 'steps', 'citations', 'feedback'):
            model = MODELS[label]
            for source in bundle.get(label, []):
                values = {'id': mapped[label, str(source['id'])]}
                timestamps = {}
                for name, value in source.items():
                    name = ALIASES.get(name, name)
                    if name in {'id', 'user', 'doctor', 'accepted_auth_user_id', 'session_id'}: continue
                    if name in {'created_at', 'updated_at', 'start_time', 'end_time'}:
                        if value is not None:
                            parsed = parse_datetime(value)
                            if parsed is None or parsed.tzinfo is None: raise CopyError('Timezone-aware timestamp required.')
                            timestamps[name] = parsed
                        continue
                    if name in RELATIONS and model._meta.get_field(name).is_relation:
                        if isinstance(value, dict): value = value.get('id')
                        destination = mapped.get((RELATIONS[name], str(value))) if value is not None else None
                        if label == 'entries' and name == 'analysis':
                            deferred.append((values['id'], destination)); continue
                        values[name + '_id'] = destination
                    else:
                        values[name] = value
                if any(field.name == 'user' for field in model._meta.concrete_fields): values['user'] = profile
                if label == 'workflows':
                    values.update(is_historical=True, accepted_auth_user_id=None)
                    if values.get('analysis_id'):
                        target = str(values['analysis_id'])
                        candidates = [mapped.get((kind, target)) for kind in ('analyses', 'cumulative', 'questions')]
                        matches = [value for value in candidates if value is not None]
                        if len(matches) != 1: raise CopyError('Unresolved workflow result identity.')
                        values['analysis_id'] = str(matches[0])
                row = model.objects.create(**values)
                if timestamps: model.objects.filter(pk=row.pk).update(**timestamps)
                DataCopyRecord.objects.create(run=run, model_label=label, source_id=source['id'], destination_id=row.pk, digest='')
        for entry_id, analysis_id in deferred:
            JournalEntry.objects.filter(pk=entry_id, user=profile).update(analysis_id=analysis_id)
        for record in run.records.all():
            record.digest = digest(snapshot(MODELS[record.model_label].objects.get(pk=record.destination_id)))
            record.save(update_fields=['digest'])
        if selected_settings_id:
            run.active_settings_copy = mapped['settings', str(selected_settings_id)]
            profile.active_settings_id = run.active_settings_copy
            profile.save(update_fields=['active_settings'])
        run.state = 'copied'; run.save(update_fields=['active_settings_copy', 'state'])
        reconcile(run)
        return report | {'run_id': str(run.pk), 'replayed': False}


def reconcile(run):
    records = list(run.records.all())
    counts = {}
    for record in records:
        row = MODELS[record.model_label].objects.filter(pk=record.destination_id).first()
        if row is None or digest(snapshot(row)) != record.digest:
            raise CopyError('Copied record was removed or changed; manual reconciliation required.')
        counts[record.model_label] = counts.get(record.model_label, 0) + 1
    expected = {key: value for key, value in run.counts.items() if value}
    if counts != expected: raise CopyError('Record count reconciliation failed.')
    return counts


def rollback(run_id):
    guard_destination()
    with transaction.atomic():
        # Account row serializes app mutations with rollback.
        owner_id = DataCopyRun.objects.get(pk=run_id).owner_id
        profile = User.objects.select_for_update().get(pk=owner_id)
        run = DataCopyRun.objects.select_for_update().get(pk=run_id, owner=profile)
        if run.state == 'rolled_back': return {'state': run.state}
        if run.state != 'copied': raise CopyError('Run cannot be rolled back.')
        reconcile(run)
        grouped = {label: set(run.records.filter(model_label=label).values_list('destination_id', flat=True)) for label in MODELS}
        result_ids = {str(value) for label in ('analyses', 'cumulative', 'questions') for value in grouped[label]}
        if WorkflowExecution.objects.filter(analysis_id__in=result_ids).exclude(pk__in=grouped['workflows']).exists():
            raise CopyError('New workflow references depend on copied results; rollback refused.')
        for label, model in MODELS.items():
            for relation in model._meta.related_objects:
                related = relation.related_model
                foreign = related.objects.filter(**{relation.field.name + '_id__in': grouped[label]})
                permitted = set().union(*(ids for name, ids in grouped.items() if MODELS[name] is related))
                if related is User and relation.field.name == 'active_settings':
                    foreign = foreign.exclude(pk=profile.pk)
                else: foreign = foreign.exclude(pk__in=permitted)
                if foreign.exists(): raise CopyError('New references depend on copied records; rollback refused.')
        expected_pointer = run.active_settings_copy or run.settings_preimage
        if profile.active_settings_id != expected_pointer:
            raise CopyError('Active settings changed; rollback refused.')
        profile.active_settings_id = run.settings_preimage
        profile.save(update_fields=['active_settings'])
        for label in ('feedback', 'citations', 'steps', 'cumulative', 'questions', 'entries', 'analyses', 'workflows', 'settings'):
            MODELS[label].objects.filter(pk__in=grouped[label]).delete()
        run.state = 'rolled_back'; run.save(update_fields=['state'])
        return {'state': run.state, 'counts': run.counts}
