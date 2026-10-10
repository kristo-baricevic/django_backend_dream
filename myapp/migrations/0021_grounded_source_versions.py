from django.db import migrations, models
import django.db.models.deletion
from django.core.serializers.json import DjangoJSONEncoder
import hashlib
import json


LEDGER_MODELS = {'entries': 'JournalEntry', 'analyses': 'Analysis',
    'cumulative': 'CumulativeAnalysis', 'questions': 'CustomQuestion',
    'settings': 'Settings', 'workflows': 'WorkflowExecution', 'steps': 'WorkflowStep',
    'citations': 'StepCitation', 'feedback': 'AnalysisFeedback',
    'longitudinal_reflections': 'DreamReflection'}
APPROVED_ADDITIONS = {'analyses': {'grounding_metadata': {}, 'is_historical': True},
    'cumulative': {'is_historical': True}, 'questions': {'is_historical': True},
    'workflows': {'source_metadata': {}}}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        cls=DjangoJSONEncoder).encode()).hexdigest()


def rebase_unchanged_ledger(apps, schema_editor):
    """Approve schema defaults only when every original field still matches.

    Changed/missing/unknown records keep version 1 and fail reconciliation.
    No private content, source manifest, owner or timestamps are rewritten.
    """
    Record = apps.get_model('myapp', 'DataCopyRecord')
    for record in Record.objects.filter(run__state='copied', digest_version=1).select_related('run').iterator():
        name = LEDGER_MODELS.get(record.model_label)
        if name is None:
            continue
        row = apps.get_model('myapp', name).objects.filter(pk=record.destination_id).first()
        if row is None or (hasattr(row, 'user_id') and row.user_id != record.run.owner_id):
            continue
        values = {field.attname: getattr(row, field.attname) for field in row._meta.concrete_fields}
        additions = APPROVED_ADDITIONS.get(record.model_label, {})
        if any(values.get(key) != expected for key, expected in additions.items()):
            continue
        original = {key: value for key, value in values.items() if key not in additions}
        if _digest(original) != record.digest:
            continue
        record.digest = _digest(values)
        record.digest_version = 2
        record.save(update_fields=['digest', 'digest_version'])


def restore_unchanged_ledger(apps, schema_editor):
    Record = apps.get_model('myapp', 'DataCopyRecord')
    for record in Record.objects.filter(run__state='copied', digest_version=2).iterator():
        name = LEDGER_MODELS.get(record.model_label)
        if name is None:
            continue
        row = apps.get_model('myapp', name).objects.filter(pk=record.destination_id).first()
        if row is None:
            continue
        values = {field.attname: getattr(row, field.attname) for field in row._meta.concrete_fields}
        additions = APPROVED_ADDITIONS.get(record.model_label, {})
        if _digest(values) != record.digest or any(values.get(key) != expected for key, expected in additions.items()):
            continue
        record.digest = _digest({key: value for key, value in values.items() if key not in additions})
        record.digest_version = 1
        record.save(update_fields=['digest', 'digest_version'])


def refuse_lossy_reverse(apps, schema_editor):
    analysis = apps.get_model('myapp', 'Analysis')
    if analysis.objects.values('entry_id').annotate(n=models.Count('pk')).filter(n__gt=1).exists():
        raise RuntimeError('Analysis versions must be preserved; restoring uniqueness requires a reviewed migration.')
    if (analysis.objects.exclude(grounding_metadata={}).exists()
            or apps.get_model('myapp', 'WorkflowExecution').objects.exclude(source_metadata={}).exists()):
        raise RuntimeError('Grounding metadata must be preserved; reverse migration requires a reviewed export.')


def label_existing_history(apps, schema_editor):
    for name in ('Analysis', 'CumulativeAnalysis', 'CustomQuestion'):
        apps.get_model('myapp', name).objects.all().update(is_historical=True)


class Migration(migrations.Migration):
    dependencies = [('myapp', '0020_longitudinal_evidence_profile')]
    operations = [
        *[migrations.AddField(model_name=name, name='is_historical',
              field=models.BooleanField(default=False, editable=False))
          for name in ('analysis', 'cumulativeanalysis', 'customquestion')],
        migrations.RunPython(label_existing_history, migrations.RunPython.noop),
        migrations.AddField(model_name='analysis', name='grounding_metadata',
            field=models.JSONField(default=dict, editable=False)),
        migrations.AddField(model_name='workflowexecution', name='source_metadata',
            field=models.JSONField(default=dict, editable=False)),
        migrations.AddField(model_name='datacopyrecord', name='digest_version',
            field=models.PositiveSmallIntegerField(default=1, editable=False)),
        migrations.RunPython(rebase_unchanged_ledger, restore_unchanged_ledger),
        migrations.AlterField(model_name='datacopyrecord', name='digest_version',
            field=models.PositiveSmallIntegerField(default=2, editable=False)),
        migrations.AlterField(model_name='analysis', name='entry',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE,
                                    related_name='analysis_versions', to='myapp.journalentry')),
        migrations.RunPython(migrations.RunPython.noop, refuse_lossy_reverse),
    ]
