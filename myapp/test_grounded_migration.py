from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase
from uuid import uuid4, uuid5
from myapp.security.data_copy import digest, snapshot, reconcile, rollback, copy_bundle, CopyError
from myapp.test_data_copy import synthetic_bundle


class GroundedVersionMigrationTests(TransactionTestCase):
    def migrate(self, target):
        executor = MigrationExecutor(connection)
        executor.migrate([('myapp', target)])
        return executor.loader.project_state([('myapp', target)]).apps

    def foundation_copy(self, *, changed=False):
        old = self.migrate('0020_longitudinal_evidence_profile')
        bundle = synthetic_bundle()
        bundle['cumulative'] = [{'id': str(uuid4()), 'analysis': 'synthetic cumulative history'}]
        bundle['questions'] = [{'id': str(uuid4()), 'question': 'synthetic question', 'answer': 'synthetic history answer'}]
        bundle['workflows'] = [{'id': str(uuid4()), 'workflow_type': 'cumulative_analysis',
            'routine_name': 'synthetic old workflow', 'is_historical': True}]
        source = {'origin': 'https://synthetic.example.invalid'}
        owner = old.get_model('myapp', 'User').objects.create(email='old-copy@example.invalid', import_ready=False)
        Run = old.get_model('myapp', 'DataCopyRun')
        run = Run.objects.create(key='foundation-rehearsal', owner=owner, source_identity=source,
            manifest_digest=digest(bundle), state='copied', counts={k: len(v) for k, v in bundle.items()})
        mapped = {(k, row['id']): uuid5(run.pk, k + ':' + row['id']) for k, rows in bundle.items() for row in rows}
        entry = old.get_model('myapp', 'JournalEntry').objects.create(
            id=mapped['entries', bundle['entries'][0]['id']], user=owner, content='synthetic source')
        analysis = old.get_model('myapp', 'Analysis').objects.create(
            id=mapped['analyses', bundle['analyses'][0]['id']], user=owner, entry=entry,
            mood='joy', color='pink', summary='synthetic summary', interpretation='synthetic interpretation', subject='synthetic')
        old.get_model('myapp', 'JournalEntry').objects.filter(pk=entry.pk).update(analysis_id=analysis.pk)
        entry.refresh_from_db()
        rows = {'entries': entry, 'analyses': analysis}
        for label, name, fields in [('cumulative', 'CumulativeAnalysis', {'analysis': 'synthetic cumulative history'}),
                ('questions', 'CustomQuestion', {'question': 'synthetic question', 'answer': 'synthetic history answer'}),
                ('workflows', 'WorkflowExecution', {'workflow_type': 'cumulative_analysis', 'routine_name': 'synthetic old workflow', 'is_historical': True}),
                ('settings', 'Settings', {'doctor_personality': 'Academic'})]:
            rows[label] = old.get_model('myapp', name).objects.create(
                id=mapped[label, bundle[label][0]['id']], user=owner, **fields)
        run.active_settings_copy = rows['settings'].pk
        run.save(update_fields=['active_settings_copy'])
        old.get_model('myapp', 'User').objects.filter(pk=owner.pk).update(active_settings_id=rows['settings'].pk)
        for label, row in rows.items():
            row.refresh_from_db()
            old.get_model('myapp', 'DataCopyRecord').objects.create(run=run, model_label=label,
                source_id=bundle[label][0]['id'], destination_id=row.pk, digest=digest(snapshot(row)))
        if changed:
            old.get_model('myapp', 'Analysis').objects.filter(pk=analysis.pk).update(interpretation='synthetic later private edit')
        self.migrate('0021_grounded_source_versions')
        from myapp.models import DataCopyRun
        return DataCopyRun.objects.get(pk=run.pk), bundle, source

    def test_foundation_copy_survives_migration_replay_reconciliation_and_rollback(self):
        run, bundle, source = self.foundation_copy()
        from myapp.security.data_copy import MODELS
        for record in run.records.all():
            self.assertEqual(record.digest_version, 2, record.model_label)
            self.assertEqual(digest(snapshot(MODELS[record.model_label].objects.get(pk=record.destination_id))),
                record.digest, record.model_label)
        self.assertEqual(reconcile(run), {key: 1 for key in bundle})
        self.assertEqual(set(run.records.values_list('digest_version', flat=True)), {2})
        self.assertTrue(copy_bundle(bundle, owner_id=run.owner_id, key=run.key,
            source_identity=source, partial=True, dry_run=False)['replayed'])
        self.assertEqual(rollback(run.pk)['state'], 'rolled_back')
        from myapp.models import JournalEntry, Analysis
        self.assertFalse(JournalEntry.objects.filter(user_id=run.owner_id).exists())
        self.assertFalse(Analysis.objects.filter(user_id=run.owner_id).exists())

    def test_schema_rebase_never_blesses_changed_private_content(self):
        run, bundle, source = self.foundation_copy(changed=True)
        self.assertEqual(run.records.get(model_label='analyses').digest_version, 1)
        with self.assertRaises(CopyError): reconcile(run)
        with self.assertRaises(CopyError): rollback(run.pk)
        from myapp.models import Analysis
        self.assertEqual(Analysis.objects.get(user_id=run.owner_id).interpretation, 'synthetic later private edit')

    def test_existing_pair_is_preserved_and_lossy_reverse_refused(self):
        executor = MigrationExecutor(connection)
        executor.migrate([('myapp', '0020_longitudinal_evidence_profile')])
        before = executor.loader.project_state([('myapp', '0020_longitudinal_evidence_profile')]).apps
        entry = before.get_model('myapp', 'JournalEntry').objects.create(content='synthetic pre-migration record')
        old = before.get_model('myapp', 'Analysis').objects.create(entry_id=entry.pk, mood='joy', color='pink',
            summary='synthetic legacy summary', interpretation='synthetic preserved interpretation', subject='synthetic')
        before.get_model('myapp', 'JournalEntry').objects.filter(pk=entry.pk).update(analysis_id=old.pk)
        executor = MigrationExecutor(connection)
        executor.migrate([('myapp', '0021_grounded_source_versions')])
        after = executor.loader.project_state([('myapp', '0021_grounded_source_versions')]).apps
        Analysis = after.get_model('myapp', 'Analysis')
        self.assertEqual(Analysis.objects.get(pk=old.pk).interpretation, 'synthetic preserved interpretation')
        self.assertEqual(after.get_model('myapp', 'JournalEntry').objects.get(pk=entry.pk).analysis_id, old.pk)
        new = Analysis.objects.create(entry_id=entry.pk, mood='interest', color='pink', summary='synthetic new',
            interpretation='synthetic fresh version', subject='synthetic')
        with self.assertRaises(RuntimeError):
            MigrationExecutor(connection).migrate([('myapp', '0020_longitudinal_evidence_profile')])
        self.assertEqual(Analysis.objects.filter(entry_id=entry.pk).count(), 2)
        new.delete()
        # Clean reversible rehearsal after removing only this test's synthetic version.
        MigrationExecutor(connection).migrate([('myapp', '0020_longitudinal_evidence_profile')])
        MigrationExecutor(connection).migrate([('myapp', '0021_grounded_source_versions')])
