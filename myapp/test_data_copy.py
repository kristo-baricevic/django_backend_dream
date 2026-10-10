from uuid import uuid4
from django.test import TestCase
from myapp.models import User, JournalEntry, Analysis, Settings, CumulativeAnalysis, DataCopyRun
from myapp.security.data_copy import copy_bundle, rollback, CopyError


def synthetic_bundle():
    entry, analysis, setting = (str(uuid4()) for _ in range(3))
    timestamp = '2025-01-01T12:00:00Z'
    return {'entries': [{'id': entry, 'user': None, 'content': 'synthetic source',
                        'analysis': {'id': analysis}, 'created_at': timestamp, 'updated_at': timestamp}],
        'analyses': [{'id': analysis, 'user': None, 'entry': entry, 'mood': 'joy', 'color': 'pink',
                     'summary': 'synthetic summary', 'interpretation': 'synthetic interpretation',
                     'subject': 'synthetic', 'created_at': timestamp, 'updated_at': timestamp}],
        'cumulative': [], 'questions': [], 'settings': [{'id': setting, 'doctorPersonality': 'Academic',
        'doctorImage': '', 'personality': 'INTJ', 'occupation': '', 'medicalHistory': {},
        'influence': {}, 'astrology': {}}]}


class CopyTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create(email='copy-target@example.invalid', import_ready=False)
        self.bundle = synthetic_bundle()
        self.kwargs = {'owner_id': self.owner.pk, 'key': 'synthetic-copy',
                       'source_identity': {'origin': 'https://source.example.invalid'}, 'partial': True}

    def test_dry_run_preserves_source_and_creates_nothing(self):
        report = copy_bundle(self.bundle, **self.kwargs)
        self.assertEqual(report['counts']['entries'], 1)
        self.assertFalse(JournalEntry.objects.exists())
        self.assertFalse(DataCopyRun.objects.exists())

    def test_cumulative_analysis_is_preserved_as_text(self):
        self.bundle['cumulative'] = [{'id': str(uuid4()), 'user': None, 'analysis': 'synthetic cumulative text',
                                      'doctor_personality': 'Academic', 'weights': {}}]
        report = copy_bundle(self.bundle, **self.kwargs, dry_run=False)
        self.assertEqual(CumulativeAnalysis.objects.get().analysis, 'synthetic cumulative text')
        rollback(report['run_id'])
        self.assertFalse(CumulativeAnalysis.objects.exists())

    def test_empty_settings_copy_cannot_rollback_over_later_settings_selection(self):
        self.bundle['settings'] = []
        report = copy_bundle(self.bundle, **self.kwargs, dry_run=False)
        self.owner.active_settings = Settings.objects.create(user=self.owner)
        self.owner.save(update_fields=['active_settings'])
        with self.assertRaises(CopyError): rollback(report['run_id'])
        self.owner.refresh_from_db(); self.assertIsNotNone(self.owner.active_settings_id)

    def test_copy_rewires_cycle_and_preserves_timestamps_without_login(self):
        report = copy_bundle(self.bundle, **self.kwargs, dry_run=False)
        entry = JournalEntry.objects.get(user=self.owner)
        self.assertNotEqual(str(entry.pk), self.bundle['entries'][0]['id'])
        self.assertEqual(entry.analysis.entry_id, entry.pk)
        self.assertEqual(entry.analysis.user_id, self.owner.pk)
        self.assertEqual(entry.created_at.year, 2025)
        self.owner.refresh_from_db()
        self.assertIsNone(self.owner.auth_user_id)
        self.assertFalse(self.owner.import_ready)
        self.assertEqual(self.owner.active_settings.user_id, self.owner.pk)
        again = copy_bundle(self.bundle, **self.kwargs, dry_run=False)
        self.assertTrue(again['replayed'])
        self.assertEqual(JournalEntry.objects.count(), 1)
        rollback(report['run_id'])
        self.assertFalse(JournalEntry.objects.exists()); self.assertFalse(Analysis.objects.exists())
        self.owner.refresh_from_db(); self.assertIsNone(self.owner.active_settings_id)
        self.assertEqual(rollback(report['run_id'])['state'], 'rolled_back')

    def test_rollback_refuses_changed_record_and_keeps_data(self):
        report = copy_bundle(self.bundle, **self.kwargs, dry_run=False)
        JournalEntry.objects.update(content='synthetic later edit')
        with self.assertRaises(CopyError): rollback(report['run_id'])
        self.assertEqual(JournalEntry.objects.get().content, 'synthetic later edit')

    def test_bad_relation_or_mixed_owner_creates_nothing(self):
        self.bundle['analyses'][0]['entry'] = str(uuid4())
        with self.assertRaises(CopyError): copy_bundle(self.bundle, **self.kwargs, dry_run=False)
        self.assertFalse(DataCopyRun.objects.exists())

    def test_import_key_cannot_change_source_or_destination(self):
        copy_bundle(self.bundle, **self.kwargs, dry_run=False)
        self.bundle['entries'][0]['content'] = 'synthetic changed export'
        with self.assertRaises(CopyError): copy_bundle(self.bundle, **self.kwargs, dry_run=False)
        self.assertEqual(JournalEntry.objects.count(), 1)

    def test_duplicate_settings_require_explicit_selection_and_preserve_all_versions(self):
        second = dict(self.bundle['settings'][0], id=str(uuid4()))
        self.bundle['settings'].append(second)
        with self.assertRaises(CopyError): copy_bundle(self.bundle, **self.kwargs, dry_run=False)
        copy_bundle(self.bundle, **self.kwargs, selected_settings_id=second['id'], dry_run=False)
        self.assertEqual(Settings.objects.count(), 2)
        with self.assertRaises(CopyError):
            copy_bundle(self.bundle, **self.kwargs, selected_settings_id=self.bundle['settings'][0]['id'], dry_run=False)

    def test_rollback_refuses_new_string_workflow_reference(self):
        from myapp.models import WorkflowExecution
        report=copy_bundle(self.bundle, **self.kwargs, dry_run=False)
        WorkflowExecution.objects.create(user=self.owner, analysis_id=str(Analysis.objects.get().pk),
                                         workflow_type='test', routine_name='synthetic new reference')
        with self.assertRaises(CopyError): rollback(report['run_id'])
        self.assertTrue(Analysis.objects.exists())

    def test_partial_acknowledgment_and_unknown_resources_fail_closed(self):
        with self.assertRaises(CopyError): copy_bundle(self.bundle, **(self.kwargs | {'partial': False}))
        self.bundle['private_documents'] = []
        with self.assertRaises(CopyError): copy_bundle(self.bundle, **self.kwargs)
