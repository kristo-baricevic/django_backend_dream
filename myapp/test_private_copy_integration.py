from copy import deepcopy
from uuid import uuid4, uuid5, UUID
from django.test import TestCase
from myapp.test_data_copy import synthetic_bundle
from myapp.models import User, JournalEntry, DreamReflection, DataCopyRun, DataCopyRecord, ThemeReview
from myapp.longitudinal.copy_contract import COPY_POLICY
from myapp.security.data_copy import copy_bundle, rollback, reconcile, digest, snapshot, CopyError


class PrivateCopyIntegrationTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create(email='private-copy-target@example.invalid', import_ready=False)
        self.source_id = str(uuid4())
        self.bundle = synthetic_bundle()
        reflection = str(uuid4())
        self.private = {'model_labels': list(COPY_POLICY), 'index_namespaces': [],
            'source_owner_id': self.source_id, 'selected_reflection_ids': [reflection],
            'reflections': [{'id': reflection, 'user_id': self.source_id,
                'entry_id': self.bundle['entries'][0]['id'], 'text': 'synthetic explicitly selected authored reflection',
                'created_at': '2025-01-02T12:00:00Z', 'updated_at': '2025-01-03T12:00:00Z',
                'revision': 7, 'use_for_profile': True, 'use_for_chat': True}]}
        self.kwargs = {'owner_id': self.owner.pk, 'key': 'private-copy-synthetic',
            'source_identity': {'origin': 'https://source.example.invalid'}, 'partial': True, 'private_bundle': self.private}

    def test_dry_run_selection_and_inventory_are_read_only(self):
        report = copy_bundle(self.bundle, **self.kwargs)
        self.assertEqual(len(report['private_policies']), 14)
        self.assertEqual(report['counts']['longitudinal_reflections'], 1)
        self.assertFalse(DreamReflection.objects.exists())
        self.assertFalse(DataCopyRun.objects.exists())

    def test_selected_reflection_mapping_exact_digest_replay_and_clean_rollback(self):
        report = copy_bundle(self.bundle, **self.kwargs, dry_run=False)
        note = DreamReflection.objects.get(user=self.owner)
        record = DataCopyRecord.objects.get(model_label='longitudinal_reflections')
        self.assertEqual(note.pk, uuid5(UUID(report['run_id']), 'dreamreflection:' + self.private['selected_reflection_ids'][0]))
        self.assertEqual(note.entry_id, JournalEntry.objects.get(user=self.owner).pk)
        self.assertFalse(note.use_for_profile); self.assertFalse(note.use_for_chat)
        self.assertEqual(note.revision, 1)
        self.assertEqual(note.created_at.year, 2025)
        self.assertEqual(record.digest, digest(snapshot(note)))
        self.assertTrue(copy_bundle(self.bundle, **self.kwargs, dry_run=False)['replayed'])
        self.assertEqual(reconcile(DataCopyRun.objects.get(pk=report['run_id']))['longitudinal_reflections'], 1)
        rollback(report['run_id'])
        self.assertFalse(DreamReflection.objects.exists()); self.assertFalse(JournalEntry.objects.exists())
        self.owner.refresh_from_db(); self.assertIsNone(self.owner.auth_user_id)

    def test_later_private_dependency_blocks_rollback_without_cascade(self):
        report = copy_bundle(self.bundle, **self.kwargs, dry_run=False)
        note = DreamReflection.objects.get(user=self.owner)
        later = DreamReflection.objects.create(user=self.owner, entry=note.entry, text='synthetic later authored note')
        with self.assertRaises(CopyError): rollback(report['run_id'])
        self.assertTrue(DreamReflection.objects.filter(pk=note.pk).exists())
        self.assertTrue(DreamReflection.objects.filter(pk=later.pk).exists())

    def test_wrong_owner_missing_entry_or_unknown_namespace_stops_all_copy(self):
        for kind in ('owner', 'entry', 'namespace', 'model', 'timestamp'):
            value = deepcopy(self.private)
            if kind == 'owner': value['reflections'][0]['user_id'] = str(uuid4())
            if kind == 'entry': value['reflections'][0]['entry_id'] = str(uuid4())
            if kind == 'namespace': value['index_namespaces'] = ['shared-faiss-private']
            if kind == 'model': value['model_labels'].append('myapp.UnknownPrivate')
            if kind == 'timestamp': value['reflections'][0]['updated_at'] = '2025-01-03T12:00:00'
            with self.subTest(kind=kind), self.assertRaises(CopyError):
                copy_bundle(self.bundle, **(self.kwargs | {'private_bundle': value}), dry_run=False)
            self.assertFalse(DataCopyRun.objects.exists())

    def test_selection_change_cannot_broaden_idempotent_run(self):
        copy_bundle(self.bundle, **self.kwargs, dry_run=False)
        changed = deepcopy(self.private)
        changed['selected_reflection_ids'] = []
        with self.assertRaises(CopyError):
            copy_bundle(self.bundle, **(self.kwargs | {'private_bundle': changed}), dry_run=False)
        self.assertEqual(DreamReflection.objects.count(), 1)
