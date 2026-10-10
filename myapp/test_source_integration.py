from django.test import TestCase
from django.utils import timezone
from datetime import timedelta
from myapp import test_auth_security as auth_helpers
from myapp.models import JournalEntry
from myapp.security.contracts import OwnerContext
from myapp.longitudinal import services
from myapp.longitudinal_models import EntryAuditPreference, DreamReflection, LongitudinalState
from myapp.security.freshness import source_preimage


class SourceIntegrationTests(TestCase):
    make_account = auth_helpers.AccountIsolationTests.make_account

    def setUp(self):
        self.auth, self.profile, self.client, self.csrf = self.make_account('source-integration')
        self.owner = OwnerContext(self.profile.pk, self.auth.pk, False, timezone.now() + timedelta(hours=1))
        self.entry = JournalEntry.objects.create(user=self.profile, content='synthetic original river')
        self.headers = {'HTTP_X_CSRFTOKEN': self.csrf, 'HTTP_X_DREAM_PRINCIPAL': str(self.profile.pk)}

    def ticket(self):
        return self.client.get('/api/longitudinal/source-reset-preview/', **self.headers).json()['revision']

    def patch(self, revision, **extra):
        return self.client.patch(f'/api/entries/{self.entry.pk}/update/',
            {'content': 'synthetic changed forest', 'expected_revision': revision,
             'expected_source': source_preimage(self.entry), 'expected_operation': 'update',
             'expected_session': self.client.get('/api/auth/session/').json()['loginDeadline'], **extra},
            content_type='application/json', **self.headers)

    def test_missing_or_stale_warning_revision_rejects_source_write(self):
        self.assertEqual(self.patch(None).status_code, 409)
        old = self.ticket()
        services.create_reflection(self.owner, 'synthetic independent reflection')
        self.assertEqual(self.patch(old).status_code, 409)
        self.entry.refresh_from_db()
        self.assertEqual(self.entry.content, 'synthetic original river')

    def test_valid_edit_atomically_invalidates_and_excludes_generated_source(self):
        services.set_preferences(self.owner, enabled=True)
        services.set_entry_preferences(self.owner, self.entry.pk, included=True, origin='user_reported', language='en')
        linked = services.create_reflection(self.owner, 'synthetic linked reflection', entry_id=self.entry.pk)
        global_note = services.create_reflection(self.owner, 'synthetic global reflection')
        old = self.ticket()
        self.assertEqual(self.patch(old, source_origin='synthetic').status_code, 200)
        self.entry.refresh_from_db()
        self.assertEqual(self.entry.content, 'synthetic changed forest')
        pref = EntryAuditPreference.objects.get(entry=self.entry)
        self.assertEqual(pref.origin, 'synthetic')
        self.assertFalse(pref.included)
        self.assertFalse(DreamReflection.objects.filter(pk=linked.pk).exists())
        self.assertTrue(DreamReflection.objects.filter(pk=global_note.pk).exists())
        self.assertGreater(LongitudinalState.objects.get(user=self.profile).source_revision, old['source'])

    def test_wrong_principal_cannot_write_after_account_switch(self):
        revision = self.ticket()
        self.headers['HTTP_X_DREAM_PRINCIPAL'] = '00000000-0000-0000-0000-000000000001'
        self.assertEqual(self.patch(revision).status_code, 403)
        self.entry.refresh_from_db()
        self.assertEqual(self.entry.content, 'synthetic original river')

    def test_delete_invalidates_before_source_and_preserves_global_notes(self):
        services.create_reflection(self.owner, 'synthetic linked reflection', entry_id=self.entry.pk)
        global_note = services.create_reflection(self.owner, 'synthetic global reflection')
        revision = self.ticket()
        response = self.client.delete(f'/api/entries/{self.entry.pk}/delete/',
            {'expected_revision': revision, 'expected_source': source_preimage(self.entry),
             'expected_operation': 'delete', 'expected_session': self.client.get('/api/auth/session/').json()['loginDeadline']},
            content_type='application/json', **self.headers)
        self.assertEqual(response.status_code, 204)
        self.assertFalse(JournalEntry.objects.filter(pk=self.entry.pk).exists())
        self.assertTrue(DreamReflection.objects.filter(pk=global_note.pk).exists())
        self.assertGreater(LongitudinalState.objects.get(user=self.profile).source_revision, revision['source'])

    def test_fresh_preview_does_not_bless_stale_displayed_source(self):
        JournalEntry.objects.filter(pk=self.entry.pk).update(content='synthetic concurrent newer source')
        self.assertEqual(self.patch(self.ticket()).status_code, 409)
        self.assertEqual(JournalEntry.objects.get(pk=self.entry.pk).content, 'synthetic concurrent newer source')

    def test_noop_keeps_timestamp_and_linked_reflection(self):
        note = services.create_reflection(self.owner, 'synthetic linked reflection', entry_id=self.entry.pk)
        before = self.entry.updated_at
        revision = self.ticket()
        self.assertEqual(self.patch(revision, content=self.entry.content).status_code, 200)
        self.entry.refresh_from_db()
        self.assertEqual(self.entry.updated_at, before)
        self.assertTrue(DreamReflection.objects.filter(pk=note.pk).exists())
        self.assertEqual(self.ticket(), revision)
