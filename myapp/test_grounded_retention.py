"""Actual ORM/session regression matrix; all source text is synthetic."""
from copy import deepcopy
from datetime import timedelta
from unittest.mock import patch
from django.test import TestCase, override_settings
from django.utils import timezone
from django.contrib.sessions.models import Session
from myapp import test_auth_security as auth_helpers
from myapp.models import (Analysis, JournalEntry, WorkflowExecution, WorkflowStep,
                          CumulativeAnalysis, AnalysisFeedback, DoctorProfile)
from myapp.security.owner import require_owner, owned_atomic_write
from myapp.security.contracts import OwnerContext
from myapp.security.analysis_input import analysis_input
from myapp.security.freshness import lock_state, snapshot_current, source_preimage
from myapp.security.jobs import JobContext, current_job, job_write
from myapp.longitudinal import services


class GroundedRetentionTests(TestCase):
    make_account = auth_helpers.AccountIsolationTests.make_account

    def setUp(self):
        self.auth, self.profile, self.client, self.csrf = self.make_account('grounded-owner')
        session = self.client.get('/api/auth/session/').json()
        self.owner = OwnerContext(self.profile.pk, self.auth.pk, False,
            timezone.datetime.fromisoformat(session['loginDeadline']), self.client.session.session_key)
        self.entry = JournalEntry.objects.create(user=self.profile, content='synthetic exact river source')
        _, _, self.snapshot = analysis_input(self.owner, [self.entry.pk], with_snapshot=True)
        self.analysis = self.make_analysis(self.snapshot)
        self.workflow = WorkflowExecution.objects.create(user=self.profile, accepted_auth_user_id=self.auth.pk,
            source_metadata=self.snapshot, workflow_type='cumulative_analysis', routine_name='synthetic',
            status='completed', final_result='synthetic sealed answer')
        self.cumulative = CumulativeAnalysis.objects.create(user=self.profile, workflow_execution=self.workflow,
            analysis='synthetic sealed answer')
        self.workflow.analysis_id = str(self.cumulative.pk)
        self.workflow.save(update_fields=['analysis_id'])
        WorkflowStep.objects.create(execution=self.workflow, step_number=2, name='synthetic generation',
            step_type='answer_generation', status='completed', output_data={
                'coverage': {'source_dependencies': self.snapshot['sources']}, 'claims': [], 'evidence': []})
        AnalysisFeedback.objects.create(user=self.profile, analysis=self.analysis, rating='bad',
            comment='synthetic sealed private feedback')

    def make_analysis(self, snapshot):
        row = Analysis.objects.create(user=self.profile, entry=self.entry, mood='joy', color='pink',
            summary='synthetic sealed summary', interpretation='synthetic sealed interpretation',
            subject='synthetic sealed subject', symbols=['synthetic sealed symbol'], sentiment_score=75,
            grounding_metadata={'source_snapshot': snapshot, 'coverage': {'source_dependencies': snapshot['sources']}})
        self.entry.analysis = row
        self.entry.save(update_fields=['analysis'])
        return row

    def assert_sealed_fields_hidden(self):
        for path in ['/api/data/', '/api/symbols/', '/api/entries/moods/', '/api/cumulative-analyses/',
                     '/api/feedback/stats/', '/api/feedback/preferences/', '/api/feedback/recommendations/']:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertNotIn('synthetic sealed', response.content.decode(), path)
        for key, text in [('title', 'sealed subject'), ('moods', 'joy'), ('symbols', 'sealed symbol'), ('analysis', 'sealed summary')]:
            response = self.client.get('/api/entries/', {key: text})
            self.assertEqual(response.json()['count'], 0, key)
        response = self.client.get(f'/api/entries/{self.entry.pk}/')
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()['analysis'])
        self.assertEqual(self.client.get(f'/api/workflows/{self.workflow.pk}/').status_code, 404)

    def test_original_snapshot_is_visible_and_exact_hash_edit_is_revoked(self):
        self.assertIn('synthetic sealed', self.client.get('/api/data/').content.decode())
        JournalEntry.objects.filter(pk=self.entry.pk).update(content='synthetic externally changed source')
        self.assert_sealed_fields_hidden()

    def test_privacy_and_exclusion_changes_revoke_even_without_text_change(self):
        services.set_preferences(self.owner, enabled=True)
        self.assert_sealed_fields_hidden()

    def test_settings_and_database_doctor_profile_changes_revoke(self):
        self.profile.refresh_from_db()
        setting = self.profile.active_settings
        setting.occupation = 'synthetic changed occupation'
        setting.save()
        self.assert_sealed_fields_hidden()
        # A fresh accepted snapshot includes the database persona contents.
        _, _, fresh = analysis_input(self.owner, [self.entry.pk], with_snapshot=True)
        doctor = DoctorProfile.objects.create(name='Academic', full_prompt='synthetic profile revision')
        self.assertFalse(owned_atomic_write(owner_id=self.profile.pk, expected_auth_user_id=self.auth.pk,
            workflow_id=None, lock_state=lock_state, write=lambda profile: snapshot_current(fresh, self.owner)))

    def test_selected_but_not_packed_source_is_also_a_dependency(self):
        other = JournalEntry.objects.create(user=self.profile, content='synthetic omitted selected record')
        _, _, snapshot = analysis_input(self.owner, [self.entry.pk, other.pk], with_snapshot=True)
        self.analysis.grounding_metadata = {'source_snapshot': snapshot}
        self.analysis.save(update_fields=['grounding_metadata'])
        other.delete()
        self.assertIsNone(self.client.get(f'/api/entries/{self.entry.pk}/').json()['analysis'])

    def test_cross_owner_or_malformed_metadata_never_becomes_history(self):
        bad = deepcopy(self.snapshot)
        bad['owner_id'] = '00000000-0000-0000-0000-000000000001'
        self.analysis.grounding_metadata = {'source_snapshot': bad}
        self.analysis.save(update_fields=['grounding_metadata'])
        self.assertIsNone(self.client.get(f'/api/entries/{self.entry.pk}/').json()['analysis'])

    def test_invalid_dependency_shape_and_workflow_identity_fail_closed(self):
        for source in [{'entry_id': [], 'content_hash': 'a' * 64},
                {'entry_id': str(self.entry.pk), 'content_hash': []}]:
            bad = deepcopy(self.snapshot); bad['sources'] = [source]
            self.analysis.grounding_metadata = {'source_snapshot': bad}
            self.analysis.save(update_fields=['grounding_metadata'])
            self.assertIsNone(self.client.get(f'/api/entries/{self.entry.pk}/').json()['analysis'])
        self.workflow.analysis_id = 'malformed-result-id'
        self.workflow.save(update_fields=['analysis_id'])
        self.assertEqual(self.client.get(f'/api/workflows/{self.workflow.pk}/').status_code, 404)
        self.workflow.analysis_id = str(self.cumulative.pk)
        self.workflow.save(update_fields=['analysis_id'])
        self.workflow.steps.update(output_data={'coverage': []})
        self.assertEqual(self.client.get(f'/api/workflows/{self.workflow.pk}/').status_code, 404)
        self.analysis.grounding_metadata = {'version': 1}
        self.analysis.save(update_fields=['grounding_metadata'])
        self.assertIsNone(self.client.get(f'/api/entries/{self.entry.pk}/').json()['analysis'])

    def test_job_cannot_publish_after_capture_changes_or_request_logout(self):
        token = current_job.set(JobContext(self.owner, (self.entry.pk,), self.snapshot, True))
        try:
            job_write(lambda profile: None)
            self.client.post('/api/auth/logout/', HTTP_X_CSRFTOKEN=self.csrf)
            with self.assertRaises(Exception):
                job_write(lambda profile: Analysis.objects.update(summary='must never commit'))
        finally:
            current_job.reset(token)
        self.analysis.refresh_from_db()
        self.assertEqual(self.analysis.summary, 'synthetic sealed summary')

    def test_accepted_durable_job_still_checks_revisions_after_logout(self):
        durable_owner = OwnerContext(self.profile.pk, self.auth.pk, False, self.owner.login_deadline)
        token = current_job.set(JobContext(durable_owner, (self.entry.pk,), self.snapshot))
        try:
            self.client.post('/api/auth/logout/', HTTP_X_CSRFTOKEN=self.csrf)
            self.assertEqual(job_write(lambda profile: 'accepted'), 'accepted')
            JournalEntry.objects.filter(pk=self.entry.pk).update(content='synthetic later source')
            with self.assertRaises(Exception):
                job_write(lambda profile: Analysis.objects.update(summary='must never commit'))
        finally:
            current_job.reset(token)

    @override_settings(DREAM_ANALYZER_URL='')
    def test_source_edit_retains_old_version_and_new_version_is_possible(self):
        old_id = self.analysis.pk
        revision = self.client.get('/api/longitudinal/source-reset-preview/', HTTP_X_DREAM_PRINCIPAL=str(self.profile.pk)).json()['revision']
        response = self.client.patch(f'/api/entries/{self.entry.pk}/update/', {
            'content': 'synthetic fresh version', 'expected_source': source_preimage(self.entry),
            'expected_operation': 'update', 'expected_session': self.owner.login_deadline.isoformat(),
            'expected_revision': revision}, content_type='application/json', HTTP_X_CSRFTOKEN=self.csrf,
            HTTP_X_DREAM_PRINCIPAL=str(self.profile.pk))
        self.assertEqual(response.status_code, 200)
        self.entry.refresh_from_db()
        self.assertIsNone(self.entry.analysis_id)
        old = Analysis.objects.get(pk=old_id)
        self.assertEqual(old.interpretation, 'synthetic sealed interpretation')
        self.assertEqual(old.grounding_metadata, {'invalidated': True})
        _, _, snapshot = analysis_input(self.owner, [self.entry.pk], with_snapshot=True)
        new = self.make_analysis(snapshot)
        self.assertNotEqual(new.pk, old_id)
        self.assertEqual(Analysis.objects.filter(entry=self.entry).count(), 2)
        self.assertEqual(self.client.get(f'/api/entries/{self.entry.pk}/').json()['analysis']['id'], str(new.pk))

    def test_historical_rows_are_labelled_and_excluded_from_current_statistics(self):
        self.analysis.grounding_metadata = {}
        self.analysis.is_historical = True
        self.analysis.save(update_fields=['grounding_metadata', 'is_historical'])
        self.assertEqual(self.client.get(f'/api/entries/{self.entry.pk}/').json()['analysis']['source_validation'], 'historical_unvalidated')
        self.assertEqual(self.client.get('/api/data/').json()['analyses'], [])
        self.cumulative.is_historical = True
        self.cumulative.save(update_fields=['is_historical'])
        AnalysisFeedback.objects.create(user=self.profile, cumulative_analysis=self.cumulative, rating='good')
        self.assertEqual(self.client.get('/api/feedback/stats/').json()['total_feedback'], 0)
        self.assertEqual(self.client.get('/api/cumulative-analyses/').json()['results'][0]['source_validation'], 'historical_unvalidated')

    def test_linked_doctor_id_is_captured_even_when_display_name_conflicts(self):
        selected = DoctorProfile.objects.create(name='Synthetic Linked', full_prompt='synthetic selected prompt')
        DoctorProfile.objects.create(name='Academic', full_prompt='synthetic name fallback')
        self.profile.refresh_from_db()
        setting = self.profile.active_settings
        setting.doctor = selected; setting.doctor_personality = 'Academic'; setting.save()
        _, config, capture = analysis_input(self.owner, [self.entry.pk], with_snapshot=True)
        self.assertEqual(config['doctor_id'], str(selected.pk))
        selected.full_prompt = 'synthetic later linked revision'; selected.save()
        self.assertFalse(owned_atomic_write(owner_id=self.profile.pk, expected_auth_user_id=self.auth.pk,
            workflow_id=None, lock_state=lock_state, write=lambda profile: snapshot_current(capture, self.owner)))

    def test_new_orphaned_workflow_result_never_becomes_legacy_history(self):
        self.workflow.delete()
        JournalEntry.objects.filter(pk=self.entry.pk).update(content='synthetic later changed source')
        self.assertNotIn('synthetic sealed', self.client.get('/api/cumulative-analyses/').content.decode())
        self.cumulative.refresh_from_db()
        self.assertFalse(self.cumulative.is_historical)
        self.assertIsNone(self.cumulative.workflow_execution_id)

    def test_request_bound_job_rejects_changed_password_and_expired_deadline(self):
        token = current_job.set(JobContext(self.owner, (self.entry.pk,), self.snapshot, True))
        try:
            self.auth.set_password('synthetic changed password'); self.auth.save()
            with self.assertRaises(Exception): job_write(lambda profile: None)
        finally: current_job.reset(token)
        expired = OwnerContext(self.profile.pk, self.auth.pk, False, timezone.now() - timedelta(seconds=1), self.owner.session_key)
        token = current_job.set(JobContext(expired, (self.entry.pk,), self.snapshot, True))
        try:
            with self.assertRaises(Exception): job_write(lambda profile: None)
        finally: current_job.reset(token)

    @override_settings(DREAM_ANALYZER_URL='http://isolated-analyzer.invalid')
    def test_single_bridge_saves_metadata_and_rejects_late_privacy_change(self):
        from types import SimpleNamespace
        def response(*args, **kwargs):
            _, _, snapshot = analysis_input(self.owner, [self.entry.pk], with_snapshot=True)
            result = {key: getattr(self.analysis, key) for key in ('mood', 'summary', 'color', 'interpretation',
                'negative', 'subject', 'sentiment_score', 'doctor_personality', 'weights', 'symbols')}
            result.update(source_snapshot=snapshot, coverage={'source_dependencies': snapshot['sources']},
                          evidence=[], claims=[], provenance={}, calibration_status='unavailable')
            return SimpleNamespace(status_code=200, json=lambda: result)
        def save(content):
            self.entry.refresh_from_db()
            revision = self.client.get('/api/longitudinal/source-reset-preview/', HTTP_X_DREAM_PRINCIPAL=str(self.profile.pk)).json()['revision']
            return self.client.patch(f'/api/entries/{self.entry.pk}/update/', {
                'content': content, 'expected_source': source_preimage(self.entry), 'expected_revision': revision,
                'expected_operation': 'update', 'expected_session': self.owner.login_deadline.isoformat()},
                content_type='application/json', HTTP_X_CSRFTOKEN=self.csrf, HTTP_X_DREAM_PRINCIPAL=str(self.profile.pk))
        with patch('myapp.views.requests.post', side_effect=response):
            result = save('synthetic successfully reanalyzed new version')
        self.assertEqual(result.status_code, 200)
        self.entry.refresh_from_db()
        self.assertIsNotNone(self.entry.analysis_id)
        self.assertIn('source_snapshot', self.entry.analysis.grounding_metadata)
        self.assertEqual(Analysis.objects.filter(entry=self.entry).count(), 2)
        def late_response(*args, **kwargs):
            result = response(*args, **kwargs)
            services.set_preferences(self.owner, enabled=True)
            return result
        with patch('myapp.views.requests.post', side_effect=late_response):
            result = save('synthetic late privacy revoked version')
        self.assertEqual(result.status_code, 200)  # Raw draft saved; no stale derived result committed.
        self.entry.refresh_from_db()
        self.assertIsNone(self.entry.analysis_id)
        self.assertEqual(Analysis.objects.filter(entry=self.entry).count(), 2)

    @override_settings(DREAM_ANALYZER_URL='http://isolated-analyzer.invalid')
    def test_unchanged_source_reanalyzes_after_settings_change_without_reset(self):
        from types import SimpleNamespace
        self.profile.refresh_from_db()
        setting = self.profile.active_settings
        setting.occupation = 'synthetic changed context'; setting.save()
        before = self.client.get('/api/longitudinal/source-reset-preview/').json()['revision']
        def respond(*args, **kwargs):
            _, _, capture = analysis_input(self.owner, [self.entry.pk], with_snapshot=True)
            fields = {key: getattr(self.analysis, key) for key in ('mood', 'summary', 'color', 'interpretation',
                'negative', 'subject', 'sentiment_score', 'doctor_personality', 'weights', 'symbols')}
            return SimpleNamespace(status_code=200, json=lambda: fields | {'source_snapshot': capture,
                'coverage': {'source_dependencies': capture['sources']}})
        with patch('myapp.views.requests.post', side_effect=respond) as provider:
            response = self.client.patch(f'/api/entries/{self.entry.pk}/update/', {
                'content': self.entry.content, 'expected_source': source_preimage(self.entry),
                'expected_revision': before, 'expected_operation': 'update', 'expected_session': self.owner.login_deadline.isoformat()},
                content_type='application/json', HTTP_X_CSRFTOKEN=self.csrf, HTTP_X_DREAM_PRINCIPAL=str(self.profile.pk))
        self.assertEqual(response.status_code, 200, response.content)
        self.assertEqual(provider.call_count, 1)
        self.entry.refresh_from_db()
        self.assertNotEqual(self.entry.analysis_id, self.analysis.pk)
        self.assertEqual(self.client.get('/api/longitudinal/source-reset-preview/').json()['revision'], before)
