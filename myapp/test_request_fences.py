"""Actual PostgreSQL sessions, account intent and late commit fences.

No provider/HTTP mock grants authorization. Mocks only interrupt a request
after its genuine cookie authorization or replace the external analyzer.
"""
from datetime import timedelta
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch
from django.contrib.sessions.models import Session
from django.db import close_old_connections, connection, transaction
from django.test import TestCase, TransactionTestCase, override_settings
from django.utils import timezone
from rest_framework.exceptions import PermissionDenied
from myapp import test_auth_security as auth_helpers
from myapp.models import JournalEntry, Analysis, AnalysisFeedback
from myapp.security.contracts import OwnerContext
from myapp.security.owner import require_owner, validate_request_session
from myapp.security.jobs import JobContext, current_job, job_write
from myapp.security.analysis_input import analysis_input
from myapp.security.freshness import source_preimage


class RequestFenceTests(TestCase):
    make_account = auth_helpers.AccountIsolationTests.make_account

    def setUp(self):
        self.auth, self.profile, self.client, self.csrf = self.make_account('request-fence')
        data = self.client.get('/api/auth/session/').json()
        self.owner = OwnerContext(self.profile.pk, self.auth.pk, False,
            timezone.datetime.fromisoformat(data['loginDeadline']), self.client.session.session_key)
        self.entry = JournalEntry.objects.create(user=self.profile, content='synthetic request source')

    def test_old_tab_input_cannot_target_switched_account_cookie(self):
        other_auth, other, other_client, other_csrf = self.make_account('request-fence-other')
        other.refresh_from_db()
        previous = other.active_settings.occupation
        hints = {'HTTP_X_DREAM_PRINCIPAL': str(self.profile.pk),
            'HTTP_X_DREAM_SESSION': self.owner.login_deadline.isoformat(), 'HTTP_X_CSRFTOKEN': other_csrf}
        create = other_client.post('/api/entries/create/', {'content': 'synthetic account A draft'},
            content_type='application/json', **hints)
        self.assertEqual(create.status_code, 403)
        settings = other_client.post('/api/settings/update/', {'occupation': 'synthetic account A occupation'},
            content_type='application/json', **hints)
        self.assertEqual(settings.status_code, 403)
        self.assertFalse(JournalEntry.objects.filter(user=other).exists())
        other.active_settings.refresh_from_db()
        self.assertEqual(other.active_settings.occupation, previous)
        self.assertEqual(other_client.get('/api/entries/', **hints).status_code, 403)
        self.assertEqual(other_client.get('/api/auth/session/').status_code, 200,
            'reject-only intent mismatch must not log out the actual current account')

    def test_password_change_after_initial_auth_blocks_session_validation(self):
        def change_password(request):
            owner = require_owner(request)
            self.auth.set_password('synthetic later password'); self.auth.save()
            return owner
        with patch('myapp.auth_views.require_owner', side_effect=change_password):
            response = self.client.post('/api/auth/validate/', HTTP_X_CSRFTOKEN=self.csrf)
        self.assertEqual(response.status_code, 401)
        self.assertNotIn('contextFence', response.json())

    def test_logout_after_initial_auth_blocks_feedback_write(self):
        analysis = Analysis.objects.create(user=self.profile, entry=self.entry, is_historical=True,
            mood='joy', color='pink', summary='synthetic', interpretation='synthetic', subject='synthetic')
        self.entry.analysis = analysis; self.entry.save(update_fields=['analysis'])
        def logout_after_auth(request):
            owner = require_owner(request)
            Session.objects.filter(session_key=owner.session_key).delete()
            return owner
        with patch('myapp.views.require_owner', side_effect=logout_after_auth):
            response = self.client.post('/api/feedback/submit/', {'analysis_id': str(analysis.pk), 'rating': 'good'},
                content_type='application/json', HTTP_X_CSRFTOKEN=self.csrf)
        self.assertEqual(response.status_code, 403)
        self.assertFalse(AnalysisFeedback.objects.exists())

    @override_settings(DREAM_ANALYZER_URL='http://synthetic-analyzer.invalid')
    def test_logout_during_external_analysis_blocks_commit_and_response(self):
        revision = self.client.get('/api/longitudinal/source-reset-preview/').json()['revision']
        def external(*args, **kwargs):
            _, _, capture = analysis_input(self.owner, [self.entry.pk], with_snapshot=True)
            Session.objects.filter(session_key=self.owner.session_key).delete()
            data = {'mood': 'joy', 'summary': 'synthetic late derived text', 'color': 'pink',
                'interpretation': 'synthetic late derived text', 'negative': False, 'subject': 'synthetic',
                'sentiment_score': 0, 'doctor_personality': 'Academic', 'weights': {}, 'symbols': [],
                'source_snapshot': capture, 'coverage': {'source_dependencies': capture['sources']}}
            return SimpleNamespace(status_code=200, json=lambda: data)
        with patch('myapp.views.requests.post', side_effect=external):
            response = self.client.patch(f'/api/entries/{self.entry.pk}/update/', {
                'content': 'synthetic accepted raw draft', 'expected_revision': revision,
                'expected_source': source_preimage(self.entry), 'expected_operation': 'update',
                'expected_session': self.owner.login_deadline.isoformat()}, content_type='application/json',
                HTTP_X_CSRFTOKEN=self.csrf, HTTP_X_DREAM_PRINCIPAL=str(self.profile.pk))
        self.assertEqual(response.status_code, 403)
        self.assertNotIn(b'synthetic late derived text', response.content)
        self.entry.refresh_from_db()
        self.assertEqual(self.entry.content, 'synthetic accepted raw draft')
        self.assertIsNone(self.entry.analysis_id)
        self.assertFalse(Analysis.objects.exists())


class SessionLockTests(TransactionTestCase):
    make_account = auth_helpers.AccountIsolationTests.make_account

    def test_session_that_expires_while_waiting_for_row_lock_is_denied(self):
        self.assertEqual(connection.vendor, 'postgresql', 'row-lock gate requires PostgreSQL')
        auth, profile, client, csrf = self.make_account('session-row-wait')
        data = client.get('/api/auth/session/').json()
        owner = OwnerContext(profile.pk, auth.pk, False,
            timezone.datetime.fromisoformat(data['loginDeadline']), client.session.session_key)
        expiry = timezone.now() + timedelta(seconds=0.6)
        Session.objects.filter(session_key=owner.session_key).update(expire_date=expiry)
        started, finished = Event(), Event()
        outcome = []
        def writer():
            close_old_connections()
            token = current_job.set(JobContext(owner, (), None, True))
            try:
                started.set()
                job_write(lambda profile: outcome.append('published'))
            except PermissionDenied:
                outcome.append('denied')
            except Exception as error:
                outcome.append(type(error).__name__)
            finally:
                current_job.reset(token); close_old_connections(); finished.set()
        with transaction.atomic():
            Session.objects.select_for_update().get(session_key=owner.session_key)
            thread = Thread(target=writer); thread.start()
            self.assertTrue(started.wait(2))
            # The SELECT begins with a fresh session but its lock wait crosses
            # expiry. Rechecking only the eight-hour deadline would publish.
            self.assertFalse(finished.wait(0.8))
        self.assertTrue(finished.wait(3))
        thread.join(timeout=1)
        self.assertEqual(outcome, ['denied'])
