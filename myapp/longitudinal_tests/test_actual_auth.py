"""Synthetic integration tests; run only with the actual additive auth models/helpers."""
from datetime import timedelta
from io import StringIO
from threading import Event, Thread
from unittest import skipUnless
from unittest.mock import patch
from uuid import uuid4
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.db import close_old_connections, connection
from django.test import Client, TransactionTestCase
from django.utils import timezone
from rest_framework.exceptions import PermissionDenied
from myapp.models import User, JournalEntry
from myapp.longitudinal import services
from myapp.longitudinal_models import AuditRun, DreamObservation, DreamReflection, EntryAudit, ProfileSnapshot, ThemeReview


@skipUnless(any(f.name == 'auth_user' for f in User._meta.fields), 'Requires task12 actual auth integration')
class ActualAuthTests(TransactionTestCase):
    def setUp(self):
        from myapp.security.contracts import OwnerContext
        self.auth = get_user_model().objects.create_user('synthetic-a', password='synthetic-password')
        self.user = User.objects.create(auth_user=self.auth, email='synthetic-a@example.invalid')
        self.owner = OwnerContext(self.user.pk, self.auth.pk, False, timezone.now() + timedelta(hours=1))
        self.client = Client(enforce_csrf_checks=True)
        csrf = self.client.get('/api/auth/csrf/').json()['csrfToken']
        response = self.client.post('/api/auth/login/', {'username': 'synthetic-a', 'password': 'synthetic-password'}, content_type='application/json', HTTP_X_CSRFTOKEN=csrf)
        self.assertEqual(response.status_code, 200)
        self.csrf = response.json()['csrfToken']
        self.headers = {'HTTP_X_CSRFTOKEN': self.csrf, 'HTTP_X_DREAM_PRINCIPAL': str(self.user.pk)}

    def run_fixture(self):
        services.set_preferences(self.owner, enabled=True)
        entries = [JournalEntry.objects.create(user=self.user, content=text) for text in ('A river.', 'The river.')]
        for entry in entries:
            services.set_entry_preferences(self.owner, entry.pk, origin='user_reported', language='en', included=True)
        return services.queue_audit(self.owner, [e.pk for e in entries], str(uuid4()))

    def test_actual_session_csrf_owner_fence_and_private_cors_contract(self):
        revision = self.client.get('/api/longitudinal/preferences/', **self.headers).json()['revision']
        self.assertEqual(self.client.patch('/api/longitudinal/preferences/', {'enabled': True, 'expected_revision': revision}, content_type='application/json', HTTP_X_DREAM_PRINCIPAL=str(self.user.pk)).status_code, 403)
        response = self.client.patch('/api/longitudinal/preferences/', {'enabled': True, 'expected_revision': revision}, content_type='application/json', **self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['X-Dream-Principal'], str(self.user.pk))
        self.assertIn('no-store', response['Cache-Control'])
        preflight = self.client.options('/api/longitudinal/profile/', HTTP_ORIGIN='http://localhost:3100', HTTP_ACCESS_CONTROL_REQUEST_METHOD='GET', HTTP_ACCESS_CONTROL_REQUEST_HEADERS='x-dream-principal')
        self.assertIn('x-dream-principal', preflight['Access-Control-Allow-Headers'])
        response = self.client.get('/api/longitudinal/profile/', HTTP_ORIGIN='http://localhost:3100', **self.headers)
        self.assertIn('X-Dream-Principal', response['Access-Control-Expose-Headers'])
        other_auth = get_user_model().objects.create_user('synthetic-b')
        other = User.objects.create(auth_user=other_auth, email='synthetic-b@example.invalid')
        stolen = JournalEntry.objects.create(user=other, content='Private synthetic B record')
        self.assertEqual(self.client.get(f'/api/entries/{stolen.pk}/audit/', **self.headers).status_code, 404)
        self.assertEqual(self.client.post('/api/longitudinal/reflections/', {'text': 'Synthetic A draft'}, content_type='application/json', HTTP_X_CSRFTOKEN=self.csrf, HTTP_X_DREAM_PRINCIPAL=str(other.pk)).status_code, 403)
        self.assertFalse(DreamReflection.objects.exists())

    def test_accepted_job_outlives_logout_but_revoked_actor_cannot_commit(self):
        run = self.run_fixture()
        self.client.post('/api/auth/logout/', **self.headers)
        self.assertEqual(services.process_run(run.pk).status, 'complete')
        self.assertEqual(services.confirmed_context(self.owner)['status'], 'disabled')
        second = services.queue_audit(self.owner, [e.pk for e in JournalEntry.objects.filter(user=self.user)], str(uuid4()))
        self.auth.is_active = False
        self.auth.save(update_fields=['is_active'])
        with self.assertRaises(PermissionDenied):
            services.process_run(second.pk)
        second.refresh_from_db()
        self.assertEqual(second.status, 'queued')

    def test_crash_rolls_back_entire_local_job_then_worker_recovers(self):
        run = self.run_fixture()
        original = services.core.extract_local
        calls = []
        def crash(content, **kwargs):
            calls.append(content)
            if len(calls) == 2:
                raise RuntimeError('synthetic worker crash')
            return original(content, **kwargs)
        with patch.object(services.core, 'extract_local', side_effect=crash):
            with self.assertRaises(RuntimeError):
                services.process_run(run.pk)
        run.refresh_from_db()
        self.assertEqual(run.status, 'queued')
        self.assertFalse(DreamObservation.objects.exists())
        self.assertFalse(ProfileSnapshot.objects.exists())
        call_command('process_dream_audits', limit=1, stdout=StringIO())
        run.refresh_from_db()
        self.assertEqual(run.status, 'complete')
        self.assertEqual(DreamObservation.objects.count(), 2)

    def test_revoked_old_jobs_do_not_starve_new_valid_job(self):
        valid = self.run_fixture()
        from myapp.security.contracts import OwnerContext
        for index in range(10):
            auth = get_user_model().objects.create_user(f'revoked-{index}')
            user = User.objects.create(auth_user=auth, email=f'revoked-{index}@example.invalid')
            owner = OwnerContext(user.pk, auth.pk, False, self.owner.login_deadline)
            services.set_preferences(owner, enabled=True)
            entry = JournalEntry.objects.create(user=user, content='A river.')
            services.set_entry_preferences(owner, entry.pk, included=True, origin='user_reported', language='en')
            stale = services.queue_audit(owner, [entry.pk], str(uuid4()))
            AuditRun.objects.filter(pk=stale.pk).update(created_at=valid.created_at - timedelta(days=1))
            auth.is_active = False
            auth.save(update_fields=['is_active'])
        call_command('process_dream_audits', limit=1, stdout=StringIO())
        valid.refresh_from_db()
        self.assertEqual(valid.status, 'complete')
        self.assertEqual(DreamObservation.objects.filter(user=self.user).count(), 2)
        self.assertEqual(DreamObservation.objects.exclude(user=self.user).count(), 0)

    def test_privacy_withdrawal_waits_for_worker_then_removes_derived_commit(self):
        if connection.vendor != 'postgresql':
            self.skipTest('Row locking requires PostgreSQL')
        run = self.run_fixture()
        entered, release, done = Event(), Event(), Event()
        errors = []
        original = services.core.extract_local
        def blocked(content, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('synthetic timeout')
            return original(content, **kwargs)
        def worker():
            close_old_connections()
            try:
                services.process_run(run.pk)
            except BaseException as error:
                errors.append(error)
            finally:
                close_old_connections()
        def withdraw():
            close_old_connections()
            try:
                services.set_preferences(self.owner, enabled=False)
                done.set()
            except BaseException as error:
                errors.append(error)
            finally:
                close_old_connections()
        with patch.object(services.core, 'extract_local', side_effect=blocked):
            first = Thread(target=worker); first.start()
            self.assertTrue(entered.wait(3))
            second = Thread(target=withdraw); second.start()
            self.assertFalse(done.wait(.1))
            release.set(); first.join(5); second.join(5)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(done.is_set())
        self.assertFalse(EntryAudit.objects.exists())
        self.assertFalse(DreamObservation.objects.exists())
        self.assertFalse(ProfileSnapshot.objects.exists())
        self.assertFalse(ThemeReview.objects.exists())
        self.assertEqual(JournalEntry.objects.count(), 2)
