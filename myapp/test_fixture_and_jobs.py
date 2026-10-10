from datetime import datetime, timedelta, timezone
from uuid import uuid4
from django.test import TestCase, TransactionTestCase, override_settings
from django.core.management import call_command
from django.core.management.base import CommandError
from django.contrib.auth import get_user_model
from rest_framework.exceptions import PermissionDenied
from django.http import Http404
from myapp.models import User, JournalEntry, WorkflowExecution
from myapp.security.contracts import OwnerContext
from myapp.security.jobs import current_job, JobContext, job_write
from myapp.security.owner import owned_atomic_write
from django.db import connections, close_old_connections, transaction, OperationalError
from threading import Event, Thread


class FixtureTests(TestCase):
    def test_fixture_is_rejected_without_explicit_local_configuration(self):
        with self.assertRaises(CommandError): call_command('create_local_admin_fixture', confirm_local_only=True)
        self.assertFalse(get_user_model().objects.filter(username='admin').exists())

    @override_settings(DREAM_ENV='local', DREAM_ENABLE_LOCAL_FIXTURE=True)
    def test_synthetic_fixture_hashes_password_and_grants_only_domain_view_permissions(self):
        call_command('create_local_admin_fixture', confirm_local_only=True, verbosity=0)
        auth=get_user_model().objects.get(username='admin')
        self.assertTrue(auth.check_password('admin'));self.assertNotEqual(auth.password,'admin')
        self.assertFalse(auth.is_superuser)
        self.assertTrue(all(permission.codename.startswith('view_') and permission.content_type.app_label=='myapp'
                            for permission in auth.user_permissions.all()))
        self.assertEqual(auth.user_permissions.count(),7)
        original=auth.password
        with self.assertRaises(CommandError): call_command('create_local_admin_fixture', confirm_local_only=True)
        auth.refresh_from_db();self.assertEqual(auth.password,original)


class JobCommitTests(TestCase):
    def setUp(self):
        self.auth=get_user_model().objects.create_user('job-owner',password='synthetic-password')
        self.owner=User.objects.create(auth_user=self.auth,email='job-owner@example.invalid')
        self.entry=JournalEntry.objects.create(user=self.owner,content='synthetic source')
        self.workflow=WorkflowExecution.objects.create(user=self.owner,accepted_auth_user_id=self.auth.pk,
            workflow_type='test',routine_name='synthetic accepted job')
        owner=OwnerContext(self.owner.pk,self.auth.pk,False,datetime.now(timezone.utc)+timedelta(hours=1))
        self.token=current_job.set(JobContext(owner,(self.entry.pk,)))
    def tearDown(self): current_job.reset(self.token)
    def write(self):
        return job_write(lambda profile: JournalEntry.objects.create(user=profile,content='synthetic result'),self.workflow.pk)
    def test_deleted_or_reassigned_source_prevents_job_commit(self):
        self.entry.delete()
        with self.assertRaises(Http404): self.write()
        self.assertFalse(JournalEntry.objects.filter(content='synthetic result').exists())
    def test_relinked_auth_identity_or_withdrawn_import_state_prevents_commit(self):
        replacement=get_user_model().objects.create_user('replacement',password='synthetic-password')
        self.owner.auth_user=replacement;self.owner.save(update_fields=['auth_user'])
        with self.assertRaises(PermissionDenied): self.write()
        self.owner.auth_user=self.auth;self.owner.import_ready=False;self.owner.save(update_fields=['auth_user','import_ready'])
        with self.assertRaises(PermissionDenied): self.write()
        self.assertFalse(JournalEntry.objects.filter(content='synthetic result').exists())
    def test_historical_or_foreign_workflow_cannot_authorize_write(self):
        self.workflow.is_historical=True;self.workflow.save(update_fields=['is_historical'])
        with self.assertRaises(PermissionDenied): self.write()


class ConcurrentCommitTests(TransactionTestCase):
    def test_auth_owner_source_hook_and_workflow_locks_are_held_through_commit(self):
        auth = get_user_model().objects.create_user('concurrent-owner', password='synthetic-password')
        owner = User.objects.create(auth_user=auth, email='concurrent@example.invalid')
        entry = JournalEntry.objects.create(user=owner, content='synthetic original')
        workflow = WorkflowExecution.objects.create(user=owner, accepted_auth_user_id=auth.pk,
            routine_name='synthetic concurrent job', workflow_type='test')
        acquired, release = Event(), Event()
        errors = []

        def run():
            close_old_connections()
            try:
                def lock_source(profile):
                    JournalEntry.objects.select_for_update().get(pk=entry.pk, user=profile)
                def write(profile):
                    acquired.set()
                    if not release.wait(5):
                        raise AssertionError('Synthetic concurrent test timed out')
                    JournalEntry.objects.filter(pk=entry.pk, user=profile).update(content='synthetic committed')
                owned_atomic_write(owner_id=owner.pk, expected_auth_user_id=auth.pk,
                    workflow_id=workflow.pk, lock_state=lock_source, write=write)
            except Exception as exc:
                errors.append(exc)
            finally:
                connections.close_all()

        worker = Thread(target=run)
        worker.start()
        try:
            self.assertTrue(acquired.wait(5))
            for queryset, update in [
                (get_user_model().objects.filter(pk=auth.pk), {'is_active': False}),
                (User.objects.filter(pk=owner.pk), {'import_ready': False}),
                (JournalEntry.objects.filter(pk=entry.pk), {'content': 'synthetic racing mutation'}),
                (WorkflowExecution.objects.filter(pk=workflow.pk), {'is_historical': True}),
            ]:
                with self.assertRaises(OperationalError):
                    with transaction.atomic():
                        with connections['default'].cursor() as cursor:
                            cursor.execute("SET LOCAL lock_timeout = '100ms'")
                        queryset.update(**update)
        finally:
            release.set()
            worker.join(6)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        entry.refresh_from_db()
        self.assertEqual(entry.content, 'synthetic committed')
        get_user_model().objects.filter(pk=auth.pk).update(is_active=False)
        with self.assertRaises(PermissionDenied):
            owned_atomic_write(owner_id=owner.pk, expected_auth_user_id=auth.pk, workflow_id=workflow.pk,
                write=lambda profile: JournalEntry.objects.filter(pk=entry.pk).update(content='synthetic rejected'))
        entry.refresh_from_db()
        self.assertEqual(entry.content, 'synthetic committed')
