from asgiref.sync import async_to_sync
from datetime import datetime, timezone
from uuid import uuid4
from django.contrib.auth import get_user_model
from django.test import Client, TransactionTestCase
from rest_framework.exceptions import PermissionDenied
from starlette.requests import Request
from fastapi import HTTPException
from myapp.models import User, WorkflowExecution, JournalEntry
from myapp.security.service_auth import authenticate_service_request
from myapp.security.owner import owned_atomic_write, load_owned_entries, require_owner


class ServiceBoundaryTests(TransactionTestCase):
    def setUp(self):
        self.auth = get_user_model().objects.create_user('service',password='synthetic-password')
        self.owner = User.objects.create(auth_user=self.auth,email='service@example.invalid')
        self.client = Client(enforce_csrf_checks=True)
        token = self.client.get('/api/auth/csrf/').json()['csrfToken']
        self.token = self.client.post('/api/auth/login/',{'username':'service','password':'synthetic-password'},
            content_type='application/json',HTTP_X_CSRFTOKEN=token).json()['csrfToken']

    def service_request(self, method='POST', scheme='http', **changes):
        headers = {'host':'testserver','cookie':self.client.cookies.output(header='',sep=';').strip(),
                   'x-csrftoken':self.token,'origin':'http://testserver'}
        for key,value in changes.items():
            if value is None:headers.pop(key,None)
            else:headers[key]=value
        return Request({'type':'http','method':method,'scheme':scheme,'path':'/private',
            'headers':[(k.encode(),v.encode()) for k,v in headers.items()],
            'client':('127.0.0.1',1234),'server':('testserver',80)})

    def test_direct_service_requires_real_session_and_csrf(self):
        owner = async_to_sync(authenticate_service_request)(self.service_request())
        self.assertEqual(owner.domain_user_id,self.owner.id)
        for change,want in [({'cookie':''},401),({'x-csrftoken':''},403),
                            ({'origin':'https://untrusted.example.invalid'},403),
                            ({'host':'untrusted.example.invalid'},400)]:
            with self.subTest(change=list(change)):
                with self.assertRaises(HTTPException) as raised:
                    async_to_sync(authenticate_service_request)(self.service_request(**change))
                self.assertEqual(raised.exception.status_code,want)

    def test_https_without_origin_requires_trusted_referer(self):
        with self.assertRaises(HTTPException) as raised:
            async_to_sync(authenticate_service_request)(self.service_request(scheme='https',origin=None))
        self.assertEqual(raised.exception.status_code,403)

    def test_safe_request_authenticates_without_csrf_token(self):
        owner = async_to_sync(authenticate_service_request)(self.service_request('GET',**{'x-csrftoken':None,'origin':None}))
        self.assertEqual(owner.domain_user_id,self.owner.id)

    def test_service_revalidates_logout_and_deactivation(self):
        self.client.post('/api/auth/logout/',HTTP_X_CSRFTOKEN=self.token)
        with self.assertRaises(HTTPException) as raised:
            async_to_sync(authenticate_service_request)(self.service_request('GET'))
        self.assertEqual(raised.exception.status_code,401)

    def test_worker_commit_requires_original_active_auth_linkage(self):
        workflow = WorkflowExecution.objects.create(user=self.owner,accepted_auth_user_id=self.auth.id,
            workflow_type='synthetic',routine_name='synthetic')
        write = lambda profile: JournalEntry.objects.create(user=profile,content='synthetic worker result')
        self.auth.is_active=False;self.auth.save(update_fields=['is_active'])
        with self.assertRaises(PermissionDenied):
            owned_atomic_write(owner_id=self.owner.id,expected_auth_user_id=self.auth.id,
                               workflow_id=workflow.id,write=write)
        self.assertEqual(JournalEntry.objects.count(),0)
        self.auth.is_active=True;self.auth.save(update_fields=['is_active'])
        other = get_user_model().objects.create_user('relinked',password='synthetic-password')
        self.owner.auth_user=other;self.owner.save(update_fields=['auth_user'])
        with self.assertRaises(PermissionDenied):
            owned_atomic_write(owner_id=self.owner.id,expected_auth_user_id=self.auth.id,
                               workflow_id=workflow.id,write=write)
        self.assertEqual(JournalEntry.objects.count(),0)
