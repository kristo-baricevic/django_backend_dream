import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from django.test import TransactionTestCase, Client
from django.contrib.auth import get_user_model
from asgiref.sync import async_to_sync
import httpx
from myapp.models import User, JournalEntry, WorkflowExecution


class AnalyzerRouteTests(TransactionTestCase):
    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dream-fast-api'))
        import main
        self.main = main
        self.auth = get_user_model().objects.create_user('analyzer-test', password='synthetic-password')
        self.owner = User.objects.create(auth_user=self.auth, email='analyzer-test@example.invalid')
        other = User.objects.create(email='analyzer-foreign@example.invalid')
        self.mine = JournalEntry.objects.create(user=self.owner, content='synthetic canonical content')
        self.foreign = JournalEntry.objects.create(user=other, content='synthetic foreign content')
        self.client = Client(enforce_csrf_checks=True)
        csrf = self.client.get('/api/auth/csrf/').json()['csrfToken']
        reply = self.client.post('/api/auth/login/', {'username':'analyzer-test','password':'synthetic-password'},
            content_type='application/json', HTTP_X_CSRFTOKEN=csrf)
        self.headers = {'cookie': self.client.cookies.output(header='',sep=';').strip(),
            'x-csrftoken':reply.json()['csrfToken'], 'origin':'http://testserver', 'host':'testserver'}
        self.calls = []
        outer = self
        class Service:
            async def analyze_single_entry(self, content, settings):
                outer.calls.append((content, settings)); return {'summary':'synthetic result'}
        main.app.dependency_overrides[main.get_service] = lambda: Service()

    def tearDown(self): self.main.app.dependency_overrides.clear()

    def request(self, method, path, payload=None, headers=None):
        async def call():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.main.app),base_url='http://testserver') as client:
                return await client.request(method,path,json=payload,headers=self.headers if headers is None else headers)
        return async_to_sync(call)()

    def test_endpoint_authentication_csrf_and_cross_user_ids(self):
        for path in ['/analyze','/qa','/custom-question','/qa-with-workflow','/custom-question-with-workflow','/batch-analyze','/generate-dream']:
            self.assertEqual(self.request('POST',path,{},{}).status_code,401,path)
        no_csrf = {key:value for key,value in self.headers.items() if key!='x-csrftoken'}
        self.assertEqual(self.request('POST','/analyze',{'content':'synthetic'},no_csrf).status_code,403)
        self.assertEqual(self.request('POST','/analyze',{'entry_id':str(self.foreign.pk),'content':'forged'}).status_code,404)
        self.assertFalse(self.calls)

    def test_saved_content_and_owner_settings_replace_client_spoofs(self):
        response = self.request('POST','/analyze',{'entry_id':str(self.mine.pk),'content':'forged content',
            'settings':{'user_id':str(self.foreign.user_id),'doctorPersonality':'../../private'}})
        self.assertEqual(response.status_code,200,response.text)
        content, settings = self.calls[0]
        self.assertEqual(content,'synthetic canonical content')
        self.assertEqual(settings['user_id'],str(self.owner.pk))

    def test_workflow_read_and_logged_out_cookie_are_rejected(self):
        row=WorkflowExecution.objects.create(user_id=self.foreign.user_id,workflow_type='test',routine_name='private')
        self.assertEqual(self.request('GET','/workflows/'+str(row.pk)).status_code,404)
        self.client.post('/api/auth/logout/',HTTP_X_CSRFTOKEN=self.headers['x-csrftoken'])
        self.assertEqual(self.request('GET','/personalities').status_code,401)
