import os
os.environ.setdefault('DJANGO_API_URL', 'https://django.example.invalid/api')
os.environ.setdefault('DREAM_ANALYZER_URL', 'https://analyzer.example.invalid')
import unittest
import httpx
from fastapi import HTTPException
from app.services.drf_client import APIClient
from app.services.session_client import SessionClient
from app.config import settings


class ChatSessionTests(unittest.IsolatedAsyncioTestCase):
    async def make_api(self, cookie, handler):
        api = APIClient({'cookie':cookie, 'x-csrftoken':'synthetic-csrf', 'origin':'https://browser.example.invalid',
                         'authorization':'untrusted', 'x-owner-id':'forged'})
        await api.client.http.aclose()
        api.client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler), headers=api.client.headers,
                                            follow_redirects=False)
        self.addAsyncCleanup(api.close)
        return api

    def principal(self, request, auth_id=1):
        return httpx.Response(200,json={'user':{'id':request.headers['cookie']},'authUserId':auth_id,
                                       'loginDeadline':'2026-10-10T12:00:00Z'})

    async def test_each_tool_call_has_separate_cookie_and_validation(self):
        observed=[]
        def handler(request):
            self.assertNotIn('authorization',request.headers)
            self.assertNotIn('x-owner-id',request.headers)
            observed.append((request.url.path,request.headers['cookie']))
            if request.url.path.endswith('/auth/validate/'): return self.principal(request)
            return httpx.Response(200,json={'results':[]})
        a=await self.make_api('synthetic-a',handler);b=await self.make_api('synthetic-b',handler)
        await a.get_journal_entries();await b.get_journal_entries();await a.get_journal_entries()
        self.assertEqual([cookie for path,cookie in observed if path.endswith('/entries/')],['synthetic-a','synthetic-b','synthetic-a'])
        self.assertEqual(sum(path.endswith('/auth/validate/') for path,cookie in observed),3)

    async def test_logout_and_account_relink_block_next_tool_before_private_request(self):
        state={'active':True,'auth_id':1,'private':0}
        def handler(request):
            if request.url.path.endswith('/auth/validate/'):
                return self.principal(request,state['auth_id']) if state['active'] else httpx.Response(401)
            state['private']+=1;return httpx.Response(200,json={'results':[]})
        api=await self.make_api('synthetic-a',handler);await api.get_journal_entries()
        state['auth_id']=2
        with self.assertRaises(HTTPException) as denied: await api.get_journal_entries()
        self.assertEqual(denied.exception.status_code,401);self.assertEqual(state['private'],1)
        state['active']=False
        with self.assertRaises(HTTPException): await api.get_journal_entries()
        self.assertEqual(state['private'],1)

    async def test_redirect_and_untrusted_destination_never_receive_forwarded_cookie(self):
        observed=[]
        def handler(request):
            observed.append(request.url.host)
            if request.url.path.endswith('/auth/validate/'): return self.principal(request)
            return httpx.Response(302,headers={'location':'https://external.example.invalid/'})
        api=await self.make_api('synthetic-a',handler)
        with self.assertRaises(HTTPException): await api.get_journal_entries()
        with self.assertRaises(HTTPException): await api.client.get('https://external.example.invalid/')
        self.assertTrue(all(host=='django.example.invalid' for host in observed))

    async def test_production_rejects_plain_http_service_urls(self):
        old=settings.DJANGO_API_URL
        try:
            settings.DJANGO_API_URL='http://django.example.invalid/api'
            with self.assertRaises(RuntimeError): SessionClient({'cookie':'synthetic-a'})
        finally: settings.DJANGO_API_URL=old
