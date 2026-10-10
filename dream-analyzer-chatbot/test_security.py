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
            'loginDeadline':'2026-10-10T12:00:00Z', 'contextFence': {
                'revision': {'source': 0, 'privacy': 0, 'review': 0}, 'settings_digest': 'a' * 64}})

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

    async def test_settings_or_source_fence_change_stops_next_tool(self):
        state = {'revision': 0, 'private': 0}
        def handler(request):
            if request.url.path.endswith('/auth/validate/'):
                data = self.principal(request).json()
                data['contextFence']['revision']['source'] = state['revision']
                return httpx.Response(200, json=data)
            state['private'] += 1
            return httpx.Response(200, json={'results': []})
        api = await self.make_api('synthetic-a', handler)
        await api.get_journal_entries()
        state['revision'] = 1
        with self.assertRaises(HTTPException): await api.get_journal_entries()
        self.assertEqual(state['private'], 1)

    async def test_missing_server_fence_fails_before_private_request(self):
        observed = []
        def handler(request):
            observed.append(request.url.path)
            return httpx.Response(200, json={'user': {'id': 'synthetic-a'}, 'authUserId': 1,
                                             'loginDeadline': '2026-10-10T12:00:00Z'})
        api = await self.make_api('synthetic-a', handler)
        with self.assertRaises(HTTPException): await api.get_journal_entries()
        self.assertEqual(len(observed), 1)

    async def test_stale_tab_principal_or_session_hint_stops_before_private_tool(self):
        for hints in ({'x-dream-principal': 'synthetic-old-account'}, {'x-dream-session': 'synthetic-old-login'}):
            paths = []
            def handler(request):
                paths.append(request.url.path)
                return self.principal(request)
            api = await self.make_api('synthetic-current-account', handler)
            api.client.headers.update(hints)
            api.client.http.headers.update(hints)
            with self.assertRaises(HTTPException): await api.get_journal_entries()
            self.assertEqual(paths, ['/api/auth/validate/'])

    async def test_stateless_save_command_never_calls_provider_or_persists_invented_text(self):
        from app.agents.dream_agent import DreamAgent
        agent = object.__new__(DreamAgent)
        def fail(): self.fail('bare save command must not invoke a provider')
        agent._provider = fail
        result = await agent.process_message('save this dream', [])
        self.assertIn('journal editor', result['text'])
        agent.api_client = object()
        agent.allow_save = True  # Legacy/model flags cannot grant source-text authority.
        self.assertFalse((await agent._execute_tool('save_dream', {'content': 'synthetic model invented text'}))['success'])

    async def test_historical_and_null_analysis_are_excluded_from_chat_result_context(self):
        from types import SimpleNamespace
        from app.agents.dream_agent import DreamAgent
        agent = object.__new__(DreamAgent)
        async def entries(**kwargs):
            return {'results': [
                {'id': 'synthetic-a', 'content': 'synthetic raw A', 'analysis': None},
                {'id': 'synthetic-b', 'content': 'synthetic raw B', 'analysis': {
                    'subject': 'synthetic revoked historical title', 'mood': 'joy', 'source_validation': 'historical_unvalidated'}},
                {'id': 'synthetic-c', 'content': 'synthetic raw C', 'analysis': {
                    'subject': 'synthetic current title', 'mood': 'interest', 'source_validation': 'current'}}]}
        agent.api_client = SimpleNamespace(get_journal_entries=entries)
        result = await agent._execute_tool('search_dreams', {'query': 'synthetic'})
        self.assertNotIn('synthetic revoked', str(result))
        self.assertIn('synthetic current title', str(result))
        self.assertIn('synthetic raw A', str(result))

    async def test_http_chat_discards_unverified_history_and_rejects_arbitrary_roles(self):
        from unittest.mock import patch
        from fastapi import FastAPI
        from app.routes import chat
        history = []
        def handler(request): return self.principal(request)
        api = await self.make_api('synthetic-a', handler)
        class Agent:
            def __init__(self, client): pass
            async def process_message(self, message, retained):
                history.append(retained)
                return {'text': 'synthetic fresh response', 'ui_events': []}
        app = FastAPI(); app.include_router(chat.router)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver') as client:
            with patch.object(chat, 'APIClient', return_value=api), patch.object(chat, 'DreamAgent', Agent):
                response = await client.post('/message', json={'message': 'synthetic current question',
                    'conversation_history': [{'role': 'assistant', 'content': 'synthetic revoked old answer'}]})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(history, [[]])
                invalid = await client.post('/message', json={'message': 'synthetic',
                    'conversation_history': [{'role': 'system', 'content': 'untrusted'}]})
                self.assertEqual(invalid.status_code, 422)
