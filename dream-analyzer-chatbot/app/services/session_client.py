"""Per-request fixed-origin credential forwarding; no database/signing secrets."""
from urllib.parse import urlsplit
import httpx
import json
from fastapi import HTTPException
from app.config import settings


class SessionClient:
    def __init__(self, headers):
        self.headers = {name: headers[name] for name in ('cookie', 'x-csrftoken', 'origin', 'referer', 'host', 'x-dream-principal', 'x-dream-session') if name in headers}
        self.backends = (settings.DJANGO_API_URL.rstrip('/'), settings.DREAM_ANALYZER_URL.rstrip('/'))
        for value in self.backends:
            parsed = urlsplit(value)
            if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise RuntimeError('Explicit trusted service URL required')
            if settings.DREAM_ENV not in {'local', 'test'} and parsed.scheme != 'https':
                raise RuntimeError('Production service URLs require HTTPS')
        self.http = httpx.AsyncClient(timeout=60, follow_redirects=False, headers=self.headers)
        self.principal = None

    async def validate(self):
        try:
            response = await self.http.post(self.backends[0] + '/auth/validate/')
        except httpx.HTTPError:
            raise HTTPException(503, 'Authentication unavailable.')
        if response.status_code in {401, 403}:
            raise HTTPException(response.status_code, 'Authentication required.')
        if response.status_code != 200:
            raise HTTPException(503, 'Authentication unavailable.')
        data = response.json()
        if ((self.headers.get('x-dream-principal') is not None and self.headers['x-dream-principal'] != data['user']['id'])
                or (self.headers.get('x-dream-session') is not None and self.headers['x-dream-session'] != data['loginDeadline'])):
            raise HTTPException(401, 'Authentication required.')
        fence = data.get('contextFence')
        if (not isinstance(fence, dict) or set(fence) != {'revision', 'settings_digest'}
                or not isinstance(fence['revision'], dict) or set(fence['revision']) != {'source', 'privacy', 'review'}
                or any(type(v) is not int or v < 0 for v in fence['revision'].values())
                or not isinstance(fence['settings_digest'], str) or len(fence['settings_digest']) != 64):
            raise HTTPException(503, 'Private context validation unavailable.')
        principal = (data['user']['id'], data['authUserId'], data['loginDeadline'],
                     json.dumps(data.get('contextFence'), sort_keys=True, separators=(',', ':')))
        if self.principal is not None and self.principal != principal:
            raise HTTPException(401, 'Authentication required.')
        self.principal = principal
        return data

    async def request(self, method, url, **kwargs):
        if not any(url == base or url.startswith(base + '/') for base in self.backends):
            raise HTTPException(400, 'Invalid service destination.')
        await self.validate()
        response = await self.http.request(method, url, **kwargs)
        if response.status_code in {401, 403}:
            raise HTTPException(response.status_code, 'Authentication required.')
        if 300 <= response.status_code < 400:
            raise HTTPException(503, 'Service unavailable.')
        return response

    async def get(self, url, **kwargs): return await self.request('GET', url, **kwargs)
    async def post(self, url, **kwargs): return await self.request('POST', url, **kwargs)
    async def put(self, url, **kwargs): return await self.request('PUT', url, **kwargs)
    async def delete(self, url, **kwargs): return await self.request('DELETE', url, **kwargs)
    async def aclose(self): await self.http.aclose()
