"""Analyzer-side Django session/CSRF validation without a Django HTTP callback."""
from django.http import HttpResponse
from django.core.handlers.wsgi import WSGIRequest
from io import BytesIO
from django.contrib.sessions.middleware import SessionMiddleware
from django.contrib.auth.middleware import AuthenticationMiddleware
from django.middleware.csrf import CsrfViewMiddleware
from django.db import connections, close_old_connections
from django.core.exceptions import DisallowedHost
from rest_framework.exceptions import APIException
from fastapi import HTTPException
from starlette.requests import Request
from starlette.concurrency import run_in_threadpool
from .owner import require_owner


def _validate(method, headers, scheme, address):
    close_old_connections()
    try:
        environ = {'REQUEST_METHOD': method, 'PATH_INFO': '/', 'SERVER_NAME': 'localhost',
                   'SERVER_PORT': '443' if scheme == 'https' else '80',
                   'wsgi.url_scheme': scheme, 'wsgi.input': BytesIO(), 'REMOTE_ADDR': address}
        for key, value in headers.items():
            environ['HTTP_' + key.upper().replace('-', '_')] = value
        request = WSGIRequest(environ)
        request.get_host()  # Validate even safe-method requests.
        response = lambda req: HttpResponse()
        SessionMiddleware(response).process_request(request)
        AuthenticationMiddleware(response).process_request(request)
        owner = require_owner(request)
        csrf = CsrfViewMiddleware(response)
        csrf.process_request(request)
        denial = csrf.process_view(request, response, (), {})
        if denial is not None:
            raise HTTPException(status_code=403, detail='CSRF verification failed.')
        return owner
    except APIException:
        raise HTTPException(status_code=401, detail='Authentication required.')
    except DisallowedHost:
        raise HTTPException(status_code=400, detail='Invalid host.')
    finally:
        connections.close_all()


async def authenticate_service_request(request: Request):
    headers = {name: request.headers[name] for name in (
        'cookie', 'x-csrftoken', 'origin', 'referer', 'host') if name in request.headers}
    try:
        return await run_in_threadpool(_validate, request.method, headers,
                                      request.scope['scheme'], request.client.host if request.client else '')
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail='Authentication unavailable.')
